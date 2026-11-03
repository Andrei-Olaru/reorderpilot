"""Eval A: baselines vs model on the TEST cutoffs (touched only here), calibration plot, abstention curve.

Run after training: `python eval/run_model_eval.py`.
Writes artifacts/metrics.json and eval/results/{model_eval.md, model_eval.json, calibration.png,
abstention_curve.png, abstention_curve.csv}.
"""

from __future__ import annotations

import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from reorderpilot.config import (  # noqa: E402
    ARTIFACTS_DIR,
    EVAL_RESULTS_DIR,
    MIN_PRIOR_ORDERS,
    TEST_CUTOFFS,
    get_settings,
)
from reorderpilot.data.loader import load_transactions  # noqa: E402
from reorderpilot.features import build_dataset, customers_ordering_in_window  # noqa: E402
from reorderpilot.model.baselines import BASELINES  # noqa: E402
from reorderpilot.model.evaluate import (  # noqa: E402
    abstention_curve,
    calibration_table,
    pair_metrics,
    top_k,
    topk_metrics,
)
from reorderpilot.model.predict import load_bundle  # noqa: E402
from reorderpilot.model.train import TAU_GRID  # noqa: E402

K = 5
# Reference palette (dataviz skill): categorical slots 1-2, light surface, muted ink, hairline grid.
BLUE, ORANGE, SURFACE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def evaluate_methods(test: pd.DataFrame, prob: pd.Series, raw: pd.Series) -> dict:
    """Ranking + pair metrics for B1-B3 and the model on the same (eligible) candidate pairs."""
    out = {}
    for name, (score_fn, eligible_fn) in BASELINES.items():
        score = score_fn(test)
        eligible = eligible_fn(test) if eligible_fn else None
        out[name] = {**topk_metrics(test, score, K, eligible), **pair_metrics(test["label"], score)}
    out["HGB_calibrated"] = {**topk_metrics(test, prob, K), **pair_metrics(test["label"], prob, prob=True),
                             "brier_uncalibrated": pair_metrics(test["label"], raw, prob=True)["brier"]}
    return out


def per_cutoff_precision(test: pd.DataFrame, prob: pd.Series) -> dict:
    """precision@5 per test month for the model and B2 (stability check)."""
    res = {}
    for c, idx in test.groupby("cutoff").groups.items():
        sub = test.loc[idx]
        res[str(pd.Timestamp(c).date())] = {
            "HGB_calibrated": topk_metrics(sub, prob.loc[idx], K)[f"precision@{K}"],
            "B2_most_frequent": topk_metrics(sub, BASELINES["B2_most_frequent"][0](sub), K)[f"precision@{K}"],
            "n_queries": topk_metrics(sub, prob.loc[idx], K)["n_queries"],
        }
    return res


def plot_calibration(test: pd.DataFrame, prob: pd.Series, raw: pd.Series, kind: str, path) -> None:
    """Reliability diagram: mean predicted vs observed re-order rate in 20 equal-size bins."""
    cal, cal_raw = calibration_table(test["label"], prob, 20), calibration_table(test["label"], raw, 20)
    top = max(cal["mean_pred"].max(), cal["observed"].max(), cal_raw["mean_pred"].max()) * 1.05
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=SURFACE)
    ax.plot([0, top], [0, top], color=MUTED, linewidth=1, linestyle="--")
    ax.plot(cal_raw["mean_pred"], cal_raw["observed"], color=ORANGE, linewidth=2, marker="o", markersize=5,
            label="raw model")
    ax.plot(cal["mean_pred"], cal["observed"], color=BLUE, linewidth=2, marker="o", markersize=5,
            label=f"calibrated ({kind}, fit on validation)")
    ax.annotate("perfect calibration", (top * 0.62, top * 0.55), color=MUTED, fontsize=9, rotation=38)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    _style(ax, "Calibration on test cutoffs (20 equal-size bins)", "mean predicted probability",
           "observed re-order rate")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_abstention(curve: pd.DataFrame, tau: float, b2_point: tuple[float, float], path) -> None:
    """Precision of recommended items vs fraction of customers covered, as tau sweeps."""
    c = curve.dropna(subset=["precision"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=SURFACE)
    ax.plot(c["coverage"], c["precision"], color=BLUE, linewidth=2, marker="o", markersize=4,
            label="model, top-5 items with p ≥ τ")
    chosen = c.loc[(c["tau"] - tau).abs().idxmin()]
    ax.plot(chosen["coverage"], chosen["precision"], "o", color=BLUE, markersize=10, markeredgecolor=SURFACE,
            markeredgewidth=2)
    ax.annotate(f"chosen τ = {tau:.2f}", (chosen["coverage"], chosen["precision"]), textcoords="offset points",
                xytext=(10, 6), color=INK, fontsize=9)
    ax.plot(*b2_point, "s", color=ORANGE, markersize=8, label="B2 most-frequent, top-5 (no abstention)")
    for t in (0.2, 0.4, 0.6):
        row = c.loc[(c["tau"] - t).abs().idxmin()]
        ax.annotate(f"τ={row['tau']:.1f}", (row["coverage"], row["precision"]), textcoords="offset points",
                    xytext=(8, 6), color=MUTED, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax, "Abstention trade-off on test cutoffs", "fraction of customers covered (incl. cold-start)",
           "precision of recommended items")
    ax.legend(frameon=False, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def markdown_table(methods: dict) -> str:
    """Methods x metrics markdown table (Brier only for the probabilistic model)."""
    cols = [f"precision@{K}", f"recall@{K}", f"hit_rate@{K}", "pr_auc", "roc_auc", "brier"]
    lines = ["| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in methods.items():
        cells = [f"{m[c]:.3f}" if c in m else "n/a" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> dict:
    tx = load_transactions()
    bundle = load_bundle()
    train_report = json.loads((ARTIFACTS_DIR / "train_report.json").read_text())

    full = build_dataset(tx, TEST_CUTOFFS, min_prior_orders=0)       # every customer with any prior order
    n_customers_total = full.groupby(["cutoff", "customer_id"]).ngroups
    test = full[full["cust_n_orders"] >= MIN_PRIOR_ORDERS].reset_index(drop=True)
    raw = pd.Series(bundle.model.predict_proba(test[bundle.features].astype(float))[:, 1])
    prob = pd.Series(bundle.predict_proba(test))

    methods = evaluate_methods(test, prob, raw)
    curve = abstention_curve(test, prob, TAU_GRID, K, n_customers_total=n_customers_total)
    chosen = curve.loc[(curve["tau"] - bundle.tau).abs().idxmin()].to_dict()

    # Population facts, so the reader knows who is (not) in the per-customer metrics.
    ordering = sum(len(customers_ordering_in_window(tx, c)) for c in TEST_CUTOFFS)
    pos_queries = int((test.groupby(["cutoff", "customer_id"])["label"].sum() > 0).sum())
    cold = full[full["cust_n_orders"] < MIN_PRIOR_ORDERS]
    population = {
        "customer_cutoff_pairs_with_history": int(n_customers_total),
        "eligible_pairs_(>=3_orders)": int(test.groupby(["cutoff", "customer_id"]).ngroups),
        "cold_start_pairs_abstained": int(cold.groupby(["cutoff", "customer_id"]).ngroups),
        "customers_ordering_anything_in_window": int(ordering),
        "eligible_customers_reordering_>=1_prior_product": pos_queries,
        "n_test_pairs": int(len(test)), "test_base_rate": float(test["label"].mean()),
    }

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_calibration(test, prob, raw, train_report["calibration"]["chosen"], EVAL_RESULTS_DIR / "calibration.png")
    b2_point = (population["eligible_pairs_(>=3_orders)"] / n_customers_total,
                float(top_k_precision_all(test, BASELINES["B2_most_frequent"][0](test))))
    plot_abstention(curve, bundle.tau, b2_point, EVAL_RESULTS_DIR / "abstention_curve.png")
    curve.to_csv(EVAL_RESULTS_DIR / "abstention_curve.csv", index=False)

    metrics = {
        "generated_by": "eval/run_model_eval.py", "as_of_default": str(get_settings().as_of_date),
        "test_cutoffs": [str(c) for c in TEST_CUTOFFS], "k": K,
        "population": population,
        "test": methods,
        "test_precision_at_5_by_cutoff": per_cutoff_precision(test, prob),
        "abstention_at_chosen_tau": {k: (float(v) if isinstance(v, int | float) else v) for k, v in chosen.items()},
        "validation": {"best_params": train_report["best_params"], "calibration": train_report["calibration"],
                       "tau": train_report["tau"], "tau_rule": train_report["tau_rule"],
                       "metrics_calibrated": train_report["valid_metrics_calibrated"]},
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (EVAL_RESULTS_DIR / "model_eval.json").write_text(json.dumps(metrics, indent=2))

    md = [f"# Model evaluation (test cutoffs {', '.join(metrics['test_cutoffs'])})", "",
          f"Generated {date.today()} by `eval/run_model_eval.py`. Customers with < {MIN_PRIOR_ORDERS} prior orders "
          "are abstained (cold start) and excluded from the table; per-customer metrics use only customers who "
          "re-ordered >= 1 previously bought product in the 30-day window.", "",
          markdown_table(methods), "",
          f"B2 shown at full coverage; precision@5 of B2 over all eligible customers = {b2_point[1]:.3f}.", "",
          f"**Abstention at tau={bundle.tau:.2f}:** coverage {chosen['coverage']:.1%} of customers with history, "
          f"precision of recommended items {chosen['precision']:.3f}, {int(chosen['n_recommended'])} items.", "",
          "## Population", "", *[f"- {k}: {v}" for k, v in population.items()], "",
          "Plots: `calibration.png`, `abstention_curve.png`."]
    (EVAL_RESULTS_DIR / "model_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(markdown_table(methods))
    print(f"\ntau={bundle.tau:.2f}: coverage {chosen['coverage']:.3f}, precision {chosen['precision']:.3f}")
    print(json.dumps(population, indent=1))
    return metrics


def top_k_precision_all(test: pd.DataFrame, score: pd.Series) -> float:
    """Precision of top-5 items over ALL eligible customers (comparable to the abstention curve's y-axis)."""
    rec = top_k(test, score, K)
    return float(rec["label"].mean())


if __name__ == "__main__":
    main()
