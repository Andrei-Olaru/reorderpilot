"""Train HistGradientBoosting on train cutoffs, select + calibrate on validation, save artifacts/.

Run: `python -m reorderpilot.model.train`. Test cutoffs are NOT used here (only in eval/run_model_eval.py).
Outputs: artifacts/model.joblib, artifacts/calibrator.joblib, artifacts/train_report.json.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from reorderpilot.config import ARTIFACTS_DIR, MIN_PRIOR_ORDERS, RANDOM_SEED, TRAIN_CUTOFFS, VALID_CUTOFFS
from reorderpilot.data.loader import load_transactions
from reorderpilot.features import FEATURE_COLUMNS, build_dataset
from reorderpilot.model.evaluate import abstention_curve, pair_metrics, topk_metrics

log = logging.getLogger(__name__)

PARAM_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": 300}
    for lr in (0.05, 0.1) for leaves in (15, 31)
]
TAU_GRID = [round(t, 2) for t in np.arange(0.0, 0.81, 0.02)]


class PlattCalibrator:
    """Platt scaling: logistic regression on the logit of the raw score. Same interface as IsotonicRegression."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        self.lr = LogisticRegression().fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(np.asarray(p)).reshape(-1, 1))[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_calibrator(kind: str):
    """'isotonic' (monotone step function) or 'platt' (sigmoid)."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0) if kind == "isotonic" else PlattCalibrator()


@dataclass
class ModelBundle:
    """What predict.py needs: the raw model, the calibrator, the abstention threshold, and the feature order."""

    model: HistGradientBoostingClassifier
    calibrator: object
    tau: float
    features: list[str]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(df[self.features].astype(float))[:, 1]
        return np.asarray(self.calibrator.predict(raw), dtype=float)


def fit_hgb(train: pd.DataFrame, params: dict) -> HistGradientBoostingClassifier:
    """Fit one HistGradientBoosting model (fixed seed, no internal random early-stopping split)."""
    model = HistGradientBoostingClassifier(**params, early_stopping=False, random_state=RANDOM_SEED)
    return model.fit(train[FEATURE_COLUMNS].astype(float), train["label"])


def choose_calibration(valid: pd.DataFrame, raw: np.ndarray) -> tuple[str, dict]:
    """Cross-fit over the validation months: fit on one month, Brier on the other; pick the lower average."""
    months = sorted(valid["cutoff"].unique())
    scores: dict[str, float] = {}
    for kind in ("isotonic", "platt"):
        briers = []
        for held_out in months:
            fit_mask = (valid["cutoff"] != held_out).to_numpy()
            cal = make_calibrator(kind).fit(raw[fit_mask], valid["label"].to_numpy()[fit_mask])
            briers.append(brier_score_loss(valid["label"][~fit_mask], cal.predict(raw[~fit_mask])))
        scores[kind] = float(np.mean(briers))
    scores["uncalibrated"] = float(brier_score_loss(valid["label"], raw))
    return min(("isotonic", "platt"), key=lambda k: scores[k]), scores


def train(artifacts_dir=ARTIFACTS_DIR) -> dict:
    """Full training run; returns the report that is also written to train_report.json."""
    tx = load_transactions()
    t0 = time.time()
    train_df = build_dataset(tx, TRAIN_CUTOFFS, MIN_PRIOR_ORDERS)
    valid_df = build_dataset(tx, VALID_CUTOFFS, MIN_PRIOR_ORDERS)
    log.info("train %s, valid %s (%.0fs)", train_df.shape, valid_df.shape, time.time() - t0)

    grid = []
    best = None
    for params in PARAM_GRID:
        t = time.time()
        model = fit_hgb(train_df, params)
        raw = model.predict_proba(valid_df[FEATURE_COLUMNS].astype(float))[:, 1]
        row = {"params": params, **pair_metrics(valid_df["label"], raw),
               **topk_metrics(valid_df, pd.Series(raw), k=5), "fit_seconds": round(time.time() - t, 1)}
        grid.append(row)
        log.info("%s -> PR-AUC %.4f P@5 %.4f", params, row["pr_auc"], row["precision@5"])
        if best is None or row["pr_auc"] > best[0]["pr_auc"]:
            best = (row, model, raw)
    best_row, model, raw = best

    kind, cal_scores = choose_calibration(valid_df, raw)
    calibrator = make_calibrator(kind).fit(raw, valid_df["label"].to_numpy())
    prob = calibrator.predict(raw)

    curve = abstention_curve(valid_df, pd.Series(prob), TAU_GRID, k=5)
    tau = float(curve.loc[curve["f1"].idxmax(), "tau"])
    bundle = ModelBundle(model=model, calibrator=calibrator, tau=tau, features=list(FEATURE_COLUMNS))

    report = {
        "train_cutoffs": [str(c) for c in TRAIN_CUTOFFS], "valid_cutoffs": [str(c) for c in VALID_CUTOFFS],
        "n_train_pairs": len(train_df), "n_valid_pairs": len(valid_df),
        "train_base_rate": float(train_df["label"].mean()),
        "grid": grid, "best_params": best_row["params"],
        "calibration": {"chosen": kind, "cross_fit_brier": cal_scores},
        "tau": tau, "tau_rule": "max F1 of recommended items (top-5 with p >= tau) on validation",
        "valid_metrics_calibrated": {**pair_metrics(valid_df["label"], prob, prob=True),
                                     **topk_metrics(valid_df, pd.Series(prob), k=5)},
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifacts_dir / "model.joblib")
    joblib.dump(calibrator, artifacts_dir / "calibrator.joblib")
    (artifacts_dir / "train_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = train()
    print(f"best {r['best_params']} | calibration {r['calibration']} | tau {r['tau']}")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in r["valid_metrics_calibrated"].items()})
