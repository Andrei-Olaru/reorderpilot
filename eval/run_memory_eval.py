"""Eval B: memory ON vs OFF ablation on repeat-recommendation of declined items -> eval/results/memory_eval.*.

For each sampled customer: take the top-1 predicted product, log declined/overstocked, ask for a briefing.
ON = tools apply memory; OFF = tools ignore memory. Metric: how often the declined item is still recommended.
Usage: python eval/run_memory_eval.py [--n 30] [--limit N] [--model NAME] [--no-cache]
"""

from __future__ import annotations

import numpy as np

from reorderpilot.agent import evaluation as ev
from reorderpilot.agent import tools
from reorderpilot.agent.loop import run_turn
from reorderpilot.api.schemas import FeedbackEvent, PredictReorderArgs
from reorderpilot.config import MIN_PRIOR_ORDERS, RANDOM_SEED, get_settings
from reorderpilot.memory import store


def sample_customers(n: int, as_of) -> list[tuple[str, str, str]]:
    """n non-abstained customers (fixed seed) with their raw top-1 product: (customer_id, stock_code, description)."""
    with ev.temp_ctx(as_of, memory_enabled=False) as ctx:
        scored = tools._scored(ctx)
        tau = ctx.bundle_loader().tau
        best = scored[scored["cust_n_orders"] >= MIN_PRIOR_ORDERS].groupby("customer_id")["probability"].max()
        eligible = sorted(best[best >= tau].index)
        picked = np.random.default_rng(RANDOM_SEED).choice(eligible, n, replace=False)
        out = []
        for cid in picked:
            top = tools.predict_reorder(PredictReorderArgs(customer_id=str(cid), top_k=1), ctx).recommendations[0]
            out.append((str(cid), top.stock_code, top.description))
    return out


def run_case(cid: str, code: str, desc: str, memory_on: bool, llm, as_of) -> dict:
    with ev.temp_ctx(as_of, memory_enabled=memory_on) as ctx:
        store.record_feedback(ctx.db_path, FeedbackEvent(customer_id=cid, outcome="declined", stock_code=code,
                                                         reason="overstocked"), as_of)
        out = run_turn(f"Give me a briefing for customer {cid}.", llm, ctx)
    flagged = any(r.get("stock_code") == code and r.get("suppressed")
                  for t in out.tool_outputs if t["name"] == "predict_reorder"
                  for r in t["result"].get("recommendations", []))
    return {"mention": ev.classify_mention(out.response.answer, code, desc), "tool_flagged_suppressed": flagged,
            "tools": [t["name"] for t in out.tool_outputs], "stop": out.stop_reason, "answer": out.response.answer}


def main() -> None:
    p = ev.common_args(__doc__)
    p.add_argument("--n", type=int, default=30, help="customers to sample (SPEC: 30)")
    args = p.parse_args()
    as_of = get_settings().as_of_date
    llm = ev.make_llm(args.model, not args.no_cache)
    customers = sample_customers(args.n, as_of)[: args.limit]
    rows = []
    for cid, code, desc in customers:
        row = {"customer_id": cid, "declined_stock_code": code, "description": desc}
        for label, on in (("on", True), ("off", False)):
            row[label] = run_case(cid, code, desc, on, llm, as_of)
        rows.append(row)
        print(f"B {cid} {code}: ON={row['on']['mention']} OFF={row['off']['mention']}")

    n = len(rows) or 1

    def rate(cond: str, key: str, value) -> float:
        return round(sum(r[cond][key] == value for r in rows) / n, 3)

    summary = {cond: {"repeat_recommendation_rate": rate(cond, "mention", "recommended"),
                      "mentions_suppression_rate": rate(cond, "mention", "mentioned_suppressed"),
                      "absent_rate": rate(cond, "mention", "absent"),
                      "tool_flagged_suppressed_rate": rate(cond, "tool_flagged_suppressed", True),
                      "finished_rate": rate(cond, "stop", "answer")} for cond in ("on", "off")}
    md = [f"# Eval B: memory ablation (n={len(rows)} customers, model `{llm.config.model}`)\n",
          "Each customer's top-1 product was logged as declined/overstocked before asking for a briefing.\n",
          "| metric | memory ON | memory OFF |", "|---|---|---|"]
    for k in summary["on"]:
        md.append(f"| {k} | {summary['on'][k]:.0%} | {summary['off'][k]:.0%} |")
    md += ["", "`recommended` = the declined item appears in a line without a suppression word (heuristic: "
           "see `agent/evaluation.py::classify_mention`; raw answers are in memory_eval.json).",
           f"\nLLM usage: {llm.usage}"]
    bad = [r["customer_id"] for r in rows if r["on"]["mention"] == "recommended"]
    md.append(f"\nMemory-ON failures (declined item still recommended): {bad or 'none'}")
    res = {"summary": summary, "cases": rows, "meta": {"model": llm.config.model, "n": args.n, "limit": args.limit,
                                                       "usage": llm.usage}}
    print(ev.write_results("memory_eval", res, "\n".join(md) + "\n"))


if __name__ == "__main__":
    main()
