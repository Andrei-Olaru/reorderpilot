"""Evals C + D: feedback-extraction accuracy and agent behaviour scenarios -> eval/results/agent_eval.{json,md}.

Usage: python eval/run_agent_eval.py [--part C|D|all] [--limit N] [--model NAME] [--no-cache]
"""

from __future__ import annotations

from pathlib import Path

import yaml

from reorderpilot.agent import evaluation as ev
from reorderpilot.agent.loop import run_turn
from reorderpilot.api.schemas import FeedbackEvent
from reorderpilot.config import get_settings
from reorderpilot.memory import store

CASES = Path(__file__).parent / "scenarios.yaml"
FIELDS = ["customer_id", "outcome", "stock_code", "reason"]
REFUSAL_MAX_CHARS = 600
ABSTAIN_WORDS = ("not enough", "abstain", "insufficient", "limited", "only", "too few", "no reliable", "cannot")


def eval_c(cases: list[dict], llm, as_of) -> dict:
    """Accuracy of write_memory arguments vs. the expected structured event."""
    rows = []
    for case in cases:
        exp = case["expect"]
        with ev.temp_ctx(as_of) as ctx:
            out = run_turn(case["message"], llm, ctx)
        writes = [t["args"] for t in out.tool_outputs if t["name"] == "write_memory"]
        got = writes[-1] if writes else {}
        correct = {f: got.get(f) == exp.get(f) for f in FIELDS}
        rows.append({"id": case["id"], "message": case["message"], "expected": exp, "got": got,
                     "n_writes": len(writes), "correct": correct, "exact": all(correct.values()),
                     "tools": [t["name"] for t in out.tool_outputs], "stop": out.stop_reason,
                     "answer": out.response.answer})
        print(f"C {case['id']}: exact={rows[-1]['exact']} got={got}")
    n = len(rows) or 1
    summary = {f"acc_{f}": round(sum(r["correct"][f] for r in rows) / n, 3) for f in FIELDS}
    summary["exact_match"] = round(sum(r["exact"] for r in rows) / n, 3)
    summary["n"] = len(rows)
    return {"summary": summary, "cases": rows}


def _check(case: dict, out, cat: dict[str, str]) -> dict[str, bool]:
    """All automatic checks for one behaviour scenario (only the ones the case asks for, + hallucination)."""
    called = [t["name"] for t in out.tool_outputs]
    answer, low = out.response.answer, out.response.answer.lower()
    checks = {"no_hallucinated_products": not ev.hallucinated_products(answer, out.tool_outputs, cat, case["message"]),
              "finished": out.stop_reason == "answer"}
    if "expect_tools" in case:
        checks["expected_tools_called"] = all(t in called for t in case["expect_tools"])
    if "forbid_tools" in case:
        checks["forbidden_tools_not_called"] = (not called if "*" in case["forbid_tools"]
                                                else not any(t in called for t in case["forbid_tools"]))
    if case.get("expect_abstain"):
        abstained = any(t["name"] == "predict_reorder" and t["result"].get("abstain") for t in out.tool_outputs)
        checks["abstention_respected"] = (abstained and not ev.mentioned_products(answer, cat)
                                          and any(w in low for w in ABSTAIN_WORDS))
    if case.get("expect_refusal"):
        checks["short_refusal"] = len(answer) <= REFUSAL_MAX_CHARS
    if "expect_text_any" in case:
        checks["expected_text"] = any(w.lower() in low for w in case["expect_text_any"])
    if "forbid_text" in case:
        checks["forbidden_text_absent"] = not any(w.lower() in low for w in case["forbid_text"])
    if "not_recommended" in case:
        checks["declined_item_not_recommended"] = all(
            ev.classify_mention(answer, c, cat.get(c, "")) != "recommended" for c in case["not_recommended"])
    if "expect_write" in case:
        writes = [t["args"] for t in out.tool_outputs if t["name"] == "write_memory"]
        checks["write_args_correct"] = bool(writes) and all(writes[-1].get(k) == v
                                                            for k, v in case["expect_write"].items())
    return checks


def eval_d(cases: list[dict], llm, as_of) -> dict:
    """Run each behaviour scenario on a fresh memory and apply its checks."""
    cat, rows = ev.catalog(), []
    for case in cases:
        with ev.temp_ctx(as_of) as ctx:
            for cid, note in (case.get("setup_notes") or {}).items():
                store.add_customer_note(ctx.db_path, cid, note, as_of)
            for fb in case.get("setup_feedback") or []:
                store.record_feedback(ctx.db_path, FeedbackEvent(**fb), as_of)
            out = run_turn(case["message"], llm, ctx)
        checks = _check(case, out, cat)
        rows.append({"id": case["id"], "category": case["category"], "message": case["message"],
                     "passed": all(checks.values()), "checks": checks,
                     "tools": [t["name"] for t in out.tool_outputs], "stop": out.stop_reason, "steps": out.steps,
                     "answer": out.response.answer})
        failed = [k for k, v in checks.items() if not v]
        print(f"D {case['id']} [{case['category']}]: {'PASS' if not failed else 'FAIL ' + ','.join(failed)}")
    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    summary = {"pass_rate": round(sum(r["passed"] for r in rows) / (len(rows) or 1), 3), "n": len(rows),
               "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in by_cat.items()}}
    return {"summary": summary, "cases": rows}


def to_markdown(res: dict, model: str, usage: dict) -> str:
    lines = [f"# Agent eval (C + D)\n\nModel: `{model}`. LLM usage: {usage}.\n"]
    if "C" in res:
        s = res["C"]["summary"]
        lines += [f"## C. Feedback extraction (n={s['n']})\n", "| field | accuracy |", "|---|---|"]
        lines += [f"| {f} | {s[f'acc_{f}']:.0%} |" for f in FIELDS] + [f"| **all 4 exact** | {s['exact_match']:.0%} |"]
        bad = [r for r in res["C"]["cases"] if not r["exact"]]
        lines += ["", "Failures:" if bad else "No failures."]
        lines += [f"- {r['id']}: expected {r['expected']}, got {r['got'] or 'no write'}" for r in bad]
    if "D" in res:
        s = res["D"]["summary"]
        lines += [f"\n## D. Behaviour scenarios (n={s['n']}): pass rate {s['pass_rate']:.0%}\n",
                  "| category | passed |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["by_category"].items()]
        bad = [r for r in res["D"]["cases"] if not r["passed"]]
        lines += ["", "Failures:" if bad else "No failures."]
        lines += [f"- {r['id']} ({r['category']}): failed {[k for k, v in r['checks'].items() if not v]}; "
                  f"tools={r['tools']}" for r in bad]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = ev.common_args(__doc__)
    p.add_argument("--part", choices=["C", "D", "all"], default="all")
    args = p.parse_args()
    cases = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    llm, as_of = ev.make_llm(args.model, not args.no_cache), get_settings().as_of_date
    res: dict = {}
    if args.part in ("C", "all"):
        res["C"] = eval_c(cases["feedback_extraction"][: args.limit], llm, as_of)
    if args.part in ("D", "all"):
        res["D"] = eval_d(cases["scenarios"][: args.limit], llm, as_of)
    res["meta"] = {"model": llm.config.model, "limit": args.limit, "usage": llm.usage}
    print(ev.write_results("agent_eval", res, to_markdown(res, llm.config.model, llm.usage)))


if __name__ == "__main__":
    main()
