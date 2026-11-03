# Eval B: memory ablation (n=4 customers, model `openai/gpt-oss-120b`)

Each customer's top-1 product was logged as declined/overstocked before asking for a briefing.

| metric | memory ON | memory OFF |
|---|---|---|
| repeat_recommendation_rate | 0% | 100% |
| mentions_suppression_rate | 100% | 0% |
| absent_rate | 0% | 0% |
| tool_flagged_suppressed_rate | 100% | 0% |
| finished_rate | 100% | 100% |

`recommended` = the declined item appears in a line without a suppression word (heuristic: see `agent/evaluation.py::classify_mention`; raw answers are in memory_eval.json).

LLM usage: {'calls': 0, 'cache_hits': 24, 'prompt_tokens': 0, 'completion_tokens': 0}

Memory-ON failures (declined item still recommended): none
