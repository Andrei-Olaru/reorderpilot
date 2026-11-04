# Eval B: memory ablation (n=30 customers, model `openai/gpt-oss-120b`)

Each customer's top-1 product was logged as declined/overstocked before asking for a briefing.

| metric | memory ON | memory OFF |
|---|---|---|
| repeat_recommendation_rate | 10% | 100% |
| mentions_suppression_rate | 90% | 0% |
| absent_rate | 0% | 0% |
| tool_flagged_suppressed_rate | 100% | 0% |
| finished_rate | 100% | 100% |

`recommended` = the declined item appears in a line without a suppression word (heuristic: see `agent/evaluation.py::classify_mention`; raw answers are in memory_eval.json).

LLM usage: {'calls': 150, 'cache_hits': 30, 'prompt_tokens': 179495, 'completion_tokens': 26387}

Memory-ON failures (declined item still recommended): ['12908', '16933', '16637']
