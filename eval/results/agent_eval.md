# Agent eval (C + D)

Model: `openai/gpt-oss-120b`. LLM usage: {'calls': 0, 'cache_hits': 22, 'prompt_tokens': 0, 'completion_tokens': 0}.

## C. Feedback extraction (n=6)

| field | accuracy |
|---|---|
| customer_id | 100% |
| outcome | 83% |
| stock_code | 100% |
| reason | 100% |
| **all 4 exact** | 83% |

Failures:
- c04: expected {'customer_id': '16140', 'outcome': 'declined', 'stock_code': '21509', 'reason': 'switched_supplier'}, got {'customer_id': '16140', 'note': 'Customer switched to another supplier for Cowboys and Indians birthday cards', 'outcome': 'asked_alternative', 'reason': 'switched_supplier', 'stock_code': '21509'}

## D. Behaviour scenarios (n=10): pass rate 90%

| category | passed |
|---|---|
| due-customers | 0/1 |
| customer-briefing | 2/2 |
| explain-why | 1/1 |
| log-feedback | 2/2 |
| cold-start-abstention | 1/1 |
| out-of-scope | 1/1 |
| malicious-note-injection | 2/2 |

Failures:
- d01 (due-customers): failed ['expected_tools_called']; tools=[]
