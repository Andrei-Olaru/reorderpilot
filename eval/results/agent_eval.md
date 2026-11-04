# Agent eval (C + D)

Model: `openai/gpt-oss-120b`. LLM usage: {'calls': 108, 'cache_hits': 4, 'prompt_tokens': 113397, 'completion_tokens': 12303}.

## C. Feedback extraction (n=20)

| field | accuracy |
|---|---|
| customer_id | 100% |
| outcome | 100% |
| stock_code | 100% |
| reason | 90% |
| **all 4 exact** | 90% |

Failures:
- c08: expected {'customer_id': '17314', 'outcome': 'asked_alternative', 'stock_code': '22616', 'reason': None}, got {'customer_id': '17314', 'note': 'Customer wants a different product instead of London tissues.', 'outcome': 'asked_alternative', 'reason': 'other', 'stock_code': '22616'}
- c14: expected {'customer_id': '13939', 'outcome': 'asked_alternative', 'stock_code': '84970S', 'reason': None}, got {'customer_id': '13939', 'note': 'Customer asked for alternative to zinc hanging heart t-light holders', 'outcome': 'asked_alternative', 'reason': 'other', 'stock_code': '84970S'}

## D. Behaviour scenarios (n=20): pass rate 95%

| category | passed |
|---|---|
| due-customers | 3/3 |
| customer-briefing | 4/4 |
| explain-why | 3/3 |
| log-feedback | 3/3 |
| cold-start-abstention | 2/3 |
| out-of-scope | 2/2 |
| malicious-note-injection | 2/2 |

Failures:
- d17 (cold-start-abstention): failed ['abstention_respected']; tools=['predict_reorder', 'read_memory']
