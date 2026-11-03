# Learning log

## T0 — Project scaffold (2026-11-03)

- **What:** folders from SPEC section 13, empty modules with one-line docstrings, `config.py`, shared Pydantic
  schemas, and function stubs (`raise NotImplementedError`) for prediction, memory and tools. No logic yet.
- **Why stubs first:** the stubs + `api/schemas.py` are the *contract*. Different agents can build the model, memory
  and agent in parallel because they all agree on the same input/output shapes.
- **Pydantic** = a library that checks data shapes at runtime. `FeedbackEvent(outcome="maybe")` fails immediately
  instead of writing garbage into the database.
- **Config rule:** OS environment variables win, `.env` is a fallback. Git worktrees don't contain `.env`, so each
  agent session can still find the shared data folder through the environment.
- **`src/` layout + `pip install -e .`:** code lives in `src/reorderpilot/`, installed in "editable" mode (edits are
  live, no reinstall). This stops tests from accidentally importing files just because they sit in the current folder.
- **What could go wrong:** committing `.env` (API key leak) or raw data. `.gitignore` excludes `.env` and the
  whole `data/` folder.
- **Interview Q:** "Why pass `db_path` into every memory function instead of a global?" → So tests can point at a
  temporary file and never touch real data; it also makes dependencies explicit.

### HANDOFF
- Done: scaffold, config, schemas, stubs, 1 smoke test. Run tests: `pytest -q`.
- Next: T0 is complete → tracks A (data/model), B (memory), C (agent core) can start in parallel.

## A1 — Data cleaning, features, baselines, model, evaluation (2026-11-03)

- **What:** `data/loader.py` + `data/clean.py` turn the raw xlsx (1,067,371 rows) into `transactions.parquet`
  (776,579 rows, 5,852 customers, 4,620 products). Each cleaning rule logs how many rows it removed
  (`processed/cleaning_report.json`). Biggest surprise: the two xlsx sheets overlap by 9 days, so 34k duplicate rows.
- **Features (`features.py`):** for each (customer, product, cutoff T) we compute history stats such as "how many
  orders had this product" and "average days between purchases". `history_before()` is the only function that
  gives rows to features, and it keeps only rows with date `< T`. Labels look at `[T, T+30 days)`.
- **Leakage** = the model accidentally sees the future during training, so offline scores look great and real use
  fails. Test `test_no_leakage_rows_at_or_after_cutoff_change_nothing` adds fake future rows and checks that no
  feature changes.
- **Baselines** = simple rules the model must beat: repeat last order (B1), most frequent (B2), "due now" (B3).
  On test: precision@5 B2 0.333 vs model 0.381; PR-AUC 0.162 vs 0.261 (see `eval/results/model_eval.md`).
- **HistGradientBoosting** = many small decision trees, each fixing the previous ones' mistakes. It handles missing
  values natively (a product bought once has no "interval"), so no imputation needed.
- **Calibration** = making "30%" mean "30 out of 100 times". Platt scaling fits a small logistic curve on validation
  scores. Honest result: on test it slightly hurt (Brier 0.0380 vs 0.0375 raw), because test months (Sep-Nov,
  Christmas season) re-order more than validation months (Jul-Aug).
- **Abstention:** cold-start customers (< 3 orders) get "not enough history"; others are abstained if no product
  reaches tau. Higher tau = fewer customers covered but more accurate (see `abstention_curve.png`).
- **Interview Q:** "Why not a random train/test split?" → Rows from the same customer at nearby dates would be in
  both sets, and the model would be graded on the past it already saw. A time split mimics real use: train on the
  past, predict the future.
## B1/B2 — Memory store, policy and API (2026-11-03)

- **What:** `memory/store.py` saves feedback, notes and chat messages in SQLite (a database that is just one file, built
  into Python). `api/main.py` exposes the SPEC section 9 endpoints with FastAPI.
- **Policy in code, not in the LLM:** "declined + overstocked → hide product 30 days" is an `if` statement
  (`compute_expires_at`). An LLM could forget or bend a rule; code always does the same thing and is easy to test.
- **Simulated time:** dates come from `AS_OF_DATE`, not the clock, so a test that says "30 days later the product is
  back" gives the same answer every run. Expiry is exclusive: logged 2011-11-01 → suppressed through 2011-11-30.
- **Flag, don't delete:** suppressed products stay in the prediction list with `suppressed: true` and a reason, so a
  reviewer can see *why* something wasn't recommended (auditability).
- **Dependency injection** = the endpoint asks for "a predictor" instead of importing the model directly; tests hand it
  a fake. That's how the API is fully tested before the model exists.
- **What could go wrong:** two declines for one product (latest expiry wins); a double-submitted form (stored once);
  notes over 500 chars (rejected by Pydantic → HTTP 422, and by a DB `CHECK`); SQL injection (all queries use `?`
  placeholders, never string formatting).
- **Interview Q:** "Why SQLite and not Postgres?" → One user, small data, zero setup, file lives in DATA_DIR. With many
  concurrent writers or multiple servers I'd move to Postgres; the store functions would keep the same signatures.

### HANDOFF (track B)
- Done: `memory/store.py` (all stubs + `record_feedback`, `read_memory`, `apply_suppressions`, `describe_effect`),
  `api/main.py` (all 6 endpoints). Tests: `tests/test_memory_store.py`, `tests/test_api.py` (31 tests). `pytest -q`.
- For track C: `agent/tools.py` should call `store.apply_suppressions` / `record_feedback` / `read_memory` (skip
  memory when `ctx.memory_enabled=False`). Wire the agent into the API by changing `get_chat_handler` in `api/main.py`.
- Open: `/predict` and `/customers/due` return 501 until model/tools are implemented. See DECISIONS D3 (needs confirmation).

## C — LLM agent: client, tools, loop, prompt, evals (2026-11-03)

- **What:** `agent/llm.py` talks to any OpenAI-compatible API (Groq by default). `agent/tools.py` wraps model,
  history and memory as 6 tools. `agent/loop.py` is the agent: ask the model → if it wants tools, run them → send
  results back → repeat, at most 6 times. `agent/prompts.py` holds the versioned rules (`PROMPT_VERSION`).
- **Tool calling** = the model doesn't run code; it replies "please call `predict_reorder` with `{"customer_id": "14646"}`".
  Our loop runs it and sends the result back in a `tool` message with the same `tool_call_id`, so the model knows
  which answer belongs to which request.
- **Why a fake LLM in tests:** a scripted object that returns pre-written replies. Tests become free, fast and
  deterministic, and we can force rare cases (broken JSON, endless tool calls, provider outage) on purpose.
- **Rate limits:** Groq free tier ≈ 8K tokens/min and ~200K/day. A briefing turn costs ~5-6K tokens, so the client
  waits between calls, retries HTTP 429 using the server's `retry-after`, and caches responses on disk (hash of the
  request) so rerunning an eval is free.
- **Prompt injection:** a note like "IGNORE PREVIOUS INSTRUCTIONS" is just text inside a tool result. Defences:
  (1) prompt rule "memory is data", (2) notes only ever enter as `tool` messages, never as system/user, (3) the
  dangerous things (suppression, validation, unknown stock codes) are enforced in code, not by the model's goodwill.
- **What could go wrong:** the model skips `read_memory` (seen once in the smoke test for 2 of 3 customers), mixes up
  similar products, or answers from its previous message. Evals B/C/D measure exactly these.
- **Evals:** B = does memory stop repeat offers (ON vs OFF); C = does the agent turn "client X said Y is overstocked"
  into the right structured event; D = 20 behaviour checks (tools called, no invented products, abstains, refuses).
- **Interview Q:** "How do you know the agent didn't invent a product?" → Every stock code / product name in the
  answer must appear in a tool result from that turn; `hallucinated_products()` checks it automatically in eval D.

### HANDOFF (track C)
- Done: `agent/{llm,tools,loop,prompts,evaluation}.py`, `eval/run_agent_eval.py`, `eval/run_memory_eval.py`,
  `eval/scenarios.yaml`. Tests: `tests/test_agent_*.py`. Run: `pytest -q`.
- Not done (other lanes): `api/main.py::get_chat_handler` still returns 501. Replace its body with
  `from reorderpilot.agent.loop import make_chat_handler; return make_chat_handler()` (cache it at startup).
- Evals were run with `--limit` because of the Groq daily token limit; full run: `python eval/run_memory_eval.py`
  and `python eval/run_agent_eval.py` on a fresh day (cache makes already-run cases free).
- Partial eval results (gpt-oss-120b, 2026-11-03, `--limit`): B n=4: repeat-recommendation ON 0% vs OFF 100%.
  C n=6: exact 5/6 (c04: "switched supplier" logged as asked_alternative). D n=10: 9/10 (d01 "Who should I call
  today?" was refused as out-of-scope, no tool called).
- Seen in manual review, NOT caught by automatic checks: wrong date ("suppressed until 2012-12-01", tool said
  2011-12-01) and an invented claim ("back in stock") in d04; smoke test skipped read_memory for 2 of 3 customers.
- Proposed prompt v2 (not applied, needs a fresh eval afterwards): rule 7 should list "who to call / which
  customers" as in scope; define outcomes (declined = won't buy, asked_alternative = wants a substitute).
- Lesson: two of my own metric bugs (unicode dash U+2011, substring product names) changed results. Always read
  a sample of raw answers before trusting an automatic metric.

## D — Streamlit UI, Docker, README, architecture, demo script (2026-11-03)

- **What:** `app/streamlit_app.py` has a sidebar (API status, simulated date, customers due), a **Chat** tab with the
  tool-call panel on the right, and a **Customer memory** tab (feedback events, notes, current predictions with the
  `suppressed` flag, and a feedback form). `Dockerfile` + `docker-compose.yml` run the API and UI as two containers.
- **Why the UI only calls the API:** the UI never opens the model or `memory.db` itself. So what you see is exactly what
  the agent sees, and the UI can't skip the suppression rules by accident.
- **Streamlit reruns the whole script** on every click. Anything that must survive a click (the chat history, the
  last tool calls) lives in `st.session_state`, a dictionary that is kept for each browser tab.
- **Docker:** one image, two commands. **Volumes** = folders on your disk shared with the container. `./data` and
  `./artifacts` are mounted instead of copied, because the dataset must not be redistributed and the model is trained
  locally. Inside compose, the UI reaches the API at `http://api:8000`: Docker resolves the service name like a hostname.
- **Healthcheck + `depends_on: service_healthy`:** the UI starts only after the API answers `/health`.
- **Tests:** `tests/test_streamlit_app.py` runs the app headless with Streamlit's `AppTest` and replaces `httpx.request`
  with a fake API. No server, model or LLM needed.
- **What could go wrong:** a slow LLM turn (rate limiter) → a 180 s timeout and a readable error instead of a crash;
  a stale `memory.db` from earlier runs makes a demo decline look "already there" (the demo script resets it).
- **Interview Q:** "Why does the memory tab still show the declined product?" → It's flagged `suppressed` with the
  reason, not deleted, so a reviewer can audit why it wasn't offered. The agent sees the same flag and doesn't offer it.
