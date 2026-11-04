# Decisions

Format per entry: context, options considered, choice, why, trade-offs.

## D1 — Package layout and install (2026-11-03)
- **Context:** SPEC uses `src/reorderpilot/`, and commands like `python -m reorderpilot.model.train` must work.
- **Options:** (a) `pyproject.toml` + `pip install -e .`; (b) set `PYTHONPATH=src` by hand; (c) flat layout.
- **Choice:** (a), plus `pythonpath = ["src"]` for pytest so tests also run without the install.
- **Why:** standard, works the same in Docker, worktrees and CI.
- **Trade-off:** one extra setup step (`pip install -e .`).

## D2 — Shared tool-result shapes (2026-11-03)
- **Context:** SPEC says predict_reorder returns "a list of recommendations, or `{abstain: true, why}`".
- **Options:** (a) return either a list or a dict; (b) one `PredictionResult` object with `abstain`, `why`, `recommendations`.
- **Choice:** (b). Also adds `price_sensitive` (SPEC: price decline marks the customer in the briefing context).
- **Why:** one fixed type is easier to validate, test and show in the UI.
- **Trade-off:** small departure from the literal SPEC wording; the fields themselves are unchanged.

## D3 — Cleaning: duplicates and non-product codes (2026-11-03)
- **Context:** the xlsx has two sheets that overlap on 2010-12-01..09 (~22.8k rows appear twice) and ~11.5k more
  identical lines inside sheets. SPEC section 5 asks to drop non-product StockCodes after inspection.
- **Options:** (a) keep duplicates; (b) drop only the sheet overlap by date; (c) drop all exact duplicate rows.
- **Choice:** (c), as the first cleaning rule (34,335 rows). Non-product list after inspecting every code that is not
  "5 digits + letters": POST, DOT, C2, M, D, ADJUST, ADJUST2, BANK CHARGES, PADS, TEST001, TEST002 (+ AMAZONFEE, B,
  CRUK, S for safety). SP1002 (kid's chalkboard) looked odd but is a real product, kept.
- **Why:** without (c) every December 2010 order is counted twice, which inflates n_orders and creates 0-day intervals.
- **Trade-off:** a genuine identical line repeated in one invoice is merged; for "did c buy p" this changes nothing.

## D4 — Evaluation population and metric definitions (2026-11-03)
- **Context:** SPEC says per-customer metrics use "customers who actually ordered in the window", but a customer who
  only bought *new* products has no correct answer among the candidates (recall is undefined).
- **Choice:** per-customer metrics use (cutoff, customer) pairs with >= 3 prior orders and >= 1 re-ordered candidate.
  precision@5 divides by 5 even if fewer items are shown. B1 recommends only last-order products (empty slots are
  misses) so it does not quietly turn into B2. Model and baselines are scored on the same pairs. Brier only for the
  model (baseline scores are not probabilities). How many customers ordered at all is reported in `population`.
- **Trade-off:** numbers look better than "all customers who ordered anything"; the population counts make that visible.

## D5 — Model selection, calibration and abstention threshold (2026-11-03)
- **Options:** isotonic vs Platt; tau fixed by hand vs chosen by a rule.
- **Choice:** 4-config HGB grid, best by validation PR-AUC. Calibrator picked by cross-fitting the two validation
  months (fit on one, Brier on the other): Platt 0.029191 vs isotonic 0.029218, so Platt; then refit on both months.
  tau = value maximizing F1 of recommended items on validation (0.08).
- **Why:** every choice uses validation only; test is touched once in `eval/run_model_eval.py`.
- **Trade-off:** the gap between calibrators is tiny; on test, calibration slightly hurt (seasonal base-rate shift).
  F1 is a generic rule; the real tau is a business choice (cost of a wasted call vs a missed sale).

## D6 — Where metrics.json comes from (2026-11-03)
- **Context:** SPEC 15 says `python -m reorderpilot.model.train` reproduces metrics.json; the task also says test
  cutoffs are touched only in the final eval script.
- **Choice:** `train` writes `artifacts/train_report.json` (validation only); `eval/run_model_eval.py` writes
  `artifacts/metrics.json` (test + a copy of the validation summary). Reproduce = train, then eval.
- **Trade-off:** two commands instead of one. Needs a SPEC 15 wording update (human decision).
## D3 — Memory visibility: same-day events count (2026-11-03)
- **Context:** SPEC section 4 says memory lookups use information "strictly before AS_OF_DATE", but section 7 stamps
  `created_at = AS_OF_DATE`. Taken literally, feedback logged today would be invisible until tomorrow, and the memory
  ablation (eval B: log a decline, then ask for a briefing at the same date) could never work.
- **Options:** (a) strict `<` for memory too; (b) `created_at <= as_of` for memory, strict `<` stays for model features.
- **Choice:** (b). Suppression is active while `as_of < expires_at` (expiry date is exclusive).
- **Why:** the leakage rule protects the model from future *transactions*; feedback the rep gave today is legitimately
  known today. Events dated after `as_of` are still invisible (tested).
- **Trade-off:** a different comparison for memory than for features — documented in `memory/store.py` docstring.
  **Needs human confirmation** (SPEC wording conflict).

## D4 — Duplicate feedback events are idempotent per simulated day (2026-11-03)
- **Context:** a double-clicked form or an agent retry can send the same feedback twice.
- **Options:** (a) store every copy; (b) unique DB constraint and return an error; (c) if an identical event
  (all fields equal, same `created_at`) exists, return the existing row.
- **Choice:** (c). The response effect says "duplicate ignored". The same event on a *later* day is a new row, and for
  overlapping suppressions of one product the latest `expires_at` wins.
- **Why:** callers (API, agent) never need special error handling, and the audit trail has no noise.
- **Trade-off:** a rep can't deliberately log the exact same fact twice on one day (no real use for that).

## D5 — Shared memory helpers + dependency injection in the API (2026-11-03)
- **Context:** both `GET /predict` and the agent tool `predict_reorder` must apply the same suppression policy; the API
  must be testable while the model and agent are built by other tracks.
- **Choice:** `memory/store.py` exposes `record_feedback`, `read_memory`, `apply_suppressions` (return existing schemas,
  no contract change); `agent/tools.py` should call these. The API gets the predictor, due-lister, chat handler, DB path
  and `as_of` via FastAPI `Depends`, overridden in tests with fakes.
- **Alternatives:** duplicate policy code in tools and API (drift risk); monkeypatching module globals in tests (brittle).
- **Trade-off:** suppressed items are *flagged*, not removed, so they still take a slot in `top_k` (auditable, but the
  rep may see fewer actionable items).

## D7 — Agent loop: hand-written, malformed-JSON and error policy (2026-11-03)
- **Context:** SPEC asks for a ~100-150 line loop, max 6 steps, one retry on malformed tool-call JSON, clear errors.
- **Options:** (a) framework (LangChain/LangGraph agents); (b) provider "auto tool runner"; (c) hand-written loop.
- **Choice:** (c). Malformed JSON (our parse fails, or Groq returns 400 `tool_use_failed`) shares ONE retry budget per
  turn; the second time the user gets a fixed message. Pydantic validation errors are NOT retries: they go back to the
  model as a ToolError so it can fix the call (they still consume steps). Hitting 6 steps returns a fixed message
  instead of making a 7th "summarize" call (saves tokens on a rate-limited free tier).
- **Why:** every branch is visible and unit-tested with a scripted fake LLM.
- **Trade-off:** no streaming, no parallel tool execution; we maintain the message format ourselves.

## D8 — Conversation memory replays only user + final assistant texts (2026-11-03)
- **Context:** `messages` table gives short-term memory; tool results are large and the free tier is ~8K tokens/min.
- **Options:** (a) replay full transcripts incl. tool calls; (b) replay last 10 user/assistant texts only.
- **Choice:** (b). **Trade-off:** in a follow-up question the model may reuse numbers from its own earlier answer
  instead of re-calling a tool (seen in the smoke test). The system prompt rule 1 still forces fresh tool calls for briefings.

## D9 — Product resolution: word coverage instead of WRatio (2026-11-03)
- **Context:** `find_product` must map "spaceboy lunch boxes" to a stock_code and say when it is ambiguous.
- **Options:** (a) rapidfuzz WRatio top match; (b) share of query words found in the description (with plural
  stemming), tie-break by whole-string similarity; (c) embeddings (out of scope: no vector DB).
- **Choice:** (b), searching only the customer's own past purchases when customer_id is given. WRatio gave ~85.5 to any
  product sharing one word, so "london tissues" was ambiguous with every "... TISSUES" product.
- **Trade-off / honesty:** (b) was tuned on the eval-C phrases (22/22 resolved offline), so eval C's stock_code accuracy is
  optimistic. A fresh held-out set of phrases is needed for an unbiased number. Adds `customer_id` to the agent-side
  `FindProductToolArgs` only (subclass in tools.py); the shared `FindProductArgs` contract is unchanged.

## D10 — Ranking customers with suppressions (2026-11-03)
- **Context:** `list_customers_due` should not push a customer whose value comes from a just-declined product.
- **Choice:** compute values for all customers, then subtract suppressed items only for the top 3 x limit before re-sorting.
- **Why:** one SQLite lookup per shown customer instead of per customer (~2,000). **Trade-off:** a customer ranked
  just outside 3 x limit can't move up; acceptable for a ranked call list.

## D11 — UI is an HTTP-only client; Docker mounts data and model (2026-11-03)
- **Context:** SPEC section 10 (Streamlit) and M5 (Docker). The UI could import the model/memory directly or call the API.
- **Options:** (a) Streamlit imports `reorderpilot` and calls functions; (b) Streamlit calls the FastAPI endpoints.
  For Docker: (c) bake data + model into the image; (d) mount `./data` and `./artifacts` as volumes.
- **Choice:** (b) and (d). One image (`python:3.12-slim`, the version the pins were tested on) runs both services.
  The UI imports only the `Outcome`/`Reason` types from `api/schemas.py`, so the form choices match the contract.
- **Why:** the demo shows exactly what the API returns, and the UI can't bypass the memory policy. Data is gitignored
  and must not be redistributed (CC BY 4.0 attribution, raw data stays out); the model is gitignored too.
- **Trade-offs:** the UI needs the API running (it shows a clear error otherwise); a clean clone must build data
  and train once before `docker compose up`. SPEC says Python 3.11; the image uses 3.12 to match `requirements.txt`.

## D12 — Prompt v2: "who to call" is in scope (2026-11-03)
- **Context:** with prompt v1, "Who should I call today?" was refused without any tool call (eval D d01, and seen
  live in the UI). Rule 7 ("requests not about customers, products or orders → refuse") didn't name call planning,
  so gpt-oss treated it as off-topic. Also seen: a "who to call" answer that briefed one customer without read_memory.
- **Options:** (a) keyword router in code that forces list_customers_due; (b) state the scope + the tool in the prompt.
- **Choice:** (b): an explicit "your job" paragraph mapping "who should I call?" to list_customers_due; rule 7 now
  refuses only clearly unrelated requests ("when unsure, use the tools"); outcome definitions added (eval C c04).
- **Why:** smallest change; a router would need its own phrase list and still miss wordings.
- **Trade-off:** looser refusal rule → slightly higher risk of answering borderline off-topic requests. Manual check
  on 5 phrasings + 1 off-topic (weather, still refused). Evals C/D must be rerun for prompt v2 numbers.

## D13 — Styled UI beyond SPEC's "clean layout" (2026-11-03)
- **Context:** SPEC section 10 says "no styling work beyond a clean layout". The human explicitly asked for a polished,
  enterprise-style UI (KPI cards, tool accordions, memory dashboard). The human's request overrides that SPEC line.
- **Options:** (a) a `.streamlit/config.toml` theme; (b) a CSS block injected by the app; (c) a component library.
- **Choice:** (b) a small CSS string + Streamlit built-ins (`st.metric`, `st.badge`, bordered containers, Material icons
  via `:material/name:`). No new dependency, no API contract change.
- **Why:** everything stays in one file and works in light and dark mode (translucent greys, no hard-coded backgrounds).
- **Trade-offs:** CSS targets Streamlit's internal `data-testid` selectors, which can change between Streamlit versions
  (the page still works, it just looks plainer). "Suppressed items" is per selected customer, because the API has no
  global count endpoint; adding one would change the contract.

## D14 — Runtime grounding guard for stock codes (2026-11-04)
- **Context:** prompt rule 2 tells the LLM to state only tool data, and eval D *measures* hallucinated products after
  the fact. Nothing stopped an invented stock code from reaching the rep at runtime.
- **Options:** (a) stronger prompt wording only; (b) re-ask the LLM to rewrite when an invented code is found;
  (c) deterministic filter in the loop: drop answer lines citing codes not seen earlier in the transcript;
  (d) raise an error and return no answer.
- **Choice:** (c), in `agent/grounding.py`, applied to every final answer in `loop._loop`. "Seen earlier" = tool
  results, the rep's messages and replayed earlier turns. Prompt left at v2 (a bump would invalidate cached eval runs).
- **Why:** a prompt is a request, not a guarantee; (b) costs an extra LLM call per incident (Groq rate limit); (d)
  throws away a mostly-good answer. A line filter is cheap, testable and auditable (`TurnResult.removed_codes`).
- **Trade-offs:** only code-shaped tokens (5 digits + up to 2 letters) are checked; an invented product named only by
  description slips through (eval D still catches those). A whole line is dropped, even if half of it was fine. Because
  the guard rewrites answers, eval D now reflects the guarded output; `removed_codes` shows what the raw LLM did.
  No API contract change (`ChatResponse` is unchanged).

## D15 — Docker: one-shot `init` service and data dir from `.env` (2026-11-04)
- **Context:** compose mounted `./data`, but `.env` may point `REORDERPILOT_DATA_DIR` elsewhere (it does on the dev
  machine), so the API container started with no data. Running from scratch also took 4 commands.
- **Options:** (a) document the manual `docker compose run` steps; (b) build data + model in the API's startup command;
  (c) a separate one-shot `init` service that the API waits for (`service_completed_successfully`).
- **Choice:** (c), idempotent (each step runs only if its output file is missing), plus the bind mount source
  `${REORDERPILOT_DATA_DIR:-./data}` read from the same `.env`. UI waits for the API health check.
- **Why:** (b) would mix a 3-minute training job into the web server's start and its health check; (c) keeps "prepare"
  and "serve" separate and makes the whole run `docker compose build` + `docker compose up`.
- **Trade-offs:** a stale model is not retrained automatically (delete `artifacts/model.joblib` to force it). The
  existence check is per file, so a half-written file from a crashed run would be trusted; rerun the module by hand.

## D16 — LangSmith: accept both env names, return trace_url, never trace from tests (2026-11-04)
- **Context:** `run_turn` passes `tracing_context(enabled=settings.langsmith_tracing)`, and that explicit value
  overrides LangSmith's own env check. The flag read only `LANGSMITH_TRACING`, so `LANGCHAIN_TRACING_V2=true` (the
  name in many guides) silently disabled tracing. `ChatResponse.trace_url` (SPEC section 9) was never filled. And the
  API test built a handler from the real settings, so `pytest` sent traces to the developer's LangSmith project.
- **Options:** (a) drop `tracing_context` and let the SDK read env vars; (b) keep the explicit flag, accept both
  names; for tests: (c) patch each test, (d) one autouse fixture in `tests/conftest.py`.
- **Choice:** (b) + (d). `trace_url` is read from the active run (`get_current_run_tree().get_url()`), wrapped so a
  tracing error can never break an answer.
- **Why:** an explicit flag in `Settings` is visible and testable; one fixture protects every current and future test.
- **Trade-offs:** `get_url()` makes one small LangSmith API call per turn when tracing is on (cached per run).

## D17 — Full agent evals on prompt v2: report as measured, do not fix the metric afterwards (2026-11-04)
- **Context:** evals B-D were rerun on the full sets (B n = 30, C n = 20, D n = 20) with prompt v2 and the grounding
  guard. Eval D's only failure (d17) is a metric false negative: the agent abstained ("there isn't enough ordering
  history"), but the keyword list in `run_agent_eval.py` has "not enough", not "isn't enough". Eval B showed the LLM
  ignored the `suppressed` flag in 3 of 30 briefings.
- **Options:** (a) add "isn't enough" to the keyword list and rerun; (b) report 95% as measured and document the
  false negative; for B: (c) add a deterministic answer filter for suppressed items now; (d) report and list it as next.
- **Choice:** (b) and (d).
- **Why:** changing a check after seeing which case failed is tuning the metric on the test set; the README states the
  raw score and what manual review found. A new answer filter is a functional change the day before the deadline,
  and it would need its own tests and a fresh eval run that the free LLM quota cannot cover today.
- **Trade-offs:** the published D score understates behaviour by one case, and the 3/30 suppression gap stays open
  until the filter exists (it is the first item in README "What I'd do next").
