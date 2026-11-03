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
