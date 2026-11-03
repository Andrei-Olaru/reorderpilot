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
