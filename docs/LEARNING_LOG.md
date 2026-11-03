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
