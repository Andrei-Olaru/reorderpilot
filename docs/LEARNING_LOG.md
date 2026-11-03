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
