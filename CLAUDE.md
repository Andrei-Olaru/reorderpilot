# CLAUDE.md — ReorderPilot

Read `SPEC.md` first. It is the source of truth. If this file and SPEC.md disagree, SPEC.md wins; tell the human.

## What this project is
B2B re-order prediction + LLM sales agent with persistent feedback memory. Python 3.11, FastAPI, scikit-learn,
SQLite, Streamlit, pytest. Dataset: UCI Online Retail II. The human is a junior engineer who must be able to
explain every part in a job interview.

## Commands
- Setup: `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`) then `pip install -r requirements.txt`
- Tests: `pytest -q`    Lint: `ruff check .`
- Train: `python -m reorderpilot.model.train`
- API: `uvicorn reorderpilot.api.main:app --reload`
- UI: `streamlit run app/streamlit_app.py`
- Eval: `python eval/run_model_eval.py`, `python eval/run_memory_eval.py`, `python eval/run_agent_eval.py`

## Rules (important)
1. **Plan before coding** for any task bigger than ~30 lines: write a short plan, wait for approval if the human asked for it.
2. **Stay in your lane.** Only edit files in the area you were assigned. Interfaces/contracts come from SPEC.md;
   do not change a contract silently — propose the change to the human.
3. **No data leakage.** Features for cutoff T use only rows strictly before T. Test cutoffs are used once, at the end.
4. **Never invent results.** Metrics come from running code. If a run fails, report it; do not fabricate numbers.
5. **Tests with every module.** Small, meaningful tests. Run `pytest -q` before saying a task is done.
6. **Simple over clever.** No new dependency without asking. No frameworks for the agent loop (hand-written).
7. **Secrets:** never commit `.env`, API keys, or raw data. Use `.env.example`.
8. **Git:** small commits, message format `type: short description` (feat, fix, test, docs, refactor).
   Work on your own branch; do not push to `main` directly.
9. **Teach as you go.** After finishing each task, append to `docs/LEARNING_LOG.md` (5-8 plain-language bullets):
   what you built, why this approach, what could go wrong, and one question an interviewer might ask about it
   with a short answer. Write for a beginner; no jargon without a one-line explanation.
10. **Decisions:** when you choose between alternatives, add an entry to `DECISIONS.md`
    (context, options considered, choice, why, trade-offs).
11. Prefer type hints, small functions, docstrings with one-line purpose. Fixed random seeds. Paths via `config.py`.

## Token economy (the human has a limited usage quota)
- Never print whole dataframes or long logs: use `.shape`, `.head(5)`, `.describe()`, and truncate command output (`| head -50`).
- Do not re-read large files you already know; read only the line ranges you need.
- Do not reformat or rewrite files you were not asked to change. No "polish passes" unless requested.
- Keep answers short: summary of what changed + how to run it. Long explanations go into `docs/LEARNING_LOG.md`, once.
- If a task looks bigger than expected, stop and propose splitting it instead of continuing for a long time.

## Definition of done for any task
Code + tests pass + lint clean + LEARNING_LOG entry (+ DECISIONS entry if applicable) + short summary to the human
(what changed, how to run it, what is still missing).
