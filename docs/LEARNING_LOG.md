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
