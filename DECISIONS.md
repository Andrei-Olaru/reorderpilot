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
