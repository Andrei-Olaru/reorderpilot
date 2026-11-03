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
