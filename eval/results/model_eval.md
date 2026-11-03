# Model evaluation (test cutoffs 2011-09-01, 2011-10-01, 2011-11-01)

Generated 2026-11-03 by `eval/run_model_eval.py`. Customers with < 3 prior orders are abstained (cold start) and excluded from the table; per-customer metrics use only customers who re-ordered >= 1 previously bought product in the 30-day window.

| method | precision@5 | recall@5 | hit_rate@5 | pr_auc | roc_auc | brier |
|---|---|---|---|---|---|---|
| B1_repeat_last_order | 0.311 | 0.171 | 0.702 | 0.117 | 0.703 | n/a |
| B2_most_frequent | 0.333 | 0.191 | 0.751 | 0.162 | 0.716 | n/a |
| B3_overdue | 0.233 | 0.150 | 0.627 | 0.105 | 0.706 | n/a |
| HGB_calibrated | 0.381 | 0.222 | 0.796 | 0.261 | 0.838 | 0.038 |

B2 shown at full coverage; precision@5 of B2 over all eligible customers = 0.105.

**Abstention at tau=0.08:** coverage 32.3% of customers with history, precision of recommended items 0.203, 21100 items.

## Population

- customer_cutoff_pairs_with_history: 16269
- eligible_pairs_(>=3_orders): 8671
- cold_start_pairs_abstained: 7598
- customers_ordering_anything_in_window: 4242
- eligible_customers_reordering_>=1_prior_product: 2732
- n_test_pairs: 1030502
- test_base_rate: 0.04474227124255945

Plots: `calibration.png`, `abstention_curve.png`.
