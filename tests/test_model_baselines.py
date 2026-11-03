"""Baselines and ranking metrics on a tiny hand-made candidate table with known answers."""

import numpy as np
import pandas as pd
import pytest

from reorderpilot.model.baselines import b1_eligible, b1_repeat_last_order, b2_most_frequent, b3_overdue
from reorderpilot.model.evaluate import abstention_curve, top_k, topk_metrics


@pytest.fixture
def cands():
    # One customer, one cutoff, 4 candidate products.
    return pd.DataFrame({
        "cutoff": pd.Timestamp("2011-03-01"), "customer_id": "A",
        "stock_code": ["P1", "P2", "P3", "P4"],
        "in_last_order": [False, True, False, False],
        "n_orders_p": [9, 1, 4, 2],
        "n_units_p": [90, 1, 40, 20],
        "overdue_ratio": [3.0, np.nan, 0.9, 1.5],
        "label": [0, 0, 1, 1],
    })


def _ranking(df, score, k=4, eligible=None):
    return list(top_k(df, score, k, eligible)["stock_code"])


def test_baseline_rankings(cands):
    assert _ranking(cands, b1_repeat_last_order(cands), eligible=b1_eligible(cands)) == ["P2"]  # only last order
    assert _ranking(cands, b2_most_frequent(cands)) == ["P1", "P3", "P4", "P2"]
    assert _ranking(cands, b3_overdue(cands)) == ["P3", "P4", "P1", "P2"]  # |r-1|: .1, .5, 2, NaN last


def test_topk_metrics_known_values(cands):
    m = topk_metrics(cands, b2_most_frequent(cands), k=2)   # top-2 = P1 (0), P3 (1)
    assert m["precision@2"] == 0.5 and m["recall@2"] == 0.5 and m["hit_rate@2"] == 1.0
    m1 = topk_metrics(cands, b1_repeat_last_order(cands), k=2, eligible=b1_eligible(cands))
    assert m1["precision@2"] == 0.0 and m1["hit_rate@2"] == 0.0   # empty slot counts as a miss


def test_abstention_curve_coverage_and_precision(cands):
    prob = pd.Series([0.1, 0.05, 0.6, 0.3])
    curve = abstention_curve(cands, prob, taus=[0.0, 0.5, 0.9], k=5, n_customers_total=2).set_index("tau")
    assert curve.loc[0.0, "coverage"] == 0.5 and curve.loc[0.0, "precision"] == 0.5
    assert curve.loc[0.5, "precision"] == 1.0 and curve.loc[0.5, "n_recommended"] == 1
    assert curve.loc[0.9, "coverage"] == 0.0
