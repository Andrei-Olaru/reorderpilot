"""predict_for: cold-start abstention, low-confidence abstention, ranking and template explanations."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from reorderpilot.model.predict import explain, predict_for

AS_OF = date(2011, 3, 1)


class FakeBundle:
    """Stands in for the trained model: probability = share_of_customer_orders_with_p (deterministic)."""

    def __init__(self, tau=0.1):
        self.tau = tau

    def predict_proba(self, df):
        return np.asarray(df["share_of_customer_orders_with_p"], dtype=float)


def _tx(rows):
    return pd.DataFrame([{"invoice": inv, "stock_code": sc, "description": f"DESC {sc}", "quantity": 1,
                          "invoice_date": pd.Timestamp(d), "unit_price": 1.0, "customer_id": c, "country": "UK"}
                         for inv, c, sc, d in rows])


@pytest.fixture
def tx():
    return _tx([
        ("1", "A", "P1", "2011-01-01"), ("1", "A", "P2", "2011-01-01"),
        ("2", "A", "P1", "2011-01-21"),
        ("3", "A", "P1", "2011-02-10"),
        ("4", "B", "P1", "2011-01-05"), ("5", "B", "P1", "2011-02-05"),   # only 2 orders -> cold start
        ("6", "A", "P3", "2011-03-02"),                                    # after as_of -> ignored
    ])


def test_cold_start_and_unknown_customer_abstain(tx):
    r = predict_for(tx, FakeBundle(), "B", 5, AS_OF)
    assert r.abstain and "not enough history" in r.why and r.recommendations == []
    assert predict_for(tx, FakeBundle(), "ZZZ", 5, AS_OF).abstain


def test_ranked_recommendations_with_explanations(tx):
    r = predict_for(tx, FakeBundle(tau=0.1), "A", 5, AS_OF)
    assert not r.abstain
    assert [x.stock_code for x in r.recommendations] == ["P1", "P2"]        # P3 is after as_of: not a candidate
    assert r.recommendations[0].probability == 1.0
    assert r.recommendations[0].reason_text == "bought 3 times, usually every ~20 days, last bought 19 days ago"
    assert r.recommendations[0].description == "DESC P1"


def test_below_tau_items_dropped_and_all_below_abstains(tx):
    r = predict_for(tx, FakeBundle(tau=0.5), "A", 5, AS_OF)
    assert [x.stock_code for x in r.recommendations] == ["P1"]               # P2 = 1/3 < 0.5
    r2 = predict_for(tx, FakeBundle(tau=1.01), "A", 5, AS_OF)
    assert r2.abstain and "low confidence" in r2.why


def test_explain_overdue_flag():
    row = pd.Series({"n_orders_p": 4, "days_since_last_p": 30, "mean_interval_p": 21.4, "overdue_ratio": 1.4})
    assert explain(row) == "bought 4 times, usually every ~21 days, last bought 30 days ago (due now)"
