"""Feature builder: hand-checked values, the no-leakage guarantee, and label construction."""

from datetime import date

import pandas as pd
import pytest

from reorderpilot.features import FEATURE_COLUMNS, KEY, build_dataset, build_features, build_labels

T = date(2011, 3, 1)


def _tx(rows):
    return pd.DataFrame(
        [{"invoice": inv, "stock_code": sc, "description": sc, "quantity": q, "invoice_date": pd.Timestamp(d),
          "unit_price": 2.0, "customer_id": c, "country": "UK"} for inv, c, sc, q, d in rows])


@pytest.fixture
def tx():
    return _tx([
        ("1", "A", "P1", 10, "2011-01-01 10:00"),
        ("1", "A", "P2", 5, "2011-01-01 10:00"),
        ("2", "A", "P1", 20, "2011-01-21 09:00"),
        ("3", "A", "P1", 30, "2011-02-10 12:00"),
        ("4", "B", "P1", 1, "2010-12-15 08:00"),   # 76 days before T -> inside the 90-day popularity window
    ])


def test_feature_values_hand_checked(tx):
    f = build_features(tx, T).set_index(KEY)
    a1 = f.loc[("A", "P1")]
    assert a1["n_orders_p"] == 3 and a1["n_units_p"] == 60 and a1["avg_qty_p"] == 20
    assert a1["mean_interval_p"] == 20 and a1["std_interval_p"] == 0
    assert a1["days_since_last_p"] == 19                     # 2011-02-10 -> 2011-03-01
    assert a1["overdue_ratio"] == pytest.approx(19 / 20)
    assert a1["share_of_customer_orders_with_p"] == 1.0
    assert a1["cust_n_orders"] == 3 and a1["cust_tenure_days"] == 59
    assert a1["product_popularity_90d"] == 2 and a1["month"] == 3
    a2 = f.loc[("A", "P2")]
    assert pd.isna(a2["mean_interval_p"]) and pd.isna(a2["overdue_ratio"])  # bought once -> no interval
    assert a2["share_of_customer_orders_with_p"] == pytest.approx(1 / 3)
    assert bool(a1["in_last_order"]) and not bool(a2["in_last_order"])


def test_no_leakage_rows_at_or_after_cutoff_change_nothing(tx):
    """Adding any data at/after T (same customers, existing and new products) must not change any feature."""
    future = _tx([
        ("9", "A", "P1", 999, "2011-03-01 00:00"),   # exactly at T -> must be excluded
        ("10", "A", "P2", 999, "2011-03-05 10:00"),
        ("11", "A", "P9", 999, "2011-04-01 10:00"),  # new product -> must not become a candidate
        ("12", "C", "P1", 999, "2011-03-02 10:00"),  # new customer -> must not change popularity
    ])
    before = build_features(tx, T)
    after = build_features(pd.concat([tx, future], ignore_index=True), T)
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
    assert set(FEATURE_COLUMNS) <= set(after.columns)


def test_labels_window_is_half_open():
    tx = _tx([
        ("1", "A", "P1", 1, "2011-02-28 23:59"),  # before T -> not a label
        ("2", "A", "P2", 1, "2011-03-01 00:00"),  # at T -> label
        ("3", "A", "P3", 1, "2011-03-30 23:59"),  # T+29d -> label
        ("4", "A", "P4", 1, "2011-03-31 00:00"),  # T+30d -> outside
    ])
    assert set(build_labels(tx, T)["stock_code"]) == {"P2", "P3"}


def test_dataset_candidates_are_prior_products_only(tx):
    later = _tx([("5", "A", "P1", 1, "2011-03-10"), ("6", "A", "P7", 1, "2011-03-10")])
    ds = build_dataset(pd.concat([tx, later], ignore_index=True), [T], min_prior_orders=3)
    assert set(ds["customer_id"]) == {"A"}                      # B has 1 order -> filtered
    assert dict(zip(ds["stock_code"], ds["label"], strict=True)) == {"P1": 1, "P2": 0}  # P7 is new -> not a candidate
