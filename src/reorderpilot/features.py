"""Build (customer, product, cutoff T) features using only rows strictly before T, plus labels in [T, T+30d).

An "order" is one invoice. Intervals are measured between distinct order *days* (two invoices on the same day
are one purchase occasion, otherwise we would get 0-day intervals).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd

from reorderpilot.config import LABEL_WINDOW_DAYS

KEY = ["customer_id", "stock_code"]
POPULARITY_WINDOW_DAYS = 90

FEATURE_COLUMNS = [
    "n_orders_p", "n_units_p", "days_since_last_p", "mean_interval_p", "std_interval_p", "overdue_ratio",
    "share_of_customer_orders_with_p", "avg_qty_p", "avg_price_p",
    "cust_n_orders", "cust_days_since_last_order", "cust_mean_order_interval", "cust_tenure_days",
    "product_popularity_90d", "month",
]


def _ts(cutoff: date | pd.Timestamp) -> pd.Timestamp:
    """Cutoff as a midnight Timestamp, so "before T" means "before day T starts"."""
    return pd.Timestamp(cutoff).normalize()


def history_before(tx: pd.DataFrame, cutoff: date | pd.Timestamp) -> pd.DataFrame:
    """The ONLY place features get their rows: transactions strictly before the cutoff."""
    return tx[tx["invoice_date"] < _ts(cutoff)]


def _interval_stats(days: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    """Mean/std of gaps (in days) between consecutive distinct purchase days, per group."""
    d = days[keys + ["day"]].drop_duplicates().sort_values(keys + ["day"])
    # diff() within each group: first purchase -> NaN gap, so a pair bought once has NaN mean (handled downstream).
    gap = d.groupby(keys)["day"].diff().dt.days
    g = d.assign(gap=gap).groupby(keys)["gap"]
    return pd.DataFrame({f"mean_interval{prefix}": g.mean(), f"std_interval{prefix}": g.std(ddof=0)})


def build_features(tx: pd.DataFrame, cutoff: date | pd.Timestamp,
                   customer_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """One row per (customer, product the customer bought before T) with FEATURE_COLUMNS + helper columns.

    Helper columns (not model features): cutoff, in_last_order, description.
    """
    T = _ts(cutoff)
    hist = history_before(tx, T)   # leakage guard: nothing on or after T is visible below this line
    # Popularity uses ALL customers (before filtering to customer_ids): it is a property of the product.
    popularity = (hist[hist["invoice_date"] >= T - pd.Timedelta(days=POPULARITY_WINDOW_DAYS)]
                  .groupby("stock_code")["customer_id"].nunique().rename("product_popularity_90d"))
    if customer_ids is not None:
        hist = hist[hist["customer_id"].isin(set(customer_ids))]
    if hist.empty:
        return pd.DataFrame(columns=KEY + FEATURE_COLUMNS + ["cutoff", "in_last_order"])

    # One line per (customer, invoice, product): an invoice can list the same product twice.
    lines = (hist.assign(day=hist["invoice_date"].dt.normalize())
             .groupby(KEY + ["invoice"], as_index=False)
             .agg(qty=("quantity", "sum"), price=("unit_price", "mean"), day=("day", "min")))

    # Customer-level features (all products).
    cg = lines.groupby("customer_id")
    cust = pd.DataFrame({
        "cust_n_orders": cg["invoice"].nunique(),
        "cust_last_day": cg["day"].max(),
        "cust_first_day": cg["day"].min(),
    })
    cust["cust_days_since_last_order"] = (T - cust["cust_last_day"]).dt.days
    cust["cust_tenure_days"] = (T - cust["cust_first_day"]).dt.days
    cust = cust.join(_interval_stats(lines, ["customer_id"], "_c")["mean_interval_c"]
                     .rename("cust_mean_order_interval"))

    # Pair-level features.
    pg = lines.groupby(KEY)
    pair = pd.DataFrame({
        "n_orders_p": pg["invoice"].nunique(),
        "n_units_p": pg["qty"].sum(),
        "avg_qty_p": pg["qty"].mean(),
        "avg_price_p": pg["price"].mean(),
        "last_day_p": pg["day"].max(),
    })
    pair = pair.join(_interval_stats(lines, KEY, "_p")).reset_index()
    pair = pair.merge(cust.reset_index(), on="customer_id", how="left")
    pair["days_since_last_p"] = (T - pair["last_day_p"]).dt.days
    # > 1 means "it has been longer than this customer's usual gap for this product". NaN when there is no gap yet.
    pair["overdue_ratio"] = pair["days_since_last_p"] / pair["mean_interval_p"].replace(0, np.nan)
    pair["share_of_customer_orders_with_p"] = pair["n_orders_p"] / pair["cust_n_orders"]
    pair["in_last_order"] = pair["last_day_p"] == pair["cust_last_day"]
    pair = pair.merge(popularity.reset_index(), on="stock_code", how="left")
    pair["product_popularity_90d"] = pair["product_popularity_90d"].fillna(0)   # not sold in the last 90 days
    pair["month"] = T.month
    pair["cutoff"] = T
    return pair[KEY + ["cutoff"] + FEATURE_COLUMNS + ["in_last_order"]]


def build_labels(tx: pd.DataFrame, cutoff: date | pd.Timestamp, window_days: int = LABEL_WINDOW_DAYS) -> pd.DataFrame:
    """(customer, product) pairs bought in [T, T + window_days). Label construction only; never used as a feature."""
    T = _ts(cutoff)
    win = tx[(tx["invoice_date"] >= T) & (tx["invoice_date"] < T + pd.Timedelta(days=window_days))]
    return win[KEY].drop_duplicates().assign(label=1)


def build_dataset(tx: pd.DataFrame, cutoffs: Iterable[date], min_prior_orders: int = 0) -> pd.DataFrame:
    """Features + label for every candidate pair at every cutoff; keeps customers with >= min_prior_orders orders."""
    parts = []
    for c in cutoffs:
        feats = build_features(tx, c)
        feats = feats[feats["cust_n_orders"] >= min_prior_orders]
        # Left join: candidates not bought in the window get NaN -> label 0. New products are not candidates.
        feats = feats.merge(build_labels(tx, c), on=KEY, how="left")
        feats["label"] = feats["label"].fillna(0).astype("int8")
        parts.append(feats)
    return pd.concat(parts, ignore_index=True)


def customers_ordering_in_window(tx: pd.DataFrame, cutoff: date | pd.Timestamp,
                                 window_days: int = LABEL_WINDOW_DAYS) -> set[str]:
    """Customers with any order in [T, T + window_days) (used to report how many re-order nothing old)."""
    T = _ts(cutoff)
    win = tx[(tx["invoice_date"] >= T) & (tx["invoice_date"] < T + pd.Timedelta(days=window_days))]
    return set(win["customer_id"])
