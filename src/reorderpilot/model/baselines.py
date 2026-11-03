"""Baselines B1 repeat-last-order, B2 most-frequent, B3 overdue heuristic (SPEC section 6).

Each baseline returns a score per candidate pair (higher = recommend first). Scores are only for ranking;
they are not probabilities. `eligible` marks which pairs a baseline is allowed to recommend at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def b1_repeat_last_order(df: pd.DataFrame) -> pd.Series:
    """B1: products from the customer's last order day first (ties broken by how often they were bought)."""
    return df["in_last_order"].astype(float) * 1e6 + df["n_orders_p"]


def b1_eligible(df: pd.DataFrame) -> pd.Series:
    """B1 only recommends products from the last order; if it had < k products, the other slots stay empty."""
    return df["in_last_order"].astype(bool)


def b2_most_frequent(df: pd.DataFrame) -> pd.Series:
    """B2: number of orders containing the product (ties broken by units bought)."""
    return df["n_orders_p"] + df["n_units_p"] / (df["n_units_p"].max() + 1.0)


def b3_overdue(df: pd.DataFrame) -> pd.Series:
    """B3: overdue_ratio closest to 1 first; products bought only once (no interval) go last."""
    score = -(df["overdue_ratio"] - 1.0).abs()
    return score.fillna(score.min() - 1.0 if score.notna().any() else -1.0).replace(-np.inf, -1e9)


BASELINES = {
    "B1_repeat_last_order": (b1_repeat_last_order, b1_eligible),
    "B2_most_frequent": (b2_most_frequent, None),
    "B3_overdue": (b3_overdue, None),
}
