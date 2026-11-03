"""Cleaning rules for Online Retail II (SPEC section 5). Each rule logs how many rows it removes."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

log = logging.getLogger(__name__)

# Service / accounting lines, not products. Found by listing every StockCode that is not
# "5 digits + optional letters" after the other rules (see DECISIONS.md D3). SP1002 is a real product (kept).
# The last group does not survive the other rules in this dataset but is listed for safety.
NON_PRODUCT_CODES: set[str] = {
    "POST", "DOT", "C2", "M", "D", "ADJUST", "ADJUST2", "BANK CHARGES", "PADS", "TEST001", "TEST002",
    "AMAZONFEE", "B", "CRUK", "S",
}

OUTPUT_COLUMNS = ["invoice", "stock_code", "description", "quantity", "invoice_date", "unit_price", "customer_id",
                  "country"]


def _drop(df: pd.DataFrame, mask: pd.Series, rule: str, report: list[dict]) -> pd.DataFrame:
    """Remove rows where mask is True and record the count."""
    n = int(mask.sum())
    report.append({"rule": rule, "rows_removed": n})
    log.info("%-45s removed %8d rows", rule, n)
    return df[~mask]


def normalize_customer_id(s: pd.Series) -> pd.Series:
    """12345.0 / '12345.0' / 12345 -> '12345'."""
    return pd.to_numeric(s).astype("int64").astype(str)


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply cleaning rules in order; return (clean df, report of rows removed per rule)."""
    report: list[dict] = [{"rule": "input rows", "rows_removed": 0, "rows": len(raw)}]
    df = raw.copy()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Invoice"] = df["Invoice"].astype(str).str.strip()

    rules: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("exact duplicate rows (incl. xlsx sheet overlap)", lambda d: d.duplicated()),
        ("missing Customer ID", lambda d: d["Customer ID"].isna()),
        ("cancellation (Invoice starts with C)", lambda d: d["Invoice"].str.startswith("C")),
        ("Quantity <= 0", lambda d: d["Quantity"] <= 0),
        ("Price <= 0", lambda d: d["Price"] <= 0),
        ("non-product StockCode", lambda d: d["StockCode"].isin(NON_PRODUCT_CODES)),
    ]
    for name, rule in rules:
        df = _drop(df, rule(df), name, report)

    # One description per StockCode: the most frequent one (ties -> alphabetical, deterministic).
    desc = (df.dropna(subset=["Description"])
            .assign(Description=lambda d: d["Description"].str.strip())
            .groupby(["StockCode", "Description"]).size().reset_index(name="n")
            .sort_values(["StockCode", "n", "Description"], ascending=[True, False, True])
            .drop_duplicates("StockCode").set_index("StockCode")["Description"])

    out = pd.DataFrame({
        "invoice": df["Invoice"],
        "stock_code": df["StockCode"],
        "description": df["StockCode"].map(desc).fillna(""),
        "quantity": df["Quantity"].astype("int64"),
        "invoice_date": pd.to_datetime(df["InvoiceDate"]),
        "unit_price": df["Price"].astype(float),
        "customer_id": normalize_customer_id(df["Customer ID"]),
        "country": df["Country"].astype(str),
    })[OUTPUT_COLUMNS].sort_values(["invoice_date", "invoice", "stock_code"]).reset_index(drop=True)
    report.append({"rule": "output rows", "rows_removed": 0, "rows": len(out)})
    return out, report
