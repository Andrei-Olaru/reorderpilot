"""Cleaning rules (SPEC section 5) on a tiny hand-made raw table."""

import pandas as pd

from reorderpilot.data.clean import clean_transactions


def _raw_row(invoice="500001", code="85099b", desc="JUMBO BAG", qty=2, price=1.5, cust=12345.0, when="2011-01-05"):
    return {"Invoice": invoice, "StockCode": code, "Description": desc, "Quantity": qty,
            "InvoiceDate": pd.Timestamp(when), "Price": price, "Customer ID": cust, "Country": "UK"}


def test_cleaning_rules_and_report():
    raw = pd.DataFrame([
        _raw_row(),                                   # kept
        _raw_row(),                                   # exact duplicate -> dropped
        _raw_row(code="22423", desc="CAKESTAND"),     # kept
        _raw_row(code="22423", desc="cake stand ?", invoice="500002"),  # kept, rare description
        _raw_row(code="22423", desc="CAKESTAND", invoice="500003"),     # kept
        _raw_row(cust=None),                          # missing customer
        _raw_row(invoice="C500004", qty=-2),          # cancellation
        _raw_row(invoice="500005", qty=0),            # quantity <= 0
        _raw_row(invoice="500006", price=0.0),        # price <= 0
        _raw_row(invoice="500007", code="POST", desc="POSTAGE"),  # non-product
    ])
    clean, report = clean_transactions(raw)
    removed = {r["rule"]: r["rows_removed"] for r in report}

    assert len(clean) == 4
    assert removed["exact duplicate rows (incl. xlsx sheet overlap)"] == 1
    assert removed["missing Customer ID"] == 1
    assert removed["cancellation (Invoice starts with C)"] == 1
    assert removed["Quantity <= 0"] == 1
    assert removed["Price <= 0"] == 1
    assert removed["non-product StockCode"] == 1

    assert list(clean.columns) == ["invoice", "stock_code", "description", "quantity", "invoice_date", "unit_price",
                                   "customer_id", "country"]
    assert set(clean["customer_id"]) == {"12345"}             # no ".0"
    assert "85099B" in set(clean["stock_code"])                # uppercased
    assert set(clean.loc[clean.stock_code == "22423", "description"]) == {"CAKESTAND"}  # most frequent
