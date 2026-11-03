"""Read the raw Online Retail II file, clean it, and save/load `transactions.parquet`.

Run: `python -m reorderpilot.data.loader` (reads <DATA_DIR>/raw/, writes <DATA_DIR>/processed/).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from reorderpilot.config import get_settings
from reorderpilot.data.clean import clean_transactions

log = logging.getLogger(__name__)

RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
RAW_DTYPES = {"Invoice": str, "StockCode": str, "Description": str, "Country": str}


def find_raw_file(raw_dir: Path) -> Path:
    """Return the single raw data file (.csv preferred, else .xlsx) in raw_dir."""
    for pattern in ("*.csv", "*.xlsx"):
        files = sorted(raw_dir.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"No .csv or .xlsx file in {raw_dir}. Download Online Retail II there (see README).")


def read_raw(raw_dir: Path, cache_path: Path | None = None) -> pd.DataFrame:
    """Read the raw file (all xlsx sheets concatenated). Cached as parquet because xlsx parsing takes minutes."""
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path)
    path = find_raw_file(raw_dir)
    log.info("Reading %s", path)
    if path.suffix == ".csv":
        df = pd.read_csv(path, dtype=RAW_DTYPES, encoding="unicode_escape")
    else:
        sheets = pd.read_excel(path, sheet_name=None, dtype=RAW_DTYPES)
        df = pd.concat(sheets.values(), ignore_index=True)
    df = df[RAW_COLUMNS]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def build_transactions(data_dir: Path | None = None) -> pd.DataFrame:
    """Read raw data, apply cleaning rules, write transactions.parquet + cleaning_report.json."""
    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    processed = data_dir / "processed"
    raw = read_raw(data_dir / "raw", cache_path=processed / "raw_cache.parquet")
    clean, report = clean_transactions(raw)
    processed.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(processed / "transactions.parquet", index=False)
    (processed / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    return clean


@lru_cache(maxsize=2)
def load_transactions(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned transactions (cached in memory; treat the result as read-only)."""
    path = path or get_settings().transactions_path
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m reorderpilot.data.loader")
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = build_transactions()
    print(f"transactions: {df.shape}, {df['customer_id'].nunique()} customers, {df['stock_code'].nunique()} products")
