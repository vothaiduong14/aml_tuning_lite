"""Load and standardize transaction data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aml_mvp.data.schema import to_standard_transaction_schema


def load_raw_transactions(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load raw IBM AML CSV data as strings to preserve account and bank IDs."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw transaction file not found: {csv_path}")
    return pd.read_csv(csv_path, dtype=str, nrows=max_rows)


def ingest_transactions(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load raw CSV data and return the standard transaction schema."""

    raw_df = load_raw_transactions(path, max_rows=max_rows)
    return to_standard_transaction_schema(raw_df)

