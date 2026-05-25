"""Data quality reporting for standardized transactions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Build a JSON-serializable data quality report."""

    label_counts = df["is_laundering"].value_counts(dropna=False).sort_index()
    missing_values = df.isna().sum().sort_values(ascending=False)

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "date_min": _iso_or_none(df["timestamp"].min()),
        "date_max": _iso_or_none(df["timestamp"].max()),
        "label_count": int(df["is_laundering"].sum()),
        "label_rate": float(df["is_laundering"].mean()) if len(df) else 0.0,
        "label_distribution": {str(key): int(value) for key, value in label_counts.items()},
        "duplicate_transaction_id_count": int(df["transaction_id"].duplicated().sum()),
        "missing_values": {column: int(count) for column, count in missing_values.items()},
        "amount_min": float(df["amount"].min()) if len(df) else None,
        "amount_max": float(df["amount"].max()) if len(df) else None,
        "amount_mean": float(df["amount"].mean()) if len(df) else None,
    }


def write_quality_report(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_quality_report(df)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def _iso_or_none(value: object) -> str | None:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)

