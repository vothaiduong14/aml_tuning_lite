"""Dataset profiling outputs for Phase 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_profile(df: pd.DataFrame, top_n: int = 20) -> dict[str, Any]:
    """Build a compact JSON profile for the standardized transaction table."""

    profile: dict[str, Any] = {
        "row_count": int(len(df)),
        "date_min": _iso_or_none(df["timestamp"].min()),
        "date_max": _iso_or_none(df["timestamp"].max()),
        "label_rate": float(df["is_laundering"].mean()) if len(df) else 0.0,
        "payment_format_counts": _value_counts(df, "payment_format", top_n),
        "currency_pair_counts": _value_counts(df, "currency_pair", top_n),
        "from_bank_counts": _value_counts(df, "from_bank", top_n),
        "to_bank_counts": _value_counts(df, "to_bank", top_n),
        "amount_summary": _numeric_summary(df["amount"]),
    }

    if "split" in df.columns:
        split_summary = (
            df.groupby("split", dropna=False)
            .agg(
                row_count=("transaction_id", "count"),
                label_count=("is_laundering", "sum"),
                label_rate=("is_laundering", "mean"),
                timestamp_min=("timestamp", "min"),
                timestamp_max=("timestamp", "max"),
            )
            .reset_index()
        )
        profile["split_summary"] = [
            {
                "split": str(row["split"]),
                "row_count": int(row["row_count"]),
                "label_count": int(row["label_count"]),
                "label_rate": float(row["label_rate"]),
                "timestamp_min": _iso_or_none(row["timestamp_min"]),
                "timestamp_max": _iso_or_none(row["timestamp_max"]),
            }
            for _, row in split_summary.iterrows()
        ]

    return profile


def profile_transactions(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_profile(df), indent=2), encoding="utf-8")
    return output_path


def _value_counts(df: pd.DataFrame, column: str, top_n: int) -> dict[str, int]:
    counts = df[column].value_counts(dropna=False).head(top_n)
    return {str(key): int(value) for key, value in counts.items()}


def _numeric_summary(series: pd.Series) -> dict[str, float | None]:
    if series.empty:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
    }


def _iso_or_none(value: object) -> str | None:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)

