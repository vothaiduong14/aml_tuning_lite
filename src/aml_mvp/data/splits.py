"""Temporal splitting for standardized AML transaction data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def create_temporal_splits(
    df: pd.DataFrame,
    split_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign train/validation/test splits by transaction timestamp."""

    config = split_config or {}
    train_ratio = float(config.get("train_ratio", 0.60))
    validation_ratio = float(config.get("validation_ratio", 0.20))
    test_ratio = float(config.get("test_ratio", 0.20))
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    train_label = str(config.get("train_label", "train"))
    validation_label = str(config.get("validation_label", "validation"))
    test_label = str(config.get("test_label", "test"))

    output = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True).copy()
    row_count = len(output)
    train_end = int(row_count * train_ratio)
    validation_end = train_end + int(row_count * validation_ratio)

    output["split"] = test_label
    output.loc[: train_end - 1, "split"] = train_label
    output.loc[train_end : validation_end - 1, "split"] = validation_label

    manifest = build_split_manifest(output, [train_label, validation_label, test_label])
    manifest["ratios"] = {
        train_label: train_ratio,
        validation_label: validation_ratio,
        test_label: test_ratio,
    }
    return output, manifest


def build_split_manifest(df: pd.DataFrame, split_order: list[str]) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split_name in split_order:
        split_df = df[df["split"] == split_name]
        splits[split_name] = {
            "row_count": int(len(split_df)),
            "label_count": int(split_df["is_laundering"].sum()) if len(split_df) else 0,
            "label_rate": float(split_df["is_laundering"].mean()) if len(split_df) else 0.0,
            "timestamp_min": _iso_or_none(split_df["timestamp"].min()) if len(split_df) else None,
            "timestamp_max": _iso_or_none(split_df["timestamp"].max()) if len(split_df) else None,
        }

    return {
        "method": "temporal_ratio",
        "row_count": int(len(df)),
        "splits": splits,
        "non_overlapping": validate_split_non_overlap(df, split_order),
    }


def validate_split_non_overlap(df: pd.DataFrame, split_order: list[str]) -> bool:
    previous_max = None
    for split_name in split_order:
        split_df = df[df["split"] == split_name]
        if split_df.empty:
            continue
        current_min = split_df["timestamp"].min()
        current_max = split_df["timestamp"].max()
        if previous_max is not None and current_min < previous_max:
            return False
        previous_max = current_max
    return True


def write_split_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def _iso_or_none(value: object) -> str | None:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)

