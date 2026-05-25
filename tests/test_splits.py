from __future__ import annotations

import pandas as pd

from aml_mvp.data.splits import create_temporal_splits, validate_split_non_overlap


def test_create_temporal_splits_assigns_every_row_once() -> None:
    df = _transactions(10)
    split_df, manifest = create_temporal_splits(
        df,
        {"train_ratio": 0.60, "validation_ratio": 0.20, "test_ratio": 0.20},
    )

    assert len(split_df) == 10
    assert split_df["split"].value_counts().to_dict() == {"train": 6, "validation": 2, "test": 2}
    assert manifest["row_count"] == 10
    assert manifest["non_overlapping"] is True


def test_create_temporal_splits_sorts_before_assigning() -> None:
    df = _transactions(5).sample(frac=1, random_state=42).reset_index(drop=True)
    split_df, _ = create_temporal_splits(df)

    assert split_df["timestamp"].is_monotonic_increasing
    assert validate_split_non_overlap(split_df, ["train", "validation", "test"])


def test_create_temporal_splits_rejects_bad_ratios() -> None:
    df = _transactions(5)

    try:
        create_temporal_splits(df, {"train_ratio": 0.50, "validation_ratio": 0.20, "test_ratio": 0.20})
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid split ratios")


def _transactions(row_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": range(row_count),
            "timestamp": pd.date_range("2022-01-01", periods=row_count, freq="h"),
            "is_laundering": [0, 1] * (row_count // 2) + [0] * (row_count % 2),
            "amount": [100.0 + i for i in range(row_count)],
        }
    )

