from __future__ import annotations

import json

import pandas as pd

from aml_mvp.data.profile import build_profile, profile_transactions
from aml_mvp.data.quality_checks import build_quality_report, write_quality_report


def test_build_quality_report_contains_phase_2_gate_fields() -> None:
    df = _transactions()
    report = build_quality_report(df)

    assert report["row_count"] == 3
    assert report["label_count"] == 1
    assert report["duplicate_transaction_id_count"] == 0
    assert "missing_values" in report


def test_write_quality_report_outputs_json(tmp_path) -> None:
    output_path = write_quality_report(_transactions(), tmp_path / "dq.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["row_count"] == 3


def test_build_profile_contains_phase_3_fields() -> None:
    profile = build_profile(_transactions())

    assert profile["row_count"] == 3
    assert profile["payment_format_counts"]["ACH"] == 2
    assert profile["currency_pair_counts"]["US Dollar -> US Dollar"] == 3
    assert len(profile["split_summary"]) == 2


def test_profile_transactions_outputs_json(tmp_path) -> None:
    output_path = profile_transactions(_transactions(), tmp_path / "profile.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["amount_summary"]["max"] == 300.0


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "timestamp": pd.date_range("2022-01-01", periods=3, freq="h"),
            "payment_format": ["ACH", "Wire", "ACH"],
            "currency_pair": ["US Dollar -> US Dollar"] * 3,
            "from_bank": ["001", "001", "002"],
            "to_bank": ["002", "001", "003"],
            "amount": [100.0, 200.0, 300.0],
            "is_laundering": [0, 1, 0],
            "split": ["train", "train", "validation"],
        }
    )

