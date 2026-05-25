from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from aml_mvp.config import load_config
from aml_mvp.data.schema import (
    STANDARD_COLUMNS,
    standardize_columns,
    to_standard_transaction_schema,
    validate_source_schema,
)


def test_config_loader_reads_phase_1_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "data_config.yaml"
    config = load_config(config_path)

    assert config["data"]["dataset_name"] == "LI-Small"
    assert config["splits"]["train_ratio"] == 0.60


def test_standardize_columns_handles_duplicate_account_name() -> None:
    raw = _raw_transactions()
    standardized = standardize_columns(raw)

    assert "account" in standardized.columns
    assert "account_1" in standardized.columns


def test_to_standard_transaction_schema_creates_required_columns() -> None:
    output = to_standard_transaction_schema(_raw_transactions())

    assert list(output.columns) == STANDARD_COLUMNS
    assert output["timestamp"].is_monotonic_increasing
    assert output["sender_account_id"].iloc[0] == "A1"
    assert output["receiver_account_id"].iloc[0] == "B1"
    assert output["currency_pair"].iloc[0] == "US Dollar -> Euro"
    assert output["cross_bank_flag"].iloc[0] == 1
    assert output["is_laundering"].sum() == 1


def test_validate_source_schema_reports_missing_columns() -> None:
    raw = standardize_columns(_raw_transactions()).drop(columns=["amount_paid"])
    result = validate_source_schema(raw)

    assert not result.is_valid
    assert result.missing_columns == ["amount_paid"]


def test_to_standard_transaction_schema_rejects_bad_timestamp() -> None:
    raw = _raw_transactions()
    raw.loc[0, "Timestamp"] = "not-a-date"

    with pytest.raises(ValueError, match="timestamp"):
        to_standard_transaction_schema(raw)


def _raw_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2022/09/01 00:00", "001", "A1", "002", "B1", "100.00", "Euro", "120.00", "US Dollar", "ACH", "0"],
            ["2022/09/01 00:01", "001", "A2", "001", "B2", "50.00", "US Dollar", "50.00", "US Dollar", "Wire", "1"],
            ["2022/09/01 00:02", "003", "A3", "004", "B3", "75.00", "US Dollar", "75.00", "US Dollar", "ACH", "0"],
        ],
        columns=[
            "Timestamp",
            "From Bank",
            "Account",
            "To Bank",
            "Account.1",
            "Amount Received",
            "Receiving Currency",
            "Amount Paid",
            "Payment Currency",
            "Payment Format",
            "Is Laundering",
        ],
    )
