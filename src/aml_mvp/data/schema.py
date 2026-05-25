"""Schema contracts and standardization for IBM AML transaction data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_COLUMNS = [
    "transaction_id",
    "timestamp",
    "from_bank",
    "to_bank",
    "sender_account_id",
    "receiver_account_id",
    "amount_paid",
    "amount_received",
    "payment_currency",
    "receiving_currency",
    "payment_format",
    "currency_pair",
    "cross_bank_flag",
    "amount",
    "log_amount",
    "is_laundering",
]

MANDATORY_SOURCE_COLUMNS = [
    "timestamp",
    "from_bank",
    "account",
    "to_bank",
    "account_1",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]


@dataclass(frozen=True)
class SchemaValidationResult:
    is_valid: bool
    missing_columns: list[str]
    row_count: int


def normalize_column_name(column: object) -> str:
    """Normalize a source column into snake_case.

    Pandas renames duplicate CSV headers as `Account.1`; this function maps
    that shape to `account_1`.
    """

    name = str(column).strip().lower()
    name = name.replace(".1", "_1")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized source column names."""

    output = df.copy()
    output.columns = [normalize_column_name(column) for column in output.columns]
    return output


def validate_source_schema(df: pd.DataFrame) -> SchemaValidationResult:
    missing = [column for column in MANDATORY_SOURCE_COLUMNS if column not in df.columns]
    return SchemaValidationResult(
        is_valid=not missing,
        missing_columns=missing,
        row_count=len(df),
    )


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def to_standard_transaction_schema(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw IBM AML transactions to the MVP standard schema."""

    df = standardize_columns(raw_df)
    validation = validate_source_schema(df)
    if not validation.is_valid:
        raise ValueError(f"Missing required source columns: {validation.missing_columns}")

    output = pd.DataFrame(index=df.index)
    output["transaction_id"] = np.arange(len(df), dtype=np.int64)
    output["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y/%m/%d %H:%M",
        errors="coerce",
    )
    output["from_bank"] = df["from_bank"].astype("string").str.strip()
    output["to_bank"] = df["to_bank"].astype("string").str.strip()
    output["sender_account_id"] = df["account"].astype("string").str.strip()
    output["receiver_account_id"] = df["account_1"].astype("string").str.strip()
    output["amount_paid"] = pd.to_numeric(df["amount_paid"], errors="coerce")
    output["amount_received"] = pd.to_numeric(df["amount_received"], errors="coerce")
    output["payment_currency"] = df["payment_currency"].astype("string").str.strip()
    output["receiving_currency"] = df["receiving_currency"].astype("string").str.strip()
    output["payment_format"] = df["payment_format"].astype("string").str.strip()
    output["currency_pair"] = (
        output["payment_currency"].fillna("UNKNOWN")
        + " -> "
        + output["receiving_currency"].fillna("UNKNOWN")
    )
    output["cross_bank_flag"] = (output["from_bank"] != output["to_bank"]).astype("int8")
    output["amount"] = output["amount_paid"]
    output["log_amount"] = np.log1p(output["amount"].clip(lower=0))
    output["is_laundering"] = pd.to_numeric(df["is_laundering"], errors="coerce").astype("Int64")

    validate_standard_schema(output)
    return output[STANDARD_COLUMNS]


def validate_standard_schema(df: pd.DataFrame) -> None:
    require_columns(df, STANDARD_COLUMNS)
    if df["timestamp"].isna().any():
        raise ValueError("Column `timestamp` contains unparsable values.")
    if df["is_laundering"].isna().any():
        raise ValueError("Column `is_laundering` contains null or unparsable values.")
    if df["amount"].isna().any():
        raise ValueError("Column `amount` contains null or unparsable values.")
    if (df["amount"] < 0).any():
        raise ValueError("Column `amount` contains negative values.")
