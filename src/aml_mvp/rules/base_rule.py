"""Common helpers for rule output rows."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


SEVERITY_SCORE = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def build_rule_hits(
    transactions: pd.DataFrame,
    mask: pd.Series,
    rule_id: str,
    rule_name: str,
    severity: str,
    trigger_values: pd.Series,
    threshold_values: pd.Series,
    rationale_values: pd.Series,
) -> pd.DataFrame:
    """Create rule hit rows following the MVP contract."""

    hit_transactions = transactions.loc[mask].copy()
    if hit_transactions.empty:
        return empty_rule_hits()

    trigger = trigger_values.loc[mask].map(_to_json)
    threshold = threshold_values.loc[mask].map(_to_json)
    rationale = rationale_values.loc[mask].astype(str)

    hits = pd.DataFrame(
        {
            "transaction_id": hit_transactions["transaction_id"].values,
            "alert_timestamp": hit_transactions["timestamp"].values,
            "rule_id": rule_id,
            "rule_name": rule_name,
            "severity": severity,
            "trigger_values_json": trigger.values,
            "threshold_json": threshold.values,
            "rationale": rationale.values,
            "is_laundering": hit_transactions["is_laundering"].astype(int).values,
            "split": hit_transactions["split"].values,
        }
    )
    hits.insert(0, "rule_hit_id", [f"{rule_id}:{transaction_id}" for transaction_id in hits["transaction_id"]])
    return hits


def build_rule_hits_from_records(
    hit_transactions: pd.DataFrame,
    rule_id: str,
    rule_name: str,
    severity: str,
    trigger_records: list[dict[str, Any]],
    threshold_records: list[dict[str, Any]],
    rationale_values: list[str],
) -> pd.DataFrame:
    """Create rule hit rows from already-filtered hit records.

    This avoids constructing evidence JSON for every source transaction when
    only a small subset becomes rule hits.
    """

    if hit_transactions.empty:
        return empty_rule_hits()
    if not (len(hit_transactions) == len(trigger_records) == len(threshold_records) == len(rationale_values)):
        raise ValueError("Hit transactions and evidence records must have matching lengths.")

    hits = pd.DataFrame(
        {
            "transaction_id": hit_transactions["transaction_id"].values,
            "alert_timestamp": hit_transactions["timestamp"].values,
            "rule_id": rule_id,
            "rule_name": rule_name,
            "severity": severity,
            "trigger_values_json": [_to_json(record) for record in trigger_records],
            "threshold_json": [_to_json(record) for record in threshold_records],
            "rationale": rationale_values,
            "is_laundering": hit_transactions["is_laundering"].astype(int).values,
            "split": hit_transactions["split"].values,
        }
    )
    hits.insert(0, "rule_hit_id", [f"{rule_id}:{transaction_id}" for transaction_id in hits["transaction_id"]])
    return hits


def empty_rule_hits() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "rule_hit_id",
            "transaction_id",
            "alert_timestamp",
            "rule_id",
            "rule_name",
            "severity",
            "trigger_values_json",
            "threshold_json",
            "rationale",
            "is_laundering",
            "split",
        ]
    )


def evidence_series(index: pd.Index, rows: list[dict[str, Any]]) -> pd.Series:
    return pd.Series(rows, index=index, dtype="object")


def _to_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)
