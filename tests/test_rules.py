from __future__ import annotations

import pandas as pd

from aml_mvp.rules.pass_through_rule import apply_pass_through_rule
from aml_mvp.rules.network_rules import apply_fan_out_rule
from aml_mvp.rules.rule_engine import run_rule_engine


def test_rule_engine_runs_r1_r2_and_consolidates_alerts() -> None:
    transactions = _transactions()
    config = {
        "rules": {
            "enabled": ["R1_AMOUNT", "R2_NOVELTY"],
            "segmentation_fields": ["payment_format", "currency_pair"],
            "fallback_percentile": 0.50,
            "min_segment_transactions": 1,
        },
        "R1_AMOUNT": {"percentile": 0.50, "severity": "medium"},
        "R2_NOVELTY": {"require_cross_bank": True, "severity": "medium"},
    }

    rule_hits, alerts = run_rule_engine(transactions, config)

    assert {"R1_AMOUNT", "R2_NOVELTY"}.issubset(set(rule_hits["rule_id"]))
    assert not alerts.empty
    assert alerts["alert_id"].str.startswith("ALERT-").all()
    assert alerts["rule_count"].ge(1).all()
    assert alerts["rationale"].str.len().gt(0).all()


def test_pass_through_rule_flags_outgoing_after_recent_incoming() -> None:
    transactions = _transactions()

    hits = apply_pass_through_rule(
        transactions,
        {"window": "24h", "min_in_out_ratio": 0.80, "severity": "critical"},
    )

    assert 2 in set(hits["transaction_id"])
    assert "R4_PASS_THROUGH" in set(hits["rule_id"])


def test_fan_out_rule_uses_prior_incoming_funders_for_current_sender() -> None:
    transactions = _fan_out_transactions()

    hits = apply_fan_out_rule(
        transactions,
        {
            "window": "24h",
            "min_unique_receivers": 2,
            "min_prior_unique_senders": 2,
            "severity": "high",
        },
    )

    assert 4 in set(hits["transaction_id"])
    assert "R6_FAN_OUT" in set(hits["rule_id"])


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5, 6],
            "timestamp": pd.to_datetime(
                [
                    "2022-01-01 00:00",
                    "2022-01-01 01:00",
                    "2022-01-01 02:00",
                    "2022-01-01 03:00",
                    "2022-01-01 04:00",
                    "2022-01-01 05:00",
                ]
            ),
            "from_bank": ["001", "001", "002", "003", "004", "005"],
            "to_bank": ["001", "002", "003", "003", "003", "003"],
            "sender_account_id": ["F1", "A", "A", "S1", "S2", "S3"],
            "receiver_account_id": ["A", "B", "C", "R", "R", "R"],
            "amount_paid": [100.0, 90.0, 200.0, 25.0, 30.0, 35.0],
            "amount_received": [100.0, 90.0, 200.0, 25.0, 30.0, 35.0],
            "payment_currency": ["USD"] * 6,
            "receiving_currency": ["USD"] * 6,
            "payment_format": ["ACH"] * 6,
            "currency_pair": ["USD -> USD"] * 6,
            "cross_bank_flag": [0, 1, 1, 0, 0, 0],
            "amount": [100.0, 90.0, 200.0, 25.0, 30.0, 35.0],
            "log_amount": [0.0] * 6,
            "is_laundering": [0, 1, 0, 0, 1, 0],
            "split": ["train", "validation", "validation", "train", "validation", "test"],
        }
    )


def _fan_out_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4],
            "timestamp": pd.to_datetime(
                [
                    "2022-01-01 00:00",
                    "2022-01-01 00:30",
                    "2022-01-01 01:00",
                    "2022-01-01 02:00",
                ]
            ),
            "from_bank": ["001", "002", "003", "003"],
            "to_bank": ["003", "003", "004", "005"],
            "sender_account_id": ["F1", "F2", "A", "A"],
            "receiver_account_id": ["A", "A", "B", "C"],
            "amount_paid": [100.0, 150.0, 40.0, 50.0],
            "amount_received": [100.0, 150.0, 40.0, 50.0],
            "payment_currency": ["USD"] * 4,
            "receiving_currency": ["USD"] * 4,
            "payment_format": ["ACH"] * 4,
            "currency_pair": ["USD -> USD"] * 4,
            "cross_bank_flag": [1] * 4,
            "amount": [100.0, 150.0, 40.0, 50.0],
            "log_amount": [0.0] * 4,
            "is_laundering": [0, 0, 1, 1],
            "split": ["train", "train", "validation", "validation"],
        }
    )
