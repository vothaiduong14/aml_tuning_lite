from __future__ import annotations

import pandas as pd

from aml_mvp.rules.rule_tuning import tune_rules


def test_tune_rules_selects_candidate_meeting_guardrails() -> None:
    transactions = _transactions()
    config = {
        "rules": {
            "enabled": ["R1_AMOUNT"],
            "segmentation_fields": ["payment_format", "currency_pair"],
            "fallback_percentile": 0.50,
            "min_segment_transactions": 1,
        },
        "R1_AMOUNT": {
            "percentile": 0.90,
            "percentile_grid": [0.50, 0.90],
            "severity": "medium",
        },
        "tuning": {
            "rules": ["R1_AMOUNT"],
            "split": "validation",
            "min_recall_floor": 0.80,
            "max_alert_rate": 1.00,
        },
    }

    candidates, selected, audit = tune_rules(transactions, config)

    assert len(candidates) == 2
    assert selected.loc[0, "selected_percentile"] == 0.50
    assert selected.loc[0, "selection_reason"] == "met_recall_floor_and_alert_rate_cap"
    assert audit["tuned_config"]["R1_AMOUNT"]["percentile"] == 0.50


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5, 6],
            "timestamp": pd.date_range("2022-01-01", periods=6, freq="h"),
            "from_bank": ["001"] * 6,
            "to_bank": ["002"] * 6,
            "sender_account_id": [f"S{i}" for i in range(6)],
            "receiver_account_id": [f"R{i}" for i in range(6)],
            "amount_paid": [10.0, 20.0, 100.0, 60.0, 95.0, 15.0],
            "amount_received": [10.0, 20.0, 100.0, 60.0, 95.0, 15.0],
            "payment_currency": ["USD"] * 6,
            "receiving_currency": ["USD"] * 6,
            "payment_format": ["ACH"] * 6,
            "currency_pair": ["USD -> USD"] * 6,
            "cross_bank_flag": [1] * 6,
            "amount": [10.0, 20.0, 100.0, 60.0, 95.0, 15.0],
            "log_amount": [0.0] * 6,
            "is_laundering": [0, 0, 1, 1, 1, 0],
            "split": ["train", "train", "train", "validation", "validation", "validation"],
        }
    )

