from __future__ import annotations

import pandas as pd


def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5],
            "timestamp": pd.date_range("2022-01-01", periods=5, freq="h"),
            "from_bank": ["001", "001", "002", "003", "004"],
            "to_bank": ["002", "003", "004", "005", "006"],
            "sender_account_id": ["S1", "S1", "S2", "S3", "S1"],
            "receiver_account_id": ["R1", "R2", "R2", "R3", "R4"],
            "amount_paid": [10.0, 100.0, 200.0, 50.0, 75.0],
            "amount_received": [10.0, 100.0, 200.0, 50.0, 75.0],
            "payment_currency": ["USD"] * 5,
            "receiving_currency": ["USD"] * 5,
            "payment_format": ["ACH", "ACH", "Wire", "ACH", "Wire"],
            "currency_pair": ["USD -> USD"] * 5,
            "cross_bank_flag": [1, 1, 1, 1, 1],
            "amount": [10.0, 100.0, 200.0, 50.0, 75.0],
            "log_amount": [2.4, 4.6, 5.3, 3.9, 4.3],
            "is_laundering": [0, 1, 1, 0, 0],
            "split": ["train", "train", "validation", "test", "test"],
        }
    )


def alerts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "alert_id": ["A1", "A2", "A3", "A4"],
            "transaction_id": [1, 2, 3, 4],
            "alert_timestamp": pd.date_range("2022-01-01", periods=4, freq="h"),
            "triggered_rules": ["R1_AMOUNT", "R1_AMOUNT,R2_NOVELTY", "R3_VELOCITY", "R4_PASS_THROUGH"],
            "rule_count": [1, 2, 1, 1],
            "max_rule_severity": ["medium", "medium", "high", "critical"],
            "rule_priority_score": [2.0, 2.1, 3.0, 4.0],
            "rationale": ["r1", "r1 r2", "r3", "r4"],
            "is_laundering": [0, 1, 1, 0],
            "split": ["train", "train", "validation", "test"],
        }
    )


def rule_hits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rule_hit_id": ["h1", "h2", "h3", "h4", "h5"],
            "transaction_id": [1, 2, 2, 3, 4],
            "alert_timestamp": pd.date_range("2022-01-01", periods=5, freq="h"),
            "rule_id": ["R1_AMOUNT", "R1_AMOUNT", "R2_NOVELTY", "R3_VELOCITY", "R4_PASS_THROUGH"],
            "rule_name": ["R1", "R1", "R2", "R3", "R4"],
            "severity": ["medium", "medium", "medium", "high", "critical"],
            "trigger_values_json": ["{}"] * 5,
            "threshold_json": ["{}"] * 5,
            "rationale": [""] * 5,
            "is_laundering": [0, 1, 1, 1, 0],
            "split": ["train", "train", "train", "validation", "test"],
        }
    )


def scored_alerts() -> pd.DataFrame:
    features = alerts().copy()
    features["target"] = features["is_laundering"]
    features["feature_rule_priority_score"] = features["rule_priority_score"]
    features["feature_amount"] = [10.0, 100.0, 200.0, 50.0]
    features["model_score"] = [0.1, 0.9, 0.8, 0.2]
    return features

