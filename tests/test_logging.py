from __future__ import annotations

import logging

import pandas as pd

from aml_mvp.logging_utils import setup_logging
from aml_mvp.rules.rule_engine import run_rule_engine
from aml_mvp.rules.rule_tuning import tune_rules


def test_setup_logging_writes_info_message_to_file(tmp_path) -> None:
    logger, log_path = setup_logging(
        "run-rules",
        tmp_path,
        "INFO",
        "outputs/run_logs/custom.log",
    )

    logger.info("hello progress")
    for handler in logging.getLogger("aml_mvp").handlers:
        handler.flush()

    assert log_path.exists()
    assert "hello progress" in log_path.read_text(encoding="utf-8")


def test_setup_logging_respects_log_level(tmp_path) -> None:
    logger, log_path = setup_logging(
        "run-rules",
        tmp_path,
        "WARNING",
        "outputs/run_logs/warn.log",
    )

    logger.info("hidden message")
    logger.warning("visible message")
    for handler in logging.getLogger("aml_mvp").handlers:
        handler.flush()

    text = log_path.read_text(encoding="utf-8")
    assert "visible message" in text
    assert "hidden message" not in text


def test_rule_engine_logs_rule_start_and_completion(caplog) -> None:
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
    logger = logging.getLogger("test.rule_engine")

    with caplog.at_level(logging.INFO, logger="test.rule_engine"):
        run_rule_engine(transactions, config, logger=logger)

    messages = [record.getMessage() for record in caplog.records]
    assert any("Starting R1_AMOUNT" in message for message in messages)
    assert any("Completed R1_AMOUNT: hits=" in message for message in messages)
    assert any("Starting R2_NOVELTY" in message for message in messages)
    assert any("Completed R2_NOVELTY: hits=" in message for message in messages)


def test_tuning_logs_each_candidate_and_selection(caplog) -> None:
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
    logger = logging.getLogger("test.tuning")

    with caplog.at_level(logging.INFO, logger="test.tuning"):
        tune_rules(transactions, config, logger=logger)

    messages = [record.getMessage() for record in caplog.records]
    assert any("Starting R1_AMOUNT candidate percentile=0.5000" in message for message in messages)
    assert any("Completed R1_AMOUNT candidate percentile=0.9000" in message for message in messages)
    assert any("Selected R1_AMOUNT percentile=" in message for message in messages)


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

