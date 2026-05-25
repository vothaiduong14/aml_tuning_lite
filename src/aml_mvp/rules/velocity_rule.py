"""R3 velocity / structuring rule."""

from __future__ import annotations

import logging
import time

import pandas as pd

from aml_mvp.rules.base_rule import build_rule_hits_from_records


def apply_velocity_rule(
    transactions: pd.DataFrame,
    training_transactions: pd.DataFrame,
    rule_config: dict,
) -> pd.DataFrame:
    window = str(rule_config.get("window", "24h"))
    count_threshold = int(rule_config.get("count_threshold", 5))
    amount_percentile = float(rule_config.get("amount_percentile", 0.90))

    scored = _add_sender_rolling_metrics(transactions, window)
    train_scored = _add_sender_rolling_metrics(training_transactions, window)
    amount_threshold = float(train_scored["sender_amount_sum_window"].quantile(amount_percentile))

    mask = (
        scored["sender_txn_count_window"].ge(count_threshold)
        & scored["sender_amount_sum_window"].ge(amount_threshold)
    )
    hits = scored.loc[mask]
    trigger_records = [
        {
            "sender_account_id": row.sender_account_id,
            "rolling_count": int(row.sender_txn_count_window),
            "rolling_sum": row.sender_amount_sum_window,
            "window": window,
        }
        for row in hits.itertuples(index=False)
    ]
    threshold_records = [
        {
            "count_threshold": count_threshold,
            "amount_threshold": amount_threshold,
            "amount_percentile": amount_percentile,
        }
        for _ in range(len(hits))
    ]
    rationale_values = [
            (
                f"Sender made {int(row.sender_txn_count_window)} transactions totaling "
                f"{row.sender_amount_sum_window:.2f} in {window}."
            )
            for row in hits.itertuples(index=False)
    ]

    return build_rule_hits_from_records(
        hits,
        "R3_VELOCITY",
        "Velocity / structuring",
        str(rule_config.get("severity", "high")),
        trigger_records,
        threshold_records,
        rationale_values,
    )


def _add_sender_rolling_metrics(transactions: pd.DataFrame, window: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    ordered = transactions.sort_values(["sender_account_id", "timestamp", "transaction_id"]).copy()
    logger.debug(
        "Starting sender rolling metrics rows=%s sender_groups=%s window=%s",
        len(ordered),
        ordered["sender_account_id"].nunique(),
        window,
    )
    rolling = (
        ordered.groupby("sender_account_id")
        .rolling(window, on="timestamp")["amount"]
        .agg(["count", "sum"])
        .reset_index(level=0, drop=True)
    )
    ordered["sender_txn_count_window"] = rolling["count"].to_numpy()
    ordered["sender_amount_sum_window"] = rolling["sum"].to_numpy()
    logger.debug("Completed sender rolling metrics elapsed=%.2fs", time.perf_counter() - start)
    return ordered.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
