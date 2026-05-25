"""R4 rapid pass-through rule."""

from __future__ import annotations

import logging
import time

import pandas as pd

from aml_mvp.rules.base_rule import build_rule_hits_from_records


def apply_pass_through_rule(transactions: pd.DataFrame, rule_config: dict) -> pd.DataFrame:
    window = str(rule_config.get("window", "24h"))
    min_ratio = float(rule_config.get("min_in_out_ratio", 0.80))
    scored = transactions.sort_values(["timestamp", "transaction_id"]).copy()
    scored["sender_incoming_amount_window"] = rolling_incoming_amount_to_sender(scored, window)
    denominator = scored["sender_incoming_amount_window"].where(scored["sender_incoming_amount_window"].ne(0))
    scored["pass_through_ratio"] = (scored["amount"] / denominator).fillna(0.0)

    mask = scored["sender_incoming_amount_window"].gt(0) & scored["pass_through_ratio"].ge(min_ratio)
    hits = scored.loc[mask]
    trigger_records = [
        {
            "sender_account_id": row.sender_account_id,
            "outgoing_amount": row.amount,
            "incoming_amount_window": row.sender_incoming_amount_window,
            "pass_through_ratio": row.pass_through_ratio,
            "window": window,
        }
        for row in hits.itertuples(index=False)
    ]
    threshold_records = [{"min_in_out_ratio": min_ratio, "window": window} for _ in range(len(hits))]
    rationale_values = [
            (
                f"Sender moved {row.pass_through_ratio:.2f} of recent incoming funds "
                f"within {window}."
            )
            for row in hits.itertuples(index=False)
    ]

    return build_rule_hits_from_records(
        hits,
        "R4_PASS_THROUGH",
        "Rapid pass-through",
        str(rule_config.get("severity", "critical")),
        trigger_records,
        threshold_records,
        rationale_values,
    )


def rolling_incoming_amount_to_sender(transactions: pd.DataFrame, window: str) -> pd.Series:
    """For each outgoing row, sum recent incoming amount to its sender account."""

    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    incoming_events = transactions[
        ["timestamp", "transaction_id", "receiver_account_id", "amount"]
    ].rename(columns={"receiver_account_id": "account_id", "amount": "incoming_amount"})
    incoming_events["_is_query"] = False
    incoming_events["_query_transaction_id"] = pd.NA

    query_events = transactions[
        ["timestamp", "transaction_id", "sender_account_id"]
    ].rename(columns={"sender_account_id": "account_id", "transaction_id": "_query_transaction_id"})
    query_events["transaction_id"] = query_events["_query_transaction_id"]
    query_events["incoming_amount"] = 0.0
    query_events["_is_query"] = True

    events = pd.concat(
        [
            incoming_events[
                ["timestamp", "transaction_id", "account_id", "incoming_amount", "_is_query", "_query_transaction_id"]
            ],
            query_events[
                ["timestamp", "transaction_id", "account_id", "incoming_amount", "_is_query", "_query_transaction_id"]
            ],
        ],
        ignore_index=True,
    ).sort_values(["account_id", "timestamp", "_is_query", "transaction_id"])

    logger.debug(
        "Starting incoming-to-sender rolling amount rows=%s sender_groups=%s receiver_groups=%s window=%s",
        len(transactions),
        transactions["sender_account_id"].nunique(),
        transactions["receiver_account_id"].nunique(),
        window,
    )
    rolled = (
        events.groupby("account_id")
        .rolling(window, on="timestamp")["incoming_amount"]
        .sum()
        .reset_index(level=0, drop=True)
    )
    events["sender_incoming_amount_window"] = rolled.to_numpy()
    query_results = events.loc[events["_is_query"], ["_query_transaction_id", "sender_incoming_amount_window"]]
    result = query_results.set_index("_query_transaction_id")["sender_incoming_amount_window"]
    logger.debug("Completed incoming-to-sender rolling amount elapsed=%.2fs", time.perf_counter() - start)
    return transactions["transaction_id"].map(result).fillna(0.0).astype(float)
