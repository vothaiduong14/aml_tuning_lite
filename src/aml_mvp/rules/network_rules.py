"""R5 fan-in and R6 fan-out/gather-scatter-lite rules."""

from __future__ import annotations

from collections import Counter, deque
import logging
import time

import pandas as pd

from aml_mvp.rules.base_rule import build_rule_hits_from_records


def apply_fan_in_rule(
    transactions: pd.DataFrame,
    training_transactions: pd.DataFrame,
    rule_config: dict,
) -> pd.DataFrame:
    window = str(rule_config.get("window", "24h"))
    min_unique_senders = int(rule_config.get("min_unique_senders", 5))
    amount_percentile = float(rule_config.get("amount_percentile", 0.90))

    scored = transactions.sort_values(["timestamp", "transaction_id"]).copy()
    scored["receiver_unique_senders_window"] = rolling_unique_count(
        scored,
        group_col="receiver_account_id",
        item_col="sender_account_id",
        window=window,
    )
    scored["receiver_incoming_amount_window"] = rolling_amount_sum(
        scored,
        group_col="receiver_account_id",
        amount_col="amount",
        window=window,
    )

    train_scored = training_transactions.sort_values(["timestamp", "transaction_id"]).copy()
    train_scored["receiver_incoming_amount_window"] = rolling_amount_sum(
        train_scored,
        group_col="receiver_account_id",
        amount_col="amount",
        window=window,
    )
    amount_threshold = float(train_scored["receiver_incoming_amount_window"].quantile(amount_percentile))

    mask = (
        scored["receiver_unique_senders_window"].ge(min_unique_senders)
        & scored["receiver_incoming_amount_window"].ge(amount_threshold)
    )
    hits = scored.loc[mask]
    trigger_records = [
        {
            "receiver_account_id": row.receiver_account_id,
            "unique_senders": int(row.receiver_unique_senders_window),
            "incoming_amount": row.receiver_incoming_amount_window,
            "window": window,
        }
        for row in hits.itertuples(index=False)
    ]
    threshold_records = [
        {
            "min_unique_senders": min_unique_senders,
            "amount_threshold": amount_threshold,
            "amount_percentile": amount_percentile,
        }
        for _ in range(len(hits))
    ]
    rationale_values = [
            (
                f"Receiver had {int(row.receiver_unique_senders_window)} unique senders "
                f"and {row.receiver_incoming_amount_window:.2f} incoming amount in {window}."
            )
            for row in hits.itertuples(index=False)
    ]

    return build_rule_hits_from_records(
        hits,
        "R5_FAN_IN",
        "Fan-in concentration",
        str(rule_config.get("severity", "high")),
        trigger_records,
        threshold_records,
        rationale_values,
    )


def apply_fan_out_rule(transactions: pd.DataFrame, rule_config: dict) -> pd.DataFrame:
    window = str(rule_config.get("window", "24h"))
    min_unique_receivers = int(rule_config.get("min_unique_receivers", 5))
    min_prior_unique_senders = int(rule_config.get("min_prior_unique_senders", 2))

    scored = transactions.sort_values(["timestamp", "transaction_id"]).copy()
    scored["sender_unique_receivers_window"] = rolling_unique_count(
        scored,
        group_col="sender_account_id",
        item_col="receiver_account_id",
        window=window,
    )
    scored["sender_prior_unique_senders_window"] = rolling_incoming_unique_senders_to_sender(scored, window)

    mask = (
        scored["sender_unique_receivers_window"].ge(min_unique_receivers)
        & scored["sender_prior_unique_senders_window"].ge(min_prior_unique_senders)
    )
    hits = scored.loc[mask]
    trigger_records = [
        {
            "sender_account_id": row.sender_account_id,
            "unique_receivers": int(row.sender_unique_receivers_window),
            "prior_unique_senders": int(row.sender_prior_unique_senders_window),
            "window": window,
        }
        for row in hits.itertuples(index=False)
    ]
    threshold_records = [
        {
            "min_unique_receivers": min_unique_receivers,
            "min_prior_unique_senders": min_prior_unique_senders,
        }
        for _ in range(len(hits))
    ]
    rationale_values = [
            (
                f"Sender had {int(row.sender_unique_receivers_window)} unique receivers "
                f"after {int(row.sender_prior_unique_senders_window)} recent funding sources in {window}."
            )
            for row in hits.itertuples(index=False)
    ]

    return build_rule_hits_from_records(
        hits,
        "R6_FAN_OUT",
        "Fan-out / gather-scatter-lite",
        str(rule_config.get("severity", "high")),
        trigger_records,
        threshold_records,
        rationale_values,
    )


def rolling_unique_count(
    transactions: pd.DataFrame,
    group_col: str,
    item_col: str,
    window: str,
    event_group_col: str | None = None,
) -> pd.Series:
    """Return rolling unique item counts aligned to input rows.

    `event_group_col` lets a transaction be counted under a different account
    role than the row's output grouping. R6 uses this to count incoming funding
    sources for the current sender account.
    """

    window_delta = pd.Timedelta(window)
    event_col = event_group_col or group_col
    events = transactions[["timestamp", "transaction_id", event_col, item_col]].copy()
    events = events.rename(columns={event_col: "_event_group", item_col: "_item"})
    events = events.sort_values(["_event_group", "timestamp", "transaction_id"])
    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    logger.debug(
        "Starting rolling unique count rows=%s groups=%s group_col=%s item_col=%s window=%s",
        len(events),
        events["_event_group"].nunique(),
        group_col,
        item_col,
        window,
    )

    result_by_transaction_id: dict[object, int] = {}
    for _, group in events.groupby("_event_group", sort=False):
        queue: deque[tuple[pd.Timestamp, object]] = deque()
        counts: Counter = Counter()
        for current_time, transaction_id, _, current_item in group.itertuples(index=False, name=None):
            queue.append((current_time, current_item))
            counts[current_item] += 1
            cutoff = current_time - window_delta
            while queue and queue[0][0] < cutoff:
                _, old_item = queue.popleft()
                counts[old_item] -= 1
                if counts[old_item] <= 0:
                    del counts[old_item]
            result_by_transaction_id[transaction_id] = len(counts)

    logger.debug("Completed rolling unique count elapsed=%.2fs", time.perf_counter() - start)
    return transactions["transaction_id"].map(result_by_transaction_id).fillna(0).astype(int)


def rolling_incoming_unique_senders_to_sender(transactions: pd.DataFrame, window: str) -> pd.Series:
    """For each outgoing row, count recent unique funders of its sender account."""

    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    window_delta = pd.Timedelta(window)
    incoming_by_account = {
        account_id: group.sort_values(["timestamp", "transaction_id"])
        for account_id, group in transactions.groupby("receiver_account_id", sort=False)
    }
    logger.debug(
        "Starting incoming unique funders rows=%s sender_groups=%s receiver_groups=%s window=%s",
        len(transactions),
        transactions["sender_account_id"].nunique(),
        len(incoming_by_account),
        window,
    )
    result: dict[object, int] = {}

    outgoing_groups = transactions.sort_values(["timestamp", "transaction_id"]).groupby(
        "sender_account_id",
        sort=False,
    )
    for sender_account_id, outgoing in outgoing_groups:
        incoming = incoming_by_account.get(sender_account_id)
        if incoming is None:
            for transaction_id in outgoing["transaction_id"]:
                result[transaction_id] = 0
            continue

        incoming_rows = list(
            incoming[["timestamp", "transaction_id", "sender_account_id"]].itertuples(index=False)
        )
        pointer = 0
        queue: deque[tuple[pd.Timestamp, object]] = deque()
        counts: Counter = Counter()

        for row in outgoing.itertuples(index=False):
            current_time = row.timestamp
            while pointer < len(incoming_rows) and incoming_rows[pointer].timestamp <= current_time:
                incoming_row = incoming_rows[pointer]
                funder = incoming_row.sender_account_id
                queue.append((incoming_row.timestamp, funder))
                counts[funder] += 1
                pointer += 1

            cutoff = current_time - window_delta
            while queue and queue[0][0] < cutoff:
                _, old_funder = queue.popleft()
                counts[old_funder] -= 1
                if counts[old_funder] <= 0:
                    del counts[old_funder]

            result[row.transaction_id] = len(counts)

    logger.debug("Completed incoming unique funders elapsed=%.2fs", time.perf_counter() - start)
    return transactions["transaction_id"].map(result).fillna(0).astype(int)


def rolling_amount_sum(
    transactions: pd.DataFrame,
    group_col: str,
    amount_col: str,
    window: str,
) -> pd.Series:
    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    window_delta = pd.Timedelta(window)
    events = transactions[["timestamp", "transaction_id", group_col, amount_col]].copy()
    events = events.sort_values([group_col, "timestamp", "transaction_id"])
    logger.debug(
        "Starting rolling amount sum rows=%s groups=%s group_col=%s amount_col=%s window=%s",
        len(events),
        events[group_col].nunique(),
        group_col,
        amount_col,
        window,
    )

    result_by_transaction_id: dict[object, float] = {}
    for _, group in events.groupby(group_col, sort=False):
        queue: deque[tuple[pd.Timestamp, float]] = deque()
        total = 0.0
        for row in group.itertuples(index=False):
            current_time = row.timestamp
            current_amount = float(getattr(row, amount_col))
            queue.append((current_time, current_amount))
            total += current_amount
            cutoff = current_time - window_delta
            while queue and queue[0][0] < cutoff:
                _, old_amount = queue.popleft()
                total -= old_amount
            result_by_transaction_id[row.transaction_id] = total

    logger.debug("Completed rolling amount sum elapsed=%.2fs", time.perf_counter() - start)
    return transactions["transaction_id"].map(result_by_transaction_id).fillna(0.0).astype(float)
