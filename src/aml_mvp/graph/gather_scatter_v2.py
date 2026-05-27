"""Gather-scatter v2 graph rule."""

from __future__ import annotations

from typing import Any

import pandas as pd


def detect_gather_scatter_v2(transactions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    settings = dict(config.get("gather_scatter_v2", config.get("graph_rules", {}).get("gather_scatter_v2", {})))
    min_sources = int(settings.get("min_unique_sources", 3))
    min_destinations = int(settings.get("min_unique_destinations", 3))
    min_ratio = float(settings.get("min_in_out_amount_ratio", 0.5))
    max_ratio = float(settings.get("max_in_out_amount_ratio", 1.5))
    frame = transactions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    incoming = (
        frame.groupby("receiver_account_id")
        .agg(
            unique_sources=("sender_account_id", "nunique"),
            incoming_amount=("amount", "sum"),
            first_in=("timestamp", "min"),
            last_in=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"receiver_account_id": "hub_account_id"})
    )
    outgoing = (
        frame.groupby("sender_account_id")
        .agg(
            unique_destinations=("receiver_account_id", "nunique"),
            outgoing_amount=("amount", "sum"),
            first_out=("timestamp", "min"),
            last_out=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"sender_account_id": "hub_account_id"})
    )
    hubs = incoming.merge(outgoing, on="hub_account_id", how="inner")
    hubs["amount_ratio"] = hubs["outgoing_amount"] / hubs["incoming_amount"].replace(0, pd.NA)
    hubs = hubs[
        hubs["unique_sources"].ge(min_sources)
        & hubs["unique_destinations"].ge(min_destinations)
        & hubs["amount_ratio"].fillna(0).between(min_ratio, max_ratio)
    ]
    if hubs.empty:
        return pd.DataFrame(columns=["hub_account_id", "unique_sources", "unique_destinations", "incoming_amount", "outgoing_amount", "amount_ratio", "evidence_json"])
    hubs["evidence_json"] = hubs.apply(
        lambda row: {
            "hub_account_id": row["hub_account_id"],
            "unique_sources": int(row["unique_sources"]),
            "unique_destinations": int(row["unique_destinations"]),
            "incoming_amount": float(row["incoming_amount"]),
            "outgoing_amount": float(row["outgoing_amount"]),
            "amount_ratio": float(row["amount_ratio"]),
        },
        axis=1,
    )
    return hubs.reset_index(drop=True)

