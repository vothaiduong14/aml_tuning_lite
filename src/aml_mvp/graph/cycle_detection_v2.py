"""Constrained short-cycle detection for remediation graph rules."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import pandas as pd


def detect_cycles_v2(transactions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    settings = dict(config.get("cycle_v2", config.get("graph_rules", {}).get("cycle_v2", {})))
    min_len = int(settings.get("min_path_length", 3))
    max_len = int(settings.get("max_path_length", 5))
    max_rows = int(settings.get("max_evaluated_transactions", 50000))
    max_window_hours = float(settings.get("max_cycle_window_hours", 72))
    min_similarity = float(settings.get("min_amount_similarity_ratio", 0.5))
    max_retention = float(settings.get("max_cycle_retention_ratio", 1.2))
    require_high_risk = bool(settings.get("require_high_risk_rule_flag", False))
    min_amount_percentile = float(settings.get("min_total_cycle_amount_percentile", 0.0))
    frame = transactions.sort_values(["timestamp", "transaction_id"]).head(max_rows).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    amount_threshold = float(frame["amount"].quantile(min_amount_percentile)) if "amount" in frame and len(frame) else 0.0
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        sender = str(row.sender_account_id)
        receiver = str(row.receiver_account_id)
        path = _path(receiver, sender, adjacency, max_len - 1)
        amount = float(getattr(row, "amount", 0.0) or 0.0)
        timestamp = pd.to_datetime(row.timestamp)
        if path and len(path["nodes"]) + 1 >= min_len:
            amounts = [amount, *path["amounts"]]
            timestamps = [timestamp, *path["timestamps"]]
            total_amount = float(sum(amounts))
            min_amount = min(amounts) if amounts else 0.0
            max_amount = max(amounts) if amounts else 0.0
            similarity = min_amount / max_amount if max_amount else 0.0
            duration_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600 if timestamps else 0.0
            retention_ratio = amount / max(sum(path["amounts"]), 1.0)
            high_risk_ok = True
            if require_high_risk and hasattr(row, "high_risk_rule_flag"):
                high_risk_ok = bool(getattr(row, "high_risk_rule_flag", 0))
            constraints_ok = (
                duration_hours <= max_window_hours
                and similarity >= min_similarity
                and retention_ratio <= max_retention
                and total_amount >= amount_threshold
                and high_risk_ok
            )
        else:
            constraints_ok = False
        if path and constraints_ok:
            cycle_path = [sender, receiver, *path["nodes"][1:]]
            records.append(
                {
                    "transaction_id": row.transaction_id,
                    "alert_timestamp": row.timestamp,
                    "cycle_path": "->".join(cycle_path),
                    "path_length": len(cycle_path),
                    "duration_hours": duration_hours,
                    "amount_similarity_ratio": similarity,
                    "cycle_retention_ratio": retention_ratio,
                    "total_cycle_amount": total_amount,
                    "cycle_status": str(settings.get("status", "research")),
                    "eligible_for_p1_override": bool(settings.get("eligible_for_p1_override", False)),
                    "is_laundering": int(getattr(row, "is_laundering", 0) or 0),
                    "split": getattr(row, "split", None),
                }
            )
        adjacency[sender].append({"to": receiver, "timestamp": timestamp, "amount": amount})
    return pd.DataFrame(records)


def _path(start: str, target: str, adjacency: dict[str, list[dict[str, Any]]], max_depth: int) -> dict[str, Any] | None:
    queue: deque[tuple[str, list[str], list[float], list[pd.Timestamp]]] = deque([(start, [start], [], [])])
    while queue:
        node, path, amounts, timestamps = queue.popleft()
        if len(path) > max_depth:
            continue
        for edge in adjacency.get(node, []):
            next_node = str(edge["to"])
            next_amounts = [*amounts, float(edge["amount"])]
            next_timestamps = [*timestamps, pd.to_datetime(edge["timestamp"])]
            if next_node == target:
                return {"nodes": [*path, next_node], "amounts": next_amounts, "timestamps": next_timestamps}
            if next_node not in path:
                queue.append((next_node, [*path, next_node], next_amounts, next_timestamps))
    return None
