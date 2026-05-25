"""Point-in-time graph feature engineering for alert-level triage."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from aml_mvp.graph.cycle_detection import add_edge, has_short_cycle_before_edge, new_adjacency
from aml_mvp.storage import write_dataframe


GRAPH_FEATURE_COLUMNS = [
    "graph_sender_out_degree",
    "graph_receiver_in_degree",
    "graph_sender_weighted_out_degree",
    "graph_receiver_weighted_in_degree",
    "graph_component_size",
    "graph_sender_pagerank",
    "graph_receiver_pagerank",
    "graph_cycle_involvement",
]


def build_graph_features(
    transactions: pd.DataFrame,
    alerts: pd.DataFrame,
    config: dict[str, Any],
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build graph features for alert transactions using prior edges only."""

    graph_config = dict(config.get("graph", {}))
    cycle_max_length = int(graph_config.get("cycle_max_length", 4))
    max_alert_rows = int(graph_config.get("max_alert_rows", 50000))

    alert_ids = set(alerts["transaction_id"]) if not alerts.empty else set()
    ordered = transactions.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    if max_alert_rows and len(alert_ids) > max_alert_rows:
        alert_ids = set(alerts.sort_values("alert_timestamp").head(max_alert_rows)["transaction_id"])

    if logger:
        logger.info("Building graph features transactions=%s alert_transactions=%s", len(ordered), len(alert_ids))

    out_neighbors: dict[str, set[str]] = defaultdict(set)
    in_neighbors: dict[str, set[str]] = defaultdict(set)
    weighted_out: dict[str, float] = defaultdict(float)
    weighted_in: dict[str, float] = defaultdict(float)
    directed_adjacency = new_adjacency()
    union_find = _UnionFind()
    total_degree = 0
    records: list[dict[str, Any]] = []

    for row in ordered.itertuples(index=False):
        sender = str(row.sender_account_id)
        receiver = str(row.receiver_account_id)
        amount = float(getattr(row, "amount", 0.0) or 0.0)
        closes_cycle = has_short_cycle_before_edge(directed_adjacency, sender, receiver, cycle_max_length)

        if row.transaction_id in alert_ids:
            sender_degree = len(out_neighbors[sender]) + len(in_neighbors[sender])
            receiver_degree = len(out_neighbors[receiver]) + len(in_neighbors[receiver])
            records.append(
                {
                    "alert_id": f"ALERT-{row.transaction_id}",
                    "transaction_id": row.transaction_id,
                    "alert_timestamp": row.timestamp,
                    "graph_sender_out_degree": len(out_neighbors[sender]),
                    "graph_receiver_in_degree": len(in_neighbors[receiver]),
                    "graph_sender_weighted_out_degree": weighted_out[sender],
                    "graph_receiver_weighted_in_degree": weighted_in[receiver],
                    "graph_component_size": union_find.component_size_for_edge(sender, receiver),
                    "graph_sender_pagerank": _degree_pagerank_proxy(sender_degree, total_degree),
                    "graph_receiver_pagerank": _degree_pagerank_proxy(receiver_degree, total_degree),
                    "graph_cycle_involvement": int(closes_cycle),
                }
            )

        total_degree += int(receiver not in out_neighbors[sender]) + int(sender not in in_neighbors[receiver])
        out_neighbors[sender].add(receiver)
        in_neighbors[receiver].add(sender)
        weighted_out[sender] += amount
        weighted_in[receiver] += amount
        add_edge(directed_adjacency, sender, receiver)
        union_find.union(sender, receiver)

    features = pd.DataFrame(records)
    if features.empty:
        features = pd.DataFrame(columns=["alert_id", "transaction_id", "alert_timestamp"] + GRAPH_FEATURE_COLUMNS)
    return features, build_graph_feature_dictionary()


def merge_graph_features_into_alert_matrix(alert_features: pd.DataFrame, graph_features: pd.DataFrame) -> pd.DataFrame:
    """Return alert features with graph feature columns appended."""

    merged = alert_features.merge(
        graph_features[["transaction_id"] + GRAPH_FEATURE_COLUMNS],
        on="transaction_id",
        how="left",
    )
    for column in GRAPH_FEATURE_COLUMNS:
        feature_col = f"feature_{column}"
        merged[feature_col] = merged[column].fillna(0.0)
        merged = merged.drop(columns=[column])
    return merged


def build_graph_feature_dictionary() -> pd.DataFrame:
    definitions = {
        "graph_sender_out_degree": "Unique receivers previously reached by the sender.",
        "graph_receiver_in_degree": "Unique senders previously funding the receiver.",
        "graph_sender_weighted_out_degree": "Prior outgoing amount from the sender.",
        "graph_receiver_weighted_in_degree": "Prior incoming amount to the receiver.",
        "graph_component_size": "Prior connected component size around sender and receiver.",
        "graph_sender_pagerank": "Point-in-time degree-normalized PageRank proxy for the sender.",
        "graph_receiver_pagerank": "Point-in-time degree-normalized PageRank proxy for the receiver.",
        "graph_cycle_involvement": "Whether the transaction closes a short directed cycle.",
    }
    return pd.DataFrame(
        [
            {
                "feature_name": f"feature_{name}",
                "feature_group": "graph",
                "definition": definition,
                "source": "transactions up to alert timestamp",
                "point_in_time_rule": "Uses only edges observed before the alert transaction is added.",
            }
            for name, definition in definitions.items()
        ]
    )


def write_graph_feature_outputs(
    graph_features: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    artifacts: dict[str, str],
    root: Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["graph_features"] = write_dataframe(graph_features, _resolve(root, artifacts["graph_features_path"]))
    dictionary_path = _resolve(root, artifacts["graph_feature_dictionary_path"])
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    feature_dictionary.to_csv(dictionary_path, index=False)
    outputs["graph_feature_dictionary"] = dictionary_path
    return outputs


def _degree_pagerank_proxy(degree: int, total_degree: int) -> float:
    return float(degree / total_degree) if total_degree else 0.0


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, node: str) -> str:
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1
            return node
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def component_size_for_edge(self, left: str, right: str) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return self.size[left_root]
        return self.size[left_root] + self.size[right_root]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
