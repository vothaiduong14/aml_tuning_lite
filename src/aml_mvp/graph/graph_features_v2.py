"""Graph features and graph rule v2 orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aml_mvp.graph.cycle_detection_v2 import detect_cycles_v2
from aml_mvp.graph.gather_scatter_v2 import detect_gather_scatter_v2
from aml_mvp.graph.graph_features import build_graph_features, build_graph_feature_dictionary
from aml_mvp.rules.base_rule import build_rule_hits_from_records, empty_rule_hits
from aml_mvp.storage import read_dataframe, write_dataframe


def build_graph_features_v2(config: dict[str, Any], root: Path, logger=None) -> pd.DataFrame:
    artifacts = dict(config.get("artifacts", {}))
    transactions = read_dataframe(_resolve(root, artifacts["transactions_path"]))
    alerts = read_dataframe(_resolve(root, artifacts["alerts_path"]))
    graph_config = dict(config.get("graph", {}))
    max_transaction_rows = int(graph_config.get("max_transaction_rows", 750000))
    if max_transaction_rows and len(transactions) > max_transaction_rows:
        selected_alerts = alerts.sort_values("alert_timestamp").head(int(graph_config.get("max_alert_rows", 50000)))
        max_ts = pd.to_datetime(selected_alerts["alert_timestamp"]).max() if not selected_alerts.empty else pd.to_datetime(transactions["timestamp"]).min()
        limited = transactions[pd.to_datetime(transactions["timestamp"]).le(max_ts)].sort_values(["timestamp", "transaction_id"])
        if len(limited) < max_transaction_rows:
            transactions = transactions.sort_values(["timestamp", "transaction_id"]).head(max_transaction_rows)
        else:
            transactions = limited.head(max_transaction_rows)
        alerts = alerts[alerts["transaction_id"].isin(set(transactions["transaction_id"]))]
    if logger:
        logger.info("Building graph v2 features transactions=%s alerts=%s", len(transactions), len(alerts))
    features, dictionary = build_graph_features(transactions, alerts, config, logger=logger)
    features = features.rename(columns={column: f"{column}_v2" for column in features.columns if column.startswith("graph_")})
    dictionary = build_graph_feature_dictionary()
    dictionary["feature_name"] = dictionary["feature_name"].str.replace("feature_graph_", "feature_graph_v2_", regex=False)
    feature_path = write_dataframe(features, _resolve(root, artifacts["graph_features_v2_path"]))
    dictionary_path = _resolve(root, artifacts["graph_feature_dictionary_v2_path"])
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(dictionary_path, index=False)
    graph_hits, cycle_summary = build_graph_rule_hits_v2(transactions, alerts, features, config, logger=logger)
    hits_path = write_dataframe(graph_hits, _resolve(root, artifacts["graph_rule_hits_v2_path"]))
    cycle_path = _resolve(root, artifacts["cycle_candidates_v2_path"])
    cycle_path.parent.mkdir(parents=True, exist_ok=True)
    cycle_summary.to_csv(cycle_path, index=False)
    if logger:
        logger.info("Wrote graph v2 features path=%s rows=%s", feature_path, len(features))
        logger.info("Wrote graph v2 rule hits path=%s rows=%s", hits_path, len(graph_hits))
    return features


def build_graph_rule_hits_v2(
    transactions: pd.DataFrame,
    alerts: pd.DataFrame,
    graph_features: pd.DataFrame,
    config: dict[str, Any],
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gather = detect_gather_scatter_v2(transactions, config)
    cycles = detect_cycles_v2(transactions, config)
    tx_cols = ["transaction_id", "timestamp", "is_laundering", "split", "sender_account_id", "receiver_account_id"]
    tx_lookup = transactions[tx_cols].drop_duplicates("transaction_id")
    hits = []
    if not gather.empty:
        linked = tx_lookup[tx_lookup["sender_account_id"].isin(set(gather["hub_account_id"])) | tx_lookup["receiver_account_id"].isin(set(gather["hub_account_id"]))]
        if not linked.empty:
            hits.append(
                build_rule_hits_from_records(
                    linked,
                    "GR1_GATHER_SCATTER_V2",
                    "Validated gather-scatter pattern",
                    "high",
                    [{"sender_account_id": row.sender_account_id, "receiver_account_id": row.receiver_account_id} for row in linked.itertuples(index=False)],
                    [{"rule_version": "v2"}] * len(linked),
                    ["Account is linked to a constrained gather-scatter hub." for _ in range(len(linked))],
                )
            )
    if not cycles.empty:
        cycle_linked = cycles.merge(tx_lookup, on=["transaction_id", "is_laundering", "split"], how="left")
        hits.append(
            build_rule_hits_from_records(
                cycle_linked,
                "GR2_CYCLE_CANDIDATE_V2",
                "Constrained short-cycle candidate",
                "high",
                [{"cycle_path": row.cycle_path, "path_length": int(row.path_length)} for row in cycle_linked.itertuples(index=False)],
                [{"rule_version": "v2"}] * len(cycle_linked),
                ["Transaction is part of a constrained short directed cycle." for _ in range(len(cycle_linked))],
            )
        )
    graph_rule_hits = pd.concat(hits, ignore_index=True) if hits else empty_rule_hits()
    if not graph_rule_hits.empty:
        graph_rule_hits["graph_path_evidence"] = graph_rule_hits["trigger_values_json"].apply(_compact_json)
        graph_rule_hits["rule_status"] = graph_rule_hits["rule_id"].map(
            {"GR1_GATHER_SCATTER_V2": "escalation", "GR2_CYCLE_CANDIDATE_V2": "research"}
        ).fillna("research")
        graph_rule_hits["queue_eligible"] = graph_rule_hits["rule_status"].eq("escalation")
        graph_rule_hits["volume_control_reason"] = graph_rule_hits["rule_status"].map(
            {"research": "research_only_not_direct_p1_p2", "escalation": ""}
        ).fillna("")
    cycle_summary = pd.DataFrame(
        [
            {"metric": "cycle_candidate_count", "value": int(len(cycles))},
            {"metric": "cycle_candidate_label_count", "value": int(cycles["is_laundering"].sum()) if "is_laundering" in cycles else 0},
            {"metric": "gather_scatter_hub_count", "value": int(len(gather))},
        ]
    )
    if logger:
        logger.info("Graph v2 rules completed hits=%s cycle_candidates=%s gather_hubs=%s", len(graph_rule_hits), len(cycles), len(gather))
    return graph_rule_hits, cycle_summary


def _compact_json(value: object) -> str:
    try:
        return json.dumps(json.loads(str(value)), sort_keys=True)
    except Exception:
        return str(value)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
