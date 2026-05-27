from __future__ import annotations

import pandas as pd

from aml_mvp.graph.graph_features import build_graph_features, merge_graph_feature_columns_into_alert_matrix
from aml_mvp.graph.graph_rules import run_graph_rules
from aml_mvp.graph.cycle_detection_v2 import detect_cycles_v2


def test_graph_features_are_point_in_time_and_detect_cycle() -> None:
    tx = _cycle_transactions()
    alerts = pd.DataFrame({"transaction_id": [4], "alert_timestamp": [tx.loc[3, "timestamp"]]})

    features, dictionary = build_graph_features(tx, alerts, {"graph": {"cycle_max_length": 4}})

    row = features.iloc[0]
    assert row["transaction_id"] == 4
    assert row["graph_cycle_involvement"] == 1
    assert row["graph_sender_out_degree"] == 0
    assert row["graph_receiver_in_degree"] == 0
    assert "feature_graph_cycle_involvement" in set(dictionary["feature_name"])


def test_graph_rules_emit_evidence_for_cycle_and_gather_scatter() -> None:
    tx = _cycle_transactions()
    alerts = pd.DataFrame({"transaction_id": [4], "alert_timestamp": [tx.loc[3, "timestamp"]]})
    features, _ = build_graph_features(tx, alerts, {"graph": {"cycle_max_length": 4}})
    features["graph_sender_out_degree"] = 3
    features["graph_component_size"] = 4

    hits, summary = run_graph_rules(
        tx,
        alerts,
        features,
        {"graph": {"gather_scatter_min_degree": 3, "gather_scatter_min_component_size": 4}},
    )

    assert {"GR1_GATHER_SCATTER", "GR2_CYCLE_CANDIDATE"}.issubset(set(hits["rule_id"]))
    assert hits["trigger_values_json"].str.contains("graph_component_size").any()
    assert not summary.empty


def test_merge_graph_feature_columns_supports_v2_suffixes() -> None:
    alert_features = pd.DataFrame(
        {
            "alert_id": ["A1", "A2"],
            "transaction_id": [1, 2],
            "feature_graph_component_size_v2": [99.0, 99.0],
        }
    )
    graph_v2 = pd.DataFrame(
        {
            "transaction_id": [1],
            "graph_component_size_v2": [4.0],
            "graph_cycle_involvement_v2": [1.0],
        }
    )

    merged = merge_graph_feature_columns_into_alert_matrix(alert_features, graph_v2)

    assert merged.loc[merged["transaction_id"].eq(1), "feature_graph_component_size_v2"].iloc[0] == 4.0
    assert merged.loc[merged["transaction_id"].eq(2), "feature_graph_component_size_v2"].iloc[0] == 0.0
    assert "feature_graph_cycle_involvement_v2" in merged.columns


def test_cycle_v2_applies_time_and_amount_constraints() -> None:
    tx = _cycle_transactions()

    accepted = detect_cycles_v2(
        tx,
        {"cycle_v2": {"min_path_length": 3, "max_path_length": 5, "max_cycle_window_hours": 8, "min_amount_similarity_ratio": 0.2}},
    )
    rejected = detect_cycles_v2(
        tx.assign(timestamp=pd.to_datetime(["2022-01-01", "2022-01-05", "2022-01-06", "2022-01-07"])),
        {"cycle_v2": {"min_path_length": 3, "max_path_length": 5, "max_cycle_window_hours": 8, "min_amount_similarity_ratio": 0.2}},
    )

    assert not accepted.empty
    assert {"duration_hours", "amount_similarity_ratio", "cycle_status"}.issubset(accepted.columns)
    assert rejected.empty


def _cycle_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4],
            "timestamp": pd.to_datetime(
                ["2022-01-01 00:00", "2022-01-01 01:00", "2022-01-01 02:00", "2022-01-01 03:00"]
            ),
            "sender_account_id": ["A", "B", "C", "D"],
            "receiver_account_id": ["B", "C", "D", "A"],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "is_laundering": [0, 0, 0, 1],
            "split": ["train", "train", "validation", "test"],
        }
    )
