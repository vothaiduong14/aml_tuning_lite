from __future__ import annotations

import pandas as pd

from aml_mvp.graph.graph_features import build_graph_features
from aml_mvp.graph.graph_rules import run_graph_rules


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

