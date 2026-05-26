from __future__ import annotations

import pandas as pd

from tests.fixtures import alerts, rule_hits, transactions

from aml_mvp.features.feature_matrix import build_alert_feature_matrix
from aml_mvp.models.train import (
    balance_training_frame,
    select_champion_model,
    select_model_features,
    train_and_score_alerts,
)


def test_train_and_score_alerts_outputs_scores_and_metrics() -> None:
    features, _, _ = build_alert_feature_matrix(transactions(), alerts(), rule_hits())
    config = {
        "model": {
            "k_values": [1, 2],
            "max_train_rows": 100,
            "random_seed": 42,
            "tuning_trials": 2,
            "champion_selection_k": 1,
        }
    }

    scored, metrics, top_k, importance, artifact = train_and_score_alerts(features, config)

    assert len(scored) == len(features)
    assert "model_score" in scored.columns
    assert scored["model_score"].between(0, 1).all()
    assert metrics["population"] == "alert_level"
    assert {"model_score", "rule_priority", "amount_rank"}.issubset(set(top_k["ranking"]))
    assert not importance.empty
    assert "champion_name" in artifact
    assert "training" in metrics
    assert metrics["training"]["imbalance"]["strategy"] == "class_weight_undersample"
    assert metrics["training"]["feature_selection"]["selected_feature_count"] == len(artifact["feature_columns"])
    assert "selected_features_table" in artifact
    assert "tuning_trials" in artifact


def test_balance_training_frame_preserves_positives_and_caps_negatives() -> None:
    frame = _imbalanced_training_frame(positive_count=3, negative_count=100)

    balanced, metadata = balance_training_frame(
        frame,
        {
            "imbalance_strategy": "class_weight_undersample",
            "negative_to_positive_ratio": 10,
            "preserve_all_positives": True,
            "max_train_rows": 1000,
            "sampling_random_seed": 7,
        },
    )

    assert int(balanced["target"].sum()) == 3
    assert len(balanced) - int(balanced["target"].sum()) == 30
    assert metadata["sampled_negative_to_positive_ratio"] == 10.0


def test_balance_training_frame_is_deterministic() -> None:
    frame = _imbalanced_training_frame(positive_count=2, negative_count=30)
    config = {
        "imbalance_strategy": "class_weight_undersample",
        "negative_to_positive_ratio": 5,
        "max_train_rows": 1000,
        "sampling_random_seed": 99,
    }

    first, _ = balance_training_frame(frame, config)
    second, _ = balance_training_frame(frame, config)

    assert first["alert_id"].tolist() == second["alert_id"].tolist()


def test_balance_training_frame_max_rows_keeps_positives() -> None:
    frame = _imbalanced_training_frame(positive_count=5, negative_count=100)

    balanced, _ = balance_training_frame(
        frame,
        {
            "imbalance_strategy": "class_weight_undersample",
            "negative_to_positive_ratio": 20,
            "preserve_all_positives": True,
            "max_train_rows": 12,
            "sampling_random_seed": 11,
        },
    )

    assert len(balanced) == 12
    assert int(balanced["target"].sum()) == 5


def test_feature_selection_caps_selected_feature_count() -> None:
    frame = _feature_selection_frame(row_count=30, feature_count=8)
    feature_columns = [column for column in frame.columns if column.startswith("feature_")]

    selected, table = select_model_features(
        frame,
        feature_columns,
        {"feature_selection_enabled": True, "selected_feature_count": 3, "random_seed": 42},
    )

    assert len(selected) == 3
    assert table["selected"].sum() == 3


def test_feature_selection_force_includes_graph_features() -> None:
    frame = _feature_selection_frame(row_count=30, feature_count=8)
    frame["feature_graph_component_size"] = 1
    frame["feature_graph_cycle_involvement"] = 0
    feature_columns = [column for column in frame.columns if column.startswith("feature_")]

    selected, table = select_model_features(
        frame,
        feature_columns,
        {
            "feature_selection_enabled": True,
            "selected_feature_count": 3,
            "force_include_feature_prefixes": ["feature_graph_"],
            "random_seed": 42,
        },
    )

    assert "feature_graph_component_size" in selected
    assert "feature_graph_cycle_involvement" in selected
    assert table.loc[table["feature_name"].eq("feature_graph_component_size"), "forced_include"].iloc[0]


def test_train_and_score_single_class_uses_dummy_fallback() -> None:
    features, _, _ = build_alert_feature_matrix(transactions(), alerts(), rule_hits())
    features["target"] = 0

    _, metrics, _, _, artifact = train_and_score_alerts(
        features,
        {"model": {"k_values": [1], "tuning_trials": 1, "champion_selection_k": 1}},
    )

    assert artifact["champion_name"] == "logistic_regression"
    assert metrics["training"]["tuning"]["reason"] == "single_class_training"


def test_champion_selection_uses_validation_precision_at_k() -> None:
    frame = pd.DataFrame(
        {
            "alert_id": ["A1", "A2", "A3", "A4"],
            "split": ["validation"] * 4,
            "target": [1, 0, 0, 0],
            "feature_amount": [1.0, 2.0, 3.0, 4.0],
        }
    )
    models = {
        "weak_model": _FixedScoreModel([0.1, 0.9, 0.8, 0.7]),
        "strong_model": _FixedScoreModel([0.9, 0.1, 0.2, 0.3]),
    }

    champion, metadata = select_champion_model(
        models,
        frame,
        ["feature_amount"],
        {"champion_selection_metric": "precision_at_k", "champion_selection_k": 1},
    )

    assert champion == "strong_model"
    assert metadata["selection_split"] == "validation"


def _imbalanced_training_frame(positive_count: int, negative_count: int):
    total = positive_count + negative_count
    return pd.DataFrame(
        {
            "alert_id": [f"A{i}" for i in range(total)],
            "split": ["train"] * total,
            "target": [1] * positive_count + [0] * negative_count,
            "feature_amount": list(range(total)),
        }
    )


def _feature_selection_frame(row_count: int, feature_count: int):
    data = {
        "alert_id": [f"A{i}" for i in range(row_count)],
        "split": ["train"] * row_count,
        "target": [0, 1] * (row_count // 2),
    }
    for index in range(feature_count):
        data[f"feature_{index}"] = [(row + index) % 5 for row in range(row_count)]
    return pd.DataFrame(data)


class _FixedScoreModel:
    classes_ = [0, 1]

    def __init__(self, positive_scores):
        self.positive_scores = positive_scores

    def predict_proba(self, frame):
        scores = self.positive_scores[: len(frame)]
        return [[1 - score, score] for score in scores]
