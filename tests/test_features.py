from __future__ import annotations

from tests.fixtures import alerts, rule_hits, transactions

from aml_mvp.features.feature_matrix import build_alert_feature_matrix


def test_build_alert_feature_matrix_one_row_per_alert() -> None:
    features, dictionary, quality = build_alert_feature_matrix(transactions(), alerts(), rule_hits())

    assert len(features) == len(alerts())
    assert features["alert_id"].is_unique
    assert "feature_rule_count" in features.columns
    assert "feature_sender_prior_txn_count" in features.columns
    assert "feature_rule_r1_amount_flag" in features.columns
    assert features["target"].tolist() == [0, 1, 1, 0]
    assert set(dictionary["feature_name"]).issubset(set(features.columns))
    assert not quality.empty

