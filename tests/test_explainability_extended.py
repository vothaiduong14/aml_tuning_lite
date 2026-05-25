from __future__ import annotations

import warnings

from aml_mvp.explainability.reason_codes import reason_phrase
from aml_mvp.explainability.shap_explainer import explain_alerts

from tests.fixtures import scored_alerts


def test_explain_alerts_returns_reason_codes() -> None:
    features = scored_alerts()
    features["feature_rule_count"] = [1, 2, 1, 1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shap_values, reason_codes, importance = explain_alerts(
            features,
            scored_alerts(),
            {"explainability": {"max_explain_rows": 4, "top_reason_features": 2}},
        )

    assert not shap_values.empty
    assert not reason_codes.empty
    assert not importance.empty
    assert reason_codes["reason_rank"].max() <= 2
    assert not any("LightGBM binary classifier" in str(item.message) for item in caught)


def test_reason_phrase_has_aml_readable_text() -> None:
    assert "Multiple rules" in reason_phrase("feature_rule_count")
    assert "R1_AMOUNT" in reason_phrase("feature_rule_r1_amount_flag")
