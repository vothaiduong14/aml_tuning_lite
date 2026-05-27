from __future__ import annotations

import pandas as pd

from aml_mvp.calibration.band_sizing import rebuild_priority_bands
from aml_mvp.cases.case_consolidation_v2 import consolidate_cases_v2
from aml_mvp.diagnostics.rule_contribution import build_rule_contribution
from aml_mvp.explainability.shap_reason_mapper import generate_alert_reason_codes
from aml_mvp.storage import write_dataframe


def test_rule_contribution_outputs_recommended_actions(tmp_path) -> None:
    rule_hits = pd.DataFrame(
        {
            "transaction_id": [1, 2, 2],
            "rule_id": ["R1_AMOUNT", "R1_AMOUNT", "R2_NOVELTY"],
            "rule_name": ["Amount", "Amount", "Novelty"],
            "is_laundering": [1, 0, 0],
        }
    )
    alerts = pd.DataFrame({"alert_id": ["A1", "A2"], "transaction_id": [1, 2], "is_laundering": [1, 0]})
    write_dataframe(rule_hits, tmp_path / "rule_hits.parquet")
    write_dataframe(alerts, tmp_path / "alerts.parquet")

    result = build_rule_contribution(
        {
            "artifacts": {
                "rule_hits_path": "rule_hits.parquet",
                "alerts_path": "alerts.parquet",
                "rule_contribution_path": "rule_contribution.csv",
            }
        },
        tmp_path,
    )

    assert {"rule_id", "unique_alert_count", "recommended_action"}.issubset(result.columns)
    assert (tmp_path / "rule_contribution.csv").exists()


def test_rebuild_priority_bands_caps_p1(tmp_path) -> None:
    alerts = pd.DataFrame(
        {
            "alert_id": [f"A{i}" for i in range(10)],
            "transaction_id": list(range(10)),
            "rule_count": [2] * 10,
            "max_rule_severity": ["high"] * 10,
            "rule_priority_score": list(range(10)),
            "is_laundering": [1, 0] * 5,
        }
    )
    scored = alerts[["alert_id", "transaction_id"]].copy()
    scored["model_score"] = [i / 10 for i in range(10)]
    write_dataframe(alerts, tmp_path / "alerts.parquet")
    write_dataframe(scored, tmp_path / "scored.parquet")

    result = rebuild_priority_bands(
        {
            "artifacts": {
                "alerts_path": "alerts.parquet",
                "scored_alerts_path": "scored.parquet",
                "output_priority_alerts_path": "priority.parquet",
                "band_summary_path": "bands.csv",
                "capacity_simulation_path": "capacity.csv",
                "top_k_after_band_fix_path": "topk.csv",
            },
            "priority_bands": {
                "score_column": "risk_score_1000",
                "daily_capacity_assumption": {"total_daily_capacity": 2},
                "bands": {"P1": {"max_alerts": 2, "score_percentile_floor": 50}, "P2": {"score_percentile_floor": 0}},
                "critical_override_conditions": {"min_rule_count": 2, "min_score": 990, "require_score_floor": True},
            },
            "scoring": {"risk_score_column": "risk_score_1000"},
        },
        tmp_path,
    )

    assert int(result["priority_band_v2"].eq("P1").sum()) <= 2
    assert "risk_score_1000" in result.columns


def test_case_consolidation_v2_maps_each_alert_once(tmp_path) -> None:
    alerts = pd.DataFrame(
        {
            "alert_id": ["A1", "A2", "A3"],
            "transaction_id": [1, 2, 3],
            "alert_timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "triggered_rules": ["R1_AMOUNT", "R1_AMOUNT", "R4_PASS_THROUGH"],
            "is_laundering": [0, 1, 0],
        }
    )
    tx = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "timestamp": alerts["alert_timestamp"],
            "sender_account_id": ["S1", "S1", "S2"],
            "receiver_account_id": ["R1", "R2", "R3"],
            "amount": [10.0, 11.0, 12.0],
            "payment_format": ["ACH", "ACH", "Wire"],
        }
    )
    write_dataframe(alerts, tmp_path / "alerts.parquet")
    write_dataframe(tx, tmp_path / "tx.parquet")

    result = consolidate_cases_v2(
        {
            "artifacts": {
                "transactions_path": "tx.parquet",
                "alerts_path": "alerts.parquet",
                "output_cases_path": "cases.parquet",
                "alert_case_mapping_path": "mapping.parquet",
                "network_clusters_path": "clusters.parquet",
                "case_quality_metrics_path": "metrics.csv",
                "mega_case_diagnostics_path": "mega.csv",
                "case_typology_summary_path": "typology.csv",
            },
            "case_consolidation": {"max_alerts_per_case": 2},
        },
        tmp_path,
    )

    assert len(result["mapping"]) == 3
    assert result["mapping"]["alert_id"].is_unique
    assert int(result["cases"]["alert_count"].max()) <= 2


def test_generate_alert_reason_codes_uses_default_for_unknown_feature(tmp_path) -> None:
    shap_values = pd.DataFrame({"alert_id": ["A1"], "feature_name": ["feature_unknown"], "shap_value": [0.5], "feature_value": [1.0]})
    priority = pd.DataFrame({"alert_id": ["A1"], "priority_band_v2": ["P1"]})
    write_dataframe(shap_values, tmp_path / "shap.parquet")
    write_dataframe(priority, tmp_path / "priority.parquet")

    result = generate_alert_reason_codes(
        {
            "artifacts": {
                "shap_values_path": "shap.parquet",
                "priority_alerts_path": "priority.parquet",
                "alert_reason_codes_path": "reasons.csv",
            },
            "defaults": {
                "unknown_feature": {
                    "business_reason_code": "Default reason",
                    "plain_language_template": "Default plain language.",
                    "investigator_note": "Review evidence.",
                }
            },
        },
        tmp_path,
    )

    assert result.loc[0, "business_reason_code"] == "Default reason"
    assert result.loc[0, "feature_value"] == 1.0
    assert result.loc[0, "risk_direction"] == "risk_increasing"


def test_generate_alert_reason_codes_uses_negative_template(tmp_path) -> None:
    shap_values = pd.DataFrame({"alert_id": ["A1"], "feature_name": ["feature_amount"], "shap_value": [-0.25], "feature_value": [12.0]})
    write_dataframe(shap_values, tmp_path / "shap.parquet")

    result = generate_alert_reason_codes(
        {
            "artifacts": {
                "shap_values_path": "shap.parquet",
                "alert_reason_codes_path": "reasons.csv",
            },
            "features": {
                "feature_amount": {
                    "business_reason_code": "Amount",
                    "risk_increasing_template": "Amount increased risk.",
                    "risk_reducing_template": "Amount reduced risk.",
                    "investigator_note": "Review amount.",
                }
            },
        },
        tmp_path,
    )

    assert result.loc[0, "risk_direction"] == "risk_reducing"
    assert result.loc[0, "contribution_bucket"] == "moderate_negative"
    assert result.loc[0, "plain_language_reason"] == "Amount reduced risk."
