from __future__ import annotations

from aml_mvp.extended.stress_testing import build_segment_stress_summary, build_stress_test_summary, build_temporal_stress_summary

from tests.fixtures import alerts, rule_hits, scored_alerts, transactions


def test_li_only_stress_summary_marks_hi_deferred() -> None:
    summary = build_stress_test_summary(
        transactions(),
        alerts(),
        rule_hits(),
        scored_alerts().assign(priority_band=["P4", "P1", "P2", "P3"]),
        {"project": {"dataset_name": "LI-Small", "hi_comparison_status": "deferred_missing_dataset"}},
    )

    row = summary.iloc[0]
    assert row["dataset_name"] == "LI-Small"
    assert row["comparison_scope"] == "LI-only"
    assert row["hi_comparison_status"] == "deferred_missing_dataset"
    assert row["transaction_count"] == 5
    assert row["alert_count"] == 4


def test_temporal_and_segment_stress_emit_metrics() -> None:
    priority = scored_alerts().assign(priority_band=["P4", "P1", "P2", "P3"], alert_timestamp=alerts()["alert_timestamp"])
    config = {
        "stress_testing": {
            "li_temporal_stress": {"enabled": True, "test_a_fraction": 0.5, "k_values": [2]},
            "segment_stress": {"enabled": True, "min_segment_alerts": 1, "segments": ["payment_format", "amount_bucket"]},
        }
    }

    temporal = build_temporal_stress_summary(priority, config)
    segment = build_segment_stress_summary(transactions(), priority, config)

    assert {"window", "precision_at_k", "recall_at_k"}.issubset(temporal.columns)
    assert {"segment", "segment_value", "status", "pr_auc"}.issubset(segment.columns)
