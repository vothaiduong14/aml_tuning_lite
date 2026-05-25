from __future__ import annotations

from aml_mvp.extended.stress_testing import build_stress_test_summary

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

