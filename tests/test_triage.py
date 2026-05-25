from __future__ import annotations

from tests.fixtures import scored_alerts

from aml_mvp.triage.capacity_simulation import simulate_capacity
from aml_mvp.triage.priority_banding import assign_priority_bands


def test_assign_priority_bands_applies_critical_rule_override() -> None:
    priority_alerts, band_summary = assign_priority_bands(
        scored_alerts(),
        {
            "triage": {
                "p1_quantile": 0.99,
                "p2_quantile": 0.95,
                "p3_quantile": 0.80,
                "critical_rules": ["R4_PASS_THROUGH"],
            }
        },
    )

    critical_row = priority_alerts[priority_alerts["alert_id"].eq("A4")].iloc[0]
    assert critical_row["priority_band"] == "P1"
    assert set(band_summary["priority_band"]) == {"P1", "P2", "P3", "P4"}


def test_simulate_capacity_outputs_baseline_and_model_rankings() -> None:
    priority_alerts, _ = assign_priority_bands(scored_alerts(), {"triage": {"critical_rules": ["R4_PASS_THROUGH"]}})
    capacity = simulate_capacity(priority_alerts, {"triage": {"capacity_k_values": [1, 2]}})

    assert {"model_score", "rule_priority", "amount_rank"}.issubset(set(capacity["ranking"]))
    assert set(capacity["k"]) == {1, 2}

