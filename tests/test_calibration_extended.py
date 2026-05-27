from __future__ import annotations

import pandas as pd

from aml_mvp.calibration.calibration import calibrate_scores
from aml_mvp.calibration.priority_bands import assign_calibrated_priority_bands


def test_calibration_adds_scores_and_band_metrics() -> None:
    frame = pd.DataFrame(
        {
            "alert_id": [f"A{i}" for i in range(20)],
            "split": ["validation"] * 20,
            "target": [0] * 10 + [1] * 10,
            "model_score": [i / 20 for i in range(20)],
            "triggered_rules": [""] * 20,
        }
    )

    calibrated, metrics = calibrate_scores(frame, {"calibration": {"method": "isotonic"}})

    assert calibrated["calibrated_score"].between(0, 1).all()
    assert calibrated["risk_score_1000"].between(0, 1000).all()
    assert calibrated.sort_values("calibrated_score")["risk_score_1000"].is_monotonic_increasing
    assert "P1" in set(calibrated["calibrated_priority_band"])
    assert set(metrics["priority_band"]) == {"P1", "P2", "P3", "P4"}


def test_critical_rule_overrides_priority_band() -> None:
    frame = pd.DataFrame(
        {
            "alert_id": ["A1", "A2"],
            "calibrated_score": [0.01, 0.02],
            "triggered_rules": ["GR2_CYCLE_CANDIDATE", ""],
        }
    )

    output = assign_calibrated_priority_bands(
        frame,
        {"calibration": {"critical_rules": ["GR2_CYCLE_CANDIDATE"], "p1_quantile": 0.99}},
    )

    assert output.loc[output["alert_id"].eq("A1"), "calibrated_priority_band"].iloc[0] == "P1"
