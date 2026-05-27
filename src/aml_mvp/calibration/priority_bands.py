"""Priority-band assignment for calibrated extended scores."""

from __future__ import annotations

from typing import Any

import pandas as pd


def assign_calibrated_priority_bands(scored_alerts: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    calibration = dict(config.get("calibration", {}))
    p1_quantile = float(calibration.get("p1_quantile", 0.99))
    p2_quantile = float(calibration.get("p2_quantile", 0.95))
    p3_quantile = float(calibration.get("p3_quantile", 0.80))
    critical_rules = set(calibration.get("critical_rules", []))

    output = scored_alerts.copy()
    scoring = dict(config.get("scoring", {}))
    score_col = str(scoring.get("calibrated_probability_column", "calibrated_score"))
    if score_col not in output:
        score_col = "calibrated_score" if "calibrated_score" in output else "model_score"
    scores = output[score_col].fillna(0.0)
    thresholds = {
        "P1": float(scores.quantile(p1_quantile)) if len(scores) else 1.0,
        "P2": float(scores.quantile(p2_quantile)) if len(scores) else 1.0,
        "P3": float(scores.quantile(p3_quantile)) if len(scores) else 1.0,
    }

    output["calibrated_priority_band"] = "P4"
    output.loc[scores >= thresholds["P3"], "calibrated_priority_band"] = "P3"
    output.loc[scores >= thresholds["P2"], "calibrated_priority_band"] = "P2"
    output.loc[scores >= thresholds["P1"], "calibrated_priority_band"] = "P1"
    if "triggered_rules" in output and critical_rules:
        critical_mask = output["triggered_rules"].fillna("").map(
            lambda value: bool(set(str(value).split(",")) & critical_rules)
        )
        output.loc[critical_mask, "calibrated_priority_band"] = "P1"
    output["calibrated_priority_rank"] = output["calibrated_priority_band"].map({"P1": 1, "P2": 2, "P3": 3, "P4": 4})
    return output.sort_values(["calibrated_priority_rank", score_col], ascending=[True, False]).reset_index(drop=True)


def summarize_calibrated_bands(alerts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_positives = int(alerts["target"].sum()) if "target" in alerts else 0
    for band in ["P1", "P2", "P3", "P4"]:
        group = alerts[alerts["calibrated_priority_band"].eq(band)]
        positives = int(group["target"].sum()) if "target" in group else 0
        rows.append(
            {
                "priority_band": band,
                "alert_count": int(len(group)),
                "label_count": positives,
                "precision": float(positives / len(group)) if len(group) else 0.0,
                "recall": float(positives / total_positives) if total_positives else 0.0,
                "avg_calibrated_score": float(group["calibrated_score"].mean()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows)
