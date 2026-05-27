"""Capacity-aware top-K simulations for remediation priority bands."""

from __future__ import annotations

from typing import Any

import pandas as pd

from aml_mvp.models.evaluate import lift_at_k, precision_at_k, recall_at_k


def run_capacity_simulation(config: dict[str, Any], priority_alerts: pd.DataFrame) -> pd.DataFrame:
    priority_config = dict(config.get("priority_bands", {}))
    capacity_config = dict(priority_config.get("daily_capacity_assumption", {}))
    total_capacity = int(
        capacity_config.get(
            "total_daily_capacity",
            int(capacity_config.get("investigators", 3)) * int(capacity_config.get("alerts_per_investigator_per_day", 40)),
        )
    )
    k_values = [100, 500, 1000, 5000, total_capacity]
    label_col = "target" if "target" in priority_alerts.columns else "is_laundering"
    score_col = str(priority_config.get("score_column", "priority_score_v2"))
    if score_col not in priority_alerts.columns:
        score_col = "model_score" if "model_score" in priority_alerts.columns else "priority_score_v2"
    labels = priority_alerts[label_col].fillna(0).astype(int) if label_col in priority_alerts else pd.Series([0] * len(priority_alerts))
    scores = priority_alerts[score_col].fillna(0.0) if score_col in priority_alerts else pd.Series([0.0] * len(priority_alerts))
    rows = []
    for k in sorted(set(int(value) for value in k_values if int(value) > 0)):
        rows.append(
            {
                "scenario": "daily_capacity" if k == total_capacity else f"top_{k}",
                "k": k,
                "precision_at_k": precision_at_k(labels, scores, k),
                "recall_at_k": recall_at_k(labels, scores, k),
                "lift_at_k": lift_at_k(labels, scores, k),
                "coverage_at_capacity": recall_at_k(labels, scores, k) if k == total_capacity else None,
            }
        )
    return pd.DataFrame(rows)

