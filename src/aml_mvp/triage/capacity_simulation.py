"""Investigator capacity simulation for alert rankings."""

from __future__ import annotations

from typing import Any

import pandas as pd

from aml_mvp.models.evaluate import lift_at_k, precision_at_k, recall_at_k


def simulate_capacity(priority_alerts: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    triage_config = dict(config.get("triage", {}))
    k_values = [int(value) for value in triage_config.get("capacity_k_values", [100, 500, 1000])]
    rankings = {
        "model_score": priority_alerts["model_score"].fillna(0.0),
        "rule_priority": priority_alerts["feature_rule_priority_score"].fillna(0.0),
        "amount_rank": priority_alerts["feature_amount"].fillna(0.0),
    }
    rows = []
    y_true = priority_alerts["target"].astype(int)
    for ranking_name, scores in rankings.items():
        for k in k_values:
            rows.append(
                {
                    "ranking": ranking_name,
                    "k": int(k),
                    "reviewed_alerts": int(min(k, len(priority_alerts))),
                    "precision_at_k": precision_at_k(y_true, scores, int(k)),
                    "recall_at_k": recall_at_k(y_true, scores, int(k)),
                    "lift_at_k": lift_at_k(y_true, scores, int(k)),
                }
            )
    return pd.DataFrame(rows)

