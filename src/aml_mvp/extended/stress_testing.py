"""LI-first stress-test summary for the extended build."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_stress_test_summary(
    transactions: pd.DataFrame,
    alerts: pd.DataFrame,
    rule_hits: pd.DataFrame,
    priority_alerts: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize the current MVP run as the LI-only benchmark baseline."""

    project = dict(config.get("project", {}))
    return pd.DataFrame(
        [
            {
                "dataset_name": project.get("dataset_name", "LI-Small"),
                "comparison_scope": "LI-only",
                "hi_comparison_status": project.get("hi_comparison_status", "deferred_missing_dataset"),
                "transaction_count": int(len(transactions)),
                "alert_count": int(len(alerts)),
                "rule_hit_count": int(len(rule_hits)),
                "alert_rate": float(len(alerts) / len(transactions)) if len(transactions) else 0.0,
                "label_count": int(transactions["is_laundering"].sum()) if "is_laundering" in transactions else 0,
                "alert_label_count": int(alerts["is_laundering"].sum()) if "is_laundering" in alerts else 0,
                "p1_alert_count": _priority_count(priority_alerts, "P1"),
                "p2_alert_count": _priority_count(priority_alerts, "P2"),
                "p3_alert_count": _priority_count(priority_alerts, "P3"),
                "p4_alert_count": _priority_count(priority_alerts, "P4"),
            }
        ]
    )


def write_stress_summary(summary: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path


def _priority_count(priority_alerts: pd.DataFrame, band: str) -> int:
    if priority_alerts.empty or "priority_band" not in priority_alerts:
        return 0
    return int(priority_alerts["priority_band"].eq(band).sum())

