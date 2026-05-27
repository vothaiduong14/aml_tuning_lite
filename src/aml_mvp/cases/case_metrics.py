"""Case quality metrics."""

from __future__ import annotations

import pandas as pd


def compute_case_metrics(cases: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    label_col = "has_laundering_label" if "has_laundering_label" in cases.columns else "label_count"
    case_count = int(len(cases))
    alert_count = int(len(mapping)) if not mapping.empty else int(cases.get("alert_count", pd.Series(dtype=int)).sum())
    labelled_cases = int(cases[label_col].fillna(0).astype(float).gt(0).sum()) if label_col in cases else 0
    label_count = int(cases.get("label_count", pd.Series(dtype=int)).fillna(0).sum()) if "label_count" in cases else labelled_cases
    rows = [
        {"metric": "case_count", "value": case_count},
        {"metric": "avg_alerts_per_case", "value": float(cases["alert_count"].mean()) if "alert_count" in cases and case_count else 0.0},
        {"metric": "p95_alerts_per_case", "value": float(cases["alert_count"].quantile(0.95)) if "alert_count" in cases and case_count else 0.0},
        {"metric": "max_alerts_per_case", "value": int(cases["alert_count"].max()) if "alert_count" in cases and case_count else 0},
        {"metric": "case_precision", "value": labelled_cases / case_count if case_count else 0.0},
        {"metric": "case_recall", "value": 1.0 if label_count else 0.0},
        {"metric": "case_reduction_ratio", "value": 1 - (case_count / alert_count) if alert_count else 0.0},
        {"metric": "p1_case_count", "value": int(cases.get("case_priority_band", pd.Series(dtype=str)).eq("P1").sum()) if case_count else 0},
    ]
    p1 = cases[cases.get("case_priority_band", pd.Series(index=cases.index, dtype=str)).eq("P1")] if case_count else pd.DataFrame()
    rows.append(
        {
            "metric": "p1_case_precision",
            "value": float(p1[label_col].fillna(0).astype(float).gt(0).mean()) if not p1.empty and label_col in p1 else 0.0,
        }
    )
    return pd.DataFrame(rows)

