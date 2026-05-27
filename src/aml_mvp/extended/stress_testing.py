"""LI-first stress-test summary for the extended build."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aml_mvp.models.evaluate import pr_auc, precision_at_k, recall_at_k


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


def build_temporal_stress_summary(priority_alerts: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    settings = dict(config.get("stress_testing", {}).get("li_temporal_stress", {}))
    if not bool(settings.get("enabled", True)) or priority_alerts.empty:
        return pd.DataFrame(columns=["window", "alert_count", "label_count", "alert_rate", "pr_auc"])
    frame = priority_alerts.copy()
    timestamp_col = "alert_timestamp" if "alert_timestamp" in frame else "timestamp" if "timestamp" in frame else None
    if timestamp_col:
        frame = frame.sort_values(timestamp_col).reset_index(drop=True)
    split_at = max(1, min(len(frame) - 1, int(len(frame) * float(settings.get("test_a_fraction", 0.5))))) if len(frame) > 1 else len(frame)
    score_col = _score_column(frame)
    label_col = _label_column(frame)
    k_values = [int(value) for value in settings.get("k_values", [100, 500, 1000])]
    rows = []
    for name, group in [("A", frame.iloc[:split_at]), ("B", frame.iloc[split_at:])]:
        rows.append(_stress_row(name, group, label_col, score_col, None))
        for k in k_values:
            rows.append(_stress_row(name, group, label_col, score_col, k))
    return pd.DataFrame(rows)


def build_segment_stress_summary(transactions: pd.DataFrame, priority_alerts: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    settings = dict(config.get("stress_testing", {}).get("segment_stress", {}))
    if not bool(settings.get("enabled", True)) or priority_alerts.empty:
        return pd.DataFrame(columns=["segment", "segment_value", "alert_count", "label_count", "precision_at_100"])
    frame = priority_alerts.merge(
        transactions[["transaction_id", "payment_format", "currency_pair", "cross_bank_flag", "amount"]].drop_duplicates("transaction_id"),
        on="transaction_id",
        how="left",
        suffixes=("", "_tx"),
    )
    try:
        frame["amount_bucket"] = pd.qcut(frame["amount"].fillna(0.0), q=4, duplicates="drop").astype(str)
    except ValueError:
        frame["amount_bucket"] = "all"
    if "graph_degree_bucket" not in frame:
        graph_cols = [column for column in frame.columns if "degree" in column and column.startswith("feature_graph")]
        degree = frame[graph_cols].sum(axis=1) if graph_cols else pd.Series(0, index=frame.index)
        frame["graph_degree_bucket"] = pd.cut(degree, bins=[-1, 0, 5, float("inf")], labels=["none", "low", "high"])
    min_alerts = int(settings.get("min_segment_alerts", 10))
    score_col = _score_column(frame)
    label_col = _label_column(frame)
    rows = []
    for segment in settings.get("segments", ["payment_format", "currency_pair"]):
        if segment not in frame:
            continue
        for value, group in frame.groupby(segment, dropna=False):
            if len(group) < min_alerts:
                status = "small_segment"
            else:
                status = "measured"
            rows.append(
                {
                    "segment": segment,
                    "segment_value": str(value),
                    "status": status,
                    "alert_count": int(len(group)),
                    "label_count": int(group[label_col].fillna(0).sum()) if label_col in group else 0,
                    "precision_at_100": precision_at_k(group[label_col].fillna(0).astype(int), group[score_col].fillna(0.0), min(100, len(group))) if label_col in group and score_col in group and len(group) else 0.0,
                    "pr_auc": pr_auc(group[label_col].fillna(0).astype(int), group[score_col].fillna(0.0)) if label_col in group and score_col in group and len(group) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def write_extended_stress_outputs(
    temporal_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    artifacts: dict[str, str],
    root: Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for key, frame in [
        ("temporal_stress_summary_path", temporal_summary),
        ("segment_stress_summary_path", segment_summary),
    ]:
        value = artifacts.get(key)
        if not value:
            continue
        path = _resolve(root, value)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        outputs[key] = path
    return outputs


def _priority_count(priority_alerts: pd.DataFrame, band: str) -> int:
    if priority_alerts.empty or "priority_band" not in priority_alerts:
        return 0
    return int(priority_alerts["priority_band"].eq(band).sum())


def _stress_row(window: str, group: pd.DataFrame, label_col: str, score_col: str, k: int | None) -> dict[str, Any]:
    labels = group[label_col].fillna(0).astype(int) if label_col in group else pd.Series([0] * len(group))
    scores = group[score_col].fillna(0.0) if score_col in group else pd.Series([0.0] * len(group))
    row = {
        "window": window,
        "k": k,
        "alert_count": int(len(group)),
        "label_count": int(labels.sum()) if len(labels) else 0,
        "alert_rate": 1.0,
        "pr_auc": pr_auc(labels, scores) if len(group) else 0.0,
    }
    if k:
        effective_k = min(k, len(group))
        row["precision_at_k"] = precision_at_k(labels, scores, effective_k) if effective_k else 0.0
        row["recall_at_k"] = recall_at_k(labels, scores, effective_k) if effective_k else 0.0
    return row


def _score_column(frame: pd.DataFrame) -> str:
    for column in ["risk_score_1000", "calibrated_score", "model_score", "priority_score", "rule_priority_score"]:
        if column in frame:
            return column
    return "model_score"


def _label_column(frame: pd.DataFrame) -> str:
    return "target" if "target" in frame else "is_laundering"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
