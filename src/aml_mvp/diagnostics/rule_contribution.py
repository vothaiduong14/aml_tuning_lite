"""Rule contribution diagnostics for alert-volume remediation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aml_mvp.storage import read_dataframe


def build_rule_contribution(config: dict[str, Any], root: Path, logger=None) -> pd.DataFrame:
    artifacts = dict(config.get("artifacts", {}))
    rule_hits = read_dataframe(_resolve(root, artifacts["rule_hits_path"]))
    alerts = read_dataframe(_resolve(root, artifacts["alerts_path"]))
    if logger:
        logger.info("Building rule contribution rule_hits=%s alerts=%s", len(rule_hits), len(alerts))

    label_col = _label_column(rule_hits)
    total_labels = int(alerts[_label_column(alerts)].sum()) if not alerts.empty and _label_column(alerts) in alerts else 0
    tx_rule_counts = rule_hits.groupby("transaction_id")["rule_id"].nunique()
    rule_order = _rank_rules(rule_hits, label_col)
    seen_transactions: set[Any] = set()
    rows: list[dict[str, Any]] = []

    for rule_id in rule_order:
        group = rule_hits[rule_hits["rule_id"].eq(rule_id)]
        tx_ids = set(group["transaction_id"].tolist())
        new_tx_ids = tx_ids - seen_transactions
        unique_alert_count = int(len(tx_ids))
        label_count = int(group.drop_duplicates("transaction_id")[label_col].sum()) if label_col in group else 0
        incremental = group[group["transaction_id"].isin(new_tx_ids)].drop_duplicates("transaction_id")
        incremental_label_count = int(incremental[label_col].sum()) if label_col in incremental else 0
        overlap_count = int(group["transaction_id"].map(tx_rule_counts).fillna(0).gt(1).sum())
        precision = _safe_divide(label_count, unique_alert_count)
        rows.append(
            {
                "rule_id": rule_id,
                "rule_name": str(group["rule_name"].iloc[0]) if "rule_name" in group and not group.empty else rule_id,
                "alert_count": int(len(group)),
                "unique_alert_count": unique_alert_count,
                "label_count": label_count,
                "precision": precision,
                "recall_contribution": _safe_divide(label_count, total_labels),
                "incremental_label_count": incremental_label_count,
                "incremental_alert_count": int(len(new_tx_ids)),
                "overlap_rate": _safe_divide(overlap_count, len(group)),
                "recommended_action": _recommended_action(unique_alert_count, precision, incremental_label_count),
            }
        )
        seen_transactions |= tx_ids

    contribution = pd.DataFrame(rows)
    output_path = _resolve(root, artifacts["rule_contribution_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contribution.to_csv(output_path, index=False)
    recommendations_path_value = artifacts.get("rule_volume_recommendations_path")
    if recommendations_path_value:
        recommendations = contribution[
            [
                "rule_id",
                "unique_alert_count",
                "label_count",
                "precision",
                "incremental_label_count",
                "overlap_rate",
                "recommended_action",
            ]
        ].copy()
        recommendations["recommendation_reason"] = recommendations.apply(_recommendation_reason, axis=1)
        recommendations_path = _resolve(root, recommendations_path_value)
        recommendations_path.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(recommendations_path, index=False)
    if logger:
        logger.info("Wrote rule contribution path=%s rows=%s", output_path, len(contribution))
    return contribution


def _rank_rules(rule_hits: pd.DataFrame, label_col: str) -> list[str]:
    if rule_hits.empty:
        return []
    summary = (
        rule_hits.groupby("rule_id")
        .agg(alert_count=("transaction_id", "count"), label_count=(label_col, "sum"))
        .reset_index()
    )
    summary["precision"] = summary["label_count"] / summary["alert_count"].clip(lower=1)
    summary = summary.sort_values(["label_count", "precision", "alert_count"], ascending=[False, False, True])
    return [str(value) for value in summary["rule_id"].tolist()]


def _recommended_action(unique_alert_count: int, precision: float, incremental_label_count: int) -> str:
    if unique_alert_count == 0:
        return "review"
    if incremental_label_count == 0 and unique_alert_count > 1000:
        return "retire_candidate"
    if precision < 0.001 and unique_alert_count > 10000:
        return "tighten"
    if precision < 0.005:
        return "downgrade"
    return "keep"


def _recommendation_reason(row: pd.Series) -> str:
    action = str(row.get("recommended_action", "review"))
    if action == "retire_candidate":
        return "High-volume rule adds no incremental labelled alerts."
    if action == "tighten":
        return "High-volume rule has very low precision."
    if action == "downgrade":
        return "Rule should remain coverage-first until precision improves."
    if action == "keep":
        return "Rule has acceptable labelled contribution for current guardrails."
    return "Review rule contribution manually."


def _label_column(df: pd.DataFrame) -> str:
    return "target" if "target" in df.columns else "is_laundering"


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
