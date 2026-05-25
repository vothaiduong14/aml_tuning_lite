"""Rule orchestration, alert consolidation, and rule diagnostics."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from aml_mvp.models.evaluate import compute_binary_metrics
from aml_mvp.rules.amount_rule import apply_amount_rule
from aml_mvp.rules.base_rule import SEVERITY_SCORE, empty_rule_hits
from aml_mvp.rules.network_rules import apply_fan_in_rule, apply_fan_out_rule
from aml_mvp.rules.novelty_rule import apply_novelty_rule
from aml_mvp.rules.pass_through_rule import apply_pass_through_rule
from aml_mvp.rules.velocity_rule import apply_velocity_rule


RULE_NAMES = {
    "R1_AMOUNT": "Segmented high amount",
    "R2_NOVELTY": "New counterparty / cross-bank novelty",
    "R3_VELOCITY": "Velocity / structuring",
    "R4_PASS_THROUGH": "Rapid pass-through",
    "R5_FAN_IN": "Fan-in concentration",
    "R6_FAN_OUT": "Fan-out / gather-scatter-lite",
}


def run_rule_engine(
    transactions: pd.DataFrame,
    config: dict,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run configured rules and return rule hits plus consolidated alerts."""

    logger = logger or logging.getLogger(__name__)
    common_config = dict(config.get("rules", {}))
    enabled = set(common_config.get("enabled", RULE_NAMES.keys()))
    training_transactions = transactions[transactions["split"].eq("train")].copy()
    if training_transactions.empty:
        training_transactions = transactions.copy()

    ordered_enabled = [rule_id for rule_id in RULE_NAMES if rule_id in enabled]
    logger.info(
        "Rule engine started rows=%s train_rows=%s enabled_rules=%s",
        len(transactions),
        len(training_transactions),
        ordered_enabled,
    )

    hits = []
    if "R1_AMOUNT" in enabled:
        hits.append(
            _run_rule(
                "R1_AMOUNT",
                logger,
                apply_amount_rule,
                transactions,
                training_transactions,
                dict(config.get("R1_AMOUNT", {})),
                common_config,
            )
        )
    if "R2_NOVELTY" in enabled:
        hits.append(
            _run_rule(
                "R2_NOVELTY",
                logger,
                apply_novelty_rule,
                transactions,
                dict(config.get("R2_NOVELTY", {})),
            )
        )
    if "R3_VELOCITY" in enabled:
        hits.append(
            _run_rule(
                "R3_VELOCITY",
                logger,
                apply_velocity_rule,
                transactions,
                training_transactions,
                dict(config.get("R3_VELOCITY", {})),
            )
        )
    if "R4_PASS_THROUGH" in enabled:
        hits.append(
            _run_rule(
                "R4_PASS_THROUGH",
                logger,
                apply_pass_through_rule,
                transactions,
                dict(config.get("R4_PASS_THROUGH", {})),
            )
        )
    if "R5_FAN_IN" in enabled:
        hits.append(
            _run_rule(
                "R5_FAN_IN",
                logger,
                apply_fan_in_rule,
                transactions,
                training_transactions,
                dict(config.get("R5_FAN_IN", {})),
            )
        )
    if "R6_FAN_OUT" in enabled:
        hits.append(
            _run_rule(
                "R6_FAN_OUT",
                logger,
                apply_fan_out_rule,
                transactions,
                dict(config.get("R6_FAN_OUT", {})),
            )
        )

    rule_hits = pd.concat(hits, ignore_index=True) if hits else empty_rule_hits()
    logger.info("Consolidating alerts from rule_hits=%s", len(rule_hits))
    consolidate_start = time.perf_counter()
    alerts = consolidate_alerts(rule_hits)
    alert_rate = len(alerts) / len(transactions) if len(transactions) else 0.0
    logger.info(
        "Alert consolidation completed alerts=%s alert_rate=%.6f elapsed=%.2fs",
        len(alerts),
        alert_rate,
        time.perf_counter() - consolidate_start,
    )
    logger.info("Rule engine completed rule_hits=%s alerts=%s", len(rule_hits), len(alerts))
    return rule_hits, alerts


def _run_rule(rule_id: str, logger: logging.Logger, func, *args, **kwargs) -> pd.DataFrame:
    start = time.perf_counter()
    logger.info("Starting %s", rule_id)
    hits = func(*args, **kwargs)
    logger.info(
        "Completed %s: hits=%s elapsed=%.2fs",
        rule_id,
        len(hits),
        time.perf_counter() - start,
    )
    return hits


def consolidate_alerts(rule_hits: pd.DataFrame) -> pd.DataFrame:
    if rule_hits.empty:
        return pd.DataFrame(
            columns=[
                "alert_id",
                "transaction_id",
                "alert_timestamp",
                "triggered_rules",
                "rule_count",
                "max_rule_severity",
                "rule_priority_score",
                "rationale",
                "is_laundering",
                "split",
            ]
        )

    rows = []
    for transaction_id, group in rule_hits.groupby("transaction_id", sort=False):
        severities = group["severity"].astype(str).tolist()
        max_severity = max(severities, key=lambda value: SEVERITY_SCORE.get(value, 0))
        rule_count = int(group["rule_id"].nunique())
        priority_score = float(SEVERITY_SCORE.get(max_severity, 0) + 0.1 * max(rule_count - 1, 0))
        triggered_rules = sorted(group["rule_id"].unique().tolist())
        rows.append(
            {
                "alert_id": f"ALERT-{transaction_id}",
                "transaction_id": transaction_id,
                "alert_timestamp": group["alert_timestamp"].min(),
                "triggered_rules": ",".join(triggered_rules),
                "rule_count": rule_count,
                "max_rule_severity": max_severity,
                "rule_priority_score": priority_score,
                "rationale": " | ".join(group["rationale"].astype(str).unique().tolist()),
                "is_laundering": int(group["is_laundering"].max()),
                "split": str(group["split"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["alert_timestamp", "transaction_id"]).reset_index(drop=True)


def build_rule_performance(transactions: pd.DataFrame, rule_hits: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split_values = sorted(transactions["split"].dropna().unique().tolist())
    for split in split_values:
        split_transactions = transactions[transactions["split"].eq(split)]
        for rule_id in sorted(rule_hits["rule_id"].dropna().unique().tolist()):
            hit_ids = set(rule_hits.loc[rule_hits["rule_id"].eq(rule_id) & rule_hits["split"].eq(split), "transaction_id"])
            y_pred = split_transactions["transaction_id"].isin(hit_ids).astype(int)
            metrics = compute_binary_metrics(split_transactions["is_laundering"], y_pred)
            rows.append(_metrics_row(split, rule_id, RULE_NAMES.get(rule_id, rule_id), metrics))

        alert_ids = set(alerts.loc[alerts["split"].eq(split), "transaction_id"])
        y_pred = split_transactions["transaction_id"].isin(alert_ids).astype(int)
        metrics = compute_binary_metrics(split_transactions["is_laundering"], y_pred)
        rows.append(_metrics_row(split, "ANY_RULE", "Any rule alert", metrics))

    return pd.DataFrame(rows)


def build_rule_overlap_matrix(rule_hits: pd.DataFrame) -> pd.DataFrame:
    if rule_hits.empty:
        return pd.DataFrame()
    indicators = pd.crosstab(rule_hits["transaction_id"], rule_hits["rule_id"]).clip(upper=1)
    overlap = indicators.T.dot(indicators)
    overlap.index.name = "rule_id"
    return overlap.reset_index()


def write_rule_outputs(
    transactions: pd.DataFrame,
    rule_hits: pd.DataFrame,
    alerts: pd.DataFrame,
    artifacts: dict,
    root: Path,
    write_dataframe,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["rule_hits"] = write_dataframe(rule_hits, _resolve(root, artifacts["rule_hits_path"]))
    outputs["alerts"] = write_dataframe(alerts, _resolve(root, artifacts["alerts_path"]))
    performance = build_rule_performance(transactions, rule_hits, alerts)
    performance_path = _resolve(root, artifacts["rule_performance_path"])
    performance_path.parent.mkdir(parents=True, exist_ok=True)
    performance.to_csv(performance_path, index=False)
    outputs["rule_performance"] = performance_path
    overlap = build_rule_overlap_matrix(rule_hits)
    overlap_path = _resolve(root, artifacts["rule_overlap_path"])
    overlap_path.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(overlap_path, index=False)
    outputs["rule_overlap"] = overlap_path
    return outputs


def _metrics_row(split: str, rule_id: str, rule_name: str, metrics) -> dict:
    return {
        "split": split,
        "rule_id": rule_id,
        "rule_name": rule_name,
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "true_negatives": metrics.true_negatives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "alert_rate": metrics.alert_rate,
        "f1": metrics.f1,
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
