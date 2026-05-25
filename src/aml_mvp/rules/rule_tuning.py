"""Threshold tuning with explicit guardrails."""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from aml_mvp.models.evaluate import compute_binary_metrics
from aml_mvp.rules.amount_rule import calculate_segment_thresholds


def tune_rules(
    transactions: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Tune supported threshold rules and return candidates, selections, and audit."""

    logger = logger or logging.getLogger(__name__)
    tuning_config = dict(config.get("tuning", {}))
    rules_to_tune = set(tuning_config.get("rules", ["R1_AMOUNT"]))
    logger.info("Rule tuning started rows=%s rules=%s", len(transactions), sorted(rules_to_tune))
    candidate_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    tuned_config = copy.deepcopy(config)

    if "R1_AMOUNT" in rules_to_tune:
        candidates, selected = tune_r1_amount(transactions, config, logger=logger)
        candidate_frames.append(candidates)
        selected_rows.append(selected)
        tuned_config.setdefault("R1_AMOUNT", {})["percentile"] = float(selected["selected_percentile"])

    all_candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    selected_thresholds = pd.DataFrame(selected_rows)
    audit = {
        "status": "completed",
        "tuned_rules": sorted(rules_to_tune),
        "selection_count": len(selected_rows),
        "selected": selected_rows,
        "tuned_config": tuned_config,
    }
    logger.info("Rule tuning completed selections=%s", len(selected_rows))
    return all_candidates, selected_thresholds, audit


def tune_r1_amount(
    transactions: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tune the R1 amount percentile against the configured validation split."""

    logger = logger or logging.getLogger(__name__)
    rule_config = dict(config.get("R1_AMOUNT", {}))
    tuning_config = dict(config.get("tuning", {}))
    grid = [float(value) for value in rule_config.get("percentile_grid", [0.70, 0.85, 0.90, 0.95, 0.99])]
    evaluation_split = str(tuning_config.get("split", "validation"))
    min_recall_floor = float(tuning_config.get("min_recall_floor", 0.80))
    max_alert_rate = float(tuning_config.get("max_alert_rate", 1.00))

    logger.info(
        "Tuning R1_AMOUNT candidates=%s evaluation_split=%s min_recall_floor=%.4f max_alert_rate=%.4f",
        len(grid),
        evaluation_split,
        min_recall_floor,
        max_alert_rate,
    )
    rows = []
    common_config = dict(config.get("rules", {}))
    segmentation_fields = list(common_config.get("segmentation_fields", ["payment_format", "currency_pair"]))
    fallback_percentile = float(common_config.get("fallback_percentile", 0.95))
    min_segment_transactions = int(common_config.get("min_segment_transactions", 1000))
    training_transactions = transactions[transactions["split"].eq("train")].copy()
    if training_transactions.empty:
        training_transactions = transactions.copy()
    eval_transactions = transactions[transactions["split"].eq(evaluation_split)].copy()

    for percentile in grid:
        candidate_start = time.perf_counter()
        logger.info("Starting R1_AMOUNT candidate percentile=%.4f", percentile)
        thresholds, fallback_threshold = calculate_segment_thresholds(
            training_transactions,
            segmentation_fields,
            percentile,
            min_segment_transactions,
            fallback_percentile,
        )
        scored = eval_transactions.merge(thresholds, on=segmentation_fields, how="left")
        scored["threshold"] = scored["threshold"].fillna(fallback_threshold)
        y_pred = scored["amount"].ge(scored["threshold"]).astype(int)
        metrics = compute_binary_metrics(eval_transactions["is_laundering"], y_pred)
        logger.info(
            (
                "Completed R1_AMOUNT candidate percentile=%.4f "
                "tp=%s fp=%s fn=%s precision=%.6f recall=%.6f alert_rate=%.6f elapsed=%.2fs"
            ),
            percentile,
            metrics.true_positives,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.precision,
            metrics.recall,
            metrics.alert_rate,
            time.perf_counter() - candidate_start,
        )
        rows.append(
            {
                "rule_id": "R1_AMOUNT",
                "parameter": "percentile",
                "candidate_value": percentile,
                "split": evaluation_split,
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "true_negatives": metrics.true_negatives,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "alert_rate": metrics.alert_rate,
                "f1": metrics.f1,
                "meets_recall_floor": metrics.recall >= min_recall_floor,
                "meets_alert_rate_cap": metrics.alert_rate <= max_alert_rate,
            }
        )

    candidates = pd.DataFrame(rows)
    selected = select_candidate(candidates, min_recall_floor, max_alert_rate)
    logger.info(
        "Selected R1_AMOUNT percentile=%.4f reason=%s recall=%.6f alert_rate=%.6f precision=%.6f",
        float(selected["candidate_value"]),
        str(selected["selection_reason"]),
        float(selected["recall"]),
        float(selected["alert_rate"]),
        float(selected["precision"]),
    )
    return candidates, {
        "rule_id": "R1_AMOUNT",
        "parameter": "percentile",
        "selected_percentile": float(selected["candidate_value"]),
        "selection_reason": str(selected["selection_reason"]),
        "split": evaluation_split,
        "recall": float(selected["recall"]),
        "alert_rate": float(selected["alert_rate"]),
        "precision": float(selected["precision"]),
    }


def select_candidate(candidates: pd.DataFrame, min_recall_floor: float, max_alert_rate: float) -> pd.Series:
    """Select lowest alert-rate candidate that satisfies guardrails.

    If no candidate satisfies both guardrails, choose the highest-recall
    candidate and record that the guardrail was not met.
    """

    eligible = candidates[
        candidates["recall"].ge(min_recall_floor)
        & candidates["alert_rate"].le(max_alert_rate)
    ].copy()
    if not eligible.empty:
        selected = eligible.sort_values(
            ["alert_rate", "candidate_value"],
            ascending=[True, False],
        ).iloc[0].copy()
        selected["selection_reason"] = "met_recall_floor_and_alert_rate_cap"
        return selected

    selected = candidates.sort_values(
        ["recall", "alert_rate"],
        ascending=[False, True],
    ).iloc[0].copy()
    selected["selection_reason"] = "guardrail_not_met_selected_highest_recall"
    return selected


def write_tuning_outputs(
    candidates: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    audit: dict[str, Any],
    artifacts: dict[str, str],
    root: Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    candidates_path = _resolve(root, artifacts["tuning_candidates_path"])
    selected_path = _resolve(root, artifacts["selected_thresholds_path"])
    audit_path = _resolve(root, artifacts["tuning_audit_path"])
    tuned_config_path = _resolve(root, artifacts["tuned_rule_config_path"])

    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_path, index=False)
    selected_thresholds.to_csv(selected_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    tuned_config_path.write_text(json.dumps(audit["tuned_config"], indent=2, default=str), encoding="utf-8")

    outputs["tuning_candidates"] = candidates_path
    outputs["selected_thresholds"] = selected_path
    outputs["tuning_audit"] = audit_path
    outputs["tuned_rule_config"] = tuned_config_path
    return outputs


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
