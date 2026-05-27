"""Capacity-aware priority band rebuild for remediation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aml_mvp.calibration.capacity_simulation import run_capacity_simulation
from aml_mvp.models.evaluate import lift_at_k, precision_at_k, recall_at_k
from aml_mvp.storage import read_dataframe, write_dataframe


def rebuild_priority_bands(config: dict[str, Any], root: Path, logger=None) -> pd.DataFrame:
    artifacts = dict(config.get("artifacts", {}))
    priority_config = dict(config.get("priority_bands", {}))
    scored = _read_scored_alerts(root, artifacts)
    alerts = read_dataframe(_resolve(root, artifacts["alerts_path"]))
    if logger:
        logger.info("Rebuilding priority bands scored_alerts=%s alerts=%s", len(scored), len(alerts))
    frame = alerts.merge(scored, on=["alert_id", "transaction_id"], how="left", suffixes=("", "_score"))
    frame = _ensure_risk_score(frame, config)
    score_col = _score_column(frame, priority_config)
    frame["priority_score_v2"] = frame[score_col].fillna(0.0)
    frame["priority_band_v2"] = "P4"
    frame["priority_rank_v2"] = 4
    frame["p1_override_reason"] = ""
    frame["band_assignment_reason"] = ""

    bands = dict(priority_config.get("bands", {}))
    thresholds = {
        name: float(np.nanpercentile(frame["priority_score_v2"], float(settings.get("score_percentile_floor", 0.0))))
        for name, settings in bands.items()
    }
    for band, rank in [("P3", 3), ("P2", 2), ("P1", 1)]:
        if band in thresholds:
            mask = frame["priority_score_v2"].ge(thresholds[band])
            frame.loc[mask, "priority_band_v2"] = band
            frame.loc[mask, "priority_rank_v2"] = rank
            frame.loc[mask, "band_assignment_reason"] = f"score >= {thresholds[band]:.6g}"

    _apply_p1_overrides(frame, priority_config, thresholds.get("P1", frame["priority_score_v2"].max()))
    _apply_capacity_caps(frame, bands)
    frame["priority_band"] = frame.get("priority_band", frame["priority_band_v2"])

    priority_path = write_dataframe(frame, _resolve(root, artifacts["output_priority_alerts_path"]))
    band_summary = _band_summary(frame)
    band_summary_path = _resolve(root, artifacts["band_summary_path"])
    band_summary_path.parent.mkdir(parents=True, exist_ok=True)
    band_summary.to_csv(band_summary_path, index=False)
    capacity = run_capacity_simulation(config, frame)
    capacity_path = _resolve(root, artifacts["capacity_simulation_path"])
    capacity_path.parent.mkdir(parents=True, exist_ok=True)
    capacity.to_csv(capacity_path, index=False)
    top_k = _top_k_after_band_fix(frame)
    top_k_path = _resolve(root, artifacts["top_k_after_band_fix_path"])
    top_k_path.parent.mkdir(parents=True, exist_ok=True)
    top_k.to_csv(top_k_path, index=False)
    if logger:
        logger.info("Wrote priority bands path=%s rows=%s", priority_path, len(frame))
        logger.info("Wrote band summary path=%s", band_summary_path)
    return frame


def _read_scored_alerts(root: Path, artifacts: dict[str, str]) -> pd.DataFrame:
    for key in ["scored_alerts_path", "fallback_scored_alerts_path"]:
        value = artifacts.get(key)
        if value:
            path = _resolve(root, value)
            if path.exists() or path.with_suffix(path.suffix + ".pkl").exists():
                return read_dataframe(path)
    raise FileNotFoundError("No scored alerts artifact was available for priority band rebuild.")


def _score_column(frame: pd.DataFrame, priority_config: dict[str, Any]) -> str:
    for candidate in [priority_config.get("score_column"), priority_config.get("fallback_score_column"), "calibrated_score", "model_score"]:
        if candidate and str(candidate) in frame.columns:
            return str(candidate)
    return "rule_priority_score"


def _apply_capacity_caps(frame: pd.DataFrame, bands: dict[str, Any]) -> None:
    ordered = frame.sort_values("priority_score_v2", ascending=False)
    for band, rank in [("P1", 1), ("P2", 2)]:
        settings = dict(bands.get(band, {}))
        max_daily = settings.get("max_alerts", settings.get("max_daily_alerts"))
        if not max_daily:
            continue
        max_alerts = int(max_daily)
        band_index = ordered[ordered["priority_band_v2"].eq(band)].index
        demote_index = band_index[max_alerts:]
        if len(demote_index):
            next_band = settings.get("overflow_band") or ("P1_overflow" if band == "P1" else "P3")
            frame.loc[demote_index, "priority_band_v2"] = next_band
            frame.loc[demote_index, "priority_rank_v2"] = 2 if next_band in {"P1_overflow", "P2"} else 3
            frame.loc[demote_index, "band_assignment_reason"] = f"demoted by {band} capacity cap"


def _apply_p1_overrides(frame: pd.DataFrame, priority_config: dict[str, Any], p1_threshold: float) -> None:
    override = dict(priority_config.get("critical_override_conditions", {}))
    if not bool(override.get("enabled", True)):
        return
    min_rule_count = int(override.get("min_rule_count", 2))
    excluded = {str(value) for value in override.get("excluded_rule_ids", [])}
    eligible_statuses = {str(value) for value in override.get("eligible_rule_statuses", [])}
    triggered = frame.get("triggered_rules", pd.Series("", index=frame.index)).fillna("").astype(str)
    has_excluded_only = triggered.map(lambda value: _triggered_rules(value) and _triggered_rules(value).issubset(excluded))
    if eligible_statuses and "rule_statuses" in frame:
        eligible_rule_mask = frame["rule_statuses"].fillna("").astype(str).map(
            lambda value: bool(set(value.split(",")) & eligible_statuses)
        )
    elif "escalation_rule_count" in frame:
        eligible_rule_mask = frame["escalation_rule_count"].fillna(0).gt(0)
    else:
        eligible_rule_mask = ~has_excluded_only
    critical_mask = frame.get("max_rule_severity", pd.Series("", index=frame.index)).astype(str).isin(["critical", "high"])
    critical_mask = critical_mask & eligible_rule_mask & ~has_excluded_only
    extra_mask = frame.get("rule_count", pd.Series(0, index=frame.index)).fillna(0).ge(min_rule_count)
    if bool(override.get("require_score_floor", True)):
        min_score = float(override.get("min_score", p1_threshold))
        extra_mask = extra_mask & frame["priority_score_v2"].ge(min_score)
    promote = critical_mask & extra_mask
    frame.loc[promote, "priority_band_v2"] = "P1"
    frame.loc[promote, "priority_rank_v2"] = 1
    frame.loc[promote, "p1_override_reason"] = "critical_rule_with_configured_secondary_condition"


def _triggered_rules(value: str) -> set[str]:
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _ensure_risk_score(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    scoring = dict(config.get("scoring", {}))
    score_col = str(scoring.get("risk_score_column", "risk_score_1000"))
    if score_col in frame.columns:
        return frame
    source_col = str(scoring.get("calibrated_probability_column", "calibrated_score"))
    if source_col not in frame:
        source_col = "model_score" if "model_score" in frame else "rule_priority_score"
    source = frame[source_col].fillna(0.0).astype(float) if source_col in frame else pd.Series(0.0, index=frame.index)
    if len(source) <= 1:
        frame[score_col] = 1000 if len(source) else pd.Series(dtype="int64")
    else:
        frame[score_col] = (source.rank(method="average", pct=True) * 1000).round().clip(0, 1000).astype(int)
    return frame


def _band_summary(frame: pd.DataFrame) -> pd.DataFrame:
    label_col = "target" if "target" in frame.columns else "is_laundering"
    rows = []
    for band, group in frame.groupby("priority_band_v2", sort=True):
        labels = int(group[label_col].fillna(0).sum()) if label_col in group else 0
        rows.append(
            {
                "priority_band": band,
                "alert_count": int(len(group)),
                "label_count": labels,
                "precision": labels / len(group) if len(group) else 0.0,
                "avg_score": float(group["priority_score_v2"].mean()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _top_k_after_band_fix(frame: pd.DataFrame) -> pd.DataFrame:
    label_col = "target" if "target" in frame.columns else "is_laundering"
    labels = frame[label_col].fillna(0).astype(int) if label_col in frame else pd.Series([0] * len(frame))
    scores = frame["priority_score_v2"].fillna(0.0)
    return pd.DataFrame(
        [
            {
                "split": "all",
                "k": k,
                "precision_at_k": precision_at_k(labels, scores, k),
                "recall_at_k": recall_at_k(labels, scores, k),
                "lift_at_k": lift_at_k(labels, scores, k),
            }
            for k in [100, 500, 1000, 5000]
        ]
    )


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
