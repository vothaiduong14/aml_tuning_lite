"""Priority band assignment for scored alerts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SEVERITY_PRIORITY = {"critical": "P1", "high": "P2", "medium": "P3", "low": "P4"}


def assign_priority_bands(scored_alerts: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    triage_config = dict(config.get("triage", {}))
    p1_quantile = float(triage_config.get("p1_quantile", 0.99))
    p2_quantile = float(triage_config.get("p2_quantile", 0.95))
    p3_quantile = float(triage_config.get("p3_quantile", 0.80))
    critical_rules = set(triage_config.get("critical_rules", ["R4_PASS_THROUGH", "R5_FAN_IN", "R6_FAN_OUT"]))

    output = scored_alerts.copy()
    scores = output["model_score"].fillna(0.0)
    p1_threshold = float(scores.quantile(p1_quantile)) if len(scores) else 1.0
    p2_threshold = float(scores.quantile(p2_quantile)) if len(scores) else 1.0
    p3_threshold = float(scores.quantile(p3_quantile)) if len(scores) else 1.0

    output["priority_band"] = "P4"
    output.loc[scores >= p3_threshold, "priority_band"] = "P3"
    output.loc[scores >= p2_threshold, "priority_band"] = "P2"
    output.loc[scores >= p1_threshold, "priority_band"] = "P1"

    critical_mask = output["triggered_rules"].fillna("").map(
        lambda value: bool(set(str(value).split(",")) & critical_rules)
    )
    output.loc[critical_mask, "priority_band"] = "P1"
    output["priority_rank"] = output["priority_band"].map({"P1": 1, "P2": 2, "P3": 3, "P4": 4}).astype(int)
    output = output.sort_values(["priority_rank", "model_score", "alert_timestamp"], ascending=[True, False, True])

    band_summary = summarize_bands(output)
    return output.reset_index(drop=True), band_summary


def summarize_bands(priority_alerts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_positives = int(priority_alerts["target"].sum()) if "target" in priority_alerts else 0
    for band in ["P1", "P2", "P3", "P4"]:
        band_df = priority_alerts[priority_alerts["priority_band"].eq(band)]
        positives = int(band_df["target"].sum()) if "target" in band_df else 0
        rows.append(
            {
                "priority_band": band,
                "alert_count": int(len(band_df)),
                "label_count": positives,
                "precision": float(positives / len(band_df)) if len(band_df) else 0.0,
                "recall": float(positives / total_positives) if total_positives else 0.0,
                "avg_model_score": float(band_df["model_score"].mean()) if len(band_df) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_priority_outputs(
    priority_alerts: pd.DataFrame,
    band_summary: pd.DataFrame,
    capacity_simulation: pd.DataFrame,
    artifacts: dict[str, str],
    root: Path,
    write_dataframe,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["priority_alerts"] = write_dataframe(priority_alerts, _resolve(root, artifacts["priority_alerts_path"]))
    band_path = _resolve(root, artifacts["band_summary_path"])
    capacity_path = _resolve(root, artifacts["capacity_simulation_path"])
    band_path.parent.mkdir(parents=True, exist_ok=True)
    capacity_path.parent.mkdir(parents=True, exist_ok=True)
    band_summary.to_csv(band_path, index=False)
    capacity_simulation.to_csv(capacity_path, index=False)
    outputs["band_summary"] = band_path
    outputs["capacity_simulation"] = capacity_path
    return outputs


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()

