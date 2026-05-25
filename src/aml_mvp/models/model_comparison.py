"""Champion/challenger comparison between MVP and extended models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def compare_model_runs(
    mvp_metrics: dict[str, Any],
    extended_metrics: dict[str, Any],
    mvp_top_k: pd.DataFrame,
    extended_top_k: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return metric comparison rows and a simple promotion decision."""

    rows = []
    rows.extend(_global_metric_rows(mvp_metrics, extended_metrics))
    rows.extend(_top_k_rows(mvp_top_k, extended_top_k))
    comparison = pd.DataFrame(rows)
    selection = select_champion(comparison, config)
    return comparison, selection


def select_champion(comparison: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    comparison_config = dict(config.get("model_comparison", {}))
    primary_metric = str(comparison_config.get("primary_metric", "precision_at_k"))
    primary_k = int(comparison_config.get("primary_k", 1000))
    min_delta = float(comparison_config.get("min_delta", 0.0))

    primary = comparison[
        comparison["metric"].eq(primary_metric)
        & comparison["k"].fillna(primary_k).astype(int).eq(primary_k)
    ]
    if primary.empty:
        primary = comparison[comparison["metric"].eq("pr_auc")]
    if primary.empty:
        return {
            "selected_model": "mvp",
            "decision": "keep_mvp",
            "reason": "No comparable primary metric was available.",
            "primary_metric": primary_metric,
            "primary_k": primary_k,
            "delta": 0.0,
        }

    row = primary.iloc[0]
    delta = float(row["delta"])
    selected = "extended" if delta > min_delta else "mvp"
    metric_label = f"{row['metric']}@{int(row['k'])}" if pd.notna(row["k"]) else str(row["metric"])
    return {
        "selected_model": selected,
        "decision": "promote_extended" if selected == "extended" else "keep_mvp",
        "reason": (
            f"Extended beat MVP on {metric_label} by {delta:.6g}."
            if selected == "extended"
            else f"MVP retained because extended delta {delta:.6g} did not exceed minimum {min_delta:.6g}."
        ),
        "primary_metric": str(row["metric"]),
        "primary_k": int(row["k"]) if pd.notna(row["k"]) else None,
        "mvp_value": float(row["mvp_value"]),
        "extended_value": float(row["extended_value"]),
        "delta": delta,
        "min_delta": min_delta,
    }


def write_model_comparison_outputs(
    comparison: pd.DataFrame,
    selection: dict[str, Any],
    artifacts: dict[str, str],
    root: Path,
) -> dict[str, Path]:
    comparison_path = _resolve(root, artifacts["model_comparison_path"])
    selection_path = _resolve(root, artifacts["model_selection_path"])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    selection_path.write_text(json.dumps(selection, indent=2, default=str), encoding="utf-8")
    return {"model_comparison": comparison_path, "model_selection": selection_path}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _global_metric_rows(mvp_metrics: dict[str, Any], extended_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric in sorted(set(mvp_metrics.get("metrics", {})) | set(extended_metrics.get("metrics", {}))):
        mvp_value = float(mvp_metrics.get("metrics", {}).get(metric, 0.0))
        extended_value = float(extended_metrics.get("metrics", {}).get(metric, 0.0))
        rows.append(_comparison_row(metric, None, mvp_value, extended_value))
    return rows


def _top_k_rows(mvp_top_k: pd.DataFrame, extended_top_k: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if mvp_top_k.empty or extended_top_k.empty:
        return rows
    left = mvp_top_k[mvp_top_k["ranking"].eq("model_score")].copy()
    right = extended_top_k[extended_top_k["ranking"].eq("model_score")].copy()
    merged = left.merge(right, on=["split", "ranking", "k"], suffixes=("_mvp", "_extended"))
    for row in merged.itertuples(index=False):
        for metric in ["precision_at_k", "recall_at_k", "lift_at_k"]:
            rows.append(
                _comparison_row(
                    metric,
                    int(row.k),
                    float(getattr(row, f"{metric}_mvp")),
                    float(getattr(row, f"{metric}_extended")),
                    split=str(row.split),
                )
            )
    return rows


def _comparison_row(
    metric: str,
    k: int | None,
    mvp_value: float,
    extended_value: float,
    split: str | None = None,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "k": k,
        "split": split,
        "mvp_value": mvp_value,
        "extended_value": extended_value,
        "delta": extended_value - mvp_value,
        "winner": "extended" if extended_value > mvp_value else "mvp" if mvp_value > extended_value else "tie",
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
