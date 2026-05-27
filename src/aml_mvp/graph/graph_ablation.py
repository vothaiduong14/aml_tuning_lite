"""Graph feature ablation diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aml_mvp.models.evaluate import pr_auc, precision_at_k, recall_at_k
from aml_mvp.storage import read_dataframe


VARIANTS = [
    ("A", "mvp_only", []),
    ("B", "degree", ["degree"]),
    ("C", "component", ["degree", "component"]),
    ("D", "pagerank", ["degree", "component", "pagerank"]),
    ("E", "all_graph", ["degree", "component", "pagerank", "cycle"]),
]


def run_graph_ablation(config: dict[str, Any], root: Path, logger=None) -> pd.DataFrame:
    artifacts = dict(config.get("artifacts", {}))
    features = read_dataframe(_resolve(root, artifacts["alert_features_path"]))
    graph = read_dataframe(_resolve(root, artifacts["graph_features_v2_path"]))
    cycle_candidates = _read_optional_csv(root, artifacts.get("cycle_candidates_v2_path"))
    merged = _merge(features, graph)
    label_col = "target" if "target" in merged.columns else "is_laundering"
    base_score = merged["model_score"] if "model_score" in merged else merged.get("feature_rule_priority_score", pd.Series(0.0, index=merged.index))
    rows = []
    for variant, group, groups in VARIANTS:
        score = base_score.astype(float).copy()
        for column in _columns_for_groups(merged, groups):
            values = merged[column].fillna(0.0).astype(float)
            max_value = values.abs().max()
            if max_value:
                score = score + 0.001 * (values / max_value)
        test = merged[merged.get("split", "test").eq("test")] if "split" in merged else merged
        test_score = score.loc[test.index]
        labels = test[label_col].fillna(0).astype(int) if label_col in test else pd.Series([0] * len(test))
        rows.append(
            {
                "model_variant": variant,
                "feature_group_added": group,
                "pr_auc": pr_auc(labels, test_score),
                "precision_at_100": precision_at_k(labels, test_score, 100),
                "precision_at_500": precision_at_k(labels, test_score, 500),
                "precision_at_1000": precision_at_k(labels, test_score, 1000),
                "recall_at_1000": recall_at_k(labels, test_score, 1000),
                "cycle_candidate_count": int(_cycle_metric(cycle_candidates, "cycle_candidate_count")),
                "cycle_label_count": int(_cycle_metric(cycle_candidates, "cycle_candidate_label_count")),
            }
        )
    results = pd.DataFrame(rows)
    baseline = float(results.loc[results["model_variant"].eq("A"), "precision_at_1000"].iloc[0]) if not results.empty else 0.0
    results["precision_at_1000_delta_vs_mvp"] = results["precision_at_1000"] - baseline
    results["decision"] = results.apply(_decision, axis=1)
    output = _resolve(root, artifacts["graph_ablation_results_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    if logger:
        logger.info("Wrote graph ablation path=%s rows=%s", output, len(results))
    return results


def _merge(features: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    graph_cols = [column for column in graph.columns if column.startswith("graph_")]
    merged = features.merge(graph[["transaction_id"] + graph_cols], on="transaction_id", how="left")
    for column in graph_cols:
        merged[f"feature_{column}"] = merged[column].fillna(0.0)
        merged = merged.drop(columns=[column])
    return merged


def _columns_for_groups(frame: pd.DataFrame, groups: list[str]) -> list[str]:
    columns = []
    for column in frame.columns:
        lowered = column.lower()
        if not column.startswith("feature_graph"):
            continue
        if "degree" in groups and "degree" in lowered:
            columns.append(column)
        elif "component" in groups and "component" in lowered:
            columns.append(column)
        elif "pagerank" in groups and "pagerank" in lowered:
            columns.append(column)
        elif "cycle" in groups and "cycle" in lowered:
            columns.append(column)
    return columns


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _decision(row: pd.Series) -> str:
    if "cycle" in str(row.get("feature_group_added", "")) and int(row.get("cycle_label_count", 0)) == 0:
        return "research_only"
    return "keep" if float(row.get("precision_at_1000_delta_vs_mvp", 0.0)) >= 0 else "remove"


def _cycle_metric(cycle_candidates: pd.DataFrame, metric: str) -> float:
    if cycle_candidates.empty:
        return 0.0
    if {"metric", "value"}.issubset(cycle_candidates.columns):
        row = cycle_candidates[cycle_candidates["metric"].eq(metric)]
        return float(row["value"].iloc[0]) if not row.empty else 0.0
    if metric == "cycle_candidate_count":
        return float(len(cycle_candidates))
    if metric == "cycle_candidate_label_count" and "is_laundering" in cycle_candidates:
        return float(cycle_candidates["is_laundering"].sum())
    return 0.0


def _read_optional_csv(root: Path, value: str | None) -> pd.DataFrame:
    if not value:
        return pd.DataFrame()
    path = _resolve(root, value)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
