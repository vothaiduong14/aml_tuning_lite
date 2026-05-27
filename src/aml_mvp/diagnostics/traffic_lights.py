"""Traffic-light findings for remediation reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


def build_traffic_light_findings(config: dict[str, Any], root: Path) -> pd.DataFrame:
    artifacts = dict(config.get("artifacts", {}))
    priority = _read_csv(root, artifacts.get("priority_band_summary_path") or artifacts.get("band_summary_path"))
    stress = _read_csv(root, artifacts.get("stress_test_summary_path"))
    temporal_stress = _read_csv(root, artifacts.get("temporal_stress_summary_path"))
    segment_stress = _read_csv(root, artifacts.get("segment_stress_summary_path"))
    comparison = _read_csv(root, artifacts.get("model_comparison_path"))
    ablation = _read_csv(root, artifacts.get("graph_ablation_results_path"))
    cases = _read_csv(root, artifacts.get("case_quality_metrics_path"))
    reasons = _read_csv(root, artifacts.get("alert_reason_codes_path"))
    rows = [
        _finding("Alert rate", _alert_rate_status(stress), "Overall alert rate against remediation target."),
        _finding("P1 sizing", _p1_status(priority), "Operational priority queue size after v2 banding."),
        _finding("Top-K uplift", _uplift_status(comparison), "Champion/challenger Precision@1000 delta."),
        _finding("Graph features", _graph_status(ablation), "Graph ablation decision."),
        _finding("Case consolidation", _case_status(cases), "Operational case size cap check."),
        _finding("Explainability", "green" if not reasons.empty else "red", "Business-readable alert reason codes."),
        _finding("Stress testing", _stress_status(temporal_stress, segment_stress), "LI temporal and segment stress outputs."),
        _finding("Report automation", "green", "Remediation report generated from local artifacts."),
    ]
    findings = pd.DataFrame(rows)
    output = _resolve(root, artifacts.get("traffic_light_findings_path", "outputs/metrics/traffic_light_findings.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    findings.to_csv(output, index=False)
    return findings


def champion_decision(config: dict[str, Any], root: Path) -> dict[str, Any]:
    artifacts = dict(config.get("artifacts", {}))
    comparison = _read_csv(root, artifacts.get("model_comparison_path"))
    findings = build_traffic_light_findings(config, root)
    precision = comparison[(comparison["metric"].eq("precision_at_k")) & (comparison["k"].fillna(0).astype(int).eq(1000))] if not comparison.empty else pd.DataFrame()
    delta = float(precision["delta"].iloc[0]) if not precision.empty else 0.0
    mvp_value = float(precision["mvp_value"].iloc[0]) if not precision.empty else 0.0
    uplift = delta / mvp_value if mvp_value else 0.0
    failed = findings.loc[findings["status"].eq("red"), "area"].tolist()
    promotion_ready = uplift >= 0.05 and not failed
    decision = {
        "champion_model": "extended" if promotion_ready else "mvp",
        "challenger_model": "extended",
        "decision": "promote_extended" if promotion_ready else "keep_mvp",
        "promotion_ready": promotion_ready,
        "primary_reason": "Extended model passed remediation gates." if promotion_ready else "MVP retained until remediation gates pass.",
        "failed_gates": failed,
        "passed_gates": findings.loc[~findings["status"].eq("red"), "area"].tolist(),
        "supporting_metrics": {"precision_at_1000_delta": delta, "precision_at_1000_relative_uplift": uplift},
    }
    output = _resolve(root, artifacts.get("champion_decision_path", "outputs/metrics/champion_challenger_decision.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    import json

    output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def _finding(area: str, status: str, evidence: str) -> dict[str, Any]:
    return {"area": area, "status": status, "evidence": evidence}


def _p1_status(priority: pd.DataFrame) -> str:
    if priority.empty:
        return "red"
    row = priority[priority["priority_band"].eq("P1")]
    count = int(row["alert_count"].iloc[0]) if not row.empty else 0
    if count <= 120:
        return "green"
    if count <= 240:
        return "amber"
    return "red"


def _alert_rate_status(stress: pd.DataFrame) -> str:
    if stress.empty or "alert_rate" not in stress:
        return "amber"
    alert_rate = float(stress["alert_rate"].iloc[0])
    if alert_rate <= 0.15:
        return "green"
    if alert_rate <= 0.30:
        return "amber"
    return "red"


def _uplift_status(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "red"
    row = comparison[(comparison["metric"].eq("precision_at_k")) & (comparison["k"].fillna(0).astype(int).eq(1000))]
    if row.empty:
        return "red"
    delta = float(row["delta"].iloc[0])
    mvp = float(row["mvp_value"].iloc[0])
    rel = delta / mvp if mvp else 0.0
    if rel >= 0.05:
        return "green"
    if rel >= 0.0:
        return "amber"
    return "red"


def _graph_status(ablation: pd.DataFrame) -> str:
    if ablation.empty:
        return "amber"
    decisions = ablation.get("decision", pd.Series(dtype=str)).astype(str)
    if decisions.eq("remove").any():
        return "red"
    if decisions.eq("research_only").any():
        return "amber"
    return "green"


def _case_status(cases: pd.DataFrame) -> str:
    if cases.empty:
        return "red"
    row = cases[cases["metric"].eq("max_alerts_per_case")]
    value = float(row["value"].iloc[0]) if not row.empty else 0.0
    if value <= 100:
        return "green"
    if value <= 200:
        return "amber"
    return "red"


def _stress_status(temporal: pd.DataFrame, segment: pd.DataFrame) -> str:
    if temporal.empty or segment.empty:
        return "amber"
    severe_small = "status" in segment and segment["status"].astype(str).eq("small_segment").mean() > 0.75
    return "amber" if severe_small else "green"


def _read_csv(root: Path, value: str | None) -> pd.DataFrame:
    if not value:
        return pd.DataFrame()
    path = _resolve(root, value)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
