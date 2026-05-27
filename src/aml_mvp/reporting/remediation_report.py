"""Standalone remediation HTML report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from aml_mvp.diagnostics.traffic_lights import build_traffic_light_findings, champion_decision
from aml_mvp.reporting.executive_summary import build_executive_summary
from aml_mvp.reporting.tables import dataframe_to_html_table, key_value_table


def build_remediation_report(config: dict[str, Any], root: Path, logger=None) -> Path:
    artifacts = dict(config.get("artifacts", {}))
    decision = champion_decision(config, root)
    findings = build_traffic_light_findings(config, root)
    summary = build_executive_summary(decision)
    context = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_table": key_value_table(summary),
        "decision_table": key_value_table(decision),
        "traffic_lights_table": dataframe_to_html_table(findings, max_rows=20),
        "rule_contribution_table": dataframe_to_html_table(_read_csv(root, artifacts.get("rule_contribution_path")), max_rows=20),
        "rule_recommendations_table": dataframe_to_html_table(_read_csv(root, artifacts.get("rule_volume_recommendations_path")), max_rows=20),
        "band_summary_table": dataframe_to_html_table(
            _read_csv(root, artifacts.get("priority_band_summary_path") or artifacts.get("band_summary_path")), max_rows=10
        ),
        "capacity_table": dataframe_to_html_table(
            _read_csv(root, artifacts.get("daily_capacity_simulation_path") or artifacts.get("capacity_simulation_path")), max_rows=20
        ),
        "graph_ablation_table": dataframe_to_html_table(_read_csv(root, artifacts.get("graph_ablation_results_path")), max_rows=10),
        "case_metrics_table": dataframe_to_html_table(_read_csv(root, artifacts.get("case_quality_metrics_path")), max_rows=15),
        "reason_codes_table": dataframe_to_html_table(_read_csv(root, artifacts.get("alert_reason_codes_path")), max_rows=20),
    }
    output = _resolve(root, artifacts.get("remediation_report_path", "outputs/reports/aml_extended_remediation_report.html"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render(context), encoding="utf-8")
    if logger:
        logger.info("Wrote remediation report path=%s", output)
    return output


def _render(context: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AML Extended Remediation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #14213d; }}
    h1, h2 {{ color: #0b1f33; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #d5dde8; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; }}
    .note {{ background: #f7f9fb; border-left: 4px solid #2f5d8c; padding: 12px; }}
  </style>
</head>
<body>
  <h1>AML Extended Remediation Report</h1>
  <p class="note">Generated at {context["generated_at"]}. This report is for alert prioritisation governance only and does not introduce alert suppression.</p>
  <h2>Executive Summary</h2>{context["summary_table"]}
  <h2>Champion-Challenger Decision</h2>{context["decision_table"]}
  <h2>Traffic-Light Findings</h2>{context["traffic_lights_table"]}
  <h2>Alert and Band Sizing</h2>{context["rule_contribution_table"]}<h3>Rule Volume Recommendations</h3>{context["rule_recommendations_table"]}{context["band_summary_table"]}{context["capacity_table"]}
  <h2>Graph Diagnostics</h2>{context["graph_ablation_table"]}
  <h2>Case Consolidation</h2>{context["case_metrics_table"]}
  <h2>Explainability</h2>{context["reason_codes_table"]}
</body>
</html>
"""


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
