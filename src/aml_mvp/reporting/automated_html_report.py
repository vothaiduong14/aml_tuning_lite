"""Extended HTML report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from aml_mvp.reporting.charts import bar_chart_svg
from aml_mvp.reporting.tables import dataframe_to_html_table, key_value_table


EXTENDED_SECTIONS = [
    "Extended Summary",
    "LI-Only Stress Test",
    "Graph Feature Diagnostics",
    "Graph Rule Results",
    "Case Consolidation",
    "LightGBM and SHAP Explainability",
    "Model Comparison",
    "Calibration and Priority Bands",
    "Extended Limitations",
]


def build_extended_report_context(config: dict[str, Any], root: Path) -> dict[str, Any]:
    artifacts = dict(config.get("artifacts", {}))
    stress = _read_csv(root, artifacts.get("stress_test_summary_path"))
    graph_dictionary = _read_csv(root, artifacts.get("graph_feature_dictionary_path"))
    cycle_summary = _read_csv(root, artifacts.get("cycle_summary_path"))
    case_metrics = _read_csv(root, artifacts.get("case_metrics_path"))
    reason_codes = _read_csv(root, artifacts.get("reason_codes_path"))
    shap_importance = _read_csv(root, artifacts.get("shap_feature_importance_path"))
    model_comparison = _read_csv(root, artifacts.get("model_comparison_path"))
    model_selection = _read_json(root, artifacts.get("model_selection_path"))
    model_tuning_trials = _read_csv(root, artifacts.get("extended_model_tuning_trials_path"))
    selected_features = _read_csv(root, artifacts.get("extended_selected_features_path"))
    priority_metrics = _read_csv(root, artifacts.get("priority_band_metrics_path"))

    return {
        "title": "AML Extended Build Report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": config.get("project", {}).get("dataset_name", "LI-Small"),
        "sections": EXTENDED_SECTIONS,
        "headline_table": key_value_table(_headline(stress, case_metrics, priority_metrics)),
        "stress_table": dataframe_to_html_table(stress, max_rows=10),
        "graph_feature_table": dataframe_to_html_table(graph_dictionary, max_rows=25),
        "cycle_summary_table": dataframe_to_html_table(cycle_summary, max_rows=10),
        "case_metrics_table": dataframe_to_html_table(case_metrics, max_rows=10),
        "reason_codes_table": dataframe_to_html_table(reason_codes, max_rows=20),
        "shap_importance_chart": bar_chart_svg(
            shap_importance.rename(columns={"mean_abs_shap": "importance"}),
            "feature_name",
            "importance",
            "Top SHAP Feature Importance",
            max_bars=10,
        ),
        "model_selection_table": key_value_table(model_selection),
        "model_comparison_table": dataframe_to_html_table(model_comparison, max_rows=20),
        "model_tuning_trials_table": dataframe_to_html_table(model_tuning_trials, max_rows=20),
        "selected_features_table": dataframe_to_html_table(selected_features, max_rows=30),
        "priority_metrics_table": dataframe_to_html_table(priority_metrics, max_rows=10),
        "limitations": [
            "Extended stress testing is LI-only until HI-Small is added to data/raw.",
            "Graph features are point-in-time approximations built from local transaction history.",
            "SHAP reason codes explain alert ranking behavior; they are not investigator decisions.",
            "Calibrated priority scores still prioritize alerts only and must not auto-close or suppress alerts.",
        ],
    }


def render_extended_report(context: dict[str, Any], output_path: str | Path, template_dir: str | Path | None = None) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    template_root = Path(template_dir) if template_dir else Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_root),
        autoescape=select_autoescape(enabled_extensions=("html", "j2")),
    )
    template = env.get_template("aml_extended_report.html.j2")
    output.write_text(template.render(**context), encoding="utf-8")
    return output


def _headline(stress: pd.DataFrame, case_metrics: pd.DataFrame, priority_metrics: pd.DataFrame) -> dict[str, Any]:
    values = {
        "comparison_scope": "LI-only",
        "hi_comparison_status": "deferred_missing_dataset",
        "case_count": 0,
        "p1_alerts": 0,
    }
    if not stress.empty:
        first = stress.iloc[0].to_dict()
        values["transaction_count"] = first.get("transaction_count", 0)
        values["alert_count"] = first.get("alert_count", 0)
        values["alert_rate"] = first.get("alert_rate", 0)
        values["hi_comparison_status"] = first.get("hi_comparison_status", values["hi_comparison_status"])
    if not case_metrics.empty:
        case_count = case_metrics[case_metrics["metric"].eq("case_count")]
        values["case_count"] = int(case_count["value"].iloc[0]) if not case_count.empty else 0
    if not priority_metrics.empty:
        p1 = priority_metrics[priority_metrics["priority_band"].eq("P1")]
        values["p1_alerts"] = int(p1["alert_count"].iloc[0]) if not p1.empty else 0
    return values


def _read_csv(root: Path, value: str | None) -> pd.DataFrame:
    if not value:
        return pd.DataFrame()
    path = _resolve(root, value)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(root: Path, value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    path = _resolve(root, value)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
