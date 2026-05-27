"""Extended HTML report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError
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
    temporal_stress = _read_csv(root, artifacts.get("temporal_stress_summary_path"))
    segment_stress = _read_csv(root, artifacts.get("segment_stress_summary_path"))
    graph_dictionary = _read_csv(root, artifacts.get("graph_feature_dictionary_path"))
    cycle_summary = _read_csv(root, artifacts.get("cycle_summary_path"))
    case_metrics = _read_csv(root, artifacts.get("case_metrics_path"))
    reason_codes = _read_csv(root, artifacts.get("reason_codes_path"))
    shap_importance = _read_csv(root, artifacts.get("shap_feature_importance_path"))
    model_comparison = _read_csv(root, artifacts.get("model_comparison_path"))
    model_selection = _read_json(root, artifacts.get("model_selection_path"))
    mvp_model_metrics = _read_json(root, artifacts.get("model_metrics_path"))
    extended_model_metrics = _read_json(root, artifacts.get("extended_model_metrics_path"))
    mvp_selected_features = _read_csv(root, artifacts.get("selected_features_path"))
    model_tuning_trials = _read_csv(root, artifacts.get("extended_model_tuning_trials_path"))
    selected_features = _read_csv(root, artifacts.get("extended_selected_features_path"))
    priority_metrics = _read_csv(root, artifacts.get("priority_band_metrics_path"))
    feature_summary = build_model_feature_summary(
        mvp_model_metrics,
        extended_model_metrics,
        mvp_selected_features,
        selected_features,
    )
    settings_summary = build_model_settings_summary(mvp_model_metrics, extended_model_metrics)
    candidate_summary = build_candidate_summary(mvp_model_metrics, extended_model_metrics)

    return {
        "title": "AML Extended Build Report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": config.get("project", {}).get("dataset_name", "LI-Small"),
        "sections": EXTENDED_SECTIONS,
        "headline_table": key_value_table(_headline(stress, case_metrics, priority_metrics)),
        "stress_table": dataframe_to_html_table(stress, max_rows=10),
        "temporal_stress_table": dataframe_to_html_table(temporal_stress, max_rows=20),
        "segment_stress_table": dataframe_to_html_table(segment_stress, max_rows=30),
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
        "model_feature_summary_table": dataframe_to_html_table(feature_summary, max_rows=10),
        "model_settings_table": dataframe_to_html_table(settings_summary, max_rows=60),
        "model_candidate_table": dataframe_to_html_table(candidate_summary, max_rows=20),
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


def build_model_feature_summary(
    mvp_metrics: dict[str, Any],
    extended_metrics: dict[str, Any],
    mvp_selected_features: pd.DataFrame,
    extended_selected_features: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _feature_summary_row("mvp", mvp_metrics, mvp_selected_features),
            _feature_summary_row("extended", extended_metrics, extended_selected_features),
        ]
    )


def build_model_settings_summary(mvp_metrics: dict[str, Any], extended_metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in [("mvp", mvp_metrics), ("extended", extended_metrics)]:
        training = dict(metrics.get("training", {}))
        imbalance = dict(training.get("imbalance", {}))
        selection = dict(training.get("feature_selection", {}))
        tuning = dict(training.get("tuning", {}))
        champion = dict(training.get("champion_selection", {}))
        rows.extend(
            [
                _setting_row(model_name, "model_candidates", ", ".join(training.get("model_candidates", []) or [])),
                _setting_row(model_name, "champion_model", metrics.get("model_name") or champion.get("selected_model")),
                _setting_row(model_name, "evaluation_split", metrics.get("evaluation_split")),
                _setting_row(model_name, "imbalance_strategy", imbalance.get("strategy")),
                _setting_row(model_name, "negative_to_positive_ratio", imbalance.get("negative_to_positive_ratio")),
                _setting_row(model_name, "sampled_positive_count", imbalance.get("sampled_positive_count")),
                _setting_row(model_name, "sampled_negative_count", imbalance.get("sampled_negative_count")),
                _setting_row(model_name, "feature_selection_enabled", selection.get("enabled")),
                _setting_row(model_name, "feature_selection_method", selection.get("method")),
                _setting_row(model_name, "forced_feature_prefixes", ", ".join(selection.get("force_include_feature_prefixes", []) or [])),
                _setting_row(model_name, "tuning_enabled", tuning.get("enabled")),
                _setting_row(model_name, "tuning_backend", tuning.get("backend")),
                _setting_row(model_name, "tuning_trial_count", tuning.get("trial_count")),
                _setting_row(model_name, "tuning_objective", _objective_label(tuning)),
                _setting_row(model_name, "selection_metric", _selection_label(champion)),
            ]
        )
        for candidate_name, candidate_metadata in dict(tuning.get("models", {})).items():
            rows.append(_setting_row(model_name, f"{candidate_name}.tuning_enabled", candidate_metadata.get("enabled")))
            for param, value in dict(candidate_metadata.get("best_params", {})).items():
                rows.append(_setting_row(model_name, f"{candidate_name}.best_param.{param}", value))
        for param, value in dict(tuning.get("best_params", {})).items():
            rows.append(_setting_row(model_name, f"best_param.{param}", value))
    return pd.DataFrame(rows)


def build_candidate_summary(mvp_metrics: dict[str, Any], extended_metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in [("mvp", mvp_metrics), ("extended", extended_metrics)]:
        candidates = metrics.get("training", {}).get("champion_selection", {}).get("candidates", [])
        for candidate in candidates:
            row = {"model_run": model_name}
            row.update(candidate)
            rows.append(row)
    return pd.DataFrame(rows)


def _feature_summary_row(model_name: str, metrics: dict[str, Any], selected_features_table: pd.DataFrame) -> dict[str, Any]:
    selection = dict(metrics.get("training", {}).get("feature_selection", {}))
    selected_features = list(selection.get("selected_features", []) or [])
    selected_count = len(selected_features)
    selected_table_count = 0
    total_candidate_count = 0
    forced_count = 0
    if not selected_features_table.empty:
        total_candidate_count = int(len(selected_features_table))
        if "selected" in selected_features_table:
            selected_table_count = int(selected_features_table["selected"].astype(str).str.lower().isin(["true", "1"]).sum())
        if "forced_include" in selected_features_table:
            forced_count = int(selected_features_table["forced_include"].astype(str).str.lower().isin(["true", "1"]).sum())
    graph_selected_count = sum(1 for feature in selected_features if str(feature).startswith("feature_graph_"))
    v2_graph_selected_count = sum(1 for feature in selected_features if str(feature).startswith("feature_graph_") and str(feature).endswith("_v2"))
    return {
        "model_run": model_name,
        "champion_model": metrics.get("model_name"),
        "selected_feature_count": selected_count or selected_table_count,
        "candidate_feature_count": total_candidate_count or None,
        "graph_feature_count": graph_selected_count,
        "graph_v2_feature_count": v2_graph_selected_count,
        "forced_feature_count": forced_count,
    }


def _setting_row(model_name: str, setting: str, value: Any) -> dict[str, Any]:
    return {"model_run": model_name, "setting": setting, "value": value}


def _objective_label(tuning: dict[str, Any]) -> str:
    metric = tuning.get("objective_metric")
    k = tuning.get("objective_k")
    split = tuning.get("objective_split")
    return f"{metric}@{k} on {split}" if metric and k else str(metric or "")


def _selection_label(champion: dict[str, Any]) -> str:
    metric = champion.get("selection_metric")
    k = champion.get("selection_k")
    split = champion.get("selection_split")
    return f"{metric}@{k} on {split}" if metric and k else str(metric or "")


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
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


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
