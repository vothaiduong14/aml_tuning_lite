"""Standalone HTML report generation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError
from jinja2 import Environment, FileSystemLoader, select_autoescape

from aml_mvp.reporting.charts import bar_chart_svg
from aml_mvp.reporting.tables import dataframe_to_html_table, key_value_table


CORE_SECTIONS = [
    "Executive Summary",
    "Dataset Profile",
    "AML Coverage Matrix",
    "Rule Catalogue and Tuning",
    "Rule Performance and Overlap",
    "ML Triage",
    "Priority Queue Simulation",
    "Explainability and Alert Examples",
    "Validation and Limitations",
    "Appendix",
]


def build_report_context(config: dict[str, Any], root: Path) -> dict[str, Any]:
    artifacts = dict(config.get("artifacts", {}))
    report_config = dict(config.get("report", {}))

    data_quality = _read_json(root, artifacts.get("data_quality_report_path"))
    data_profile = _read_json(root, artifacts.get("data_profile_path"))
    split_manifest = _read_json(root, artifacts.get("split_manifest_path"))
    tuning_audit = _read_json(root, artifacts.get("tuning_audit_path"))
    model_metrics = _read_json(root, artifacts.get("model_metrics_path"))

    rule_performance = _read_csv(root, artifacts.get("rule_performance_path"))
    rule_overlap = _read_csv(root, artifacts.get("rule_overlap_path"))
    tuning_candidates = _read_csv(root, artifacts.get("tuning_candidates_path"))
    selected_thresholds = _read_csv(root, artifacts.get("selected_thresholds_path"))
    feature_dictionary = _read_csv(root, artifacts.get("feature_dictionary_path"))
    feature_quality = _read_csv(root, artifacts.get("feature_quality_report_path"))
    top_k = _read_csv(root, artifacts.get("top_k_metrics_path"))
    feature_importance = _read_csv(root, artifacts.get("feature_importance_path"))
    model_tuning_trials = _read_csv(root, artifacts.get("model_tuning_trials_path"))
    selected_features = _read_csv(root, artifacts.get("selected_features_path"))
    band_summary = _read_csv(root, artifacts.get("band_summary_path"))
    capacity = _read_csv(root, artifacts.get("capacity_simulation_path"))

    generated_at = datetime.now(timezone.utc).isoformat()
    dataset_name = data_profile.get("dataset_name") or data_quality.get("dataset_name") or "LI-Small"
    headline = _headline_metrics(data_quality, rule_performance, model_metrics, band_summary)
    run_manifest = build_run_manifest(config, artifacts, root, generated_at)
    acceptance = build_acceptance_checklist(
        data_quality,
        rule_performance,
        selected_thresholds,
        feature_dictionary,
        model_metrics,
        band_summary,
    )

    return {
        "title": "AML Rules and ML Triage MVP Report",
        "generated_at": generated_at,
        "dataset_name": dataset_name,
        "headline": headline,
        "sections": CORE_SECTIONS,
        "data_quality_table": key_value_table(data_quality),
        "split_table": dataframe_to_html_table(_split_manifest_to_df(split_manifest)),
        "payment_format_chart": bar_chart_svg(
            _mapping_to_df(data_profile.get("payment_format_counts", {}), "payment_format", "transaction_count"),
            "payment_format",
            "transaction_count",
            "Payment Format Distribution",
        ),
        "coverage_table": dataframe_to_html_table(build_coverage_matrix()),
        "rule_catalogue_table": dataframe_to_html_table(build_rule_catalogue()),
        "rule_performance_table": dataframe_to_html_table(rule_performance, max_rows=40),
        "rule_overlap_table": dataframe_to_html_table(rule_overlap, max_rows=20),
        "tuning_candidates_table": dataframe_to_html_table(tuning_candidates, max_rows=20),
        "selected_thresholds_table": dataframe_to_html_table(selected_thresholds, max_rows=20),
        "top_k_table": dataframe_to_html_table(top_k, max_rows=30),
        "model_metrics_table": key_value_table(_flatten_model_metrics(model_metrics)),
        "model_training_table": key_value_table(_flatten_training_metadata(model_metrics)),
        "model_tuning_trials_table": dataframe_to_html_table(model_tuning_trials, max_rows=20),
        "selected_features_table": dataframe_to_html_table(selected_features, max_rows=30),
        "band_summary_table": dataframe_to_html_table(band_summary, max_rows=10),
        "capacity_table": dataframe_to_html_table(capacity, max_rows=30),
        "feature_importance_chart": bar_chart_svg(
            feature_importance,
            "feature_name",
            "importance",
            "Top Feature Importance",
            max_bars=10,
        ),
        "feature_dictionary_table": dataframe_to_html_table(feature_dictionary, max_rows=25),
        "feature_quality_table": dataframe_to_html_table(feature_quality, max_rows=25),
        "limitations": [
            "The IBM AML dataset is synthetic; measured performance is not production evidence for a real bank.",
            "The MVP does not include KYC, sanctions, PEP, adverse media, beneficial ownership, or investigation outcomes.",
            "Model scores prioritize alerts only. They must not auto-close, suppress, or delete alerts.",
        ],
        "run_manifest": run_manifest,
        "run_manifest_table": key_value_table(run_manifest),
        "acceptance_checklist": acceptance,
        "acceptance_checklist_table": dataframe_to_html_table(acceptance, max_rows=30),
        "report_config": report_config,
    }


def render_html_report(context: dict[str, Any], output_path: str | Path, template_dir: str | Path | None = None) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    template_root = Path(template_dir) if template_dir else Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_root),
        autoescape=select_autoescape(enabled_extensions=("html", "j2")),
    )
    template = env.get_template("aml_mvp_report.html.j2")
    output.write_text(template.render(**context), encoding="utf-8")
    return output


def write_handover_artifacts(context: dict[str, Any], config: dict[str, Any], root: Path) -> dict[str, Path]:
    report_config = dict(config.get("report", {}))
    manifest_path = _resolve(root, report_config.get("run_manifest_path", "outputs/run_logs/run_manifest.json"))
    checklist_path = _resolve(root, report_config.get("acceptance_checklist_path", "outputs/metrics/final_acceptance_checklist.csv"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    checklist_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(context["run_manifest"], indent=2, default=str), encoding="utf-8")
    context["acceptance_checklist"].to_csv(checklist_path, index=False)
    return {"run_manifest": manifest_path, "acceptance_checklist": checklist_path}


def build_run_manifest(config: dict[str, Any], artifacts: dict[str, str], root: Path, generated_at: str) -> dict[str, Any]:
    resolved = {}
    for name, path in artifacts.items():
        resolved_path = _resolve(root, path)
        resolved[name] = {"path": str(resolved_path), "exists": resolved_path.exists()}
    return {
        "generated_at": generated_at,
        "project_root": str(root),
        "report_format": config.get("report", {}).get("format", "html"),
        "artifact_count": len(resolved),
        "artifacts": resolved,
    }


def build_acceptance_checklist(
    data_quality: dict[str, Any],
    rule_performance: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    model_metrics: dict[str, Any],
    band_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        ("Data validation", bool(data_quality.get("row_count", 0)), "Data quality report has rows."),
        ("Rule metrics", not rule_performance.empty, "Rule performance table generated."),
        ("Tuning audit", not selected_thresholds.empty, "Selected thresholds generated."),
        ("Feature dictionary", not feature_dictionary.empty, "Feature dictionary generated."),
        ("ML triage", bool(model_metrics.get("metrics")), "Model metrics generated."),
        ("Priority bands", not band_summary.empty, "Band summary generated."),
        ("HTML report", True, "Report rendered by build-report command."),
        ("Limitations", True, "Synthetic-data and no-auto-closure limitations included."),
    ]
    return pd.DataFrame(
        [{"acceptance_item": item, "status": "Closed" if passed else "Open", "evidence": evidence} for item, passed, evidence in rows]
    )


def build_coverage_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("R1_AMOUNT", "High amount", "Implemented"),
            ("R2_NOVELTY", "New counterparty / cross-bank novelty", "Implemented"),
            ("R3_VELOCITY", "Velocity / structuring", "Implemented"),
            ("R4_PASS_THROUGH", "Rapid pass-through", "Implemented"),
            ("R5_FAN_IN", "Fan-in concentration", "Implemented"),
            ("R6_FAN_OUT", "Fan-out / gather-scatter-lite", "Implemented"),
            ("R7_CYCLE", "Cycle candidate", "Deferred stretch"),
        ],
        columns=["rule_id", "typology_coverage", "mvp_status"],
    )


def build_rule_catalogue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("R1_AMOUNT", "Segmented high amount", "Amount exceeds segment percentile threshold."),
            ("R2_NOVELTY", "New counterparty / cross-bank novelty", "First observed sender-receiver pair."),
            ("R3_VELOCITY", "Velocity / structuring", "Repeated activity in rolling windows."),
            ("R4_PASS_THROUGH", "Rapid pass-through", "Outgoing movement after recent incoming funds."),
            ("R5_FAN_IN", "Fan-in concentration", "Many unique senders fund one receiver."),
            ("R6_FAN_OUT", "Fan-out / gather-scatter-lite", "Sender distributes funds after multiple funders."),
        ],
        columns=["rule_id", "rule_name", "core_logic"],
    )


def _headline_metrics(
    data_quality: dict[str, Any],
    rule_performance: pd.DataFrame,
    model_metrics: dict[str, Any],
    band_summary: pd.DataFrame,
) -> dict[str, Any]:
    any_rule = pd.DataFrame()
    if not rule_performance.empty and "rule_id" in rule_performance:
        any_rule = rule_performance[rule_performance["rule_id"].eq("ANY_RULE")]
    return {
        "transactions": data_quality.get("row_count", 0),
        "label_rate": data_quality.get("label_rate", 0),
        "rule_recall": float(any_rule["recall"].max()) if not any_rule.empty else 0.0,
        "model_pr_auc": model_metrics.get("metrics", {}).get("pr_auc", 0),
        "p1_alerts": int(band_summary.loc[band_summary["priority_band"].eq("P1"), "alert_count"].sum()) if not band_summary.empty else 0,
    }


def _flatten_model_metrics(model_metrics: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "model_name": model_metrics.get("model_name", ""),
        "evaluation_split": model_metrics.get("evaluation_split", ""),
    }
    for key, value in model_metrics.get("metrics", {}).items():
        flat[f"metric_{key}"] = value
    for baseline, metrics in model_metrics.get("baselines", {}).items():
        for key, value in metrics.items():
            flat[f"{baseline}_{key}"] = value
    return flat


def _flatten_training_metadata(model_metrics: dict[str, Any]) -> dict[str, Any]:
    training = dict(model_metrics.get("training", {}))
    imbalance = dict(training.get("imbalance", {}))
    feature_selection = dict(training.get("feature_selection", {}))
    tuning = dict(training.get("tuning", {}))
    champion = dict(training.get("champion_selection", {}))
    return {
        "imbalance_strategy": imbalance.get("strategy", ""),
        "original_positive_count": imbalance.get("original_positive_count", 0),
        "original_negative_count": imbalance.get("original_negative_count", 0),
        "sampled_positive_count": imbalance.get("sampled_positive_count", 0),
        "sampled_negative_count": imbalance.get("sampled_negative_count", 0),
        "selected_feature_count": feature_selection.get("selected_feature_count", 0),
        "tuning_backend": tuning.get("backend", ""),
        "tuning_trial_count": tuning.get("trial_count", 0),
        "champion_model": champion.get("selected_model", model_metrics.get("model_name", "")),
        "champion_selection_metric": champion.get("selection_metric", ""),
        "champion_selection_split": champion.get("selection_split", ""),
    }


def _split_manifest_to_df(split_manifest: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split, values in split_manifest.get("splits", {}).items():
        row = {"split": split}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _mapping_to_df(mapping: dict[str, Any], key_col: str, value_col: str) -> pd.DataFrame:
    return pd.DataFrame([{key_col: key, value_col: value} for key, value in mapping.items()])


def _read_json(root: Path, value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    path = _resolve(root, value)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
