from __future__ import annotations

import json

import pandas as pd

from aml_mvp.reporting.html_report import (
    build_report_context,
    render_html_report,
    write_handover_artifacts,
)


def test_html_report_renders_core_sections_without_placeholders(tmp_path) -> None:
    config = _write_report_artifacts(tmp_path)

    context = build_report_context(config, tmp_path)
    output_path = render_html_report(context, tmp_path / "outputs" / "reports" / "report.html")
    handover = write_handover_artifacts(context, config, tmp_path)

    html = output_path.read_text(encoding="utf-8")
    assert "Executive Summary" in html
    assert "Dataset Profile" in html
    assert "Rule Performance and Overlap" in html
    assert "ML Triage" in html
    assert "Priority Queue Simulation" in html
    assert "Validation and Limitations" in html
    assert "Synthetic" in html or "synthetic" in html
    assert "{{" not in html
    assert "{%" not in html
    assert handover["run_manifest"].exists()
    assert handover["acceptance_checklist"].exists()


def test_report_context_reconciles_headline_metrics(tmp_path) -> None:
    config = _write_report_artifacts(tmp_path)
    context = build_report_context(config, tmp_path)

    assert context["headline"]["transactions"] == 10
    assert context["headline"]["rule_recall"] == 0.75
    assert context["headline"]["model_pr_auc"] == 0.25
    assert context["headline"]["p1_alerts"] == 2


def _write_report_artifacts(tmp_path):
    metrics = tmp_path / "outputs" / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "data_quality_report.json").write_text(
        json.dumps({"row_count": 10, "label_rate": 0.2, "dataset_name": "LI-Small"}),
        encoding="utf-8",
    )
    (metrics / "data_profile.json").write_text(
        json.dumps({"payment_format_counts": {"ACH": 7, "Wire": 3}, "dataset_name": "LI-Small"}),
        encoding="utf-8",
    )
    (metrics / "split_manifest.json").write_text(
        json.dumps({"splits": {"train": {"row_count": 6}, "validation": {"row_count": 2}, "test": {"row_count": 2}}}),
        encoding="utf-8",
    )
    (metrics / "tuning_audit_log.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (metrics / "model_metrics.json").write_text(
        json.dumps({"model_name": "gradient_boosting", "evaluation_split": "test", "metrics": {"pr_auc": 0.25, "roc_auc": 0.5}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"split": "test", "rule_id": "ANY_RULE", "recall": 0.75, "precision": 0.1, "alert_rate": 0.2}]
    ).to_csv(metrics / "rule_performance.csv", index=False)
    pd.DataFrame([{"rule_id": "R1_AMOUNT", "R1_AMOUNT": 2}]).to_csv(metrics / "rule_overlap_matrix.csv", index=False)
    pd.DataFrame([{"rule_id": "R1_AMOUNT", "candidate_value": 0.95, "recall": 0.75}]).to_csv(
        metrics / "tuning_candidates.csv",
        index=False,
    )
    pd.DataFrame([{"rule_id": "R1_AMOUNT", "selected_percentile": 0.95}]).to_csv(
        metrics / "selected_thresholds.csv",
        index=False,
    )
    pd.DataFrame([{"feature_name": "feature_amount", "feature_group": "transaction"}]).to_csv(
        metrics / "feature_dictionary.csv",
        index=False,
    )
    pd.DataFrame([{"feature_name": "feature_amount", "null_count": 0}]).to_csv(
        metrics / "feature_quality_report.csv",
        index=False,
    )
    pd.DataFrame([{"ranking": "model_score", "k": 1, "precision_at_k": 1.0}]).to_csv(
        metrics / "top_k_metrics.csv",
        index=False,
    )
    pd.DataFrame([{"feature_name": "feature_amount", "importance": 0.9}]).to_csv(
        metrics / "feature_importance.csv",
        index=False,
    )
    pd.DataFrame([{"priority_band": "P1", "alert_count": 2}, {"priority_band": "P2", "alert_count": 3}]).to_csv(
        metrics / "band_summary.csv",
        index=False,
    )
    pd.DataFrame([{"ranking": "model_score", "k": 1, "precision_at_k": 1.0}]).to_csv(
        metrics / "capacity_simulation.csv",
        index=False,
    )
    return {
        "report": {
            "format": "html",
            "output_file": "outputs/reports/aml_mvp_report.html",
            "run_manifest_path": "outputs/run_logs/run_manifest.json",
            "acceptance_checklist_path": "outputs/metrics/final_acceptance_checklist.csv",
        },
        "artifacts": {
            "data_quality_report_path": "outputs/metrics/data_quality_report.json",
            "data_profile_path": "outputs/metrics/data_profile.json",
            "split_manifest_path": "outputs/metrics/split_manifest.json",
            "rule_performance_path": "outputs/metrics/rule_performance.csv",
            "rule_overlap_path": "outputs/metrics/rule_overlap_matrix.csv",
            "tuning_candidates_path": "outputs/metrics/tuning_candidates.csv",
            "selected_thresholds_path": "outputs/metrics/selected_thresholds.csv",
            "tuning_audit_path": "outputs/metrics/tuning_audit_log.json",
            "feature_dictionary_path": "outputs/metrics/feature_dictionary.csv",
            "feature_quality_report_path": "outputs/metrics/feature_quality_report.csv",
            "model_metrics_path": "outputs/metrics/model_metrics.json",
            "top_k_metrics_path": "outputs/metrics/top_k_metrics.csv",
            "feature_importance_path": "outputs/metrics/feature_importance.csv",
            "band_summary_path": "outputs/metrics/band_summary.csv",
            "capacity_simulation_path": "outputs/metrics/capacity_simulation.csv",
        },
    }

