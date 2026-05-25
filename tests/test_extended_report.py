from __future__ import annotations

import pandas as pd

from aml_mvp.reporting.automated_html_report import build_extended_report_context, render_extended_report


def test_extended_report_renders_without_placeholders(tmp_path) -> None:
    (tmp_path / "outputs/extended").mkdir(parents=True)
    (tmp_path / "outputs/reports").mkdir(parents=True)
    pd.DataFrame([{"dataset_name": "LI-Small", "alert_count": 2}]).to_csv(
        tmp_path / "outputs/extended/stress_test_summary.csv",
        index=False,
    )
    pd.DataFrame([{"feature_name": "feature_graph_cycle_involvement"}]).to_csv(
        tmp_path / "outputs/extended/graph_feature_dictionary.csv",
        index=False,
    )
    pd.DataFrame([{"metric": "cycle_candidate_count", "value": 1}]).to_csv(
        tmp_path / "outputs/extended/cycle_summary.csv",
        index=False,
    )
    pd.DataFrame([{"metric": "case_count", "value": 1}]).to_csv(
        tmp_path / "outputs/extended/case_metrics.csv",
        index=False,
    )
    pd.DataFrame([{"alert_id": "A1", "reason_code": "Large amount"}]).to_csv(
        tmp_path / "outputs/extended/reason_codes.csv",
        index=False,
    )
    pd.DataFrame([{"feature_name": "feature_amount", "mean_abs_shap": 1.0}]).to_csv(
        tmp_path / "outputs/extended/shap_feature_importance.csv",
        index=False,
    )
    pd.DataFrame([{"priority_band": "P1", "alert_count": 1}]).to_csv(
        tmp_path / "outputs/extended/priority_band_metrics.csv",
        index=False,
    )
    (tmp_path / "outputs/extended/model_selection.json").write_text(
        '{"selected_model": "extended", "decision": "promote_extended"}',
        encoding="utf-8",
    )
    config = {
        "project": {"dataset_name": "LI-Small"},
        "artifacts": {
            "stress_test_summary_path": "outputs/extended/stress_test_summary.csv",
            "graph_feature_dictionary_path": "outputs/extended/graph_feature_dictionary.csv",
            "cycle_summary_path": "outputs/extended/cycle_summary.csv",
            "case_metrics_path": "outputs/extended/case_metrics.csv",
            "reason_codes_path": "outputs/extended/reason_codes.csv",
            "shap_feature_importance_path": "outputs/extended/shap_feature_importance.csv",
            "model_selection_path": "outputs/extended/model_selection.json",
            "priority_band_metrics_path": "outputs/extended/priority_band_metrics.csv",
        },
    }

    context = build_extended_report_context(config, tmp_path)
    output = render_extended_report(context, tmp_path / "outputs/reports/aml_extended_report.html")
    html = output.read_text(encoding="utf-8")

    assert "AML Extended Build Report" in html
    assert "promote_extended" in html
    assert "{{" not in html
    assert "Extended Limitations" in html
