from __future__ import annotations

import pandas as pd

from aml_mvp.models.model_comparison import compare_model_runs


def test_compare_model_runs_selects_extended_when_primary_metric_wins() -> None:
    mvp_metrics = {"metrics": {"pr_auc": 0.20, "roc_auc": 0.60}}
    extended_metrics = {"metrics": {"pr_auc": 0.25, "roc_auc": 0.62}}
    mvp_top_k = _top_k(precision=0.10, recall=0.20)
    extended_top_k = _top_k(precision=0.15, recall=0.25)

    comparison, selection = compare_model_runs(
        mvp_metrics,
        extended_metrics,
        mvp_top_k,
        extended_top_k,
        {"model_comparison": {"primary_metric": "precision_at_k", "primary_k": 1000}},
    )

    assert not comparison.empty
    assert selection["selected_model"] == "extended"
    assert selection["decision"] == "promote_extended"


def test_compare_model_runs_keeps_mvp_when_extended_does_not_improve() -> None:
    comparison, selection = compare_model_runs(
        {"metrics": {"pr_auc": 0.20}},
        {"metrics": {"pr_auc": 0.20}},
        _top_k(precision=0.10, recall=0.20),
        _top_k(precision=0.10, recall=0.21),
        {"model_comparison": {"primary_metric": "precision_at_k", "primary_k": 1000, "min_delta": 0.0}},
    )

    primary = comparison[comparison["metric"].eq("precision_at_k")].iloc[0]
    assert primary["delta"] == 0.0
    assert selection["selected_model"] == "mvp"


def _top_k(precision: float, recall: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "test",
                "ranking": "model_score",
                "k": 1000,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "lift_at_k": 2.0,
            }
        ]
    )

