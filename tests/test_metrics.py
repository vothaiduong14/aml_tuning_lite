from __future__ import annotations

import pandas as pd

from aml_mvp.models.evaluate import compute_binary_metrics, lift_at_k, precision_at_k, recall_at_k


def test_compute_binary_metrics_matches_manual_counts() -> None:
    metrics = compute_binary_metrics(
        pd.Series([1, 1, 0, 0]),
        pd.Series([1, 0, 1, 0]),
    )

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.alert_rate == 0.5


def test_top_k_metrics_match_manual_ranking() -> None:
    y_true = pd.Series([0, 1, 1, 0])
    scores = pd.Series([0.2, 0.9, 0.4, 0.8])

    assert precision_at_k(y_true, scores, 2) == 0.5
    assert recall_at_k(y_true, scores, 2) == 0.5
    assert lift_at_k(y_true, scores, 2) == 1.0

