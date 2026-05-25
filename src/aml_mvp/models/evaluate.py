"""Shared evaluation metrics for rules and later ML ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    alert_rate: float
    f1: float


def compute_binary_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> BinaryMetrics:
    true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    if true.shape != pred.shape:
        raise ValueError("`y_true` and `y_pred` must have the same shape.")

    tp = int(((true == 1) & (pred == 1)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    alert_rate = _safe_divide(tp + fp, len(true))
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return BinaryMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        alert_rate=alert_rate,
        f1=f1,
    )


def precision_at_k(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray, k: int) -> float:
    ranked_true = _top_k_truth(y_true, scores, k)
    return _safe_divide(int(ranked_true.sum()), len(ranked_true))


def recall_at_k(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray, k: int) -> float:
    true = np.asarray(y_true).astype(int)
    ranked_true = _top_k_truth(true, scores, k)
    return _safe_divide(int(ranked_true.sum()), int(true.sum()))


def lift_at_k(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray, k: int) -> float:
    true = np.asarray(y_true).astype(int)
    base_rate = _safe_divide(int(true.sum()), len(true))
    return _safe_divide(precision_at_k(true, scores, k), base_rate)


def pr_auc(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    true = np.asarray(y_true).astype(int)
    if len(np.unique(true)) < 2:
        return 0.0
    return float(average_precision_score(true, np.asarray(scores)))


def roc_auc(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    true = np.asarray(y_true).astype(int)
    if len(np.unique(true)) < 2:
        return 0.0
    return float(roc_auc_score(true, np.asarray(scores)))


def top_k_table(
    y_true: pd.Series | np.ndarray,
    rankings: dict[str, pd.Series | np.ndarray],
    k_values: list[int],
    split: str,
) -> pd.DataFrame:
    rows = []
    for ranking_name, scores in rankings.items():
        for k in k_values:
            rows.append(
                {
                    "split": split,
                    "ranking": ranking_name,
                    "k": int(k),
                    "precision_at_k": precision_at_k(y_true, scores, int(k)),
                    "recall_at_k": recall_at_k(y_true, scores, int(k)),
                    "lift_at_k": lift_at_k(y_true, scores, int(k)),
                }
            )
    return pd.DataFrame(rows)


def _top_k_truth(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray, k: int) -> np.ndarray:
    true = np.asarray(y_true).astype(int)
    score_values = np.asarray(scores)
    if true.shape != score_values.shape:
        raise ValueError("`y_true` and `scores` must have the same shape.")
    if k <= 0 or len(true) == 0:
        return np.asarray([], dtype=int)
    top_k = min(k, len(true))
    order = np.argsort(-score_values, kind="mergesort")[:top_k]
    return true[order]


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
