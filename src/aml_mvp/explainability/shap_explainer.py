"""LightGBM + SHAP explainability with deterministic fallback outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

from aml_mvp.explainability.reason_codes import reason_phrase
from aml_mvp.models.train import prepare_model_frame
from aml_mvp.storage import write_dataframe


def explain_alerts(
    alert_features: pd.DataFrame,
    scored_alerts: pd.DataFrame,
    config: dict[str, Any],
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train an explainability model and emit SHAP-like reason artifacts."""

    explain_config = dict(config.get("explainability", {}))
    max_rows = int(explain_config.get("max_explain_rows", 1000))
    top_n = int(explain_config.get("top_reason_features", 3))
    random_seed = int(explain_config.get("random_seed", 42))

    model_frame, feature_columns = prepare_model_frame(alert_features)
    train = model_frame[model_frame["split"].eq("train")].copy()
    if train.empty:
        train = model_frame.copy()
    explain_frame = model_frame.sample(min(max_rows, len(model_frame)), random_state=random_seed) if len(model_frame) else model_frame

    if train["target"].nunique() < 2 or not feature_columns:
        shap_values = _zero_shap(explain_frame, feature_columns)
        importance = _importance_from_shap(shap_values)
        reasons = _reason_codes(shap_values, top_n)
        return shap_values, reasons, importance

    model, backend = _fit_explainer_model(train, feature_columns, random_seed)
    if logger:
        logger.info("Explainability model backend=%s train_rows=%s explain_rows=%s", backend, len(train), len(explain_frame))

    shap_values = _compute_shap_values(model, explain_frame, feature_columns, backend)
    if shap_values.empty:
        shap_values = _fallback_contributions(model, train, explain_frame, feature_columns, random_seed)
    importance = _importance_from_shap(shap_values)
    reasons = _reason_codes(shap_values, top_n)
    return shap_values, reasons, importance


def write_explainability_outputs(
    shap_values: pd.DataFrame,
    reason_codes: pd.DataFrame,
    feature_importance: pd.DataFrame,
    artifacts: dict[str, str],
    root: Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["shap_values"] = write_dataframe(shap_values, _resolve(root, artifacts["shap_values_path"]))
    reason_path = _resolve(root, artifacts["reason_codes_path"])
    importance_path = _resolve(root, artifacts["shap_feature_importance_path"])
    reason_path.parent.mkdir(parents=True, exist_ok=True)
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    reason_codes.to_csv(reason_path, index=False)
    feature_importance.to_csv(importance_path, index=False)
    outputs["reason_codes"] = reason_path
    outputs["shap_feature_importance"] = importance_path
    return outputs


def _fit_explainer_model(train: pd.DataFrame, feature_columns: list[str], random_seed: int):
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_seed,
            verbosity=-1,
        )
        model.fit(train[feature_columns], train["target"].astype(int))
        return model, "lightgbm"
    except Exception:
        model = GradientBoostingClassifier(random_state=random_seed)
        model.fit(train[feature_columns], train["target"].astype(int))
        return model, "sklearn_fallback"


def _compute_shap_values(model, explain_frame: pd.DataFrame, feature_columns: list[str], backend: str) -> pd.DataFrame:
    if backend != "lightgbm":
        return pd.DataFrame()
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        explanation = explainer(explain_frame[feature_columns])
        values = _positive_class_values(getattr(explanation, "values", explanation), feature_columns)
        return _long_contributions(explain_frame, feature_columns, values)
    except Exception:
        return pd.DataFrame()


def _positive_class_values(values, feature_columns: list[str]) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        if array.shape[1] == len(feature_columns):
            return array[:, :, -1]
        if array.shape[2] == len(feature_columns):
            return array[-1, :, :]
    return array


def _fallback_contributions(
    model,
    train: pd.DataFrame,
    explain_frame: pd.DataFrame,
    feature_columns: list[str],
    random_seed: int,
) -> pd.DataFrame:
    try:
        result = permutation_importance(
            model,
            train[feature_columns],
            train["target"].astype(int),
            n_repeats=3,
            random_state=random_seed,
        )
        weights = np.asarray(result.importances_mean)
    except Exception:
        estimator_weights = getattr(model, "feature_importances_", np.zeros(len(feature_columns)))
        weights = np.asarray(estimator_weights)
    if not weights.any():
        weights = np.ones(len(feature_columns))
    centered = explain_frame[feature_columns].fillna(0.0).to_numpy()
    centered = centered - np.nanmean(centered, axis=0)
    values = centered * weights
    return _long_contributions(explain_frame, feature_columns, values)


def _zero_shap(explain_frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    values = np.zeros((len(explain_frame), len(feature_columns)))
    return _long_contributions(explain_frame, feature_columns, values)


def _long_contributions(explain_frame: pd.DataFrame, feature_columns: list[str], values: np.ndarray) -> pd.DataFrame:
    records = []
    for row_index, (_, row) in enumerate(explain_frame.iterrows()):
        for col_index, feature in enumerate(feature_columns):
            records.append(
                {
                    "alert_id": row["alert_id"],
                    "feature_name": feature,
                    "shap_value": float(values[row_index, col_index]),
                    "feature_value": float(row.get(feature, 0.0) or 0.0),
                }
            )
    return pd.DataFrame(records)


def _importance_from_shap(shap_values: pd.DataFrame) -> pd.DataFrame:
    if shap_values.empty:
        return pd.DataFrame(columns=["feature_name", "mean_abs_shap"])
    return (
        shap_values.assign(abs_shap=shap_values["shap_value"].abs())
        .groupby("feature_name", as_index=False)["abs_shap"]
        .mean()
        .rename(columns={"abs_shap": "mean_abs_shap"})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def _reason_codes(shap_values: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if shap_values.empty:
        return pd.DataFrame(columns=["alert_id", "reason_rank", "feature_name", "reason_code", "contribution"])
    rows = []
    ranked = shap_values.assign(abs_shap=shap_values["shap_value"].abs()).sort_values(
        ["alert_id", "abs_shap"],
        ascending=[True, False],
    )
    for alert_id, group in ranked.groupby("alert_id", sort=False):
        for rank, row in enumerate(group.head(top_n).itertuples(index=False), start=1):
            rows.append(
                {
                    "alert_id": alert_id,
                    "reason_rank": rank,
                    "feature_name": row.feature_name,
                    "reason_code": reason_phrase(row.feature_name),
                    "contribution": float(row.shap_value),
                }
            )
    return pd.DataFrame(rows)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
