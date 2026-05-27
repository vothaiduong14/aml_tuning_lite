"""Calibrate model scores and rebuild stable priority bands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from aml_mvp.calibration.priority_bands import assign_calibrated_priority_bands, summarize_calibrated_bands
from aml_mvp.storage import write_dataframe


def calibrate_scores(
    scored_alerts: pd.DataFrame,
    config: dict[str, Any],
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add calibrated scores with isotonic regression and sparse-data fallback."""

    calibration = dict(config.get("calibration", {}))
    preferred = str(calibration.get("method", "isotonic"))
    fallback = str(calibration.get("fallback_method", "sigmoid"))
    split = str(calibration.get("split", "validation"))

    output = scored_alerts.copy()
    calibration_frame = output[output["split"].eq(split)].copy() if "split" in output else output.copy()
    if calibration_frame.empty:
        calibration_frame = output.copy()

    method_used = "identity"
    if _can_fit(calibration_frame):
        if preferred == "isotonic" and calibration_frame["target"].sum() >= 2:
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(calibration_frame["model_score"], calibration_frame["target"].astype(int))
            output["calibrated_score"] = model.predict(output["model_score"].fillna(0.0))
            method_used = "isotonic"
        elif fallback == "sigmoid":
            output["calibrated_score"] = _sigmoid_scores(output, calibration_frame)
            method_used = "sigmoid"
    else:
        output["calibrated_score"] = output["model_score"].fillna(0.0)

    output["calibration_method"] = method_used
    output = add_risk_score(output, config)
    output = assign_calibrated_priority_bands(output, config)
    metrics = summarize_calibrated_bands(output)
    metrics.insert(0, "calibration_method", method_used)
    if logger:
        logger.info("Score calibration completed method=%s rows=%s", method_used, len(output))
    return output, metrics


def add_risk_score(scored_alerts: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add a 0-1000 operational rank score without replacing calibrated probability."""

    scoring = dict(config.get("scoring", {}))
    probability_col = str(scoring.get("calibrated_probability_column", "calibrated_score"))
    score_col = str(scoring.get("risk_score_column", "risk_score_1000"))
    min_score = int(scoring.get("risk_score_min", 0))
    max_score = int(scoring.get("risk_score_max", 1000))
    source_col = probability_col if probability_col in scored_alerts else "model_score"

    output = scored_alerts.copy()
    if source_col not in output:
        output[score_col] = min_score
        return output
    source = output[source_col].fillna(0.0).astype(float)
    if len(source) <= 1:
        output[score_col] = max_score if len(source) else pd.Series(dtype="int64")
        return output
    pct_rank = source.rank(method="average", pct=True)
    scaled = min_score + pct_rank * (max_score - min_score)
    output[score_col] = scaled.round().clip(min_score, max_score).astype(int)
    return output


def write_calibration_outputs(
    calibrated_scores: pd.DataFrame,
    band_metrics: pd.DataFrame,
    artifacts: dict[str, str],
    root: Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["calibrated_scores"] = write_dataframe(calibrated_scores, _resolve(root, artifacts["calibrated_scores_path"]))
    metrics_path = _resolve(root, artifacts["priority_band_metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    band_metrics.to_csv(metrics_path, index=False)
    outputs["priority_band_metrics"] = metrics_path
    return outputs


def _can_fit(frame: pd.DataFrame) -> bool:
    return len(frame) >= 10 and "target" in frame and frame["target"].nunique() >= 2 and "model_score" in frame


def _sigmoid_scores(output: pd.DataFrame, calibration_frame: pd.DataFrame) -> pd.Series:
    model = LogisticRegression(max_iter=500)
    model.fit(calibration_frame[["model_score"]].fillna(0.0), calibration_frame["target"].astype(int))
    classes = list(model.classes_)
    probabilities = model.predict_proba(output[["model_score"]].fillna(0.0))
    return pd.Series(probabilities[:, classes.index(1)] if 1 in classes else 0.0, index=output.index)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
