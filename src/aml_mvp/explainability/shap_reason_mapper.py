"""Map SHAP contributors to AML-readable reason codes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aml_mvp.storage import read_dataframe


def generate_alert_reason_codes(config: dict[str, Any], root: Path, logger=None) -> pd.DataFrame:
    artifacts = dict(config.get("artifacts", {}))
    shap_values = read_dataframe(_resolve(root, artifacts["shap_values_path"]))
    priority = _read_optional_dataframe(root, artifacts.get("priority_alerts_path"))
    mapping = _read_optional_dataframe(root, artifacts.get("alert_case_mapping_path"))
    feature_map = {**dict(config.get("features", {})), **dict(config.get("rule_flags", {}))}
    default = dict(config.get("defaults", {}).get("unknown_feature", {}))
    if logger:
        logger.info("Generating reason codes shap_rows=%s", len(shap_values))
    shap_values = shap_values.copy()
    shap_values["abs_contribution"] = shap_values["shap_value"].abs()
    shap_values = shap_values.sort_values(["alert_id", "abs_contribution"], ascending=[True, False])
    shap_values["reason_rank"] = shap_values.groupby("alert_id").cumcount() + 1
    top = shap_values[shap_values["reason_rank"].le(int(config.get("reason_codes", {}).get("top_reasons_per_alert", 3)))].copy()
    if priority is not None and "priority_band_v2" in priority:
        p1_ids = set(priority.loc[priority["priority_band_v2"].eq("P1"), "alert_id"])
        if p1_ids:
            top = top[top["alert_id"].isin(p1_ids) | top["reason_rank"].eq(1)]
    if mapping is not None and "case_id" in mapping:
        top = top.merge(mapping[["alert_id", "case_id"]].drop_duplicates("alert_id"), on="alert_id", how="left")
    else:
        top["case_id"] = ""
    reason_rows = []
    for row in top.itertuples(index=False):
        entry = dict(feature_map.get(row.feature_name, default))
        shap_value = float(row.shap_value)
        risk_direction = _risk_direction(shap_value)
        phrase = _plain_language_reason(entry, risk_direction, row.feature_name)
        reason_rows.append(
            {
                "alert_id": row.alert_id,
                "case_id": getattr(row, "case_id", ""),
                "reason_rank": int(row.reason_rank),
                "feature_name": row.feature_name,
                "feature_value": getattr(row, "feature_value", ""),
                "shap_value": shap_value,
                "contribution": shap_value,
                "contribution_bucket": _contribution_bucket(shap_value),
                "risk_direction": risk_direction,
                "business_reason_code": entry.get("business_reason_code", "Model feature contributed to risk score"),
                "plain_language_reason": phrase,
                "investigator_note": entry.get("investigator_note", "Review transaction context and linked rule evidence."),
            }
        )
    reasons = pd.DataFrame(reason_rows)
    output = _resolve(root, artifacts["alert_reason_codes_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    reasons.to_csv(output, index=False)
    if logger:
        logger.info("Wrote alert reason codes path=%s rows=%s", output, len(reasons))
    return reasons


def _risk_direction(shap_value: float) -> str:
    if shap_value > 0:
        return "risk_increasing"
    if shap_value < 0:
        return "risk_reducing"
    return "contextual"


def _contribution_bucket(shap_value: float) -> str:
    magnitude = abs(shap_value)
    if shap_value > 0 and magnitude >= 1.0:
        return "strong_positive"
    if shap_value > 0:
        return "moderate_positive"
    if shap_value < 0 and magnitude >= 1.0:
        return "strong_negative"
    if shap_value < 0:
        return "moderate_negative"
    return "neutral"


def _plain_language_reason(entry: dict[str, Any], risk_direction: str, feature_name: str) -> str:
    direction_key = {
        "risk_increasing": "risk_increasing_template",
        "risk_reducing": "risk_reducing_template",
        "contextual": "contextual_template",
    }[risk_direction]
    return str(
        entry.get(direction_key)
        or entry.get("plain_language_template")
        or f"{feature_name} contributed to the alert score."
    )


def _read_optional_dataframe(root: Path, value: str | None) -> pd.DataFrame | None:
    if not value:
        return None
    path = _resolve(root, value)
    if path.exists() or path.with_suffix(path.suffix + ".pkl").exists():
        return read_dataframe(path)
    return None


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
