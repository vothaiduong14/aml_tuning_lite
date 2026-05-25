"""Consolidate related alerts into investigator-facing cases."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import json
import pandas as pd

from aml_mvp.cases.case_linkage import connected_components
from aml_mvp.storage import write_dataframe


SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def consolidate_cases(
    transactions: pd.DataFrame,
    alerts: pd.DataFrame,
    scored_alerts: pd.DataFrame,
    graph_rule_hits: pd.DataFrame,
    config: dict[str, Any],
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group alerts linked by shared accounts and graph-rule relationships."""

    if alerts.empty:
        return _empty_cases(), _case_metrics(_empty_cases())

    if logger:
        logger.info("Starting case consolidation alerts=%s graph_rule_hits=%s", len(alerts), len(graph_rule_hits))

    alert_frame = alerts.merge(
        transactions[["transaction_id", "sender_account_id", "receiver_account_id", "amount"]],
        on="transaction_id",
        how="left",
    )
    score_columns = [column for column in ["alert_id", "model_score", "priority_band"] if column in scored_alerts]
    if score_columns:
        alert_frame = alert_frame.merge(scored_alerts[score_columns], on="alert_id", how="left")

    alert_frame["_alert_id_str"] = alert_frame["alert_id"].astype(str)
    edges: list[tuple[str, str]] = []
    account_to_alerts: dict[str, list[str]] = defaultdict(list)
    for row in alert_frame.itertuples(index=False):
        account_to_alerts[str(row.sender_account_id)].append(str(row.alert_id))
        account_to_alerts[str(row.receiver_account_id)].append(str(row.alert_id))
    for linked_alerts in account_to_alerts.values():
        edges.extend(_pairwise(linked_alerts))

    if not graph_rule_hits.empty:
        edges.extend(_graph_evidence_edges(alert_frame, graph_rule_hits))

    for alert_id in alert_frame["_alert_id_str"]:
        edges.append((alert_id, alert_id))

    if logger:
        logger.info("Built case linkage graph accounts=%s edges=%s", len(account_to_alerts), len(edges))

    components = connected_components(edges)
    case_map = _case_map(components)
    alert_frame["case_id"] = alert_frame["_alert_id_str"].map(case_map)
    if logger:
        logger.info("Assigned alerts to connected components cases=%s", alert_frame["case_id"].nunique())

    rows = [_case_row(case_id, group) for case_id, group in alert_frame.groupby("case_id", sort=True)]
    cases = pd.DataFrame(rows).sort_values(["max_model_score", "alert_count"], ascending=[False, False])
    metrics = _case_metrics(cases)
    if logger:
        logger.info("Case consolidation completed cases=%s alerts=%s", len(cases), len(alerts))
    return cases.reset_index(drop=True), metrics


def write_case_outputs(
    cases: pd.DataFrame,
    metrics: pd.DataFrame,
    artifacts: dict[str, str],
    root: Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["consolidated_cases"] = write_dataframe(cases, _resolve(root, artifacts["consolidated_cases_path"]))
    metrics_path = _resolve(root, artifacts["case_metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    outputs["case_metrics"] = metrics_path
    return outputs


def _case_metrics(cases: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "case_count", "value": int(len(cases))},
            {"metric": "avg_alerts_per_case", "value": float(cases["alert_count"].mean()) if len(cases) else 0.0},
            {"metric": "max_alerts_per_case", "value": int(cases["alert_count"].max()) if len(cases) else 0},
            {"metric": "avg_accounts_per_case", "value": float(cases["account_count"].mean()) if len(cases) else 0.0},
        ]
    )


def _empty_cases() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "case_id",
            "alert_count",
            "account_count",
            "total_amount",
            "max_severity",
            "avg_model_score",
            "max_model_score",
            "linked_alert_ids",
            "linked_accounts",
        ]
    )


def _pairwise(values: list[str]) -> list[tuple[str, str]]:
    unique = sorted(set(values))
    if len(unique) <= 1:
        return [(value, value) for value in unique]
    return [(unique[index], unique[index + 1]) for index in range(len(unique) - 1)]


def _graph_evidence_edges(alert_frame: pd.DataFrame, graph_rule_hits: pd.DataFrame) -> list[tuple[str, str]]:
    """Link graph-rule alerts only when evidence references the same accounts.

    The previous version chained all graph-rule alerts together, which could
    create one very large case and make aggregation unnecessarily expensive.
    """

    if "trigger_values_json" not in graph_rule_hits:
        return []
    tx_to_alert = dict(zip(alert_frame["transaction_id"], alert_frame["_alert_id_str"]))
    account_to_alerts: dict[str, list[str]] = defaultdict(list)
    for row in graph_rule_hits.itertuples(index=False):
        alert_id = tx_to_alert.get(getattr(row, "transaction_id"))
        if alert_id is None:
            continue
        evidence = _parse_json(getattr(row, "trigger_values_json", "{}"))
        for account_field in ["sender_account_id", "receiver_account_id"]:
            account = evidence.get(account_field)
            if account:
                account_to_alerts[str(account)].append(str(alert_id))
    edges: list[tuple[str, str]] = []
    for linked_alerts in account_to_alerts.values():
        edges.extend(_pairwise(linked_alerts))
    return edges


def _case_map(components: list[set[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for idx, component in enumerate(sorted(components, key=lambda values: sorted(values)[0]), start=1):
        case_id = f"CASE-{idx:06d}"
        for alert_id in component:
            mapping[str(alert_id)] = case_id
    return mapping


def _case_row(case_id: str, group: pd.DataFrame) -> dict[str, Any]:
    accounts = set(group["sender_account_id"].astype(str)) | set(group["receiver_account_id"].astype(str))
    max_severity = max(group["max_rule_severity"].astype(str), key=lambda value: SEVERITY_RANK.get(value, 0))
    return {
        "case_id": case_id,
        "alert_count": int(len(group)),
        "account_count": int(len(accounts)),
        "total_amount": float(group["amount"].fillna(0.0).sum()),
        "max_severity": max_severity,
        "avg_model_score": float(group["model_score"].fillna(0.0).mean()) if "model_score" in group else 0.0,
        "max_model_score": float(group["model_score"].fillna(0.0).max()) if "model_score" in group else 0.0,
        "linked_alert_ids": ",".join(sorted(group["_alert_id_str"])),
        "linked_accounts": ",".join(sorted(accounts)),
    }


def _parse_json(value: object) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
