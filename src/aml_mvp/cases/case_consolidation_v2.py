"""Operational case consolidation v2 with explicit caps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aml_mvp.cases.case_metrics import compute_case_metrics
from aml_mvp.cases.case_typology import assign_case_typology
from aml_mvp.storage import read_dataframe, write_dataframe


SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
BAND_RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}


def consolidate_cases_v2(config: dict[str, Any], root: Path, logger=None) -> dict[str, pd.DataFrame]:
    artifacts = dict(config.get("artifacts", {}))
    settings = dict(config.get("case_consolidation", {}))
    max_alerts = int(settings.get("max_alerts_per_case", 100))
    max_accounts = int(settings.get("max_accounts_per_case", 50))
    max_transactions = int(settings.get("max_transactions_per_case", max_alerts))
    max_component = int(settings.get("max_component_size_for_case", settings.get("max_component_size_for_auto_case", 1000)))
    alerts = read_dataframe(_resolve(root, artifacts["alerts_path"]))
    tx = read_dataframe(_resolve(root, artifacts["transactions_path"]))
    scores = _read_optional_dataframe(root, artifacts.get("scored_alerts_path"))
    if scores is None:
        scores = _read_optional_dataframe(root, artifacts.get("fallback_scored_alerts_path"))
    graph_hits = _read_optional_dataframe(root, artifacts.get("graph_rule_hits_path"))
    if graph_hits is None:
        graph_hits = pd.DataFrame()
    if logger:
        logger.info(
            "Consolidating cases v2 alerts=%s max_alerts_per_case=%s max_accounts_per_case=%s max_component_size=%s",
            len(alerts),
            max_alerts,
            max_accounts,
            max_component,
        )
    frame = alerts.merge(
        tx[["transaction_id", "timestamp", "sender_account_id", "receiver_account_id", "amount", "payment_format"]],
        on="transaction_id",
        how="left",
    )
    if scores is not None:
        score_cols = [column for column in ["alert_id", "model_score", "calibrated_score", "priority_band", "priority_band_v2"] if column in scores]
        frame = frame.merge(scores[score_cols].drop_duplicates("alert_id"), on="alert_id", how="left")
    graph_tx = set(graph_hits["transaction_id"]) if not graph_hits.empty and "transaction_id" in graph_hits else set()
    frame["has_graph_rule"] = frame["transaction_id"].isin(graph_tx).astype(int)
    frame["link_key"] = _link_key(frame)
    frame = frame.sort_values(["link_key", "alert_timestamp", "alert_id"]).reset_index(drop=True)
    component_stats = _component_stats(frame)
    large_keys = set(component_stats.loc[component_stats["component_size"].gt(max_component), "link_key"])
    cluster_frame = frame[frame["link_key"].isin(large_keys)].copy()
    case_frame = frame[~frame["link_key"].isin(large_keys)].copy()
    chunk_size = max(1, min(max_alerts, max_transactions, max(1, max_accounts // 2)))
    if not case_frame.empty:
        case_frame["case_chunk"] = case_frame.groupby("link_key").cumcount() // chunk_size
        case_frame["case_id"] = case_frame.groupby(["link_key", "case_chunk"], sort=True).ngroup().add(1).map(lambda value: f"CASEV2-{value:06d}")
    else:
        case_frame["case_id"] = pd.Series(dtype="object")
    case_mapping = case_frame[["alert_id", "transaction_id", "case_id"]].copy()
    case_mapping["network_cluster_id"] = ""
    case_mapping["mapping_type"] = "operational_case"
    case_mapping["mapping_reason"] = "same_typology_priority_time_chunk_capped"
    network_clusters = _build_network_clusters(cluster_frame)
    cluster_mapping = cluster_frame[["alert_id", "transaction_id", "link_key"]].copy()
    if not cluster_mapping.empty:
        cluster_mapping = cluster_mapping.merge(
            network_clusters[["link_key", "network_cluster_id"]],
            on="link_key",
            how="left",
        ).drop(columns=["link_key"], errors="ignore")
        cluster_mapping["case_id"] = ""
        cluster_mapping["mapping_type"] = "network_intelligence_only"
        cluster_mapping["mapping_reason"] = "component_exceeded_operational_case_cap"
        cluster_mapping = cluster_mapping[["alert_id", "transaction_id", "case_id", "network_cluster_id", "mapping_type", "mapping_reason"]]
    else:
        cluster_mapping = pd.DataFrame(columns=["alert_id", "transaction_id", "case_id", "network_cluster_id", "mapping_type", "mapping_reason"])
    mapping = pd.concat([case_mapping, cluster_mapping], ignore_index=True)
    cases = _build_cases(case_frame)
    metrics = compute_case_metrics(cases, mapping)
    typology_summary = (
        cases["case_typology"].value_counts().rename_axis("case_typology").reset_index(name="case_count")
        if "case_typology" in cases
        else pd.DataFrame(columns=["case_typology", "case_count"])
    )
    _write_outputs(cases, mapping, network_clusters, metrics, typology_summary, artifacts, root)
    if logger:
        logger.info("Case v2 completed cases=%s network_clusters=%s mappings=%s", len(cases), len(network_clusters), len(mapping))
    return {"cases": cases, "mapping": mapping, "network_clusters": network_clusters, "metrics": metrics}


def _link_key(frame: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(frame["alert_timestamp"])
    day_bucket = timestamp.dt.floor("D").astype(str)
    typology = frame["triggered_rules"].fillna("unknown").astype(str).str.split(",").str[0]
    payment = frame.get("payment_format", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    priority = frame.get("priority_band_v2", frame.get("priority_band", pd.Series("P4", index=frame.index))).fillna("P4").astype(str)
    return typology + "|" + payment + "|" + priority + "|" + day_bucket


def _build_cases(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "case_created_at",
                "case_end_at",
                "case_score",
                "alert_count",
                "transaction_count",
                "total_amount",
                "max_transaction_amount",
                "rule_ids",
                "has_laundering_label",
                "label_count",
                "account_count",
                "case_priority_band",
                "case_rationale",
                "case_typology",
            ]
        )
    working = frame.copy()
    score_source = "calibrated_score" if "calibrated_score" in working else "model_score" if "model_score" in working else None
    working["_case_score_source"] = working[score_source].fillna(0.0) if score_source else 0.0
    working["_priority_rank"] = working["priority_band_v2"].map(BAND_RANK) if "priority_band_v2" in working else 4
    working["_label"] = working.get("is_laundering", pd.Series(0, index=working.index)).fillna(0).astype(int)
    sender_counts = working.groupby("case_id")["sender_account_id"].nunique()
    receiver_counts = working.groupby("case_id")["receiver_account_id"].nunique()
    cases = (
        working.groupby("case_id", sort=True)
        .agg(
            case_created_at=("alert_timestamp", "min"),
            case_end_at=("alert_timestamp", "max"),
            case_score=("_case_score_source", "max"),
            alert_count=("alert_id", "count"),
            transaction_count=("transaction_id", "nunique"),
            total_amount=("amount", "sum"),
            max_transaction_amount=("amount", "max"),
            rule_ids=("triggered_rules", "first"),
            has_laundering_label=("_label", "max"),
            label_count=("_label", "sum"),
            min_priority_rank=("_priority_rank", "min"),
        )
        .reset_index()
    )
    cases["account_count"] = cases["case_id"].map(sender_counts.add(receiver_counts, fill_value=0)).fillna(0).astype(int)
    cases["case_priority_band"] = cases["min_priority_rank"].map({1: "P1", 2: "P2", 3: "P3", 4: "P4"}).fillna("P4")
    cases["case_rationale"] = cases["alert_count"].astype(str) + " linked alert(s) grouped by pair, typology, and time window."
    cases = cases.drop(columns=["min_priority_rank"])
    if not cases.empty:
        cases["case_typology"] = assign_case_typology(cases)
    return cases


def _component_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["link_key", "component_size", "alert_count", "account_count"])
    rows = []
    for link_key, group in frame.groupby("link_key", sort=False):
        accounts = set(group["sender_account_id"].dropna().astype(str)) | set(group["receiver_account_id"].dropna().astype(str))
        rows.append(
            {
                "link_key": link_key,
                "component_size": int(max(len(group), len(accounts))),
                "alert_count": int(len(group)),
                "account_count": int(len(accounts)),
            }
        )
    return pd.DataFrame(rows)


def _build_network_clusters(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["link_key", "network_cluster_id", "component_size", "alert_count", "account_count", "split_reason", "recommended_use"])
    rows = []
    for index, (link_key, group) in enumerate(frame.groupby("link_key", sort=True), start=1):
        accounts = set(group["sender_account_id"].dropna().astype(str)) | set(group["receiver_account_id"].dropna().astype(str))
        rows.append(
            {
                "link_key": link_key,
                "network_cluster_id": f"NETCLUSTER-{index:06d}",
                "component_size": int(max(len(group), len(accounts))),
                "alert_count": int(len(group)),
                "account_count": int(len(accounts)),
                "split_reason": "component_exceeded_operational_case_cap",
                "recommended_use": "network_intelligence_only",
            }
        )
    return pd.DataFrame(rows)


def _case_priority(group: pd.DataFrame) -> str:
    for column in ["priority_band_v2", "priority_band"]:
        if column in group:
            values = group[column].dropna().astype(str)
            if not values.empty:
                return min(values, key=lambda value: BAND_RANK.get(value, 9))
    return "P4"


def _write_outputs(
    cases: pd.DataFrame,
    mapping: pd.DataFrame,
    network_clusters: pd.DataFrame,
    metrics: pd.DataFrame,
    typology_summary: pd.DataFrame,
    artifacts: dict[str, str],
    root: Path,
) -> None:
    write_dataframe(cases, _resolve(root, artifacts["output_cases_path"]))
    write_dataframe(mapping, _resolve(root, artifacts["alert_case_mapping_path"]))
    write_dataframe(network_clusters, _resolve(root, artifacts["network_clusters_path"]))
    for key, df in [
        ("case_quality_metrics_path", metrics),
        ("mega_case_diagnostics_path", network_clusters),
        ("case_typology_summary_path", typology_summary),
    ]:
        path = _resolve(root, artifacts[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


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
