"""Split overly large components into operational case chunks."""

from __future__ import annotations

from typing import Any

import pandas as pd


def split_mega_components(alert_links: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    settings = dict(config.get("case_consolidation", {}))
    max_alerts = int(settings.get("max_alerts_per_case", 100))
    max_component = int(settings.get("max_component_size_for_auto_case", 1000))
    if alert_links.empty:
        return {"operational_links": alert_links, "network_clusters": pd.DataFrame()}
    rows = []
    clusters = []
    for component_id, group in alert_links.groupby("component_id", sort=True):
        if len(group) > max_component:
            cluster_id = f"NET-{len(clusters) + 1:06d}"
            clusters.append(
                {
                    "network_cluster_id": cluster_id,
                    "component_size": int(len(group)),
                    "alert_count": int(group["alert_id"].nunique()),
                    "account_count": int(group.get("account_count", pd.Series([0])).max()),
                    "split_reason": "component_exceeds_auto_case_cap",
                    "recommended_use": "network_intelligence_only",
                }
            )
            group = group.copy()
            group["network_cluster_id"] = cluster_id
        for chunk, start in enumerate(range(0, len(group), max_alerts), start=1):
            part = group.iloc[start : start + max_alerts].copy()
            part["case_chunk"] = chunk
            rows.append(part)
    return {
        "operational_links": pd.concat(rows, ignore_index=True) if rows else alert_links.head(0),
        "network_clusters": pd.DataFrame(clusters),
    }

