"""Graph-enhanced rule hits for the extended build."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aml_mvp.rules.base_rule import build_rule_hits_from_records, empty_rule_hits
from aml_mvp.storage import write_dataframe


def run_graph_rules(
    transactions: pd.DataFrame,
    alerts: pd.DataFrame,
    graph_features: pd.DataFrame,
    config: dict[str, Any],
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create graph rule hits from point-in-time graph feature evidence."""

    graph_config = dict(config.get("graph", {}))
    min_degree = int(graph_config.get("gather_scatter_min_degree", 3))
    min_component = int(graph_config.get("gather_scatter_min_component_size", 4))
    gather_severity = str(graph_config.get("gather_scatter_severity", "high"))
    cycle_severity = str(graph_config.get("cycle_rule_severity", "high"))

    frame = graph_features.merge(
        transactions[
            [
                "transaction_id",
                "timestamp",
                "is_laundering",
                "split",
                "sender_account_id",
                "receiver_account_id",
            ]
        ],
        on="transaction_id",
        how="left",
    )
    hits = []

    gather_mask = (
        (frame["graph_sender_out_degree"].fillna(0) >= min_degree)
        | (frame["graph_receiver_in_degree"].fillna(0) >= min_degree)
    ) & (frame["graph_component_size"].fillna(0) >= min_component)
    if gather_mask.any():
        gather_frame = frame[gather_mask].copy()
        hits.append(
            build_rule_hits_from_records(
                gather_frame,
                "GR1_GATHER_SCATTER",
                "Graph gather-scatter concentration",
                gather_severity,
                _evidence_records(gather_frame),
                [{"min_degree": min_degree, "min_component_size": min_component}] * len(gather_frame),
                [
                    "Graph structure shows high account connectivity consistent with gather-scatter behavior."
                    for _ in range(len(gather_frame))
                ],
            )
        )

    cycle_mask = frame["graph_cycle_involvement"].fillna(0).astype(int).eq(1)
    if cycle_mask.any():
        cycle_frame = frame[cycle_mask].copy()
        hits.append(
            build_rule_hits_from_records(
                cycle_frame,
                "GR2_CYCLE_CANDIDATE",
                "Short-cycle candidate",
                cycle_severity,
                _evidence_records(cycle_frame),
                [{"cycle_involvement": 1}] * len(cycle_frame),
                ["Transaction closes a short directed account cycle." for _ in range(len(cycle_frame))],
            )
        )

    graph_rule_hits = pd.concat(hits, ignore_index=True) if hits else empty_rule_hits()
    cycle_summary = build_cycle_summary(frame, graph_rule_hits)
    if logger:
        logger.info("Graph rules completed graph_rule_hits=%s cycle_candidates=%s", len(graph_rule_hits), len(cycle_summary))
    return graph_rule_hits, cycle_summary


def build_cycle_summary(frame: pd.DataFrame, graph_rule_hits: pd.DataFrame) -> pd.DataFrame:
    cycle_hits = graph_rule_hits[graph_rule_hits["rule_id"].eq("GR2_CYCLE_CANDIDATE")]
    return pd.DataFrame(
        [
            {"metric": "alert_transactions_evaluated", "value": int(len(frame))},
            {"metric": "cycle_candidate_count", "value": int(len(cycle_hits))},
            {
                "metric": "cycle_candidate_label_count",
                "value": int(cycle_hits["is_laundering"].sum()) if "is_laundering" in cycle_hits else 0,
            },
        ]
    )


def write_graph_rule_outputs(
    graph_rule_hits: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    artifacts: dict[str, str],
    root: Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["graph_rule_hits"] = write_dataframe(graph_rule_hits, _resolve(root, artifacts["graph_rule_hits_path"]))
    cycle_path = _resolve(root, artifacts["cycle_summary_path"])
    cycle_path.parent.mkdir(parents=True, exist_ok=True)
    cycle_summary.to_csv(cycle_path, index=False)
    outputs["cycle_summary"] = cycle_path
    return outputs


def _evidence_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    fields = [
        "sender_account_id",
        "receiver_account_id",
        "graph_sender_out_degree",
        "graph_receiver_in_degree",
        "graph_component_size",
        "graph_cycle_involvement",
    ]
    return [{field: row.get(field) for field in fields} for _, row in frame.iterrows()]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()

