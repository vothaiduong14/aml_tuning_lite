"""R1 segmented high-amount rule."""

from __future__ import annotations

import pandas as pd

from aml_mvp.rules.base_rule import build_rule_hits_from_records


def calculate_segment_thresholds(
    transactions: pd.DataFrame,
    segmentation_fields: list[str],
    percentile: float,
    min_segment_transactions: int,
    fallback_percentile: float,
) -> tuple[pd.DataFrame, float]:
    """Calculate amount thresholds from training transactions."""

    fallback_threshold = float(transactions["amount"].quantile(fallback_percentile))
    grouped = (
        transactions.groupby(segmentation_fields, dropna=False)
        .agg(segment_count=("amount", "size"), threshold=("amount", lambda values: float(values.quantile(percentile))))
        .reset_index()
    )
    grouped["threshold_source"] = "segment"
    sparse_mask = grouped["segment_count"] < min_segment_transactions
    grouped.loc[sparse_mask, "threshold"] = fallback_threshold
    grouped.loc[sparse_mask, "threshold_source"] = "fallback"
    return grouped, fallback_threshold


def apply_amount_rule(
    transactions: pd.DataFrame,
    training_transactions: pd.DataFrame,
    rule_config: dict,
    common_config: dict,
) -> pd.DataFrame:
    segmentation_fields = list(common_config.get("segmentation_fields", ["payment_format", "currency_pair"]))
    percentile = float(rule_config.get("percentile", common_config.get("fallback_percentile", 0.95)))
    fallback_percentile = float(common_config.get("fallback_percentile", 0.95))
    min_segment_transactions = int(common_config.get("min_segment_transactions", 1000))

    thresholds, fallback_threshold = calculate_segment_thresholds(
        training_transactions,
        segmentation_fields,
        percentile,
        min_segment_transactions,
        fallback_percentile,
    )
    scored = transactions.merge(thresholds, on=segmentation_fields, how="left")
    scored["threshold"] = scored["threshold"].fillna(fallback_threshold)
    scored["threshold_source"] = scored["threshold_source"].fillna("fallback")
    scored["segment_count"] = scored["segment_count"].fillna(0).astype(int)

    mask = scored["amount"] >= scored["threshold"]
    hits = scored.loc[mask]
    trigger_records = [
        {
            "amount": row.amount,
            "payment_format": row.payment_format,
            "currency_pair": row.currency_pair,
        }
        for row in hits.itertuples(index=False)
    ]
    threshold_records = [
        {
            "threshold": row.threshold,
            "percentile": percentile,
            "threshold_source": row.threshold_source,
            "segment_count": int(row.segment_count),
        }
        for row in hits.itertuples(index=False)
    ]
    rationale_values = [
            (
                f"Amount {row.amount:.2f} is at or above the R1 threshold "
                f"{row.threshold:.2f} for {row.payment_format}/{row.currency_pair}."
            )
            for row in hits.itertuples(index=False)
    ]

    return build_rule_hits_from_records(
        hits,
        "R1_AMOUNT",
        "Segmented high amount",
        str(rule_config.get("severity", "medium")),
        trigger_records,
        threshold_records,
        rationale_values,
    )
