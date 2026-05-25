"""R2 new counterparty / cross-bank novelty rule."""

from __future__ import annotations

import pandas as pd

from aml_mvp.rules.base_rule import build_rule_hits_from_records


def apply_novelty_rule(transactions: pd.DataFrame, rule_config: dict) -> pd.DataFrame:
    scored = transactions.sort_values(["timestamp", "transaction_id"]).copy()
    pair_columns = ["sender_account_id", "receiver_account_id"]
    scored["first_seen_pair_flag"] = (~scored.duplicated(pair_columns)).astype(int)
    require_cross_bank = bool(rule_config.get("require_cross_bank", True))

    mask = scored["first_seen_pair_flag"].eq(1)
    if require_cross_bank:
        mask = mask & scored["cross_bank_flag"].eq(1)

    hits = scored.loc[mask]
    trigger_records = [
        {
            "sender_account_id": row.sender_account_id,
            "receiver_account_id": row.receiver_account_id,
            "first_seen_pair_flag": int(row.first_seen_pair_flag),
            "cross_bank_flag": int(row.cross_bank_flag),
        }
        for row in hits.itertuples(index=False)
    ]
    threshold_records = [{"require_cross_bank": require_cross_bank} for _ in range(len(hits))]
    rationale_values = [
        "First observed sender/receiver pair" + (" with cross-bank movement." if require_cross_bank else ".")
        for _ in range(len(hits))
    ]

    return build_rule_hits_from_records(
        hits,
        "R2_NOVELTY",
        "New counterparty / cross-bank novelty",
        str(rule_config.get("severity", "medium")),
        trigger_records,
        threshold_records,
        rationale_values,
    )
