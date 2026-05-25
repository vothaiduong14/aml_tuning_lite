from __future__ import annotations

import pandas as pd

from aml_mvp.cases.case_consolidation import consolidate_cases
from aml_mvp.rules.base_rule import empty_rule_hits

from tests.fixtures import alerts, scored_alerts, transactions


def test_case_consolidation_groups_alerts_by_shared_accounts() -> None:
    cases, metrics = consolidate_cases(transactions(), alerts(), scored_alerts(), empty_rule_hits(), {})

    assert not cases.empty
    assert cases["alert_count"].max() >= 2
    assert "case_count" in set(metrics["metric"])


def test_case_consolidation_links_graph_rule_alerts() -> None:
    graph_hits = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "rule_id": ["GR2_CYCLE_CANDIDATE", "GR2_CYCLE_CANDIDATE"],
            "is_laundering": [0, 0],
            "trigger_values_json": [
                '{"sender_account_id": "S1", "receiver_account_id": "R1"}',
                '{"sender_account_id": "S1", "receiver_account_id": "R2"}',
            ],
        }
    )

    cases, _ = consolidate_cases(transactions(), alerts(), scored_alerts(), graph_hits, {})

    linked = cases[cases["linked_alert_ids"].str.contains("A1") & cases["linked_alert_ids"].str.contains("A2")]
    assert not linked.empty
