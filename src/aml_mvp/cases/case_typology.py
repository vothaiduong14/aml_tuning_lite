"""Case typology assignment."""

from __future__ import annotations

import pandas as pd


def assign_case_typology(case_alerts: pd.DataFrame) -> pd.Series:
    typologies = []
    for _, row in case_alerts.iterrows():
        rules = str(row.get("rule_ids", row.get("triggered_rules", ""))).upper()
        if "CYCLE" in rules:
            typologies.append("Cycle")
        elif "GATHER" in rules or "SCATTER" in rules:
            typologies.append("Gather-scatter")
        elif "R5" in rules or "FAN_IN" in rules:
            typologies.append("Fan-in")
        elif "R6" in rules or "FAN_OUT" in rules:
            typologies.append("Fan-out")
        elif "R4" in rules or "PASS" in rules:
            typologies.append("Pass-through")
        elif "R3" in rules or "VELOCITY" in rules:
            typologies.append("Structuring")
        elif "R1" in rules or "AMOUNT" in rules:
            typologies.append("High amount")
        else:
            typologies.append("Mixed")
    return pd.Series(typologies, index=case_alerts.index, name="case_typology")

