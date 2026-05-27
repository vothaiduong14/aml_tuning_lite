"""Executive summary helpers for remediation report."""

from __future__ import annotations

from typing import Any


def build_executive_summary(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_decision": decision.get("decision", "keep_mvp"),
        "primary_reason": decision.get("primary_reason", "Decision artifact was not available."),
        "risk_implication": "Use remediation outputs for prioritisation analysis only; no auto-closure or suppression.",
        "operational_implication": "P1 and case caps are evaluated against configured investigation capacity.",
        "next_approval_gate": "Promote only after Precision@1000, recall, alert sizing, case sizing, and reason-code gates pass.",
    }

