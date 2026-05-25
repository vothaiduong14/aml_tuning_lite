"""Convert model feature contributors into AML-readable reason codes."""

from __future__ import annotations


FEATURE_REASON_PHRASES = {
    "feature_amount": "Large transaction amount contributed to the score.",
    "feature_log_amount": "High transformed amount contributed to the score.",
    "feature_rule_count": "Multiple rules triggered for this alert.",
    "feature_rule_priority_score": "High deterministic rule priority contributed to the score.",
    "feature_max_rule_severity_ord": "A severe rule trigger contributed to the score.",
    "feature_graph_cycle_involvement": "Short-cycle graph evidence contributed to the score.",
    "feature_graph_component_size": "Large linked account component contributed to the score.",
    "feature_graph_sender_out_degree": "Sender had prior outgoing links to multiple receivers.",
    "feature_graph_receiver_in_degree": "Receiver had prior incoming links from multiple senders.",
}


def reason_phrase(feature_name: str) -> str:
    if feature_name in FEATURE_REASON_PHRASES:
        return FEATURE_REASON_PHRASES[feature_name]
    if feature_name.startswith("feature_rule_") and feature_name.endswith("_flag"):
        rule_id = feature_name.removeprefix("feature_rule_").removesuffix("_flag").upper()
        return f"{rule_id} triggered for this alert."
    readable = feature_name.removeprefix("feature_").replace("_", " ")
    return f"{readable.title()} contributed to the score."

