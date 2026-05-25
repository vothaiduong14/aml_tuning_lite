"""Build alert-level feature matrices for ML triage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SEVERITY_ORDINAL = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def build_alert_feature_matrix(
    transactions: pd.DataFrame,
    alerts: pd.DataFrame,
    rule_hits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return alert features, feature dictionary, and quality report."""

    if alerts.empty:
        features = _empty_feature_matrix()
        return features, build_feature_dictionary(features), build_feature_quality_report(features, alerts)

    tx = _transaction_history_features(transactions)
    alert_features = alerts.merge(
        tx,
        on="transaction_id",
        how="left",
        suffixes=("", "_transaction"),
    )
    alert_features = _add_rule_flags(alert_features, rule_hits)
    alert_features = _rename_and_select_features(alert_features)
    feature_dictionary = build_feature_dictionary(alert_features)
    quality_report = build_feature_quality_report(alert_features, alerts)
    return alert_features, feature_dictionary, quality_report


def build_feature_dictionary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    definitions = {
        "feature_rule_count": ("rule", "Number of rules triggered by the alert."),
        "feature_rule_priority_score": ("rule", "Deterministic score from max severity and rule count."),
        "feature_max_rule_severity_ord": ("rule", "Ordinal severity value: low=1, medium=2, high=3, critical=4."),
        "feature_log_amount": ("transaction", "Log-transformed transaction amount."),
        "feature_amount": ("transaction", "Raw transaction amount."),
        "feature_cross_bank_flag": ("transaction", "Whether sender and receiver banks differ."),
        "feature_sender_prior_txn_count": ("account", "Sender transaction count before this transaction."),
        "feature_receiver_prior_txn_count": ("account", "Receiver transaction count before this transaction."),
        "feature_pair_prior_txn_count": ("counterparty", "Sender/receiver pair count before this transaction."),
        "feature_sender_prior_unique_receivers": ("graph_lite", "Unique receivers previously sent to by sender."),
        "feature_receiver_prior_unique_senders": ("graph_lite", "Unique senders previously funding receiver."),
    }
    for column in features.columns:
        if not column.startswith("feature_"):
            continue
        group, definition = definitions.get(column, ("rule" if column.startswith("feature_rule_") else "derived", column))
        rows.append(
            {
                "feature_name": column,
                "feature_group": group,
                "definition": definition,
                "source": "alerts + transactions + rule_hits",
                "point_in_time_rule": "Uses alert transaction and transactions at or before alert timestamp.",
            }
        )
    return pd.DataFrame(rows)


def build_feature_quality_report(features: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in features.columns if column.startswith("feature_")]
    rows = []
    for column in feature_columns:
        rows.append(
            {
                "feature_name": column,
                "row_count": int(len(features)),
                "null_count": int(features[column].isna().sum()),
                "null_rate": float(features[column].isna().mean()) if len(features) else 0.0,
                "unique_count": int(features[column].nunique(dropna=True)),
                "all_null": bool(features[column].isna().all()) if len(features) else False,
            }
        )
    return pd.DataFrame(rows)


def write_feature_outputs(
    features: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    quality_report: pd.DataFrame,
    artifacts: dict[str, str],
    root: Path,
    write_dataframe,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["alert_features"] = write_dataframe(features, _resolve(root, artifacts["alert_features_path"]))

    dictionary_path = _resolve(root, artifacts["feature_dictionary_path"])
    quality_path = _resolve(root, artifacts["feature_quality_report_path"])
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    feature_dictionary.to_csv(dictionary_path, index=False)
    quality_report.to_csv(quality_path, index=False)
    outputs["feature_dictionary"] = dictionary_path
    outputs["feature_quality_report"] = quality_path
    return outputs


def _transaction_history_features(transactions: pd.DataFrame) -> pd.DataFrame:
    ordered = transactions.sort_values(["timestamp", "transaction_id"]).copy()
    ordered["feature_sender_prior_txn_count"] = ordered.groupby("sender_account_id").cumcount()
    ordered["feature_receiver_prior_txn_count"] = ordered.groupby("receiver_account_id").cumcount()
    ordered["feature_pair_prior_txn_count"] = ordered.groupby(["sender_account_id", "receiver_account_id"]).cumcount()
    ordered["feature_sender_prior_unique_receivers"] = _prior_unique_count(
        ordered,
        group_col="sender_account_id",
        item_col="receiver_account_id",
    )
    ordered["feature_receiver_prior_unique_senders"] = _prior_unique_count(
        ordered,
        group_col="receiver_account_id",
        item_col="sender_account_id",
    )
    return ordered[
        [
            "transaction_id",
            "timestamp",
            "from_bank",
            "to_bank",
            "sender_account_id",
            "receiver_account_id",
            "payment_format",
            "currency_pair",
            "cross_bank_flag",
            "amount",
            "log_amount",
            "feature_sender_prior_txn_count",
            "feature_receiver_prior_txn_count",
            "feature_pair_prior_txn_count",
            "feature_sender_prior_unique_receivers",
            "feature_receiver_prior_unique_senders",
        ]
    ]


def _prior_unique_count(transactions: pd.DataFrame, group_col: str, item_col: str) -> pd.Series:
    result = pd.Series(0, index=transactions.index, dtype="int64")
    for _, group in transactions.groupby(group_col, sort=False):
        seen: set[Any] = set()
        values = []
        for item in group[item_col]:
            values.append(len(seen))
            seen.add(item)
        result.loc[group.index] = values
    return result


def _add_rule_flags(alert_features: pd.DataFrame, rule_hits: pd.DataFrame) -> pd.DataFrame:
    output = alert_features.copy()
    if rule_hits.empty:
        return output
    rule_table = pd.crosstab(rule_hits["transaction_id"], rule_hits["rule_id"]).clip(upper=1)
    rule_table.columns = [f"feature_rule_{column.lower()}_flag" for column in rule_table.columns]
    output = output.merge(rule_table.reset_index(), on="transaction_id", how="left")
    for column in rule_table.columns:
        output[column] = output[column].fillna(0).astype(int)
    return output


def _rename_and_select_features(alert_features: pd.DataFrame) -> pd.DataFrame:
    output = alert_features.copy()
    output["target"] = output["is_laundering"].astype(int)
    output["feature_rule_count"] = output["rule_count"].fillna(0).astype(float)
    output["feature_rule_priority_score"] = output["rule_priority_score"].fillna(0).astype(float)
    output["feature_max_rule_severity_ord"] = (
        output["max_rule_severity"].map(SEVERITY_ORDINAL).fillna(0).astype(float)
    )
    output["feature_log_amount"] = output["log_amount"].fillna(0.0).astype(float)
    output["feature_amount"] = output["amount"].fillna(0.0).astype(float)
    output["feature_cross_bank_flag"] = output["cross_bank_flag"].fillna(0).astype(float)

    feature_columns = sorted([column for column in output.columns if column.startswith("feature_")])
    metadata_columns = [
        "alert_id",
        "transaction_id",
        "alert_timestamp",
        "split",
        "target",
        "payment_format",
        "currency_pair",
        "triggered_rules",
        "max_rule_severity",
    ]
    return output[metadata_columns + feature_columns]


def _empty_feature_matrix() -> pd.DataFrame:
    return pd.DataFrame(columns=["alert_id", "transaction_id", "alert_timestamp", "split", "target"])


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()

