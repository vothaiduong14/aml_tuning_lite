"""Command line entry points for MVP workflows."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from aml_mvp.config import load_config, project_root_from_config
from aml_mvp.data.load_data import ingest_transactions
from aml_mvp.data.profile import profile_transactions
from aml_mvp.data.quality_checks import write_quality_report
from aml_mvp.data.splits import create_temporal_splits, write_split_manifest
from aml_mvp.features.feature_matrix import build_alert_feature_matrix, write_feature_outputs
from aml_mvp.logging_utils import resolve_log_file, setup_logging
from aml_mvp.models.train import train_and_score_alerts, write_model_outputs
from aml_mvp.rules.rule_engine import run_rule_engine, write_rule_outputs
from aml_mvp.rules.rule_tuning import tune_rules, write_tuning_outputs
from aml_mvp.storage import read_dataframe, write_dataframe
from aml_mvp.triage.capacity_simulation import simulate_capacity
from aml_mvp.triage.priority_banding import assign_priority_bands, write_priority_outputs


def main() -> None:
    parser = argparse.ArgumentParser(prog="aml_mvp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-data")
    validate_parser.add_argument("--config", required=True)
    _add_logging_args(validate_parser)

    split_parser = subparsers.add_parser("create-splits")
    split_parser.add_argument("--config", required=True)
    _add_logging_args(split_parser)

    rules_parser = subparsers.add_parser("run-rules")
    rules_parser.add_argument("--config", required=True)
    _add_logging_args(rules_parser)

    tuning_parser = subparsers.add_parser("tune-rules")
    tuning_parser.add_argument("--config", required=True)
    _add_logging_args(tuning_parser)

    features_parser = subparsers.add_parser("build-features")
    features_parser.add_argument("--config", required=True)
    _add_logging_args(features_parser)

    model_parser = subparsers.add_parser("train-model")
    model_parser.add_argument("--config", required=True)
    _add_logging_args(model_parser)

    priority_parser = subparsers.add_parser("prioritize-alerts")
    priority_parser.add_argument("--config", required=True)
    _add_logging_args(priority_parser)

    report_parser = subparsers.add_parser("build-report")
    report_parser.add_argument("--config", required=True)
    _add_logging_args(report_parser)

    extended_stress_parser = subparsers.add_parser("extended-stress-test")
    extended_stress_parser.add_argument("--config", required=True)
    _add_logging_args(extended_stress_parser)

    graph_features_parser = subparsers.add_parser("build-graph-features")
    graph_features_parser.add_argument("--config", required=True)
    _add_logging_args(graph_features_parser)

    extended_model_parser = subparsers.add_parser("train-extended-model")
    extended_model_parser.add_argument("--config", required=True)
    _add_logging_args(extended_model_parser)

    compare_model_parser = subparsers.add_parser("compare-models")
    compare_model_parser.add_argument("--config", required=True)
    _add_logging_args(compare_model_parser)

    graph_rules_parser = subparsers.add_parser("run-graph-rules")
    graph_rules_parser.add_argument("--config", required=True)
    _add_logging_args(graph_rules_parser)

    cases_parser = subparsers.add_parser("consolidate-cases")
    cases_parser.add_argument("--config", required=True)
    _add_logging_args(cases_parser)

    explain_parser = subparsers.add_parser("explain-alerts")
    explain_parser.add_argument("--config", required=True)
    _add_logging_args(explain_parser)

    calibration_parser = subparsers.add_parser("calibrate-scores")
    calibration_parser.add_argument("--config", required=True)
    _add_logging_args(calibration_parser)

    extended_report_parser = subparsers.add_parser("build-extended-report")
    extended_report_parser.add_argument("--config", required=True)
    _add_logging_args(extended_report_parser)

    workflow_parser = subparsers.add_parser("run-workflow")
    workflow_parser.add_argument(
        "--preset",
        default="full",
        choices=["mvp", "extended", "remediation", "full"],
        help="Workflow preset to run when --steps is not provided.",
    )
    workflow_parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step names to run in the requested order.",
    )
    workflow_parser.add_argument(
        "--from-step",
        default=None,
        help="Start at this step within the selected preset.",
    )
    workflow_parser.add_argument(
        "--to-step",
        default=None,
        help="Stop after this step within the selected preset.",
    )
    workflow_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected commands without running them.",
    )
    workflow_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    workflow_parser.add_argument(
        "--log-file",
        default=None,
        help="Shared workflow log file. Defaults to outputs/run_logs/workflow_<preset>_<timestamp>.log.",
    )

    diagnose_parser = subparsers.add_parser("diagnose-alerts")
    diagnose_parser.add_argument("--config", required=True)
    _add_logging_args(diagnose_parser)

    band_parser = subparsers.add_parser("rebuild-priority-bands")
    band_parser.add_argument("--config", required=True)
    _add_logging_args(band_parser)

    graph_v2_parser = subparsers.add_parser("build-graph-v2")
    graph_v2_parser.add_argument("--config", required=True)
    _add_logging_args(graph_v2_parser)

    graph_ablation_parser = subparsers.add_parser("run-graph-ablation")
    graph_ablation_parser.add_argument("--config", required=True)
    _add_logging_args(graph_ablation_parser)

    cases_v2_parser = subparsers.add_parser("consolidate-cases-v2")
    cases_v2_parser.add_argument("--config", required=True)
    _add_logging_args(cases_v2_parser)

    reason_parser = subparsers.add_parser("generate-reason-codes")
    reason_parser.add_argument("--config", required=True)
    _add_logging_args(reason_parser)

    remediation_report_parser = subparsers.add_parser("build-remediation-report")
    remediation_report_parser.add_argument("--config", required=True)
    _add_logging_args(remediation_report_parser)

    remediation_parser = subparsers.add_parser("run-remediation")
    remediation_parser.add_argument("--config", required=True)
    remediation_parser.add_argument("--steps", default=None)
    remediation_parser.add_argument("--skip", default=None)
    remediation_parser.add_argument("--from-step", default=None)
    remediation_parser.add_argument("--dry-run", action="store_true")
    remediation_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    remediation_parser.add_argument(
        "--log-file",
        default=None,
        help="Shared remediation log file. Defaults to outputs/run_logs/remediation_<timestamp>.log.",
    )

    args = parser.parse_args()

    if args.command == "validate-data":
        _validate_data(Path(args.config), args)
    elif args.command == "create-splits":
        _create_splits(Path(args.config), args)
    elif args.command == "run-rules":
        _run_rules(Path(args.config), args)
    elif args.command == "tune-rules":
        _tune_rules(Path(args.config), args)
    elif args.command == "build-features":
        _build_features(Path(args.config), args)
    elif args.command == "train-model":
        _train_model(Path(args.config), args)
    elif args.command == "prioritize-alerts":
        _prioritize_alerts(Path(args.config), args)
    elif args.command == "build-report":
        _build_report(Path(args.config), args)
    elif args.command == "extended-stress-test":
        _extended_stress_test(Path(args.config), args)
    elif args.command == "build-graph-features":
        _build_graph_features(Path(args.config), args)
    elif args.command == "train-extended-model":
        _train_extended_model(Path(args.config), args)
    elif args.command == "compare-models":
        _compare_models(Path(args.config), args)
    elif args.command == "run-graph-rules":
        _run_graph_rules(Path(args.config), args)
    elif args.command == "consolidate-cases":
        _consolidate_cases(Path(args.config), args)
    elif args.command == "explain-alerts":
        _explain_alerts(Path(args.config), args)
    elif args.command == "calibrate-scores":
        _calibrate_scores(Path(args.config), args)
    elif args.command == "build-extended-report":
        _build_extended_report(Path(args.config), args)
    elif args.command == "run-workflow":
        _run_workflow(args)
    elif args.command == "diagnose-alerts":
        _diagnose_alerts(Path(args.config), args)
    elif args.command == "rebuild-priority-bands":
        _rebuild_priority_bands(Path(args.config), args)
    elif args.command == "build-graph-v2":
        _build_graph_v2(Path(args.config), args)
    elif args.command == "run-graph-ablation":
        _run_graph_ablation(Path(args.config), args)
    elif args.command == "consolidate-cases-v2":
        _consolidate_cases_v2(Path(args.config), args)
    elif args.command == "generate-reason-codes":
        _generate_reason_codes(Path(args.config), args)
    elif args.command == "build-remediation-report":
        _build_remediation_report(Path(args.config), args)
    elif args.command == "run-remediation":
        _run_remediation(args)


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default=None)


def _validate_data(config_path: Path, args: argparse.Namespace) -> None:
    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("validate-data", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    logger.info("Loading raw transactions from %s", _resolve(root, config["data"]["raw_path"]))
    df = ingest_transactions(_resolve(root, config["data"]["raw_path"]))
    logger.info("Loaded and standardized transactions rows=%s columns=%s", len(df), len(df.columns))
    table_path = write_dataframe(df, _resolve(root, config["data"]["interim_path"]))
    report_path = write_quality_report(df, _resolve(root, config["data"]["quality_report_path"]))
    logger.info("Wrote standardized transactions path=%s", table_path)
    logger.info("Wrote data quality report path=%s", report_path)
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _create_splits(config_path: Path, args: argparse.Namespace) -> None:
    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("create-splits", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    interim_path = _resolve(root, config["data"]["interim_path"])
    logger.info("Reading standardized transactions path=%s", interim_path)
    df = read_dataframe(interim_path)
    logger.info("Loaded transactions rows=%s columns=%s", len(df), len(df.columns))
    split_df, manifest = create_temporal_splits(df, config["splits"])
    logger.info("Created temporal splits split_counts=%s", split_df["split"].value_counts().to_dict())
    table_path = write_dataframe(split_df, _resolve(root, config["data"]["processed_path"]))
    manifest_path = write_split_manifest(manifest, _resolve(root, config["data"]["split_manifest_path"]))
    profile_path = profile_transactions(split_df, _resolve(root, config["data"]["profile_report_path"]))
    logger.info("Wrote split transactions path=%s", table_path)
    logger.info("Wrote split manifest path=%s", manifest_path)
    logger.info("Wrote data profile path=%s", profile_path)
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _run_rules(config_path: Path, args: argparse.Namespace) -> None:
    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("run-rules", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    artifacts = config["artifacts"]
    transactions_path = _resolve(root, artifacts["transactions_path"])
    logger.info("Reading split transactions path=%s", transactions_path)
    transactions = read_dataframe(transactions_path)
    logger.info(
        "Loaded transactions rows=%s split_counts=%s",
        len(transactions),
        _split_counts(transactions),
    )
    rule_hits, alerts = run_rule_engine(transactions, config, logger=logging.getLogger("aml_mvp.rules"))
    outputs = write_rule_outputs(transactions, rule_hits, alerts, artifacts, root, write_dataframe)
    logger.info("Wrote rule hits path=%s", outputs["rule_hits"])
    logger.info("Wrote alerts path=%s", outputs["alerts"])
    logger.info("Wrote rule performance path=%s", outputs["rule_performance"])
    logger.info("Wrote rule overlap matrix path=%s", outputs["rule_overlap"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _tune_rules(config_path: Path, args: argparse.Namespace) -> None:
    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("tune-rules", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    artifacts = config["artifacts"]
    transactions_path = _resolve(root, artifacts["transactions_path"])
    logger.info("Reading split transactions path=%s", transactions_path)
    transactions = read_dataframe(transactions_path)
    logger.info(
        "Loaded transactions rows=%s split_counts=%s",
        len(transactions),
        _split_counts(transactions),
    )
    candidates, selected_thresholds, audit = tune_rules(
        transactions,
        config,
        logger=logging.getLogger("aml_mvp.tuning"),
    )
    outputs = write_tuning_outputs(candidates, selected_thresholds, audit, artifacts, root)
    logger.info("Wrote tuning candidates path=%s", outputs["tuning_candidates"])
    logger.info("Wrote selected thresholds path=%s", outputs["selected_thresholds"])
    logger.info("Wrote tuning audit path=%s", outputs["tuning_audit"])
    logger.info("Wrote tuned rule config path=%s", outputs["tuned_rule_config"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _build_features(config_path: Path, args: argparse.Namespace) -> None:
    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("build-features", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    artifacts = config["artifacts"]
    transactions = read_dataframe(_resolve(root, artifacts["transactions_path"]))
    alerts = read_dataframe(_resolve(root, artifacts["alerts_path"]))
    rule_hits = read_dataframe(_resolve(root, artifacts["rule_hits_path"]))
    logger.info("Loaded transactions=%s alerts=%s rule_hits=%s", len(transactions), len(alerts), len(rule_hits))
    features, feature_dictionary, quality_report = build_alert_feature_matrix(transactions, alerts, rule_hits)
    outputs = write_feature_outputs(features, feature_dictionary, quality_report, artifacts, root, write_dataframe)
    logger.info("Built alert feature matrix rows=%s feature_count=%s", len(features), len(feature_dictionary))
    logger.info("Wrote alert features path=%s", outputs["alert_features"])
    logger.info("Wrote feature dictionary path=%s", outputs["feature_dictionary"])
    logger.info("Wrote feature quality report path=%s", outputs["feature_quality_report"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _train_model(config_path: Path, args: argparse.Namespace) -> None:
    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("train-model", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    artifacts = config["artifacts"]
    features = read_dataframe(_resolve(root, artifacts["alert_features_path"]))
    logger.info("Loaded alert features rows=%s split_counts=%s", len(features), _split_counts(features))
    scored_alerts, metrics, top_k_metrics, feature_importance, model_artifact = train_and_score_alerts(features, config)
    outputs = write_model_outputs(
        scored_alerts,
        metrics,
        top_k_metrics,
        feature_importance,
        model_artifact,
        artifacts,
        root,
    )
    logger.info("Wrote scored alerts path=%s", outputs["scored_alerts"])
    logger.info("Wrote model artifact path=%s", outputs["model_artifact"])
    logger.info("Wrote model metrics path=%s", outputs["model_metrics"])
    logger.info("Wrote top-k metrics path=%s", outputs["top_k_metrics"])
    logger.info("Wrote feature importance path=%s", outputs["feature_importance"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _prioritize_alerts(config_path: Path, args: argparse.Namespace) -> None:
    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("prioritize-alerts", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    artifacts = config["artifacts"]
    scored_alerts = read_dataframe(_resolve(root, artifacts["scored_alerts_path"]))
    logger.info("Loaded scored alerts rows=%s", len(scored_alerts))
    priority_alerts, band_summary = assign_priority_bands(scored_alerts, config)
    capacity = simulate_capacity(priority_alerts, config)
    outputs = write_priority_outputs(priority_alerts, band_summary, capacity, artifacts, root, write_dataframe)
    logger.info("Wrote priority alerts path=%s", outputs["priority_alerts"])
    logger.info("Wrote band summary path=%s", outputs["band_summary"])
    logger.info("Wrote capacity simulation path=%s", outputs["capacity_simulation"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _build_report(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.reporting.html_report import build_report_context, render_html_report, write_handover_artifacts

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("build-report", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    report_config = config["report"]
    context = build_report_context(config, root)
    report_path = render_html_report(context, _resolve(root, report_config["output_file"]))
    handover = write_handover_artifacts(context, config, root)
    logger.info("Wrote HTML report path=%s", report_path)
    logger.info("Wrote run manifest path=%s", handover["run_manifest"])
    logger.info("Wrote acceptance checklist path=%s", handover["acceptance_checklist"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _extended_stress_test(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.extended.stress_testing import (
        build_segment_stress_summary,
        build_stress_test_summary,
        build_temporal_stress_summary,
        write_extended_stress_outputs,
        write_stress_summary,
    )

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("extended-stress-test", root, args.log_level, args.log_file)
    start = time.perf_counter()
    artifacts = config["artifacts"]
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    transactions = read_dataframe(_resolve(root, artifacts["transactions_path"]))
    alerts = read_dataframe(_resolve(root, artifacts["alerts_path"]))
    rule_hits = read_dataframe(_resolve(root, artifacts["rule_hits_path"]))
    priority_alerts = read_dataframe(_resolve(root, artifacts["priority_alerts_path"]))
    summary = build_stress_test_summary(transactions, alerts, rule_hits, priority_alerts, config)
    temporal = build_temporal_stress_summary(priority_alerts, config)
    segment = build_segment_stress_summary(transactions, priority_alerts, config)
    output = write_stress_summary(summary, _resolve(root, artifacts["stress_test_summary_path"]))
    extra_outputs = write_extended_stress_outputs(temporal, segment, artifacts, root)
    logger.info("Wrote stress test summary path=%s rows=%s", output, len(summary))
    for key, path in extra_outputs.items():
        logger.info("Wrote %s path=%s", key, path)
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _build_graph_features(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.graph.graph_features import (
        build_graph_features,
        merge_graph_features_into_alert_matrix,
        write_graph_feature_outputs,
    )

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("build-graph-features", root, args.log_level, args.log_file)
    start = time.perf_counter()
    artifacts = config["artifacts"]
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    transactions = read_dataframe(_resolve(root, artifacts["transactions_path"]))
    alerts = read_dataframe(_resolve(root, artifacts["alerts_path"]))
    logger.info("Loaded transactions=%s alerts=%s", len(transactions), len(alerts))
    graph_features, dictionary = build_graph_features(transactions, alerts, config, logger=logger)
    outputs = write_graph_feature_outputs(graph_features, dictionary, artifacts, root)
    if "extended_alert_features_path" in artifacts:
        alert_features = read_dataframe(_resolve(root, artifacts["alert_features_path"]))
        extended_features = merge_graph_features_into_alert_matrix(alert_features, graph_features)
        extended_path = write_dataframe(extended_features, _resolve(root, artifacts["extended_alert_features_path"]))
        logger.info("Wrote extended alert features path=%s rows=%s", extended_path, len(extended_features))
    logger.info("Wrote graph features path=%s rows=%s", outputs["graph_features"], len(graph_features))
    logger.info("Wrote graph feature dictionary path=%s", outputs["graph_feature_dictionary"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _train_extended_model(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.graph.graph_features import merge_graph_feature_columns_into_alert_matrix

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("train-extended-model", root, args.log_level, args.log_file)
    start = time.perf_counter()
    artifacts = config["artifacts"]
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    features_path = _resolve(root, artifacts["extended_alert_features_path"])
    features = read_dataframe(features_path)
    graph_v2_path_value = artifacts.get("graph_features_v2_path")
    if graph_v2_path_value:
        graph_v2_path = _resolve(root, graph_v2_path_value)
        if graph_v2_path.exists() or graph_v2_path.with_suffix(graph_v2_path.suffix + ".pkl").exists():
            graph_v2 = read_dataframe(graph_v2_path)
            before_columns = set(features.columns)
            features = merge_graph_feature_columns_into_alert_matrix(features, graph_v2)
            added_columns = sorted(set(features.columns) - before_columns)
            logger.info(
                "Merged graph v2 features into extended training frame path=%s graph_rows=%s added_feature_columns=%s",
                graph_v2_path,
                len(graph_v2),
                added_columns,
            )
    logger.info("Loaded extended alert features path=%s rows=%s split_counts=%s", features_path, len(features), _split_counts(features))
    scored_alerts, metrics, top_k_metrics, feature_importance, model_artifact = train_and_score_alerts(features, config)
    outputs = write_model_outputs(
        scored_alerts,
        metrics,
        top_k_metrics,
        feature_importance,
        model_artifact,
        _extended_model_artifacts(artifacts),
        root,
    )
    logger.info("Wrote extended scored alerts path=%s", outputs["scored_alerts"])
    logger.info("Wrote extended model artifact path=%s", outputs["model_artifact"])
    logger.info("Wrote extended model metrics path=%s", outputs["model_metrics"])
    logger.info("Wrote extended top-k metrics path=%s", outputs["top_k_metrics"])
    logger.info("Wrote extended feature importance path=%s", outputs["feature_importance"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _compare_models(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.models.model_comparison import compare_model_runs, read_json, write_model_comparison_outputs

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("compare-models", root, args.log_level, args.log_file)
    start = time.perf_counter()
    artifacts = config["artifacts"]
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    _require_artifact_keys(
        artifacts,
        [
            "model_metrics_path",
            "extended_model_metrics_path",
            "top_k_metrics_path",
            "extended_top_k_metrics_path",
            "model_comparison_path",
            "model_selection_path",
        ],
    )
    mvp_metrics = read_json(_resolve(root, artifacts["model_metrics_path"]))
    extended_metrics = read_json(_resolve(root, artifacts["extended_model_metrics_path"]))
    mvp_top_k = _read_csv(_resolve(root, artifacts["top_k_metrics_path"]))
    extended_top_k = _read_csv(_resolve(root, artifacts["extended_top_k_metrics_path"]))
    comparison, selection = compare_model_runs(mvp_metrics, extended_metrics, mvp_top_k, extended_top_k, config)
    outputs = write_model_comparison_outputs(comparison, selection, artifacts, root)
    logger.info("Wrote model comparison path=%s rows=%s", outputs["model_comparison"], len(comparison))
    logger.info("Wrote model selection path=%s selected_model=%s", outputs["model_selection"], selection["selected_model"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _run_graph_rules(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.graph.graph_rules import run_graph_rules, write_graph_rule_outputs

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("run-graph-rules", root, args.log_level, args.log_file)
    start = time.perf_counter()
    artifacts = config["artifacts"]
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    transactions = read_dataframe(_resolve(root, artifacts["transactions_path"]))
    alerts = read_dataframe(_resolve(root, artifacts["alerts_path"]))
    graph_features = read_dataframe(_resolve(root, artifacts["graph_features_path"]))
    logger.info("Loaded transactions=%s alerts=%s graph_features=%s", len(transactions), len(alerts), len(graph_features))
    graph_rule_hits, cycle_summary = run_graph_rules(transactions, alerts, graph_features, config, logger=logger)
    outputs = write_graph_rule_outputs(graph_rule_hits, cycle_summary, artifacts, root)
    logger.info("Wrote graph rule hits path=%s rows=%s", outputs["graph_rule_hits"], len(graph_rule_hits))
    logger.info("Wrote cycle summary path=%s", outputs["cycle_summary"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _consolidate_cases(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.cases.case_consolidation import consolidate_cases, write_case_outputs

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("consolidate-cases", root, args.log_level, args.log_file)
    start = time.perf_counter()
    artifacts = config["artifacts"]
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    transactions = read_dataframe(_resolve(root, artifacts["transactions_path"]))
    alerts = read_dataframe(_resolve(root, artifacts["alerts_path"]))
    scored_alerts = _read_preferred_scored_alerts(root, artifacts)
    graph_rule_hits = read_dataframe(_resolve(root, artifacts["graph_rule_hits_path"]))
    cases, metrics = consolidate_cases(transactions, alerts, scored_alerts, graph_rule_hits, config, logger=logger)
    outputs = write_case_outputs(cases, metrics, artifacts, root)
    logger.info("Wrote consolidated cases path=%s rows=%s", outputs["consolidated_cases"], len(cases))
    logger.info("Wrote case metrics path=%s", outputs["case_metrics"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _explain_alerts(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.explainability.shap_explainer import explain_alerts, write_explainability_outputs
    from aml_mvp.graph.graph_features import merge_graph_features_into_alert_matrix

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("explain-alerts", root, args.log_level, args.log_file)
    start = time.perf_counter()
    artifacts = config["artifacts"]
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    extended_path = _resolve(root, artifacts.get("extended_alert_features_path", artifacts["alert_features_path"]))
    if extended_path.exists() or extended_path.with_suffix(extended_path.suffix + ".pkl").exists():
        alert_features = read_dataframe(extended_path)
    else:
        alert_features = read_dataframe(_resolve(root, artifacts["alert_features_path"]))
    scored_alerts = _read_preferred_scored_alerts(root, artifacts)
    graph_path = _resolve(root, artifacts["graph_features_path"])
    has_graph_features = any(column.startswith("feature_graph_") for column in alert_features.columns)
    if not has_graph_features and (graph_path.exists() or graph_path.with_suffix(graph_path.suffix + ".pkl").exists()):
        graph_features = read_dataframe(graph_path)
        alert_features = merge_graph_features_into_alert_matrix(alert_features, graph_features)
        logger.info("Merged graph features into explainability frame rows=%s", len(alert_features))
    shap_values, reason_codes, importance = explain_alerts(alert_features, scored_alerts, config, logger=logger)
    outputs = write_explainability_outputs(shap_values, reason_codes, importance, artifacts, root)
    logger.info("Wrote SHAP values path=%s rows=%s", outputs["shap_values"], len(shap_values))
    logger.info("Wrote reason codes path=%s rows=%s", outputs["reason_codes"], len(reason_codes))
    logger.info("Wrote SHAP feature importance path=%s", outputs["shap_feature_importance"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _calibrate_scores(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.calibration.calibration import calibrate_scores, write_calibration_outputs

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("calibrate-scores", root, args.log_level, args.log_file)
    start = time.perf_counter()
    artifacts = config["artifacts"]
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    scored_alerts = _read_preferred_scored_alerts(root, artifacts)
    calibrated_scores, metrics = calibrate_scores(scored_alerts, config, logger=logger)
    outputs = write_calibration_outputs(calibrated_scores, metrics, artifacts, root)
    logger.info("Wrote calibrated scores path=%s rows=%s", outputs["calibrated_scores"], len(calibrated_scores))
    logger.info("Wrote priority band metrics path=%s", outputs["priority_band_metrics"])
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _build_extended_report(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.reporting.automated_html_report import build_extended_report_context, render_extended_report

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("build-extended-report", root, args.log_level, args.log_file)
    start = time.perf_counter()
    artifacts = config["artifacts"]
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    context = build_extended_report_context(config, root)
    output = render_extended_report(context, _resolve(root, artifacts["extended_report_path"]))
    logger.info("Wrote extended HTML report path=%s", output)
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _diagnose_alerts(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.diagnostics.alert_waterfall import build_alert_waterfall
    from aml_mvp.diagnostics.rule_contribution import build_rule_contribution

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("diagnose-alerts", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    contribution = build_rule_contribution(config, root, logger=logger)
    waterfall = build_alert_waterfall(config, root, logger=logger)
    logger.info("Built diagnostics rule_rows=%s waterfall_rows=%s", len(contribution), len(waterfall["waterfall"]))
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _rebuild_priority_bands(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.calibration.band_sizing import rebuild_priority_bands

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("rebuild-priority-bands", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    priority = rebuild_priority_bands(config, root, logger=logger)
    logger.info("Rebuilt priority bands rows=%s band_counts=%s", len(priority), priority["priority_band_v2"].value_counts().to_dict())
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _build_graph_v2(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.graph.graph_features_v2 import build_graph_features_v2

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("build-graph-v2", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    features = build_graph_features_v2(config, root, logger=logger)
    logger.info("Built graph v2 rows=%s", len(features))
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _run_graph_ablation(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.graph.graph_ablation import run_graph_ablation

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("run-graph-ablation", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    results = run_graph_ablation(config, root, logger=logger)
    logger.info("Graph ablation completed rows=%s", len(results))
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _consolidate_cases_v2(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.cases.case_consolidation_v2 import consolidate_cases_v2

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("consolidate-cases-v2", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    outputs = consolidate_cases_v2(config, root, logger=logger)
    logger.info("Consolidated cases v2 cases=%s mappings=%s", len(outputs["cases"]), len(outputs["mapping"]))
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _generate_reason_codes(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.explainability.shap_reason_mapper import generate_alert_reason_codes

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("generate-reason-codes", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    reasons = generate_alert_reason_codes(config, root, logger=logger)
    logger.info("Generated reason codes rows=%s", len(reasons))
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


def _build_remediation_report(config_path: Path, args: argparse.Namespace) -> None:
    from aml_mvp.reporting.remediation_report import build_remediation_report

    config = load_config(config_path)
    root = project_root_from_config(config_path)
    logger, log_path = setup_logging("build-remediation-report", root, args.log_level, args.log_file)
    start = time.perf_counter()
    logger.info("Command started config=%s project_root=%s log_file=%s", config_path, root, log_path)
    output = build_remediation_report(config, root, logger=logger)
    logger.info("Wrote remediation report path=%s", output)
    logger.info("Command completed elapsed=%.2fs", time.perf_counter() - start)


WORKFLOW_STEPS = {
    "validate-data": ["validate-data", "--config", "config/data_config.yaml"],
    "create-splits": ["create-splits", "--config", "config/data_config.yaml"],
    "run-rules": ["run-rules", "--config", "config/rule_config.yaml"],
    "tune-rules": ["tune-rules", "--config", "config/rule_config.yaml"],
    "build-features": ["build-features", "--config", "config/model_config.yaml"],
    "train-model": ["train-model", "--config", "config/model_config.yaml"],
    "prioritize-alerts": ["prioritize-alerts", "--config", "config/model_config.yaml"],
    "build-report": ["build-report", "--config", "config/report_config.yaml"],
    "extended-stress-test": ["extended-stress-test", "--config", "config/extended_config.yaml"],
    "build-graph-features": ["build-graph-features", "--config", "config/extended_config.yaml"],
    "run-graph-rules": ["run-graph-rules", "--config", "config/extended_config.yaml"],
    "train-extended-model": ["train-extended-model", "--config", "config/extended_config.yaml"],
    "compare-models": ["compare-models", "--config", "config/extended_config.yaml"],
    "consolidate-cases": ["consolidate-cases", "--config", "config/extended_config.yaml"],
    "explain-alerts": ["explain-alerts", "--config", "config/extended_config.yaml"],
    "calibrate-scores": ["calibrate-scores", "--config", "config/extended_config.yaml"],
    "build-extended-report": ["build-extended-report", "--config", "config/extended_config.yaml"],
    "diagnose-alerts": ["diagnose-alerts", "--config", "config/priority_band_config.yaml"],
    "rebuild-priority-bands": ["rebuild-priority-bands", "--config", "config/priority_band_config.yaml"],
    "build-graph-v2": ["build-graph-v2", "--config", "config/graph_rule_config.yaml"],
    "run-graph-ablation": ["run-graph-ablation", "--config", "config/graph_rule_config.yaml"],
    "consolidate-cases-v2": ["consolidate-cases-v2", "--config", "config/case_consolidation_config.yaml"],
    "generate-reason-codes": ["generate-reason-codes", "--config", "config/reason_code_dictionary.yaml"],
    "build-remediation-report": ["build-remediation-report", "--config", "config/report_config.yaml"],
}

WORKFLOW_PRESETS = {
    "mvp": [
        "validate-data",
        "create-splits",
        "run-rules",
        "tune-rules",
        "build-features",
        "train-model",
        "prioritize-alerts",
        "build-report",
    ],
    "extended": [
        "extended-stress-test",
        "build-graph-features",
        "run-graph-rules",
        "train-extended-model",
        "compare-models",
        "consolidate-cases",
        "explain-alerts",
        "calibrate-scores",
        "build-extended-report",
    ],
    "remediation": [
        "diagnose-alerts",
        "rebuild-priority-bands",
        "build-graph-v2",
        "run-graph-ablation",
        "train-extended-model",
        "compare-models",
        "consolidate-cases-v2",
        "generate-reason-codes",
        "build-extended-report",
        "build-remediation-report",
    ],
}
WORKFLOW_PRESETS["full"] = WORKFLOW_PRESETS["mvp"] + WORKFLOW_PRESETS["extended"] + WORKFLOW_PRESETS["remediation"]


def _run_workflow(args: argparse.Namespace) -> None:
    steps = _selected_workflow_steps(args)
    log_path = _shared_workflow_log_path(args, "workflow")
    print("Selected workflow steps:")
    print(f"Combined log file: {log_path}")
    for index, step in enumerate(steps, start=1):
        command = _workflow_command(step, args.log_level, log_path)
        print(f"{index}. {step}: {' '.join(command)}")
    if args.dry_run:
        return
    for step in steps:
        command = _workflow_command(step, args.log_level, log_path)
        print(f"\n=== Running {step} ===", flush=True)
        subprocess.run(command, check=True)


def _selected_workflow_steps(args: argparse.Namespace) -> list[str]:
    if args.steps:
        steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    else:
        steps = list(WORKFLOW_PRESETS[args.preset])

    unknown = [step for step in steps if step not in WORKFLOW_STEPS]
    if unknown:
        raise ValueError(f"Unknown workflow step(s): {', '.join(unknown)}")

    if args.from_step:
        if args.from_step not in steps:
            raise ValueError(f"--from-step must be one of the selected steps: {args.from_step}")
        steps = steps[steps.index(args.from_step) :]
    if args.to_step:
        if args.to_step not in steps:
            raise ValueError(f"--to-step must be one of the selected steps: {args.to_step}")
        steps = steps[: steps.index(args.to_step) + 1]
    return steps


def _run_remediation(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = load_config(config_path)
    root = project_root_from_config(config_path)
    workflow = dict(config.get("workflow", {}))
    step_config = dict(config.get("steps", {}))
    default_steps = list(workflow.get("default_steps", WORKFLOW_PRESETS["remediation"]))
    steps = _select_remediation_steps(default_steps, args)
    status_path = _resolve(root, workflow.get("status_path", "outputs/metrics/remediation_workflow_status.json"))
    continue_on_error = bool(workflow.get("continue_on_error", False))
    log_path = _shared_workflow_log_path(args, "remediation")
    statuses: list[dict[str, object]] = []
    print("Selected remediation steps:")
    print(f"Combined log file: {log_path}")
    for index, step in enumerate(steps, start=1):
        command_info = step_config.get(step, {})
        command = _remediation_command(step, command_info, args.log_level, log_path)
        print(f"{index}. {step}: {' '.join(command)}")
    if args.dry_run:
        return
    for step in steps:
        command_info = step_config.get(step, {})
        command = _remediation_command(step, command_info, args.log_level, log_path)
        start = time.perf_counter()
        row: dict[str, object] = {"step": step, "command": " ".join(command), "status": "running"}
        try:
            _check_required_inputs(root, command_info)
            subprocess.run(command, check=True)
            row["status"] = "completed"
            row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
            row["outputs"] = _artifact_status(root, command_info.get("expected_outputs", []))
        except Exception as exc:
            row["status"] = "failed"
            row["elapsed_seconds"] = round(time.perf_counter() - start, 3)
            row["error"] = str(exc)
            statuses.append(row)
            _write_status(status_path, statuses)
            if not continue_on_error:
                raise
            continue
        statuses.append(row)
        _write_status(status_path, statuses)


def _select_remediation_steps(default_steps: list[str], args: argparse.Namespace) -> list[str]:
    if args.steps:
        steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    else:
        steps = list(default_steps)
    if args.skip:
        skip = {step.strip() for step in args.skip.split(",") if step.strip()}
        steps = [step for step in steps if step not in skip]
    if args.from_step:
        if args.from_step not in steps:
            raise ValueError(f"--from-step must be one of the selected remediation steps: {args.from_step}")
        steps = steps[steps.index(args.from_step) :]
    unknown = [step for step in steps if step not in WORKFLOW_STEPS]
    if unknown:
        raise ValueError(f"Unknown remediation step(s): {', '.join(unknown)}")
    return steps


def _workflow_command(step: str, log_level: str, log_path: Path) -> list[str]:
    return [sys.executable, "-m", "aml_mvp.cli", *WORKFLOW_STEPS[step], "--log-level", log_level, "--log-file", str(log_path)]


def _remediation_command(step: str, command_info: dict[str, object], log_level: str, log_path: Path | None = None) -> list[str]:
    command_name = str(command_info.get("command", step))
    config_value = str(command_info.get("config", WORKFLOW_STEPS[step][2]))
    command = [sys.executable, "-m", "aml_mvp.cli", command_name, "--config", config_value, "--log-level", log_level]
    if log_path is not None:
        command.extend(["--log-file", str(log_path)])
    return command


def _shared_workflow_log_path(args: argparse.Namespace, command_prefix: str) -> Path:
    root = Path.cwd().resolve()
    preset = str(getattr(args, "preset", command_prefix))
    command_name = f"{command_prefix}-{preset}" if command_prefix == "workflow" else command_prefix
    return resolve_log_file(command_name, root, getattr(args, "log_file", None))


def _check_required_inputs(root: Path, command_info: dict[str, object]) -> None:
    missing = []
    for value in command_info.get("required_inputs", []) or []:
        path = _resolve(root, str(value))
        if not (path.exists() or path.with_suffix(path.suffix + ".pkl").exists()):
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing required remediation input(s): {', '.join(missing)}")


def _artifact_status(root: Path, values: object) -> list[dict[str, object]]:
    rows = []
    for value in values or []:
        path = _resolve(root, str(value))
        exists = path.exists() or path.with_suffix(path.suffix + ".pkl").exists()
        rows.append({"path": str(path), "exists": exists})
    return rows


def _write_status(path: Path, statuses: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"steps": statuses}, indent=2, default=str), encoding="utf-8")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _extended_model_artifacts(artifacts: dict[str, str]) -> dict[str, str]:
    return {
        "scored_alerts_path": artifacts["extended_scored_alerts_path"],
        "model_artifact_path": artifacts["extended_model_artifact_path"],
        "model_metrics_path": artifacts["extended_model_metrics_path"],
        "top_k_metrics_path": artifacts["extended_top_k_metrics_path"],
        "feature_importance_path": artifacts["extended_feature_importance_path"],
        "model_tuning_trials_path": artifacts["extended_model_tuning_trials_path"],
        "selected_features_path": artifacts["extended_selected_features_path"],
    }


def _require_artifact_keys(artifacts: dict[str, str], keys: list[str]) -> None:
    missing = [key for key in keys if key not in artifacts]
    if missing:
        raise ValueError(f"Missing required artifact config keys: {', '.join(missing)}")


def _read_preferred_scored_alerts(root: Path, artifacts: dict[str, str]):
    extended_path_value = artifacts.get("extended_scored_alerts_path")
    if extended_path_value:
        extended_path = _resolve(root, extended_path_value)
        if extended_path.exists() or extended_path.with_suffix(extended_path.suffix + ".pkl").exists():
            return read_dataframe(extended_path)
    return read_dataframe(_resolve(root, artifacts["scored_alerts_path"]))


def _read_csv(path: Path):
    import pandas as pd
    from pandas.errors import EmptyDataError

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _split_counts(df) -> dict:
    if "split" not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df["split"].value_counts().to_dict().items()}


if __name__ == "__main__":
    main()
