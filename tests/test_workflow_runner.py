from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from aml_mvp.cli import _remediation_command, _selected_workflow_steps, _workflow_command


def test_workflow_full_preset_uses_dependency_order() -> None:
    steps = _selected_workflow_steps(Namespace(preset="full", steps=None, from_step=None, to_step=None))

    assert steps.index("build-report") < steps.index("extended-stress-test")
    assert steps.index("build-graph-features") < steps.index("run-graph-rules")
    assert steps.index("run-graph-rules") < steps.index("consolidate-cases")
    assert steps.index("train-extended-model") < steps.index("compare-models")
    assert steps.index("build-extended-report") < steps.index("diagnose-alerts")
    assert steps.index("build-graph-v2") < steps.index("run-graph-ablation")
    assert steps.index("run-graph-ablation") < steps.index("consolidate-cases-v2")
    assert steps.index("generate-reason-codes") < steps.index("build-remediation-report")
    assert steps[-1] == "build-remediation-report"


def test_workflow_remediation_preset_uses_dependency_order() -> None:
    steps = _selected_workflow_steps(Namespace(preset="remediation", steps=None, from_step=None, to_step=None))

    assert steps == [
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
    ]


def test_workflow_custom_steps_keep_requested_order() -> None:
    steps = _selected_workflow_steps(
        Namespace(
            preset="full",
            steps="build-graph-features,train-extended-model,compare-models",
            from_step=None,
            to_step=None,
        )
    )

    assert steps == ["build-graph-features", "train-extended-model", "compare-models"]


def test_workflow_from_to_slice() -> None:
    steps = _selected_workflow_steps(
        Namespace(preset="full", steps=None, from_step="build-graph-features", to_step="compare-models")
    )

    assert steps == ["build-graph-features", "run-graph-rules", "train-extended-model", "compare-models"]


def test_workflow_rejects_unknown_step() -> None:
    with pytest.raises(ValueError, match="Unknown workflow step"):
        _selected_workflow_steps(Namespace(preset="full", steps="not-a-step", from_step=None, to_step=None))


def test_workflow_command_passes_shared_log_file() -> None:
    log_path = Path("outputs/run_logs/workflow_full_20260527_000000.log")

    command = _workflow_command("validate-data", "INFO", log_path)

    assert "--log-file" in command
    assert str(log_path) in command


def test_remediation_command_passes_shared_log_file() -> None:
    log_path = Path("outputs/run_logs/remediation_20260527_000000.log")

    command = _remediation_command("diagnose-alerts", {}, "DEBUG", log_path)

    assert command[-2:] == ["--log-file", str(log_path)]
