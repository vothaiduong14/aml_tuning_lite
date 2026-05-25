from __future__ import annotations

from argparse import Namespace

import pytest

from aml_mvp.cli import _selected_workflow_steps


def test_workflow_full_preset_uses_dependency_order() -> None:
    steps = _selected_workflow_steps(Namespace(preset="full", steps=None, from_step=None, to_step=None))

    assert steps.index("build-graph-features") < steps.index("run-graph-rules")
    assert steps.index("run-graph-rules") < steps.index("consolidate-cases")
    assert steps.index("train-extended-model") < steps.index("compare-models")
    assert steps[-1] == "build-extended-report"


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

