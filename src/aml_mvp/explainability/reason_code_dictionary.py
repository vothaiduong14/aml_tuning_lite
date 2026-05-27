"""Reason-code dictionary loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aml_mvp.config import load_config


def load_reason_code_dictionary(path: str | Path) -> dict[str, Any]:
    config = load_config(path)
    return {
        "defaults": dict(config.get("defaults", {})),
        "features": dict(config.get("features", {})),
        "rule_flags": dict(config.get("rule_flags", {})),
        "artifacts": dict(config.get("artifacts", {})),
    }

