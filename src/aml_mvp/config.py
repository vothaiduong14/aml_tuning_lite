"""Small config loader for the MVP YAML files.

The current root Pixi environment does not require PyYAML, so this loader
supports the intentionally simple config shape used by this package.
"""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    PyYAML is available in the Pixi environment and is required for the
    remediation v2 configs. The fallback keeps older shallow configs readable
    in very small environments.
    """

    config_path = Path(path)
    try:
        import yaml

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return loaded or {}
    except ImportError:
        pass

    result: dict[str, Any] = {}
    current_section: str | None = None

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1]
            result[current_section] = {}
            continue

        if ":" not in stripped:
            raise ValueError(f"Unsupported config line in {config_path}: {raw_line}")

        key, value = stripped.split(":", 1)
        parsed = _parse_scalar(value.strip())
        if indent == 0:
            result[key] = parsed
        else:
            if current_section is None:
                raise ValueError(f"Nested config value without section: {raw_line}")
            section = result.setdefault(current_section, {})
            if not isinstance(section, dict):
                raise ValueError(f"Config section is not a mapping: {current_section}")
            section[key] = parsed

    return result


def _parse_scalar(value: str) -> Any:
    if value == "":
        return {}
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return literal_eval(value)
        except (SyntaxError, ValueError):
            inner = value[1:-1].strip()
            return [] if not inner else [part.strip() for part in inner.split(",")]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def project_root_from_config(config_path: str | Path) -> Path:
    """Return the project root for a config path inside `config/`."""

    path = Path(config_path).resolve()
    return path.parent.parent
