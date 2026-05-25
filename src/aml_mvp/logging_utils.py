"""Logging setup for CLI workflows."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGER_ROOT = "aml_mvp"


def setup_logging(
    command_name: str,
    project_root: Path,
    log_level: str = "INFO",
    log_file: str | Path | None = None,
) -> tuple[logging.Logger, Path]:
    """Configure console and file logging for a CLI command."""

    level = parse_log_level(log_level)
    log_path = resolve_log_file(command_name, project_root, log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_ROOT)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_aml_mvp_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    stream_handler._aml_mvp_handler = True  # type: ignore[attr-defined]

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler._aml_mvp_handler = True  # type: ignore[attr-defined]

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    command_logger = logging.getLogger(f"{LOGGER_ROOT}.{command_name}")
    command_logger.setLevel(level)
    return command_logger, log_path


def parse_log_level(log_level: str) -> int:
    level_name = str(log_level).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {log_level}")
    return level


def resolve_log_file(command_name: str, project_root: Path, log_file: str | Path | None) -> Path:
    if log_file:
        path = Path(log_file)
        return path if path.is_absolute() else (project_root / path).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_command = command_name.replace("-", "_")
    return project_root / "outputs" / "run_logs" / f"{safe_command}_{timestamp}.log"

