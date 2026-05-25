"""DataFrame storage helpers with a Parquet-first interface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame, preferring Parquet and falling back to pickle.

    Pandas needs an optional engine such as pyarrow for Parquet. The fallback
    keeps Phase 1-3 runnable in the current Pixi environment while preserving
    the intended target path when the engine is available.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
        return output_path
    except ImportError:
        fallback_path = output_path.with_suffix(output_path.suffix + ".pkl")
        df.to_pickle(fallback_path)
        return fallback_path


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Read a DataFrame written by `write_dataframe`."""

    input_path = Path(path)
    if input_path.exists():
        try:
            return pd.read_parquet(input_path)
        except ImportError:
            pass

    fallback_path = input_path.with_suffix(input_path.suffix + ".pkl")
    if fallback_path.exists():
        return pd.read_pickle(fallback_path)

    raise FileNotFoundError(f"No table found at {input_path} or {fallback_path}")

