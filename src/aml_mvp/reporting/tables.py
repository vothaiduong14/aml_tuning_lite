"""HTML table helpers for the MVP report."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    """Render a compact HTML table with escaped values."""

    if df is None or df.empty:
        return '<p class="muted">No data available.</p>'

    view = df.head(max_rows).copy()
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in view.columns)
    body_rows = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{html.escape(_format_value(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    overflow = ""
    if len(df) > max_rows:
        overflow = f'<caption>Showing {max_rows} of {len(df)} rows</caption>'
    return f'<table class="data-table">{overflow}<thead><tr>{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def key_value_table(mapping: dict[str, Any]) -> str:
    rows = []
    for key, value in mapping.items():
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(_format_value(value))}</td>"
            "</tr>"
        )
    return f'<table class="kv-table"><tbody>{"".join(rows)}</tbody></table>'


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return str(value)
    if pd.isna(value):
        return ""
    return str(value)

