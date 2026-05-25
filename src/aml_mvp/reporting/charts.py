"""Small dependency-free SVG charts for the MVP report."""

from __future__ import annotations

import html

import pandas as pd


def bar_chart_svg(df: pd.DataFrame, label_col: str, value_col: str, title: str, max_bars: int = 12) -> str:
    if df is None or df.empty or label_col not in df or value_col not in df:
        return '<p class="muted">No chart data available.</p>'

    chart_df = df[[label_col, value_col]].head(max_bars).copy()
    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce").fillna(0.0)
    max_value = float(chart_df[value_col].max()) or 1.0
    width = 760
    row_height = 30
    left_pad = 180
    bar_max = width - left_pad - 40
    height = 42 + row_height * len(chart_df)

    rows = [
        f'<text x="16" y="24" class="chart-title">{html.escape(title)}</text>',
    ]
    for i, row in enumerate(chart_df.itertuples(index=False), start=0):
        label = html.escape(str(getattr(row, label_col)))
        value = float(getattr(row, value_col))
        y = 44 + i * row_height
        bar_width = max(1.0, bar_max * value / max_value)
        rows.append(f'<text x="16" y="{y + 14}" class="chart-label">{label}</text>')
        rows.append(f'<rect x="{left_pad}" y="{y}" width="{bar_width:.2f}" height="18" rx="3"></rect>')
        rows.append(f'<text x="{left_pad + bar_width + 8:.2f}" y="{y + 14}" class="chart-value">{value:.4g}</text>')

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        + "".join(rows)
        + "</svg>"
    )

