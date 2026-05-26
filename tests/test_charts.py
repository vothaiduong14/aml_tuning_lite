from __future__ import annotations

import pandas as pd

from aml_mvp.reporting.charts import bar_chart_svg


def test_bar_chart_uses_wide_label_column_and_tooltip_for_long_labels() -> None:
    svg = bar_chart_svg(
        pd.DataFrame(
            [
                {
                    "feature_name": "feature_sender_prior_unique_receivers_with_extra_long_suffix",
                    "importance": 12.5,
                }
            ]
        ),
        "feature_name",
        "importance",
        "Feature Importance",
    )

    assert 'viewBox="0 0 980' in svg
    assert 'x="392"' in svg
    assert "<title>feature_sender_prior_unique_receivers_with_extra_long_suffix</title>" in svg
    assert "feature_sender_prior_unique_receivers_with_…" in svg
