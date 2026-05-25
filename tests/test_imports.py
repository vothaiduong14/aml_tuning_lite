from __future__ import annotations


def test_phase_1_imports() -> None:
    import aml_mvp
    from aml_mvp.data import load_data, profile, quality_checks, schema, splits

    assert aml_mvp.__version__ == "0.1.0"
    assert load_data is not None
    assert profile is not None
    assert quality_checks is not None
    assert schema is not None
    assert splits is not None

