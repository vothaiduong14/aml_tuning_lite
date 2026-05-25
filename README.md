# AML Rules and ML Triage MVP

Greenfield MVP implementation for AML rule development, alert generation, and later ML-based alert prioritization.

This package is intentionally isolated from the existing root-level `aml.py`. Existing code can be used as reference, but implementation should happen under this folder.

## Current Implemented Scope

- MVP Phase 1: package skeleton, configs, CLI, smoke-testable imports
- MVP Phase 2: local CSV ingestion, standard transaction schema, data quality report
- MVP Phase 3: temporal train/validation/test split and dataset profiling
- MVP Phase 4: R1/R2 rule engine and alert consolidation
- MVP Phase 5: R3-R6 rule expansion
- MVP Phase 6: R1 threshold tuning with guardrail audit artifacts
- MVP Phase 7: alert-level feature matrix
- MVP Phase 8: ML triage training, scoring, and top-K evaluation
- MVP Phase 9: priority banding and capacity simulation
- MVP Phase 10: standalone HTML report and handover artifacts
- Extended E1: LI-only stress-test summary
- Extended E2: graph feature generation and graph-augmented alert feature matrix
- Extended E3: graph rule hits for gather-scatter and short-cycle candidates
- Extended E4: case consolidation
- Extended E5: LightGBM/SHAP explainability and reason codes
- Extended E6: score calibration and calibrated priority bands
- Extended E7: extended standalone HTML report

## MVP Commands

Run commands from this repo root with the Pixi environment:

```bash
pixi run test
pixi run validate-data
pixi run create-splits
pixi run run-rules
pixi run tune-rules
pixi run build-features
pixi run train-model
pixi run prioritize-alerts
pixi run build-report
```

Equivalent direct commands:

```bash
PYTHONPATH=src pixi run python -m aml_mvp.cli validate-data --config config/data_config.yaml
PYTHONPATH=src pixi run python -m aml_mvp.cli create-splits --config config/data_config.yaml
PYTHONPATH=src pixi run python -m aml_mvp.cli run-rules --config config/rule_config.yaml
PYTHONPATH=src pixi run python -m aml_mvp.cli tune-rules --config config/rule_config.yaml
PYTHONPATH=src pixi run python -m aml_mvp.cli build-features --config config/model_config.yaml
PYTHONPATH=src pixi run python -m aml_mvp.cli train-model --config config/model_config.yaml
PYTHONPATH=src pixi run python -m aml_mvp.cli prioritize-alerts --config config/model_config.yaml
PYTHONPATH=src pixi run python -m aml_mvp.cli build-report --config config/report_config.yaml
```

To see detailed long-running rule progress:

```bash
pixi run python -m aml_mvp.cli run-rules --config config/rule_config.yaml --log-level DEBUG
pixi run python -m aml_mvp.cli tune-rules --config config/rule_config.yaml --log-file outputs/run_logs/tuning_debug.log
```

Logs are written to `outputs/run_logs/` by default.

## Extended Commands

The extended build is additive. It reads MVP artifacts and writes separate extended artifacts so the MVP output contracts remain stable.

```bash
pixi run extended-stress-test
pixi run build-graph-features
pixi run train-extended-model
pixi run compare-models
pixi run run-graph-rules
pixi run consolidate-cases
pixi run explain-alerts
pixi run calibrate-scores
pixi run build-extended-report
```

Full workflow from raw data through the extended report:

```bash
pixi run validate-data
pixi run create-splits
pixi run run-rules
pixi run tune-rules
pixi run build-features
pixi run train-model
pixi run prioritize-alerts
pixi run build-report
pixi run extended-stress-test
pixi run build-graph-features
pixi run train-extended-model
pixi run compare-models
pixi run run-graph-rules
pixi run consolidate-cases
pixi run explain-alerts
pixi run calibrate-scores
pixi run build-extended-report
```

## Extended Model Training

Yes, model training must be rerun if you want alert scores to use newly created features. The repo now has an explicit extended challenger path for this.

Important current behavior:

- `pixi run build-features` creates the MVP feature matrix at `data/processed/alert_features.parquet`.
- `pixi run train-model` currently reads `config/model_config.yaml`, which points to `data/processed/alert_features.parquet`.
- `pixi run build-graph-features` creates graph outputs at `data/processed/graph_features.parquet` and a graph-augmented matrix at `data/processed/extended_alert_features.parquet`.
- `pixi run train-extended-model` trains a separate challenger on `data/processed/extended_alert_features.parquet`.
- `pixi run compare-models` compares MVP vs extended metrics and writes the promotion decision.

Both MVP and extended model training now apply the same training controls:

- label-aware imbalance handling with all positives preserved and negatives downsampled to at most `10:1`
- model-importance feature selection capped at the top 25 selected features
- Optuna tuning for the LightGBM challenger
- balanced logistic regression baseline
- champion selection by validation `Precision@1000`
- test split reserved for final reporting, not tuning or feature selection

Extended model artifacts:

- `data/processed/extended_scored_alerts.parquet`
- `outputs/models/extended_model.pkl`
- `outputs/extended/extended_model_metrics.json`
- `outputs/extended/extended_top_k_metrics.csv`
- `outputs/extended/extended_feature_importance.csv`
- `outputs/extended/extended_model_tuning_trials.csv`
- `outputs/extended/extended_selected_features.csv`
- `outputs/extended/model_comparison.csv`
- `outputs/extended/model_selection.json`

Recommended ordering:

- For the stable MVP baseline: run `build-features`, `train-model`, and `prioritize-alerts`.
- For a graph-enhanced challenger: run `build-graph-features`, then `train-extended-model`, then `compare-models`.
- Promote the extended model only if `outputs/extended/model_selection.json` selects `extended`.
- `consolidate-cases`, `explain-alerts`, and `calibrate-scores` prefer `data/processed/extended_scored_alerts.parquet` when it exists; otherwise they fall back to the MVP score artifact.

## Data

The default config points to `data/raw/LI-Small_Trans.csv`. Raw data is local-only and should not be newly committed.
