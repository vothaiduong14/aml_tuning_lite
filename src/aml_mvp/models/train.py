"""Train alert-level ML triage models and produce score artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aml_mvp.models.evaluate import pr_auc, roc_auc, top_k_table
from aml_mvp.storage import write_dataframe


def train_and_score_alerts(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train baseline/challenger models and score alert features."""

    model_config = dict(config.get("model", {}))
    k_values = [int(value) for value in model_config.get("k_values", [100, 500, 1000])]
    random_seed = int(model_config.get("random_seed", 42))

    model_frame, feature_columns = prepare_model_frame(features)
    raw_train_frame = _training_frame(model_frame)
    train_frame, imbalance_metadata = balance_training_frame(raw_train_frame, model_config)

    selected_features, selected_features_table = select_model_features(train_frame, feature_columns, model_config)
    tuning_frame = _selection_frame(model_frame)
    models, tuning_trials, tuning_metadata = _fit_models(train_frame, tuning_frame, selected_features, model_config)
    champion_name, champion_selection = select_champion_model(models, model_frame, selected_features, model_config)
    champion_model = models[champion_name]

    scored = features.copy()
    encoded_all = model_frame[selected_features]
    for model_name, model in models.items():
        scored[f"score_{model_name}"] = _predict_positive_score(model, encoded_all)
    scored["model_score"] = scored[f"score_{champion_name}"]

    metrics, top_k_metrics = evaluate_rankings(scored, k_values, champion_name)
    metrics["training"] = {
        "model_candidates": list(models.keys()),
        "imbalance": imbalance_metadata,
        "feature_selection": {
            "enabled": bool(model_config.get("feature_selection_enabled", True)),
            "method": model_config.get("feature_selection_method", "model_importance_top_k"),
            "selector_model": model_config.get("feature_selector_model", "lightgbm"),
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "force_include_feature_prefixes": model_config.get("force_include_feature_prefixes", []),
        },
        "tuning": tuning_metadata,
        "champion_selection": champion_selection,
    }
    feature_importance = build_feature_importance(champion_model, selected_features, champion_name)
    artifact = {
        "champion_name": champion_name,
        "feature_columns": selected_features,
        "all_feature_columns": feature_columns,
        "selected_features_table": selected_features_table,
        "tuning_trials": tuning_trials,
        "training_metadata": metrics["training"],
        "models": models,
    }
    return scored, metrics, top_k_metrics, feature_importance, artifact


def prepare_model_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = [column for column in features.columns if column.startswith("feature_")]
    categorical_columns = [column for column in ["payment_format", "currency_pair", "max_rule_severity"] if column in features]
    model_frame = features[["alert_id", "split", "target"] + feature_columns + categorical_columns].copy()
    model_frame[feature_columns] = model_frame[feature_columns].fillna(0.0)
    encoded = pd.get_dummies(model_frame, columns=categorical_columns, dummy_na=True)
    encoded_feature_columns = [
        column
        for column in encoded.columns
        if column.startswith("feature_")
        or column.startswith("payment_format_")
        or column.startswith("currency_pair_")
        or column.startswith("max_rule_severity_")
    ]
    return encoded, encoded_feature_columns


def evaluate_rankings(scored: pd.DataFrame, k_values: list[int], champion_name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    eval_split = "test" if scored["split"].eq("test").any() else "validation"
    eval_frame = scored[scored["split"].eq(eval_split)].copy()
    if eval_frame.empty:
        eval_frame = scored.copy()
        eval_split = "all"

    rankings = {
        "rule_priority": eval_frame["feature_rule_priority_score"].fillna(0.0),
        "amount_rank": eval_frame["feature_amount"].fillna(0.0),
        "model_score": eval_frame["model_score"].fillna(0.0),
    }
    top_k_metrics = top_k_table(eval_frame["target"], rankings, k_values, eval_split)
    metrics = {
        "population": "alert_level",
        "model_name": champion_name,
        "evaluation_split": eval_split,
        "metrics": {
            "pr_auc": pr_auc(eval_frame["target"], eval_frame["model_score"]),
            "roc_auc": roc_auc(eval_frame["target"], eval_frame["model_score"]),
        },
        "baselines": {
            "rule_priority": {
                "pr_auc": pr_auc(eval_frame["target"], rankings["rule_priority"]),
                "roc_auc": roc_auc(eval_frame["target"], rankings["rule_priority"]),
            },
            "amount_rank": {
                "pr_auc": pr_auc(eval_frame["target"], rankings["amount_rank"]),
                "roc_auc": roc_auc(eval_frame["target"], rankings["amount_rank"]),
            },
        },
    }
    return metrics, top_k_metrics


def build_feature_importance(model, feature_columns: list[str], model_name: str) -> pd.DataFrame:
    estimator = model.named_steps.get("model") if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])
    else:
        values = [0.0] * len(feature_columns)
    return (
        pd.DataFrame({"model_name": model_name, "feature_name": feature_columns, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def write_model_outputs(
    scored_alerts: pd.DataFrame,
    metrics: dict[str, Any],
    top_k_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model_artifact: dict[str, Any],
    artifacts: dict[str, str],
    root: Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs["scored_alerts"] = write_dataframe(scored_alerts, _resolve(root, artifacts["scored_alerts_path"]))

    model_path = _resolve(root, artifacts["model_artifact_path"])
    metrics_path = _resolve(root, artifacts["model_metrics_path"])
    top_k_path = _resolve(root, artifacts["top_k_metrics_path"])
    feature_importance_path = _resolve(root, artifacts["feature_importance_path"])
    tuning_trials_path = _resolve(root, artifacts["model_tuning_trials_path"]) if "model_tuning_trials_path" in artifacts else None
    selected_features_path = _resolve(root, artifacts["selected_features_path"]) if "selected_features_path" in artifacts else None
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    top_k_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(model_artifact, file)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    top_k_metrics.to_csv(top_k_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)
    if tuning_trials_path:
        tuning_trials_path.parent.mkdir(parents=True, exist_ok=True)
        model_artifact.get("tuning_trials", pd.DataFrame()).to_csv(tuning_trials_path, index=False)
        outputs["model_tuning_trials"] = tuning_trials_path
    if selected_features_path:
        selected_features_path.parent.mkdir(parents=True, exist_ok=True)
        model_artifact.get("selected_features_table", pd.DataFrame()).to_csv(selected_features_path, index=False)
        outputs["selected_features"] = selected_features_path
    outputs["model_artifact"] = model_path
    outputs["model_metrics"] = metrics_path
    outputs["top_k_metrics"] = top_k_path
    outputs["feature_importance"] = feature_importance_path
    return outputs


def balance_training_frame(train_frame: pd.DataFrame, model_config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    strategy = str(model_config.get("imbalance_strategy", "class_weight_undersample"))
    ratio = int(model_config.get("negative_to_positive_ratio", 10))
    preserve_all_positives = bool(model_config.get("preserve_all_positives", True))
    random_seed = int(model_config.get("sampling_random_seed", model_config.get("random_seed", 42)))
    max_train_rows = int(model_config.get("max_train_rows", 200000))

    positives = train_frame[train_frame["target"].astype(int).eq(1)]
    negatives = train_frame[train_frame["target"].astype(int).eq(0)]
    original_positive_count = int(len(positives))
    original_negative_count = int(len(negatives))

    if strategy != "class_weight_undersample" or positives.empty or negatives.empty:
        sampled = _cap_training_rows(train_frame, max_train_rows, random_seed)
    else:
        max_negatives = min(len(negatives), max(original_positive_count * ratio, 1))
        sampled_negatives = negatives.sample(max_negatives, random_state=random_seed) if len(negatives) > max_negatives else negatives
        sampled = pd.concat([positives, sampled_negatives], ignore_index=False) if preserve_all_positives else pd.concat(
            [positives, sampled_negatives],
            ignore_index=False,
        )
        sampled = _cap_training_rows(sampled, max_train_rows, random_seed, preserve_positive_rows=preserve_all_positives)

    sampled = sampled.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    sampled_positive_count = int(sampled["target"].astype(int).sum())
    sampled_negative_count = int(len(sampled) - sampled_positive_count)
    metadata = {
        "strategy": strategy,
        "negative_to_positive_ratio": ratio,
        "preserve_all_positives": preserve_all_positives,
        "original_positive_count": original_positive_count,
        "original_negative_count": original_negative_count,
        "sampled_positive_count": sampled_positive_count,
        "sampled_negative_count": sampled_negative_count,
        "sampled_negative_to_positive_ratio": float(sampled_negative_count / sampled_positive_count) if sampled_positive_count else None,
        "max_train_rows": max_train_rows,
    }
    return sampled, metadata


def select_model_features(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    model_config: dict[str, Any],
) -> tuple[list[str], pd.DataFrame]:
    enabled = bool(model_config.get("feature_selection_enabled", True))
    top_k = int(model_config.get("selected_feature_count", 25))
    random_seed = int(model_config.get("random_seed", 42))
    if not enabled or len(feature_columns) <= top_k or train_frame["target"].nunique() < 2:
        rows = [
            {"feature_name": feature, "selection_rank": rank, "selection_importance": 0.0, "selected": True}
            for rank, feature in enumerate(feature_columns, start=1)
        ]
        return feature_columns, pd.DataFrame(rows)

    X_train = train_frame[feature_columns]
    y_train = train_frame["target"].astype(int)
    selector_model = str(model_config.get("feature_selector_model", "lightgbm"))
    selector = _new_selector_classifier(selector_model, random_seed)
    sample_weight = _balanced_sample_weight(y_train)
    try:
        selector.fit(X_train, y_train, sample_weight=sample_weight)
        importances = getattr(selector, "feature_importances_", np.zeros(len(feature_columns)))
    except Exception:
        fallback = RandomForestClassifier(n_estimators=75, random_state=random_seed, class_weight="balanced_subsample", n_jobs=-1)
        fallback.fit(X_train, y_train, sample_weight=sample_weight)
        importances = getattr(fallback, "feature_importances_", np.zeros(len(feature_columns)))

    ranking = (
        pd.DataFrame({"feature_name": feature_columns, "selection_importance": importances})
        .sort_values(["selection_importance", "feature_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranking["selection_rank"] = ranking.index + 1
    forced_prefixes = [str(prefix) for prefix in model_config.get("force_include_feature_prefixes", [])]
    ranking["forced_include"] = ranking["feature_name"].map(
        lambda feature_name: any(str(feature_name).startswith(prefix) for prefix in forced_prefixes)
    )
    ranking["selected"] = ranking["selection_rank"].le(top_k) | ranking["forced_include"]
    selected = ranking.loc[ranking["selected"], "feature_name"].tolist()
    return selected, ranking[["feature_name", "selection_rank", "selection_importance", "forced_include", "selected"]]


def select_champion_model(
    models: dict[str, Any],
    model_frame: pd.DataFrame,
    feature_columns: list[str],
    model_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    metric = str(model_config.get("champion_selection_metric", "precision_at_k"))
    k = int(model_config.get("champion_selection_k", 1000))
    selection_frame, selection_split, fallback_used = _selection_frame(model_frame)

    rows = []
    for model_name, model in models.items():
        scores = pd.Series(_predict_positive_score(model, selection_frame[feature_columns]), index=selection_frame.index)
        value = _selection_metric(selection_frame["target"], scores, metric, k)
        rows.append(
            {
                "model_name": model_name,
                "selection_metric": metric,
                "selection_k": k,
                "selection_split": selection_split,
                "metric_value": value,
                "pr_auc": pr_auc(selection_frame["target"], scores),
                "roc_auc": roc_auc(selection_frame["target"], scores),
            }
        )
    table = pd.DataFrame(rows).sort_values(["metric_value", "pr_auc", "model_name"], ascending=[False, False, True])
    champion_name = str(table.iloc[0]["model_name"])
    return champion_name, {
        "selected_model": champion_name,
        "selection_metric": metric,
        "selection_k": k,
        "selection_split": selection_split,
        "fallback_used": fallback_used,
        "candidates": rows,
    }


def _fit_models(
    train_frame: pd.DataFrame,
    tuning_frame_info: tuple[pd.DataFrame, str, bool],
    feature_columns: list[str],
    model_config: dict[str, Any],
):
    random_seed = int(model_config.get("random_seed", 42))
    candidates = _model_candidates(model_config)
    X_train = train_frame[feature_columns]
    y_train = train_frame["target"].astype(int)
    if y_train.nunique() < 2:
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X_train, y_train)
        tuning_trials = pd.DataFrame()
        tuning_metadata = {"enabled": False, "reason": "single_class_training", "trial_count": 0, "best_params": {}}
        return {candidates[0]: dummy}, tuning_trials, tuning_metadata

    models: dict[str, Any] = {}
    tuning_frames: list[pd.DataFrame] = []
    tuning_metadata: dict[str, Any] = {"enabled": bool(model_config.get("tuning_enabled", True)), "models": {}}

    if "logistic_regression" in candidates:
        logistic = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=random_seed, solver="liblinear")),
            ]
        )
        logistic.fit(X_train, y_train)
        models["logistic_regression"] = logistic
        tuning_metadata["models"]["logistic_regression"] = {"enabled": False, "best_params": {"class_weight": "balanced", "solver": "liblinear"}}

    if "random_forest" in candidates:
        params = _random_forest_params(model_config, random_seed)
        random_forest = RandomForestClassifier(**params)
        random_forest.fit(X_train, y_train, sample_weight=_balanced_sample_weight(y_train))
        models["random_forest"] = random_forest
        tuning_metadata["models"]["random_forest"] = {"enabled": False, "best_params": params}

    if "lightgbm" in candidates:
        lightgbm_model, lightgbm_trials, lightgbm_metadata = _fit_lightgbm_candidate(
            train_frame,
            tuning_frame_info,
            feature_columns,
            model_config,
        )
        models["lightgbm"] = lightgbm_model
        tuning_frames.append(lightgbm_trials.assign(model_name="lightgbm") if not lightgbm_trials.empty else lightgbm_trials)
        tuning_metadata["models"]["lightgbm"] = lightgbm_metadata

    if "xgboost" in candidates:
        xgboost_model, xgboost_trials, xgboost_metadata = _fit_xgboost_candidate(
            train_frame,
            tuning_frame_info,
            feature_columns,
            model_config,
        )
        models["xgboost"] = xgboost_model
        tuning_frames.append(xgboost_trials.assign(model_name="xgboost") if not xgboost_trials.empty else xgboost_trials)
        tuning_metadata["models"]["xgboost"] = xgboost_metadata

    if not models:
        raise ValueError(f"No supported model candidates were configured: {candidates}")

    tuning_trials = pd.concat([frame for frame in tuning_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in tuning_frames) else pd.DataFrame()
    if "lightgbm" in tuning_metadata["models"]:
        tuning_metadata.update({key: value for key, value in tuning_metadata["models"]["lightgbm"].items() if key not in {"models"}})
    elif "xgboost" in tuning_metadata["models"]:
        tuning_metadata.update({key: value for key, value in tuning_metadata["models"]["xgboost"].items() if key not in {"models"}})
    else:
        tuning_metadata.update({"trial_count": 0, "best_params": {}})
    return models, tuning_trials, tuning_metadata


def _fit_lightgbm_candidate(
    train_frame: pd.DataFrame,
    tuning_frame_info: tuple[pd.DataFrame, str, bool],
    feature_columns: list[str],
    model_config: dict[str, Any],
):
    random_seed = int(model_config.get("random_seed", 42))
    tuning_enabled = bool(model_config.get("tuning_enabled", True))
    tuning_trials = int(model_config.get("tuning_trials", 50))
    X_train = train_frame[feature_columns]
    y_train = train_frame["target"].astype(int)
    tuning_frame, tuning_split, tuning_fallback = tuning_frame_info
    sample_weight = _balanced_sample_weight(y_train)
    if not tuning_enabled:
        params = _default_lightgbm_params(random_seed)
        model = _new_lightgbm_classifier(params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model, pd.DataFrame(), {"enabled": False, "trial_count": 0, "best_params": params}

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed))

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.70, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "random_state": random_seed,
            }
            model = _new_lightgbm_classifier(params)
            model.fit(X_train, y_train, sample_weight=sample_weight)
            scores = _predict_positive_score(model, tuning_frame[feature_columns])
            return _selection_metric(
                tuning_frame["target"],
                pd.Series(scores, index=tuning_frame.index),
                str(model_config.get("champion_selection_metric", "precision_at_k")),
                int(model_config.get("champion_selection_k", 1000)),
            )

        study.optimize(objective, n_trials=tuning_trials, show_progress_bar=False)
        best_params = dict(study.best_params)
        best_params["random_state"] = random_seed
        trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    except Exception:
        best_params = _default_lightgbm_params(random_seed)
        trials = pd.DataFrame()

    model = _new_lightgbm_classifier(best_params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    metadata = {
        "enabled": tuning_enabled,
        "backend": "optuna",
        "trial_count": int(len(trials)),
        "best_params": best_params,
        "objective_metric": model_config.get("champion_selection_metric", "precision_at_k"),
        "objective_k": int(model_config.get("champion_selection_k", 1000)),
        "objective_split": tuning_split,
        "objective_fallback_used": tuning_fallback,
    }
    return model, trials, metadata


def _fit_xgboost_candidate(
    train_frame: pd.DataFrame,
    tuning_frame_info: tuple[pd.DataFrame, str, bool],
    feature_columns: list[str],
    model_config: dict[str, Any],
):
    random_seed = int(model_config.get("random_seed", 42))
    tuning_enabled = bool(model_config.get("tuning_enabled", True))
    tuning_trials = int(model_config.get("tuning_trials", 50))
    feature_name_map = _xgboost_feature_name_map(feature_columns)
    X_train = _rename_xgboost_features(train_frame[feature_columns], feature_name_map)
    y_train = train_frame["target"].astype(int)
    tuning_frame, tuning_split, tuning_fallback = tuning_frame_info
    X_tuning = _rename_xgboost_features(tuning_frame[feature_columns], feature_name_map)
    sample_weight = _balanced_sample_weight(y_train)
    if not tuning_enabled:
        params = _default_xgboost_params(random_seed)
        model = _new_xgboost_classifier(params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return _FeatureNameAdapter(model, feature_name_map), pd.DataFrame(), {
            "enabled": False,
            "backend": "xgboost",
            "trial_count": 0,
            "best_params": params,
            "feature_name_sanitized": True,
        }

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed + 13))

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.70, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "random_state": random_seed,
            }
            model = _new_xgboost_classifier(params)
            model.fit(X_train, y_train, sample_weight=sample_weight)
            scores = _predict_positive_score(model, X_tuning)
            return _selection_metric(
                tuning_frame["target"],
                pd.Series(scores, index=tuning_frame.index),
                str(model_config.get("champion_selection_metric", "precision_at_k")),
                int(model_config.get("champion_selection_k", 1000)),
            )

        study.optimize(objective, n_trials=tuning_trials, show_progress_bar=False)
        best_params = dict(study.best_params)
        best_params["random_state"] = random_seed
        trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    except Exception:
        best_params = _default_xgboost_params(random_seed)
        trials = pd.DataFrame()

    model = _new_xgboost_classifier(best_params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    metadata = {
        "enabled": tuning_enabled,
        "backend": "optuna",
        "trial_count": int(len(trials)),
        "best_params": best_params,
        "objective_metric": model_config.get("champion_selection_metric", "precision_at_k"),
        "objective_k": int(model_config.get("champion_selection_k", 1000)),
        "objective_split": tuning_split,
        "objective_fallback_used": tuning_fallback,
        "feature_name_sanitized": True,
    }
    return _FeatureNameAdapter(model, feature_name_map), trials, metadata


def _model_candidates(model_config: dict[str, Any]) -> list[str]:
    configured = model_config.get("model_candidates")
    if configured:
        return [str(value) for value in configured]
    return ["logistic_regression", "lightgbm"]


def _random_forest_params(model_config: dict[str, Any], random_seed: int) -> dict[str, Any]:
    configured = dict(model_config.get("random_forest", {}))
    params = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
        "random_state": random_seed,
        "n_jobs": -1,
    }
    params.update(configured)
    return params


def _new_selector_classifier(selector_model: str, random_seed: int):
    if selector_model == "random_forest":
        return RandomForestClassifier(
            n_estimators=75,
            random_state=random_seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    if selector_model == "xgboost":
        return _new_xgboost_classifier(_default_xgboost_params(random_seed) | {"n_estimators": 75, "max_depth": 3})
    return _new_lightgbm_classifier(
        {
            "n_estimators": 75,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "random_state": random_seed,
        }
    )


class _FeatureNameAdapter:
    """Adapter for estimators that require sanitized column names."""

    def __init__(self, model, feature_name_map: dict[str, str]) -> None:
        self.model = model
        self.feature_name_map = feature_name_map

    @property
    def classes_(self):
        return getattr(self.model, "classes_", [])

    @property
    def feature_importances_(self):
        return getattr(self.model, "feature_importances_", [])

    def predict_proba(self, frame: pd.DataFrame):
        return self.model.predict_proba(_rename_xgboost_features(frame, self.feature_name_map))


def _xgboost_feature_name_map(feature_columns: list[str]) -> dict[str, str]:
    return {feature: f"f{index}_{_sanitize_xgboost_feature_name(feature)}" for index, feature in enumerate(feature_columns)}


def _sanitize_xgboost_feature_name(feature_name: str) -> str:
    safe = []
    for char in str(feature_name):
        if char.isalnum() or char == "_":
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe)


def _rename_xgboost_features(frame: pd.DataFrame, feature_name_map: dict[str, str]) -> pd.DataFrame:
    renamed = frame.rename(columns=feature_name_map).copy()
    return renamed[[feature_name_map[column] for column in feature_name_map]]


def _predict_positive_score(model, frame: pd.DataFrame):
    probabilities = np.asarray(model.predict_proba(frame))
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps"):
        classes = list(getattr(model.named_steps["model"], "classes_", []))
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return [0.0] * len(frame)


def _training_frame(model_frame: pd.DataFrame) -> pd.DataFrame:
    train_frame = model_frame[model_frame["split"].eq("train")].copy()
    return train_frame if not train_frame.empty else model_frame.copy()


def _selection_frame(model_frame: pd.DataFrame) -> tuple[pd.DataFrame, str, bool]:
    validation_frame = model_frame[model_frame["split"].eq("validation")].copy()
    if not validation_frame.empty:
        return validation_frame, "validation", False
    train_frame = model_frame[model_frame["split"].eq("train")].copy()
    if not train_frame.empty:
        return train_frame, "train", True
    return model_frame.copy(), "all", True


def _cap_training_rows(
    frame: pd.DataFrame,
    max_train_rows: int,
    random_seed: int,
    preserve_positive_rows: bool = True,
) -> pd.DataFrame:
    if len(frame) <= max_train_rows:
        return frame.copy()
    positives = frame[frame["target"].astype(int).eq(1)]
    negatives = frame[frame["target"].astype(int).eq(0)]
    if preserve_positive_rows and not positives.empty:
        negative_slots = max(max_train_rows - len(positives), 0)
        sampled_negatives = negatives.sample(min(len(negatives), negative_slots), random_state=random_seed)
        return pd.concat([positives, sampled_negatives], ignore_index=False)
    return frame.sample(max_train_rows, random_state=random_seed)


def _balanced_sample_weight(y_train: pd.Series) -> pd.Series:
    counts = y_train.value_counts().to_dict()
    total = len(y_train)
    weights = {label: total / (len(counts) * count) for label, count in counts.items() if count}
    return y_train.map(weights).astype(float)


def _selection_metric(y_true: pd.Series, scores: pd.Series, metric: str, k: int) -> float:
    from aml_mvp.models.evaluate import precision_at_k, recall_at_k

    if metric == "precision_at_k":
        return precision_at_k(y_true, scores, k)
    if metric == "recall_at_k":
        return recall_at_k(y_true, scores, k)
    if metric == "roc_auc":
        return roc_auc(y_true, scores)
    return pr_auc(y_true, scores)


def _new_lightgbm_classifier(params: dict[str, Any]):
    try:
        from lightgbm import LGBMClassifier

        model_params = _default_lightgbm_params(int(params.get("random_state", 42)))
        model_params.update(params)
        return LGBMClassifier(**model_params)
    except Exception:
        return GradientBoostingClassifier(random_state=int(params.get("random_state", 42)))


def _new_xgboost_classifier(params: dict[str, Any]):
    try:
        from xgboost import XGBClassifier

        model_params = _default_xgboost_params(int(params.get("random_state", 42)))
        model_params.update(params)
        return XGBClassifier(**model_params)
    except Exception:
        return GradientBoostingClassifier(random_state=int(params.get("random_state", 42)))


def _default_lightgbm_params(random_seed: int) -> dict[str, Any]:
    return {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": random_seed,
        "verbosity": -1,
    }


def _default_xgboost_params(random_seed: int) -> dict[str, Any]:
    return {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": random_seed,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
