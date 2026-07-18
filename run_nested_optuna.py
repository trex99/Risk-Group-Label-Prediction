"""Leakage-safe nested Optuna tuning for one outer split and one or more models."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import optuna

from fold_isolated_pipeline import (
    RANDOM_STATE,
    RESULTS,
    build_or_load_split_matrices,
    ensure_dirs,
    indices_sha256,
    load_base,
    metrics,
)
from nested_protocol import inner_splits, outer_splits, validate_splits


MODELS = ("hgb", "lgb", "xgb", "cb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["stratified", "group", "temporal"], required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--force-history", action="store_true")
    return parser.parse_args()


def suggest_params(model_name: str, trial: optuna.Trial, threads: int) -> dict:
    if model_name == "hgb":
        return {
            "random_state": RANDOM_STATE,
            "early_stopping": True,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            "max_bins": trial.suggest_int("max_bins", 64, 255),
            "validation_fraction": trial.suggest_float("validation_fraction", 0.05, 0.2),
        }
    if model_name == "lgb":
        return {
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "n_jobs": threads,
            "deterministic": True,
            "force_col_wise": True,
            "subsample_freq": 1,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", -1, 20),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        }
    if model_name == "xgb":
        return {
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
            "n_jobs": threads,
            "eval_metric": "logloss",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-1, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        }
    if model_name == "cb":
        return {
            "random_seed": RANDOM_STATE,
            "verbose": 0,
            "allow_writing_files": False,
            "thread_count": threads,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.5, 2.0),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        }
    raise ValueError(model_name)


def build_model(model_name: str, params: dict):
    if model_name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(**params)
    if model_name == "lgb":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params)
    if model_name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(**params)
    if model_name == "cb":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params)
    raise ValueError(model_name)


def completed_trials(study: optuna.Study) -> int:
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def prepare_inner_data(scheme: str, outer_fold: int, inner_n: int, force_history: bool):
    x_base, y, groups, meta = load_base()
    outer = outer_splits(scheme, y, groups, meta)
    if not 1 <= outer_fold <= len(outer):
        raise ValueError(f"outer-fold must be in 1..{len(outer)} for {scheme}")
    outer_train, outer_valid = outer[outer_fold - 1]
    inner = inner_splits(scheme, outer_train, y, groups, meta, outer_fold, inner_n)
    validate_splits(scheme, outer_train, outer_valid, inner, groups, meta)
    datasets = []
    split_hashes = []
    for inner_fold, (train_idx, valid_idx) in enumerate(inner, start=1):
        key = f"nested_{scheme}_outer{outer_fold}_inner{inner_fold}"
        x_train, x_valid = build_or_load_split_matrices(
            x_base, meta, train_idx, valid_idx, key, force=force_history
        )
        datasets.append((x_train, y[train_idx], x_valid, y[valid_idx]))
        split_hashes.append(
            {
                "fold": inner_fold,
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "train_indices_sha256": indices_sha256(train_idx),
                "valid_indices_sha256": indices_sha256(valid_idx),
            }
        )
    protocol = {
        "scheme": scheme,
        "outer_fold": outer_fold,
        "outer_train_rows": int(len(outer_train)),
        "outer_valid_rows": int(len(outer_valid)),
        "outer_train_indices_sha256": indices_sha256(outer_train),
        "outer_valid_indices_sha256": indices_sha256(outer_valid),
        "inner_splits": split_hashes,
        "feature_count": int(x_base.shape[1]),
        "history_rule": "training-partition sources only and source.TestDate < target.TestDate",
    }
    return datasets, protocol


def make_objective(model_name: str, datasets, threads: int):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(model_name, trial, threads)
        fold_scores = []
        fold_metrics = []
        for fold, (x_train, y_train, x_valid, y_valid) in enumerate(datasets, start=1):
            model = build_model(model_name, params)
            model.fit(x_train, y_train)
            pred = model.predict_proba(x_valid)[:, 1]
            values = metrics(y_valid, pred)
            fold_scores.append(values["Score"])
            fold_metrics.append({"fold": fold, **values})
            del model, pred
            gc.collect()
        trial.set_user_attr("fold_metrics", fold_metrics)
        return float(np.mean(fold_scores))

    return objective


def tune_model(model_name: str, datasets, protocol: dict, n_trials: int, threads: int) -> dict:
    scheme = protocol["scheme"]
    outer_fold = protocol["outer_fold"]
    out_dir = RESULTS / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"{scheme}_outer{outer_fold}_{model_name}.db"
    study_name = f"leakage_safe_{scheme}_outer{outer_fold}_{model_name}"
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{db_path.as_posix()}",
        study_name=study_name,
        load_if_exists=True,
    )
    study_protocol = {
        **protocol,
        "random_seed": RANDOM_STATE,
        "threads_per_model": threads,
        "sampler": "TPESampler",
        "objective": "mean competition Score across leakage-safe inner folds",
    }
    protocol_json = json.dumps(study_protocol, ensure_ascii=False, sort_keys=True)
    existing = study.user_attrs.get("protocol_json")
    if existing is not None and existing != protocol_json:
        raise ValueError(f"Study protocol mismatch: {study_name}")
    study.set_user_attr("protocol_json", protocol_json)
    remaining = max(0, n_trials - completed_trials(study))
    if remaining:
        study.optimize(
            make_objective(model_name, datasets, threads),
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    trials_path = out_dir / f"{study_name}_trials.csv"
    study.trials_dataframe(
        attrs=("number", "value", "state", "params", "user_attrs", "duration")
    ).to_csv(trials_path, index=False, encoding="utf-8-sig")
    best_params = suggest_params(model_name, study.best_trial, threads)
    payload = {
        "model": model_name,
        "study_name": study_name,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "n_completed_trials": int(completed_trials(study)),
        "best_params": best_params,
        "protocol": study_protocol,
        "trials_csv": str(trials_path),
    }
    best_path = out_dir / f"{study_name}_best.json"
    best_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    ensure_dirs()
    datasets, protocol = prepare_inner_data(
        args.scheme, args.outer_fold, args.inner_splits, args.force_history
    )
    summary = {}
    for model_name in args.models:
        summary[model_name] = tune_model(model_name, datasets, protocol, args.n_trials, args.threads)
        print(
            f"{args.scheme} outer {args.outer_fold} {model_name}: "
            f"best={summary[model_name]['best_value']:.9f} "
            f"trials={summary[model_name]['n_completed_trials']}",
            flush=True,
        )
    out = RESULTS / "optuna" / f"leakage_safe_{args.scheme}_outer{args.outer_fold}_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
