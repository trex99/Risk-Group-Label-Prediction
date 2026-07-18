"""Leakage-safe stepwise, ablation, and temporal-time sensitivity analyses."""

from __future__ import annotations

import argparse
import gc
import json
import os

import numpy as np
import pandas as pd

from fold_isolated_pipeline import (
    ABSOLUTE_TIME_COLS,
    RESULTS,
    build_or_load_split_matrices,
    feature_groups,
    load_base,
    make_voting_classifier,
    metrics,
    relative_time_matrix,
)
from nested_protocol import outer_splits
from run_outer_evaluation import load_hp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["stepwise", "ablation", "stratified_time", "temporal_time"],
        required=True,
    )
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--threads-per-estimator", type=int, default=5)
    parser.add_argument("--voting-jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def conditions(task: str, x_train: pd.DataFrame) -> list[tuple[str, list[str] | None]]:
    groups = feature_groups(x_train.columns)
    if task == "stepwise":
        current = groups["accuracy_error"] + groups["response_time"]
        return [
            ("model_a_age_time", groups["age_time"]),
            ("model_b_age_time_history", groups["age_time"] + groups["history"]),
            ("model_c_current_cognitive_response", current),
            ("model_d_full", list(x_train.columns)),
        ]
    if task == "ablation":
        return [
            ("full", list(x_train.columns)),
            ("without_accuracy_error", [c for c in x_train if c not in groups["accuracy_error"]]),
            ("without_age_time", [c for c in x_train if c not in groups["age_time"]]),
            ("without_history", [c for c in x_train if c not in groups["history"]]),
            ("without_missingness", [c for c in x_train if c not in groups["missingness"]]),
            ("without_response_time", [c for c in x_train if c not in groups["response_time"]]),
        ]
    if task in {"stratified_time", "temporal_time"}:
        return [
            ("full", list(x_train.columns)),
            ("without_absolute_time", [c for c in x_train if c not in ABSOLUTE_TIME_COLS]),
            ("relative_time", None),
        ]
    raise ValueError(task)


def main() -> None:
    args = parse_args()
    scheme = "temporal" if args.task == "temporal_time" else "stratified"
    if args.task == "temporal_time" and args.outer_fold != 1:
        raise ValueError("temporal_time has only outer fold 1")
    os.environ["OMP_NUM_THREADS"] = str(args.threads_per_estimator)
    os.environ["MKL_NUM_THREADS"] = str(args.threads_per_estimator)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads_per_estimator)
    x_base, y, groups, meta = load_base()
    train_idx, valid_idx = outer_splits(scheme, y, groups, meta)[args.outer_fold - 1]
    hp = load_hp(scheme, args.outer_fold, train_idx, valid_idx)
    x_train, x_valid = build_or_load_split_matrices(
        x_base,
        meta,
        train_idx,
        valid_idx,
        f"outer_{scheme}_fold{args.outer_fold}",
        force=False,
    )
    out_dir = RESULTS / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition, columns in conditions(args.task, x_train):
        pred_path = out_dir / f"{args.task}_outer{args.outer_fold}_{condition}_predictions.csv.gz"
        if condition in {"full", "model_d_full"}:
            source = RESULTS / "nested_evaluation" / f"{scheme}_outer{args.outer_fold}_predictions.csv.gz"
            frame = pd.read_csv(source, usecols=["row_index", "y_true", "p_voting"])
            pred = frame["p_voting"].to_numpy()
        elif pred_path.exists() and not args.force:
            frame = pd.read_csv(pred_path)
            pred = frame["p_voting"].to_numpy()
        else:
            if condition == "relative_time":
                fit_train = relative_time_matrix(x_train)
                fit_valid = relative_time_matrix(x_valid)
            else:
                fit_train = x_train[columns]
                fit_valid = x_valid[columns]
            model = make_voting_classifier(
                hp,
                voting_jobs=args.voting_jobs,
                threads_per_estimator=args.threads_per_estimator,
            )
            model.fit(fit_train, y[train_idx])
            pred = model.predict_proba(fit_valid)[:, 1]
            frame = pd.DataFrame({"row_index": valid_idx, "y_true": y[valid_idx], "p_voting": pred})
            frame.to_csv(pred_path, index=False, compression="gzip")
            del model, fit_train, fit_valid
            gc.collect()
        rows.append(
            {
                "task": args.task,
                "scheme": scheme,
                "outer_fold": args.outer_fold,
                "condition": condition,
                "n_features": len(relative_time_matrix(x_train.iloc[:1]).columns) if condition == "relative_time" else len(columns),
                **metrics(y[valid_idx], pred),
            }
        )
    pd.DataFrame(rows).to_csv(
        out_dir / f"{args.task}_outer{args.outer_fold}_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "task": args.task,
        "scheme": scheme,
        "outer_fold": args.outer_fold,
        "model": "direct sklearn VotingClassifier(voting='soft')",
        "hyperparameters": "locked full-feature inner-Optuna parameters for the same outer fold",
        "random_seed": 42,
    }
    (out_dir / f"{args.task}_outer{args.outer_fold}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
