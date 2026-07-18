"""Fit the locked outer-fold VotingClassifier and save direct predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from fold_isolated_pipeline import (
    RANDOM_STATE,
    RESULTS,
    build_or_load_split_matrices,
    indices_sha256,
    load_base,
    make_voting_classifier,
    metrics,
)
from nested_protocol import outer_splits


MODEL_NAMES = ("hgb", "lgb", "xgb", "cb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["stratified", "group", "temporal"], required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--threads-per-estimator", type=int, default=5)
    parser.add_argument("--voting-jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def prediction_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes(order="C")).hexdigest()


def load_hp(scheme: str, outer_fold: int, outer_train: np.ndarray, outer_valid: np.ndarray) -> dict:
    hp = {}
    for name in MODEL_NAMES:
        path = RESULTS / "optuna" / f"leakage_safe_{scheme}_outer{outer_fold}_{name}_best.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload["protocol"]
        if protocol["outer_train_indices_sha256"] != indices_sha256(outer_train):
            raise ValueError(f"Outer-train hash mismatch in {path.name}")
        if protocol["outer_valid_indices_sha256"] != indices_sha256(outer_valid):
            raise ValueError(f"Outer-validation hash mismatch in {path.name}")
        if protocol["random_seed"] != RANDOM_STATE:
            raise ValueError(f"Random seed mismatch in {path.name}")
        hp[name] = payload["best_params"]
    return hp


def main() -> None:
    args = parse_args()
    if args.threads_per_estimator < 1 or args.voting_jobs < 1:
        raise ValueError("Thread counts must be positive")
    os.environ["OMP_NUM_THREADS"] = str(args.threads_per_estimator)
    os.environ["MKL_NUM_THREADS"] = str(args.threads_per_estimator)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads_per_estimator)

    out_dir = RESULTS / "nested_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scheme}_outer{args.outer_fold}"
    pred_path = out_dir / f"{stem}_predictions.csv.gz"
    metric_path = out_dir / f"{stem}_metrics.csv"
    manifest_path = out_dir / f"{stem}_manifest.json"
    if pred_path.exists() and metric_path.exists() and manifest_path.exists() and not args.force:
        print("EXISTS", flush=True)
        return

    x_base, y, groups, meta = load_base()
    splits = outer_splits(args.scheme, y, groups, meta)
    train_idx, valid_idx = splits[args.outer_fold - 1]
    hp = load_hp(args.scheme, args.outer_fold, train_idx, valid_idx)
    x_train, x_valid = build_or_load_split_matrices(
        x_base,
        meta,
        train_idx,
        valid_idx,
        f"outer_{args.scheme}_fold{args.outer_fold}",
        force=False,
    )
    model = make_voting_classifier(
        hp,
        voting_jobs=args.voting_jobs,
        threads_per_estimator=args.threads_per_estimator,
    )
    model.fit(x_train, y[train_idx])
    predictions = {
        name: model.named_estimators_[name].predict_proba(x_valid)[:, 1]
        for name in MODEL_NAMES
    }
    predictions["voting"] = model.predict_proba(x_valid)[:, 1]
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Non-finite prediction detected")

    frame = pd.DataFrame(
        {
            "row_index": valid_idx,
            "y_true": y[valid_idx],
            **{f"p_{name}": values for name, values in predictions.items()},
        }
    )
    frame.to_csv(pred_path, index=False, compression="gzip")
    rows = []
    for name, values in predictions.items():
        rows.append(
            {
                "scheme": args.scheme,
                "outer_fold": args.outer_fold,
                "model": name,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                **metrics(y[valid_idx], values),
                "prediction_sha256": prediction_sha256(values),
            }
        )
    pd.DataFrame(rows).to_csv(metric_path, index=False, encoding="utf-8-sig")
    manifest = {
        "scheme": args.scheme,
        "outer_fold": args.outer_fold,
        "random_seed": RANDOM_STATE,
        "model": "sklearn VotingClassifier(voting='soft')",
        "single_model_predictions": "fitted estimators inside the same VotingClassifier",
        "voting_prediction_source": "VotingClassifier.predict_proba",
        "threads_per_estimator": args.threads_per_estimator,
        "voting_jobs": args.voting_jobs,
        "train_indices_sha256": indices_sha256(train_idx),
        "valid_indices_sha256": indices_sha256(valid_idx),
        "prediction_file": str(pred_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
