"""Combine sensitivity folds and recompute all metrics from predictions."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from fold_isolated_pipeline import RESULTS, metrics, topk_table


TASK_SCHEME = {
    "stepwise": "stratified",
    "ablation": "stratified",
    "stratified_time": "stratified",
    "temporal_time": "temporal",
}
CONDITIONS = {
    "stepwise": [
        "model_a_age_time",
        "model_b_age_time_history",
        "model_c_current_cognitive_response",
        "model_d_full",
    ],
    "ablation": [
        "full",
        "without_accuracy_error",
        "without_age_time",
        "without_history",
        "without_missingness",
        "without_response_time",
    ],
    "stratified_time": ["full", "without_absolute_time", "relative_time"],
    "temporal_time": ["full", "without_absolute_time", "relative_time"],
}


def prediction_path(task: str, fold: int, condition: str):
    scheme = TASK_SCHEME[task]
    if condition in {"full", "model_d_full"}:
        return RESULTS / "nested_evaluation" / f"{scheme}_outer{fold}_predictions.csv.gz", "p_voting"
    return RESULTS / "sensitivity" / f"{task}_outer{fold}_{condition}_predictions.csv.gz", "p_voting"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASK_SCHEME), required=True)
    args = parser.parse_args()
    folds = range(1, 2) if args.task == "temporal_time" else range(1, 6)
    out_dir = RESULTS / "sensitivity"
    fold_metrics = pd.concat(
        [pd.read_csv(out_dir / f"{args.task}_outer{fold}_metrics.csv") for fold in folds],
        ignore_index=True,
    )
    rows = []
    combined_predictions = {}
    for condition in CONDITIONS[args.task]:
        frames = []
        for fold in folds:
            path, probability_col = prediction_path(args.task, fold, condition)
            usecols = ["row_index", "y_true", probability_col]
            frame = pd.read_csv(path, usecols=usecols).rename(columns={probability_col: "p"})
            if frame["row_index"].duplicated().any():
                raise ValueError(f"Duplicate rows inside {path.name}")
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        if combined["row_index"].duplicated().any():
            raise ValueError(f"Duplicate combined rows for {condition}")
        combined_predictions[condition] = combined
        subset = fold_metrics.loc[fold_metrics["condition"] == condition]
        result = {
            "task": args.task,
            "condition": condition,
            "folds": len(subset),
            "n_features": int(subset["n_features"].iloc[0]),
        }
        for col in ["Score", "AUC", "PR-AUC", "Brier", "ECE_uniform_10", "ECE_adaptive_10"]:
            result[f"{col}_fold_mean"] = float(subset[col].mean())
            result[f"{col}_fold_std"] = float(subset[col].std(ddof=1)) if len(subset) > 1 else 0.0
        result.update(
            {f"{key}_combined": value for key, value in metrics(combined["y_true"], combined["p"]).items()}
        )
        rows.append(result)
        if args.task == "temporal_time":
            topk_table(combined["y_true"].to_numpy(), combined["p"].to_numpy()).assign(
                condition=condition
            ).to_csv(out_dir / f"temporal_time_{condition}_topk.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"{args.task}_summary.csv", index=False, encoding="utf-8-sig")
    if args.task == "stepwise":
        b = combined_predictions["model_b_age_time_history"]
        d = combined_predictions["model_d_full"]
        if not b[["row_index", "y_true"]].equals(d[["row_index", "y_true"]]):
            raise ValueError("Model B and D rows do not align")
        mb = metrics(b["y_true"].to_numpy(), b["p"].to_numpy())
        md = metrics(d["y_true"].to_numpy(), d["p"].to_numpy())
        pd.DataFrame(
            [
                {
                    "comparison": "Model D minus Model B",
                    **{f"delta_{key}": md[key] - mb[key] for key in mb},
                }
            ]
        ).to_csv(out_dir / "stepwise_incremental_value.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"{args.task}_summary_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "task": args.task,
                "metric_source": "recomputed from saved direct VotingClassifier predictions",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
