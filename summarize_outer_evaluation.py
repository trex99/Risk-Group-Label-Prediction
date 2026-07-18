"""Recompute and summarize saved nested outer-fold predictions."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from fold_isolated_pipeline import RESULTS, load_base, metrics, reliability_table, topk_table
from nested_protocol import outer_splits


MODEL_NAMES = ("hgb", "lgb", "xgb", "cb", "voting")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["stratified", "group", "temporal"], required=True)
    args = parser.parse_args()
    x, y, groups, meta = load_base()
    splits = outer_splits(args.scheme, y, groups, meta)
    out_dir = RESULTS / "nested_evaluation"
    frames = []
    fold_rows = []
    for fold, (_, valid_idx) in enumerate(splits, start=1):
        path = out_dir / f"{args.scheme}_outer{fold}_predictions.csv.gz"
        frame = pd.read_csv(path)
        if not np.array_equal(frame["row_index"].to_numpy(dtype=np.int64), valid_idx):
            raise ValueError(f"Validation row order mismatch: {path.name}")
        if not np.array_equal(frame["y_true"].to_numpy(dtype=np.int8), y[valid_idx]):
            raise ValueError(f"Validation labels mismatch: {path.name}")
        frames.append(frame)
        for model in MODEL_NAMES:
            fold_rows.append(
                {
                    "scheme": args.scheme,
                    "outer_fold": fold,
                    "model": model,
                    **metrics(frame["y_true"].to_numpy(), frame[f"p_{model}"].to_numpy()),
                }
            )
    combined = pd.concat(frames, ignore_index=True)
    if combined["row_index"].duplicated().any():
        raise ValueError("OOF/holdout rows are duplicated")
    expected = np.concatenate([valid for _, valid in splits])
    if set(combined["row_index"].tolist()) != set(expected.tolist()):
        raise ValueError("OOF/holdout row coverage mismatch")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / f"{args.scheme}_fold_metrics_recomputed.csv", index=False, encoding="utf-8-sig")
    metric_cols = ["Score", "AUC", "PR-AUC", "Brier", "ECE_uniform_10", "ECE_adaptive_10"]
    summary_rows = []
    for model in MODEL_NAMES:
        subset = fold_df.loc[fold_df["model"] == model]
        row = {"scheme": args.scheme, "model": model, "folds": len(subset)}
        for col in metric_cols:
            row[f"{col}_fold_mean"] = float(subset[col].mean())
            row[f"{col}_fold_std"] = float(subset[col].std(ddof=1)) if len(subset) > 1 else 0.0
        row.update(
            {
                f"{key}_combined": value
                for key, value in metrics(
                    combined["y_true"].to_numpy(), combined[f"p_{model}"].to_numpy()
                ).items()
            }
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"{args.scheme}_model_summary.csv", index=False, encoding="utf-8-sig")
    topk_table(combined["y_true"].to_numpy(), combined["p_voting"].to_numpy()).to_csv(
        out_dir / f"{args.scheme}_voting_topk.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(
        [
            reliability_table(combined["y_true"].to_numpy(), combined["p_voting"].to_numpy(), "uniform"),
            reliability_table(combined["y_true"].to_numpy(), combined["p_voting"].to_numpy(), "adaptive"),
        ],
        ignore_index=True,
    ).to_csv(out_dir / f"{args.scheme}_voting_reliability.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "scheme": args.scheme,
        "rows": int(len(combined)),
        "unique_rows": int(combined["row_index"].nunique()),
        "label_rate": float(combined["y_true"].mean()),
        "metric_recomputation": "directly from saved VotingClassifier and component predictions",
        "status": "PASS",
    }
    (out_dir / f"{args.scheme}_summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
