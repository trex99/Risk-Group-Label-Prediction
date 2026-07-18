"""Paired PrimaryKey-cluster bootstrap for Model D minus Model B."""

from __future__ import annotations

import argparse
import json

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fold_isolated_pipeline import RANDOM_STATE, RESULTS, load_base


def one_bootstrap(seed: int, y, p_b, p_d, group_codes, n_groups):
    rng = np.random.default_rng(seed)
    group_weights = rng.poisson(1.0, size=n_groups).astype(float)
    weights = group_weights[group_codes]
    auc_b = roc_auc_score(y, p_b, sample_weight=weights)
    auc_d = roc_auc_score(y, p_d, sample_weight=weights)
    pr_b = average_precision_score(y, p_b, sample_weight=weights)
    pr_d = average_precision_score(y, p_d, sample_weight=weights)
    return auc_d - auc_b, pr_d - pr_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    x, y_all, groups_all, meta = load_base()
    del x, groups_all
    parts_b = []
    parts_d = []
    for fold in range(1, 6):
        b = pd.read_csv(
            RESULTS / "sensitivity" / f"stepwise_outer{fold}_model_b_age_time_history_predictions.csv.gz"
        )
        d = pd.read_csv(
            RESULTS / "nested_evaluation" / f"stratified_outer{fold}_predictions.csv.gz",
            usecols=["row_index", "y_true", "p_voting"],
        )
        parts_b.append(b)
        parts_d.append(d)
    b = pd.concat(parts_b, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    d = pd.concat(parts_d, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    if not b[["row_index", "y_true"]].equals(d[["row_index", "y_true"]]):
        raise ValueError("Bootstrap inputs do not align")
    row_idx = b["row_index"].to_numpy(dtype=np.int64)
    y = b["y_true"].to_numpy(dtype=np.int8)
    p_b = b["p_voting"].to_numpy(dtype=float)
    p_d = d["p_voting"].to_numpy(dtype=float)
    if not np.array_equal(y, y_all[row_idx]):
        raise ValueError("Bootstrap labels do not match base data")
    group_codes, unique_groups = pd.factorize(meta.loc[row_idx, "PrimaryKey"], sort=True)
    seeds = np.random.SeedSequence(RANDOM_STATE).generate_state(args.replicates, dtype=np.uint32)
    values = Parallel(n_jobs=args.jobs, backend="loky", verbose=0)(
        delayed(one_bootstrap)(int(seed), y, p_b, p_d, group_codes, len(unique_groups))
        for seed in seeds
    )
    dist = pd.DataFrame(values, columns=["delta_AUC", "delta_PR_AUC"])
    dist.insert(0, "replicate", np.arange(1, len(dist) + 1))
    out_dir = RESULTS / "sensitivity"
    dist.to_csv(out_dir / "incremental_value_cluster_bootstrap_distribution.csv", index=False)
    observed = {
        "delta_AUC": roc_auc_score(y, p_d) - roc_auc_score(y, p_b),
        "delta_PR_AUC": average_precision_score(y, p_d) - average_precision_score(y, p_b),
    }
    rows = []
    for metric in ["delta_AUC", "delta_PR_AUC"]:
        rows.append(
            {
                "metric": metric,
                "observed": observed[metric],
                "bootstrap_mean": float(dist[metric].mean()),
                "ci_2.5": float(dist[metric].quantile(0.025)),
                "ci_97.5": float(dist[metric].quantile(0.975)),
                "replicates": args.replicates,
                "cluster": "PrimaryKey",
                "method": "paired Poisson cluster bootstrap",
                "random_seed": RANDOM_STATE,
            }
        )
    pd.DataFrame(rows).to_csv(
        out_dir / "incremental_value_cluster_bootstrap_summary.csv", index=False, encoding="utf-8-sig"
    )
    (out_dir / "incremental_value_cluster_bootstrap_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "rows": int(len(y)),
                "clusters": int(len(unique_groups)),
                "replicates": args.replicates,
                "random_seed": RANDOM_STATE,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
