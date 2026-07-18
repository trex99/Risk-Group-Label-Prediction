"""Build reusable fold-isolated history matrices and leakage audit tables."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from fold_isolated_pipeline import (
    CACHE,
    HISTORY_COLS,
    RESULTS,
    compute_fold_history,
    ensure_dirs,
    fold_history_cache_path,
    load_base,
)


def splits_for(scheme: str, x, y, groups):
    if scheme == "stratified":
        return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y))
    if scheme == "group":
        return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(x, y, groups))
    raise ValueError(scheme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", nargs="+", choices=["stratified", "group"], default=["stratified", "group"])
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    x, y, groups, meta = load_base()
    audit_rows = []

    for scheme in args.schemes:
        splits = splits_for(scheme, x, y, groups)
        for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
            if fold not in args.folds:
                continue
            started = time.time()
            cache_path = fold_history_cache_path(scheme, fold)
            split_path = CACHE / f"{scheme}_fold{fold}_indices.npz"
            if cache_path.exists() and split_path.exists() and not args.force:
                history = pd.read_pickle(cache_path)
                status = "loaded"
            else:
                train_history = compute_fold_history(meta, train_idx, train_idx)
                valid_history = compute_fold_history(meta, train_idx, valid_idx)
                history = pd.concat([train_history, valid_history]).sort_index()
                if len(history) != len(x) or not history.index.equals(x.index):
                    raise ValueError(f"{scheme} fold {fold}: incomplete history matrix")
                history.to_pickle(cache_path)
                np.savez(split_path, train_idx=train_idx, valid_idx=valid_idx)
                status = "built"

            # Differences from the precomputed whole-data history quantify the
            # rows corrected by fold isolation; NaN-to-NaN is treated as equal.
            old = x[HISTORY_COLS]
            # Equivalent cumulative-variance formulas can differ at ~1e-8;
            # those roundoff differences are not substantive feature changes.
            equal = np.isclose(old.to_numpy(), history.to_numpy(), equal_nan=True, rtol=1e-9, atol=1e-7)
            changed_any = ~equal.all(axis=1)
            audit_rows.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "status": status,
                    "train_rows": len(train_idx),
                    "valid_rows": len(valid_idx),
                    "train_primarykeys": int(meta.loc[train_idx, "PrimaryKey"].nunique()),
                    "valid_primarykeys": int(meta.loc[valid_idx, "PrimaryKey"].nunique()),
                    "primarykey_overlap": int(
                        len(set(meta.loc[train_idx, "PrimaryKey"]).intersection(meta.loc[valid_idx, "PrimaryKey"]))
                    ),
                    "changed_train_rows": int(changed_any[train_idx].sum()),
                    "changed_train_rate": float(changed_any[train_idx].mean()),
                    "changed_valid_rows": int(changed_any[valid_idx].sum()),
                    "changed_valid_rate": float(changed_any[valid_idx].mean()),
                    "valid_prev_ab_label_available": int(history.loc[valid_idx, "prev_ab_all_label_mean"].notna().sum()),
                    "valid_prev_ab_label_available_rate": float(history.loc[valid_idx, "prev_ab_all_label_mean"].notna().mean()),
                    "validation_rows_used_as_history_sources": 0,
                    "elapsed_seconds": time.time() - started,
                    "cache_path": str(cache_path),
                }
            )
            print(pd.DataFrame([audit_rows[-1]]).to_string(index=False), flush=True)

    audit = pd.DataFrame(audit_rows)
    out = RESULTS / "fold_history_cache_audit.csv"
    if out.exists() and not audit.empty:
        prior = pd.read_csv(out)
        prior = prior[~prior.set_index(["scheme", "fold"]).index.isin(audit.set_index(["scheme", "fold"]).index)]
        audit = pd.concat([prior, audit], ignore_index=True).sort_values(["scheme", "fold"])
    audit.to_csv(out, index=False, encoding="utf-8-sig")
    print(out, flush=True)


if __name__ == "__main__":
    main()
