"""Code- and row-level leakage audit for every declared evaluation scheme."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fold_isolated_pipeline import (
    HISTORY_COLS,
    RESULTS,
    build_or_load_history,
    compute_fold_history,
    ensure_dirs,
    load_base,
)
from nested_protocol import inner_splits, outer_splits, validate_splits


def _stats(frame: pd.DataFrame, prefix: str, label_prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_YearMonthIndex_mean": float(frame["YearMonthIndex"].mean()) if len(frame) else np.nan,
        f"{prefix}_YearMonthIndex_sum": float(frame["YearMonthIndex"].sum()) if len(frame) else np.nan,
        f"{prefix}_YearMonthIndex_std": float(frame["YearMonthIndex"].std(ddof=1)) if len(frame) > 1 else np.nan,
        f"{label_prefix}_mean": float(frame["Label"].mean()) if len(frame) else np.nan,
        f"{label_prefix}_sum": float(frame["Label"].sum()) if len(frame) else np.nan,
        f"{label_prefix}_std": float(frame["Label"].std(ddof=1)) if len(frame) > 1 else np.nan,
    }


def brute_force_row(meta: pd.DataFrame, source: pd.DataFrame, target_idx: int) -> dict[str, float]:
    target = meta.loc[target_idx]
    eligible = source.loc[
        (source["PrimaryKey"] == target["PrimaryKey"])
        & (source["TestDate"] < target["TestDate"])
    ]
    out = {}
    out.update(_stats(eligible, "prev_ab", "prev_ab_all_label"))
    same = eligible.loc[eligible["Test"] == target["Test"]]
    out.update(_stats(same, "prev", "prev_all_label"))
    opposite = eligible.loc[eligible["Test"] != target["Test"]]
    out.update(
        {
            "other_test_Test_id_count": float(len(opposite)) if len(opposite) else np.nan,
            "other_test_YearMonthIndex_mean": float(opposite["YearMonthIndex"].mean()) if len(opposite) else np.nan,
            "other_test_YearMonthIndex_std": float(opposite["YearMonthIndex"].std(ddof=1)) if len(opposite) > 1 else np.nan,
            "other_test_Label_mean": float(opposite["Label"].mean()) if len(opposite) else np.nan,
        }
    )
    return out


def values_match(left: float, right: float) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(np.isclose(left, right, rtol=1e-10, atol=1e-10))


def audit_one_sample(
    scheme: str,
    fold: int,
    meta: pd.DataFrame,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    n: int = 100,
) -> dict:
    rng = np.random.default_rng(42000 + fold)
    sample = rng.choice(target_idx, size=min(n, len(target_idx)), replace=False).astype(np.int64)
    observed = compute_fold_history(meta, source_idx, sample)
    sample_primarykeys = set(meta.loc[sample, "PrimaryKey"].tolist())
    source = meta.loc[source_idx]
    source = source.loc[source["PrimaryKey"].isin(sample_primarykeys)]
    mismatches = []
    for idx in sample:
        expected = brute_force_row(meta, source, int(idx))
        for col in HISTORY_COLS:
            if not values_match(observed.loc[idx, col], expected[col]):
                mismatches.append({"target_index": int(idx), "column": col})
    return {
        "scheme": scheme,
        "fold": fold,
        "targets_checked": int(len(sample)),
        "values_checked": int(len(sample) * len(HISTORY_COLS)),
        "mismatches": mismatches,
    }


def main() -> None:
    ensure_dirs()
    x, y, groups, meta = load_base()
    checks = {
        "rows": int(len(y)),
        "features": int(x.shape[1]),
        "labels_binary": bool(set(np.unique(y)).issubset({0, 1})),
        "test_id_unique": bool(meta["Test_id"].is_unique),
        "history_columns_exact": [c for c in x.columns if c in HISTORY_COLS] == HISTORY_COLS,
        "same_month_rule": "source.TestDate < target.TestDate",
    }
    split_rows = []
    samples = []
    for scheme in ["stratified", "group", "temporal"]:
        outer = outer_splits(scheme, y, groups, meta)
        for fold, (outer_train, outer_valid) in enumerate(outer, start=1):
            inner = inner_splits(scheme, outer_train, y, groups, meta, fold, 3)
            validate_splits(scheme, outer_train, outer_valid, inner, groups, meta)
            split_rows.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "train_rows": len(outer_train),
                    "valid_rows": len(outer_valid),
                    "row_overlap": int(np.intersect1d(outer_train, outer_valid).size),
                    "primarykey_overlap": int(np.intersect1d(groups[outer_train], groups[outer_valid]).size),
                    "inner_outer_valid_overlap": int(
                        sum(
                            np.intersect1d(np.concatenate([tr, va]), outer_valid).size
                            for tr, va in inner
                        )
                    ),
                }
            )
            if fold == 1:
                samples.append(audit_one_sample(scheme, fold, meta, outer_train, outer_valid))

    # Validation-label mutation must leave every validation history unchanged.
    outer_train, outer_valid = outer_splits("stratified", y, groups, meta)[0]
    rng = np.random.default_rng(20260715)
    sample = rng.choice(outer_valid, size=500, replace=False).astype(np.int64)
    before = compute_fold_history(meta, outer_train, sample)
    mutated = meta.copy()
    mutated.loc[outer_valid, "Label"] = 1 - mutated.loc[outer_valid, "Label"].astype(int)
    after = compute_fold_history(mutated, outer_train, sample)
    mutation_invariant = bool(before.equals(after))

    # Materialize one manifest-protected pair and immediately reload it.
    cache_train = outer_train[:20000]
    cache_valid = outer_valid[:5000]
    first = build_or_load_history(meta, cache_train, cache_valid, "audit_manifest_probe", force=True)
    second = build_or_load_history(meta, cache_train, cache_valid, "audit_manifest_probe", force=False)
    cache_roundtrip = bool(first.equals(second))

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(RESULTS / "leakage_split_audit.csv", index=False, encoding="utf-8-sig")
    payload = {
        **checks,
        "validation_label_mutation_invariance": mutation_invariant,
        "cache_manifest_roundtrip": cache_roundtrip,
        "sample_audits": samples,
        "all_sample_values_match": all(not item["mismatches"] for item in samples),
        "all_outer_row_overlap_zero": bool((split_df["row_overlap"] == 0).all()),
        "all_inner_outer_valid_overlap_zero": bool((split_df["inner_outer_valid_overlap"] == 0).all()),
        "group_outer_primarykey_overlap_zero": bool(
            (split_df.loc[split_df["scheme"] == "group", "primarykey_overlap"] == 0).all()
        ),
    }
    required = [
        payload["labels_binary"],
        payload["test_id_unique"],
        payload["history_columns_exact"],
        payload["validation_label_mutation_invariance"],
        payload["cache_manifest_roundtrip"],
        payload["all_sample_values_match"],
        payload["all_outer_row_overlap_zero"],
        payload["all_inner_outer_valid_overlap_zero"],
        payload["group_outer_primarykey_overlap_zero"],
    ]
    payload["status"] = "PASS" if all(required) else "FAIL"
    out = RESULTS / "leakage_audit.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(payload["status"], flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
