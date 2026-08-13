"""Leakage-safe nested tuning of the logistic-regression baseline.

The fixed-C baseline outputs are read only. Every new artifact is written under
``revision`` with a distinct ``tuned_logistic_expanded`` name.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


REVISION_DIR = Path(__file__).resolve().parent
CODE_DIR = REVISION_DIR.parent
WORKSPACE_DIR = CODE_DIR
TMP = REVISION_DIR / "tmp"
TABLES = REVISION_DIR / "tables"
sys.path.insert(0, str(CODE_DIR))

from fold_isolated_pipeline import (  # noqa: E402
    CACHE,
    HISTORY_COLS,
    RANDOM_STATE,
    STRICT_DATA,
    history_sha256,
    indices_sha256,
    matrix_with_history,
    metadata_sha256,
    metrics,
)
from nested_protocol import inner_splits, outer_splits, validate_splits  # noqa: E402


C_CANDIDATES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
MAX_ITER = 1000
OUTPUT_TAG = "tuned_logistic_expanded"
METRIC_COLUMNS = (
    "Score",
    "AUC",
    "PR-AUC",
    "Brier",
    "ECE_uniform_10",
    "ECE_adaptive_10",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_history_cache(
    meta: pd.DataFrame,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    cache_key: str,
) -> pd.DataFrame:
    """Load one history matrix only after validating its provenance manifest."""
    history_path = CACHE / f"{cache_key}_history.pkl"
    manifest_path = CACHE / f"{cache_key}_history_manifest.json"
    if not history_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing history cache for {cache_key}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "cache_key": cache_key,
        "source_rows": int(len(source_indices)),
        "target_rows": int(len(target_indices)),
        "source_indices_sha256": indices_sha256(source_indices),
        "target_indices_sha256": indices_sha256(target_indices),
        "source_metadata_with_label_sha256": metadata_sha256(
            meta, source_indices, include_label=True
        ),
        "target_metadata_without_label_sha256": metadata_sha256(
            meta, target_indices, include_label=False
        ),
        "same_month_rule": "source.TestDate < target.TestDate",
        "history_columns": HISTORY_COLS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"History manifest mismatch for {cache_key}: {key}")
    history = pd.read_pickle(history_path)
    if list(history.columns) != HISTORY_COLS:
        raise ValueError(f"History schema mismatch for {cache_key}")
    if not history.index.equals(pd.Index(target_indices)):
        raise ValueError(f"History target order mismatch for {cache_key}")
    if manifest.get("history_sha256") != history_sha256(history):
        raise ValueError(f"History content hash mismatch for {cache_key}")
    return history


def verified_split_matrices(
    x_base: pd.DataFrame,
    meta: pd.DataFrame,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    cache_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if np.intersect1d(train_indices, valid_indices).size:
        raise ValueError(f"Train/validation overlap in {cache_key}")
    train_history = verify_history_cache(
        meta, train_indices, train_indices, f"{cache_key}.train"
    )
    valid_history = verify_history_cache(
        meta, train_indices, valid_indices, f"{cache_key}.valid"
    )
    train = matrix_with_history(x_base, train_history, train_indices)
    valid = matrix_with_history(x_base, valid_history, valid_indices)
    del train_history, valid_history
    return train, valid


def prepare_matrices(
    x_train: pd.DataFrame, x_valid: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Fit preprocessing on training rows only and transform both partitions."""
    imputer = SimpleImputer(strategy="median")
    train = imputer.fit_transform(x_train)
    valid = imputer.transform(x_valid)
    scaler = StandardScaler(copy=False)
    train = scaler.fit_transform(train)
    valid = scaler.transform(valid)
    if not np.isfinite(train).all() or not np.isfinite(valid).all():
        raise ValueError("Non-finite value remained after preprocessing")
    return train, valid


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    c_value: float,
) -> tuple[np.ndarray, int, float]:
    model = LogisticRegression(
        C=float(c_value),
        max_iter=MAX_ITER,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    probabilities = model.predict_proba(x_valid)[:, 1]
    n_iter = int(np.max(model.n_iter_))
    if n_iter >= MAX_ITER:
        raise RuntimeError(f"LogisticRegression reached max_iter for C={c_value}")
    del model
    return probabilities, n_iter, elapsed


def aggregate_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    means = frame.groupby(group_columns, sort=False)[list(METRIC_COLUMNS)].mean()
    stds = (
        frame.groupby(group_columns, sort=False)[list(METRIC_COLUMNS)]
        .std(ddof=1)
        .add_suffix("_SD")
    )
    return means.join(stds).reset_index()


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()

    x_path = STRICT_DATA / "features_past_only.pkl"
    y_path = STRICT_DATA / "target_past_only.npy"
    groups_path = STRICT_DATA / "groups_past_only.npy"
    meta_path = STRICT_DATA / "meta_past_only.pkl"
    x_base = pd.read_pickle(x_path)
    y = np.load(y_path)
    groups = np.load(groups_path, allow_pickle=True)
    meta = pd.read_pickle(meta_path).reset_index(drop=True)
    if "YearMonthIndex" not in meta.columns:
        meta["YearMonthIndex"] = (meta["TestDate"] // 100) * 12 + (meta["TestDate"] % 100)
    if not np.array_equal(meta["Label"].to_numpy(), y):
        raise ValueError("Metadata labels and target array differ")
    if x_base.shape != (944_767, 37):
        raise ValueError(f"Unexpected feature shape: {x_base.shape}")

    outer = outer_splits("stratified", y, groups, meta, n_splits=5)
    inner_rows: list[dict] = []
    outer_rows: list[dict] = []
    selected_rows: list[dict] = []
    prediction_paths: list[Path] = []
    split_evidence: list[dict] = []

    for outer_fold, (outer_train, outer_valid) in enumerate(outer, start=1):
        print(f"outer {outer_fold}: prepare inner splits", flush=True)
        inner = inner_splits(
            "stratified", outer_train, y, groups, meta, outer_fold, n_splits=3
        )
        validate_splits("stratified", outer_train, outer_valid, inner, groups, meta)
        split_evidence.append(
            {
                "outer_fold": outer_fold,
                "outer_train_rows": int(len(outer_train)),
                "outer_valid_rows": int(len(outer_valid)),
                "outer_train_indices_sha256": indices_sha256(outer_train),
                "outer_valid_indices_sha256": indices_sha256(outer_valid),
                "inner": [
                    {
                        "inner_fold": inner_fold,
                        "train_rows": int(len(train_idx)),
                        "valid_rows": int(len(valid_idx)),
                        "train_indices_sha256": indices_sha256(train_idx),
                        "valid_indices_sha256": indices_sha256(valid_idx),
                    }
                    for inner_fold, (train_idx, valid_idx) in enumerate(inner, start=1)
                ],
            }
        )

        for inner_fold, (train_idx, valid_idx) in enumerate(inner, start=1):
            key = f"nested_stratified_outer{outer_fold}_inner{inner_fold}"
            print(f"outer {outer_fold} inner {inner_fold}: load verified matrices", flush=True)
            x_train_df, x_valid_df = verified_split_matrices(
                x_base, meta, train_idx, valid_idx, key
            )
            x_train, x_valid = prepare_matrices(x_train_df, x_valid_df)
            del x_train_df, x_valid_df
            gc.collect()

            for c_value in C_CANDIDATES:
                probabilities, n_iter, fit_seconds = fit_logistic(
                    x_train, y[train_idx], x_valid, c_value
                )
                values = metrics(y[valid_idx], probabilities)
                inner_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "C": c_value,
                        "train_rows": int(len(train_idx)),
                        "valid_rows": int(len(valid_idx)),
                        "n_iter": n_iter,
                        "fit_seconds": fit_seconds,
                        **values,
                    }
                )
                print(
                    f"outer {outer_fold} inner {inner_fold} C={c_value:g}: "
                    f"Score={values['Score']:.9f} AUC={values['AUC']:.9f} "
                    f"fit={fit_seconds:.2f}s",
                    flush=True,
                )
                del probabilities
            del x_train, x_valid
            gc.collect()

        outer_inner = pd.DataFrame(inner_rows)
        outer_inner = outer_inner.loc[outer_inner["outer_fold"] == outer_fold]
        candidate_summary = (
            outer_inner.groupby("C", sort=False)[list(METRIC_COLUMNS)]
            .mean()
            .reset_index()
            .sort_values(["Score", "C"], ascending=[True, True], kind="mergesort")
        )
        best = candidate_summary.iloc[0]
        best_c = float(best["C"])
        selected_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_C": best_c,
                "inner_mean_Score": float(best["Score"]),
                "inner_mean_AUC": float(best["AUC"]),
                "inner_mean_PR-AUC": float(best["PR-AUC"]),
            }
        )
        print(
            f"outer {outer_fold}: selected C={best_c:g} "
            f"inner Score={best['Score']:.9f}",
            flush=True,
        )

        outer_key = f"outer_stratified_fold{outer_fold}"
        x_outer_train_df, x_outer_valid_df = verified_split_matrices(
            x_base, meta, outer_train, outer_valid, outer_key
        )
        x_outer_train, x_outer_valid = prepare_matrices(
            x_outer_train_df, x_outer_valid_df
        )
        del x_outer_train_df, x_outer_valid_df
        gc.collect()
        p_tuned, outer_n_iter, outer_fit_seconds = fit_logistic(
            x_outer_train, y[outer_train], x_outer_valid, best_c
        )
        tuned_values = metrics(y[outer_valid], p_tuned)
        del x_outer_train, x_outer_valid
        gc.collect()

        fixed_path = TMP / f"simple_baseline_outer{outer_fold}_predictions.csv.gz"
        fixed = pd.read_csv(fixed_path)
        if not np.array_equal(
            fixed["row_index"].to_numpy(dtype=np.int64), outer_valid
        ):
            raise ValueError(f"Fixed-C row order mismatch in outer fold {outer_fold}")
        if not np.array_equal(fixed["y_true"].to_numpy(), y[outer_valid]):
            raise ValueError(f"Fixed-C labels mismatch in outer fold {outer_fold}")

        comparison_predictions = {
            "LogisticRegression (C=1.0)": fixed["p_logistic"].to_numpy(),
            "LogisticRegression (inner-tuned C)": p_tuned,
            "VotingClassifier": fixed["p_voting"].to_numpy(),
        }
        for model_name, probabilities in comparison_predictions.items():
            values = metrics(y[outer_valid], probabilities)
            outer_rows.append(
                {
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "selected_C": best_c
                    if model_name == "LogisticRegression (inner-tuned C)"
                    else (1.0 if model_name == "LogisticRegression (C=1.0)" else np.nan),
                    "n_iter": outer_n_iter
                    if model_name == "LogisticRegression (inner-tuned C)"
                    else np.nan,
                    "fit_seconds": outer_fit_seconds
                    if model_name == "LogisticRegression (inner-tuned C)"
                    else np.nan,
                    **values,
                }
            )

        prediction_frame = pd.DataFrame(
            {
                "row_index": outer_valid,
                "y_true": y[outer_valid],
                "p_logistic_C1": fixed["p_logistic"].to_numpy(),
                "p_logistic_tuned": p_tuned,
                "p_voting": fixed["p_voting"].to_numpy(),
            }
        )
        prediction_path = TMP / f"{OUTPUT_TAG}_outer{outer_fold}_predictions.csv.gz"
        prediction_frame.to_csv(prediction_path, index=False, compression="gzip")
        prediction_paths.append(prediction_path)
        print(
            f"outer {outer_fold}: tuned Score={tuned_values['Score']:.9f} "
            f"AUC={tuned_values['AUC']:.9f} PR-AUC={tuned_values['PR-AUC']:.9f}",
            flush=True,
        )
        del fixed, p_tuned, prediction_frame
        gc.collect()

    inner_frame = pd.DataFrame(inner_rows)
    inner_path = TMP / f"{OUTPUT_TAG}_inner_metrics.csv"
    inner_frame.to_csv(inner_path, index=False, encoding="utf-8-sig")
    inner_summary = aggregate_metrics(inner_frame, ["outer_fold", "C"])
    inner_summary_path = TABLES / f"table_revision_{OUTPUT_TAG}_inner_summary.csv"
    inner_summary.to_csv(inner_summary_path, index=False, encoding="utf-8-sig")

    selected_frame = pd.DataFrame(selected_rows)
    selected_path = TABLES / f"table_revision_{OUTPUT_TAG}_selected_c.csv"
    selected_frame.to_csv(selected_path, index=False, encoding="utf-8-sig")

    outer_frame = pd.DataFrame(outer_rows)
    outer_path = TMP / f"{OUTPUT_TAG}_outer_metrics.csv"
    outer_frame.to_csv(outer_path, index=False, encoding="utf-8-sig")
    outer_summary = aggregate_metrics(outer_frame, ["model"])
    outer_summary_path = TABLES / f"table_revision_{OUTPUT_TAG}_summary.csv"
    outer_summary.to_csv(outer_summary_path, index=False, encoding="utf-8-sig")

    summary_by_model = outer_summary.set_index("model")
    tuned = summary_by_model.loc["LogisticRegression (inner-tuned C)"]
    fixed = summary_by_model.loc["LogisticRegression (C=1.0)"]
    voting = summary_by_model.loc["VotingClassifier"]
    comparison_rows = []
    for comparison, left, right in [
        ("tuned minus C=1.0", tuned, fixed),
        ("VotingClassifier minus tuned", voting, tuned),
    ]:
        comparison_rows.append(
            {"comparison": comparison, **{metric: float(left[metric] - right[metric]) for metric in METRIC_COLUMNS}}
        )
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_path = TABLES / f"table_revision_{OUTPUT_TAG}_differences.csv"
    comparison_frame.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    dummy_summary_path = TABLES / "table_revision_simple_baseline_summary.csv"
    if not dummy_summary_path.exists():
        raise FileNotFoundError(
            "Run revision/run_revision_analyses.py before tuning so the "
            "DummyClassifier summary is available."
        )
    dummy_summary = pd.read_csv(dummy_summary_path)
    dummy_row = dummy_summary.loc[
        dummy_summary["model"] == "Prevalence-only DummyClassifier",
        ["model", *METRIC_COLUMNS],
    ].copy()
    dummy_row["model"] = "DummyClassifier"
    tuned_row = pd.DataFrame(
        [{"model": "LogisticRegression", **{metric: float(tuned[metric]) for metric in METRIC_COLUMNS}}]
    )
    final_table = pd.concat([dummy_row, tuned_row], ignore_index=True).rename(
        columns={
            "model": "Model",
            "ECE_uniform_10": "ECE (uniform-10)",
            "ECE_adaptive_10": "Adaptive ECE (quantile-10)",
        }
    )
    existing_models = pd.read_csv(
        CODE_DIR / "tables" / "table3_single_models_and_votingclassifier.csv"
    )
    existing_models = existing_models.loc[
        ~existing_models["Model"].isin(["DummyClassifier", "LogisticRegression"])
    ]
    final_table = pd.concat([final_table, existing_models], ignore_index=True)
    final_table_path = TABLES / "table_revision_model_comparison.csv"
    final_table.to_csv(final_table_path, index=False, encoding="utf-8-sig")

    manifest = {
        "analysis": "leakage-safe nested C tuning for logistic-regression baseline",
        "random_state": RANDOM_STATE,
        "outer_split": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "inner_split": "StratifiedKFold(n_splits=3, shuffle=True, random_state=42) within each outer training fold",
        "selection_objective": "lowest mean competition Score across three leakage-safe inner folds",
        "C_candidates": list(C_CANDIDATES),
        "preprocessing": "SimpleImputer(median) -> StandardScaler, fitted separately within each training partition",
        "model": f"LogisticRegression(penalty='l2', solver='lbfgs', max_iter={MAX_ITER})",
        "feature_shape": list(x_base.shape),
        "input_files": {
            str(path.relative_to(WORKSPACE_DIR)): sha256_file(path)
            for path in [x_path, y_path, groups_path, meta_path]
        },
        "history_rule": "training-partition sources only and source.TestDate < target.TestDate",
        "selected_C_by_outer_fold": selected_rows,
        "split_evidence": split_evidence,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started_all,
        "script_sha256": sha256_file(Path(__file__)),
        "outputs": {
            "inner_metrics": str(inner_path.relative_to(WORKSPACE_DIR)),
            "inner_summary": str(inner_summary_path.relative_to(WORKSPACE_DIR)),
            "selected_C": str(selected_path.relative_to(WORKSPACE_DIR)),
            "outer_metrics": str(outer_path.relative_to(WORKSPACE_DIR)),
            "outer_summary": str(outer_summary_path.relative_to(WORKSPACE_DIR)),
            "differences": str(comparison_path.relative_to(WORKSPACE_DIR)),
            "final_model_comparison": str(final_table_path.relative_to(WORKSPACE_DIR)),
            "predictions": [str(path.relative_to(WORKSPACE_DIR)) for path in prediction_paths],
        },
    }
    manifest_path = TMP / f"{OUTPUT_TAG}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(selected_frame.to_string(index=False), flush=True)
    print(outer_summary.to_string(index=False), flush=True)
    print(comparison_frame.to_string(index=False), flush=True)
    print(f"elapsed_seconds={manifest['elapsed_seconds']:.2f}", flush=True)
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
