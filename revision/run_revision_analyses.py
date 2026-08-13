"""Run the final expanded baseline and year-by-test-type analyses.

Outputs are confined to ``revision``. The simple-model comparison reuses the
outer StratifiedKFold partitions and verified fold-isolated history caches
created by the main reproduction pipeline.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REVISION_DIR = Path(__file__).resolve().parent
CODE_DIR = REVISION_DIR.parent
WORKSPACE_DIR = CODE_DIR
sys.path.insert(0, str(CODE_DIR))

from fold_isolated_pipeline import (  # noqa: E402
    CACHE,
    HISTORY_COLS,
    RANDOM_STATE,
    RESULTS,
    STRICT_DATA,
    history_sha256,
    indices_sha256,
    matrix_with_history,
    metadata_sha256,
    metrics,
)


FINAL_PREDICTIONS = RESULTS / "nested_evaluation"
TABLES = REVISION_DIR / "tables"
TMP = REVISION_DIR / "tmp"


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
) -> tuple[pd.DataFrame, Path, Path]:
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
    return history, history_path, manifest_path


def build_year_test_table(meta: pd.DataFrame) -> pd.DataFrame:
    frame = meta[["TestDate", "Test", "Label"]].copy()
    frame["year"] = frame["TestDate"] // 100
    by_type = (
        frame.groupby(["year", "Test"], sort=True)["Label"]
        .agg(n="size", positives="sum", label_rate="mean")
        .reset_index()
    )
    total = (
        frame.groupby("year", sort=True)["Label"]
        .agg(total_n="size", total_positives="sum", overall_label_rate="mean")
        .reset_index()
    )
    pieces = []
    for test in ["A", "B"]:
        part = by_type.loc[by_type["Test"] == test, ["year", "n", "positives", "label_rate"]].copy()
        part = part.rename(
            columns={
                "n": f"{test}_n",
                "positives": f"{test}_positives",
                "label_rate": f"{test}_label_rate",
            }
        )
        pieces.append(part)
    out = total
    for part in pieces:
        out = out.merge(part, on="year", how="left", validate="1:1")
    for test in ["A", "B"]:
        out[f"{test}_n"] = out[f"{test}_n"].fillna(0).astype(int)
        out[f"{test}_positives"] = out[f"{test}_positives"].fillna(0).astype(int)
        out[f"{test}_share"] = out[f"{test}_n"] / out["total_n"]
    return out[
        [
            "year",
            "total_n",
            "overall_label_rate",
            "A_n",
            "A_share",
            "A_label_rate",
            "B_n",
            "B_share",
            "B_label_rate",
        ]
    ]


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    x_path = STRICT_DATA / "features_past_only.pkl"
    y_path = STRICT_DATA / "target_past_only.npy"
    meta_path = STRICT_DATA / "meta_past_only.pkl"
    x_base = pd.read_pickle(x_path)
    y = np.load(y_path)
    meta = pd.read_pickle(meta_path).reset_index(drop=True)
    if "YearMonthIndex" not in meta.columns:
        meta["YearMonthIndex"] = (meta["TestDate"] // 100) * 12 + (meta["TestDate"] % 100)
    if x_base.shape != (944_767, 37):
        raise ValueError(f"Unexpected feature shape: {x_base.shape}")
    if not np.array_equal(meta["Label"].to_numpy(), y):
        raise ValueError("Metadata labels and target array differ")

    year_test = build_year_test_table(meta)
    year_test_path = TABLES / "table_revision_year_by_test_type.csv"
    year_test.to_csv(year_test_path, index=False, encoding="utf-8-sig")

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_indices = np.arange(len(y), dtype=np.int64)
    fold_rows: list[dict] = []
    cache_evidence: list[dict] = []
    prediction_paths: list[Path] = []

    for fold, (train_pos, valid_pos) in enumerate(splitter.split(all_indices, y), start=1):
        train_idx = all_indices[train_pos]
        valid_idx = all_indices[valid_pos]
        print(f"fold {fold}: load verified histories", flush=True)
        train_history, train_history_path, train_manifest_path = verify_history_cache(
            meta, train_idx, train_idx, f"outer_stratified_fold{fold}.train"
        )
        valid_history, valid_history_path, valid_manifest_path = verify_history_cache(
            meta, train_idx, valid_idx, f"outer_stratified_fold{fold}.valid"
        )
        x_train = matrix_with_history(x_base, train_history, train_idx)
        x_valid = matrix_with_history(x_base, valid_history, valid_idx)

        dummy = DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
        dummy.fit(np.zeros((len(train_idx), 1), dtype=np.uint8), y[train_idx])
        p_dummy = dummy.predict_proba(np.zeros((len(valid_idx), 1), dtype=np.uint8))[:, 1]

        logistic = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        print(f"fold {fold}: fit logistic baseline", flush=True)
        logistic.fit(x_train, y[train_idx])
        p_logistic = logistic.predict_proba(x_valid)[:, 1]
        n_iter = int(np.max(logistic.named_steps["model"].n_iter_))

        final_path = FINAL_PREDICTIONS / f"stratified_outer{fold}_predictions.csv.gz"
        final = pd.read_csv(final_path)
        if not np.array_equal(final["row_index"].to_numpy(dtype=np.int64), valid_idx):
            raise ValueError(f"Final prediction row order mismatch in fold {fold}")
        if not np.array_equal(final["y_true"].to_numpy(), y[valid_idx]):
            raise ValueError(f"Final prediction labels mismatch in fold {fold}")

        for model_name, probabilities in [
            ("Prevalence-only DummyClassifier", p_dummy),
            ("LogisticRegression", p_logistic),
            ("Soft-voting VotingClassifier", final["p_voting"].to_numpy()),
        ]:
            fold_rows.append(
                {
                    "outer_fold": fold,
                    "model": model_name,
                    "train_rows": len(train_idx),
                    "valid_rows": len(valid_idx),
                    "train_label_rate": float(np.mean(y[train_idx])),
                    "valid_label_rate": float(np.mean(y[valid_idx])),
                    "logistic_n_iter": n_iter if model_name == "LogisticRegression" else np.nan,
                    **metrics(y[valid_idx], np.asarray(probabilities, dtype=float)),
                }
            )

        prediction_frame = pd.DataFrame(
            {
                "row_index": valid_idx,
                "y_true": y[valid_idx],
                "p_dummy": p_dummy,
                "p_logistic": p_logistic,
                "p_voting": final["p_voting"].to_numpy(),
            }
        )
        prediction_path = TMP / f"simple_baseline_outer{fold}_predictions.csv.gz"
        prediction_frame.to_csv(prediction_path, index=False, compression="gzip")
        prediction_paths.append(prediction_path)
        cache_evidence.append(
            {
                "fold": fold,
                "train_history": str(train_history_path.relative_to(WORKSPACE_DIR)),
                "train_manifest_sha256": sha256_file(train_manifest_path),
                "valid_history": str(valid_history_path.relative_to(WORKSPACE_DIR)),
                "valid_manifest_sha256": sha256_file(valid_manifest_path),
                "final_prediction_file": str(final_path.relative_to(WORKSPACE_DIR)),
                "final_prediction_sha256": sha256_file(final_path),
            }
        )

        del (
            train_history,
            valid_history,
            x_train,
            x_valid,
            dummy,
            logistic,
            p_dummy,
            p_logistic,
            final,
            prediction_frame,
        )
        gc.collect()

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics_path = TMP / "simple_baseline_fold_metrics.csv"
    fold_metrics.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")

    metric_columns = ["Score", "AUC", "PR-AUC", "Brier", "ECE_uniform_10", "ECE_adaptive_10"]
    summary_mean = fold_metrics.groupby("model", sort=False)[metric_columns].mean()
    summary_std = fold_metrics.groupby("model", sort=False)[metric_columns].std(ddof=1).add_suffix("_SD")
    summary = summary_mean.join(summary_std).reset_index()
    summary_path = TABLES / "table_revision_simple_baseline_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    existing_table3 = pd.read_csv(CODE_DIR / "tables" / "table3_single_models_and_votingclassifier.csv")
    existing_table3 = existing_table3.loc[
        ~existing_table3["Model"].isin(["DummyClassifier", "LogisticRegression"])
    ]
    baseline_rows = summary.loc[
        summary["model"].isin(["Prevalence-only DummyClassifier", "LogisticRegression"]),
        ["model", *metric_columns],
    ].rename(
        columns={
            "model": "Model",
            "ECE_uniform_10": "ECE (uniform-10)",
            "ECE_adaptive_10": "Adaptive ECE (quantile-10)",
        }
    )
    revised_table3 = pd.concat([baseline_rows, existing_table3], ignore_index=True)
    revised_table3_path = TABLES / "table_revision_model_comparison_fixed_c.csv"
    revised_table3.to_csv(revised_table3_path, index=False, encoding="utf-8-sig")

    manifest = {
        "analysis": "final expanded baseline and year-by-test-type analyses",
        "random_state": RANDOM_STATE,
        "outer_split": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "history_rule": "source.TestDate < target.TestDate; verified cached histories",
        "feature_shape": list(x_base.shape),
        "input_files": {
            str(x_path.relative_to(WORKSPACE_DIR)): sha256_file(x_path),
            str(y_path.relative_to(WORKSPACE_DIR)): sha256_file(y_path),
            str(meta_path.relative_to(WORKSPACE_DIR)): sha256_file(meta_path),
        },
        "models": {
            "dummy": "DummyClassifier(strategy='prior', random_state=42)",
            "logistic": "SimpleImputer(median) -> StandardScaler -> LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)",
            "final": "VotingClassifier.predict_proba predictions from the outer folds",
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "cache_evidence": cache_evidence,
        "outputs": {
            "year_by_test_type": str(year_test_path.relative_to(WORKSPACE_DIR)),
            "fold_metrics": str(fold_metrics_path.relative_to(WORKSPACE_DIR)),
            "baseline_summary": str(summary_path.relative_to(WORKSPACE_DIR)),
            "fixed_c_model_comparison": str(revised_table3_path.relative_to(WORKSPACE_DIR)),
            "prediction_files": [str(path.relative_to(WORKSPACE_DIR)) for path in prediction_paths],
        },
    }
    manifest_path = TMP / "revision_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(year_test.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"wrote {revised_table3_path}", flush=True)
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
