"""PrimaryKey separation and history-availability sensitivity analysis.

This expanded analysis separates two questions:

1. How much does performance change when the 16 history features are removed?
2. How much does performance change when people are completely disjoint after
   both sides are evaluated without history?

An optional supplementary condition evaluates sequential history availability
under PrimaryKey separation. In that condition, validation features may use
strictly earlier records from the same validation person, but never the
current, same-month, or future record. The fitted model still uses no
validation row for training.

All new code, predictions, manifests, and tables are written below
``code/revision``. Existing manuscript artifacts are read but never modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import fold_isolated_pipeline as fp  # noqa: E402
from nested_protocol import outer_splits  # noqa: E402


DATA_DIR = fp.STRICT_DATA
BASE_RESULTS = fp.RESULTS
BASE_CACHE = fp.CACHE
REVISION_ROOT = Path(__file__).resolve().parent
TMP_DIR = REVISION_ROOT / "tmp" / "primarykey_history_factorial"
TABLE_DIR = REVISION_ROOT / "tables"

MODEL_NAMES = ("hgb", "lgb", "xgb", "cb")
FINAL_CONDITIONS = (
    "stratified_no_history",
    "group_no_history",
)
SUPPLEMENTARY_CONDITIONS = (
    "group_prior_self_history",
)
NEW_CONDITIONS = FINAL_CONDITIONS + SUPPLEMENTARY_CONDITIONS

TABLE7_NEW_CONDITIONS = FINAL_CONDITIONS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=[*NEW_CONDITIONS, "summarize"],
        required=True,
    )
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--threads-per-estimator", type=int, default=2)
    parser.add_argument("--voting-jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes(order="C")).hexdigest()


def load_base() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    x = pd.read_pickle(DATA_DIR / "features_past_only.pkl")
    y = np.load(DATA_DIR / "target_past_only.npy")
    groups = np.load(DATA_DIR / "groups_past_only.npy", allow_pickle=True)
    meta = pd.read_pickle(DATA_DIR / "meta_past_only.pkl").reset_index(drop=True)
    if "YearMonthIndex" not in meta.columns:
        meta["YearMonthIndex"] = (meta["TestDate"] // 100) * 12 + (meta["TestDate"] % 100)
    if not (len(x) == len(y) == len(groups) == len(meta)):
        raise ValueError("Base artifacts have inconsistent row counts")
    if not np.array_equal(meta["Label"].to_numpy(), y):
        raise ValueError("Metadata labels and target array differ")
    missing = sorted(set(fp.HISTORY_COLS) - set(x.columns))
    if missing:
        raise ValueError(f"Missing history columns: {missing}")
    return x, y, groups, meta


def load_fold_hp(
    scheme: str,
    fold: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> tuple[dict, dict[str, str]]:
    hp: dict[str, dict] = {}
    paths: dict[str, str] = {}
    for name in MODEL_NAMES:
        path = BASE_RESULTS / "optuna" / f"leakage_safe_{scheme}_outer{fold}_{name}_best.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload["protocol"]
        if protocol["outer_train_indices_sha256"] != fp.indices_sha256(train_idx):
            raise ValueError(f"Outer-train hash mismatch: {path.name}")
        if protocol["outer_valid_indices_sha256"] != fp.indices_sha256(valid_idx):
            raise ValueError(f"Outer-validation hash mismatch: {path.name}")
        hp[name] = payload["best_params"]
        paths[name] = str(path)
    return hp, paths


def no_history_matrix(x_base: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    """Drop history while retaining a missing count for non-history features only."""
    x = x_base.iloc[np.asarray(indices, dtype=np.int64)].copy()
    history_missing = x[fp.HISTORY_COLS].isna().sum(axis=1).astype(float)
    nonhistory_missing = x["isna_sum"].astype(float) - history_missing
    if (nonhistory_missing < 0).any():
        raise ValueError("Negative non-history missing count")
    x = x.drop(columns=fp.HISTORY_COLS)
    x["isna_sum"] = nonhistory_missing
    if len(x.columns) != 21:
        raise ValueError(f"Expected 21 no-history features, got {len(x.columns)}")
    return x


def group_training_matrix(
    x_base: pd.DataFrame,
    train_idx: np.ndarray,
    fold: int,
) -> pd.DataFrame:
    path = BASE_CACHE / f"outer_group_fold{fold}.train_history.pkl"
    history = pd.read_pickle(path)
    return fp.matrix_with_history(x_base, history, train_idx)


def condition_matrices(
    condition: str,
    x_base: pd.DataFrame,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if condition in {"stratified_no_history", "group_no_history"}:
        x_train = no_history_matrix(x_base, train_idx)
        x_valid = no_history_matrix(x_base, valid_idx)
        audit = {
            "feature_count": int(len(x_train.columns)),
            "history_columns_in_model": 0,
            "validation_history_source": "none",
            "valid_rows_with_prior_history": 0,
            "valid_prior_history_rate": 0.0,
            "missing_count_rule": "non-history columns only",
        }
        return x_train, x_valid, audit

    if condition == "group_prior_self_history":
        x_train = group_training_matrix(x_base, train_idx, fold)
        # Retrospective dataset sensitivity: strictly earlier records of the
        # validation person are eligible; current/same-month/future rows are
        # excluded by compute_fold_history.
        valid_history = fp.compute_fold_history(meta, valid_idx, valid_idx)
        x_valid = fp.matrix_with_history(x_base, valid_history, valid_idx)
        available = x_valid["prev_ab_all_label_mean"].notna()
        audit = {
            "feature_count": int(len(x_train.columns)),
            "history_columns_in_model": int(len(fp.HISTORY_COLS)),
            "validation_history_source": (
                "strictly earlier records of the same validation person; "
                "current, same-month, and future rows excluded"
            ),
            "valid_rows_with_prior_history": int(available.sum()),
            "valid_prior_history_rate": float(available.mean()),
            "valid_history_sha256": fp.history_sha256(valid_history),
            "missing_count_rule": "recomputed after history replacement",
        }
        return x_train, x_valid, audit

    raise ValueError(condition)


def run_fold(args: argparse.Namespace) -> None:
    if args.outer_fold not in {1, 2, 3, 4, 5}:
        raise ValueError("--outer-fold must be 1..5")
    if args.threads_per_estimator < 1 or args.voting_jobs < 1:
        raise ValueError("Thread counts must be positive")
    os.environ["OMP_NUM_THREADS"] = str(args.threads_per_estimator)
    os.environ["MKL_NUM_THREADS"] = str(args.threads_per_estimator)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads_per_estimator)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.condition}_outer{args.outer_fold}"
    pred_path = TMP_DIR / f"{stem}_predictions.csv.gz"
    metric_path = TMP_DIR / f"{stem}_metrics.csv"
    manifest_path = TMP_DIR / f"{stem}_manifest.json"
    if pred_path.exists() and metric_path.exists() and manifest_path.exists() and not args.force:
        print(f"EXISTS {stem}", flush=True)
        return

    started = time.time()
    x_base, y, groups, meta = load_base()
    scheme = "stratified" if args.condition.startswith("stratified") else "group"
    train_idx, valid_idx = outer_splits(scheme, y, groups, meta)[args.outer_fold - 1]
    overlap = np.intersect1d(groups[train_idx], groups[valid_idx]).size
    if scheme == "group" and overlap:
        raise ValueError(f"PrimaryKey overlap in group fold {args.outer_fold}: {overlap}")
    hp, hp_paths = load_fold_hp(scheme, args.outer_fold, train_idx, valid_idx)
    x_train, x_valid, audit = condition_matrices(
        args.condition,
        x_base,
        meta,
        train_idx,
        valid_idx,
        args.outer_fold,
    )
    if list(x_train.columns) != list(x_valid.columns):
        raise ValueError("Train/validation feature schemas differ")

    model = fp.make_voting_classifier(
        hp,
        voting_jobs=args.voting_jobs,
        threads_per_estimator=args.threads_per_estimator,
    )
    model.fit(x_train, y[train_idx])
    pred = model.predict_proba(x_valid)[:, 1]
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite prediction")
    metric = fp.metrics(y[valid_idx], pred)
    frame = pd.DataFrame({"row_index": valid_idx, "y_true": y[valid_idx], "p_voting": pred})
    frame.to_csv(pred_path, index=False, compression="gzip")
    metric_row = {
        "condition": args.condition,
        "scheme": scheme,
        "outer_fold": args.outer_fold,
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "n_features": int(len(x_train.columns)),
        **metric,
        "valid_prior_history_rate": audit["valid_prior_history_rate"],
        "prediction_sha256": sha256_array(pred),
    }
    pd.DataFrame([metric_row]).to_csv(metric_path, index=False, encoding="utf-8-sig")
    manifest = {
        **metric_row,
        **audit,
        "random_seed": fp.RANDOM_STATE,
        "train_indices_sha256": fp.indices_sha256(train_idx),
        "valid_indices_sha256": fp.indices_sha256(valid_idx),
        "primarykey_overlap": int(overlap),
        "hyperparameter_source": hp_paths,
        "hyperparameter_policy": (
            "locked full-feature inner-Optuna parameters for the same scheme and outer fold"
        ),
        "prediction_file": str(pred_path),
        "elapsed_seconds": float(time.time() - started),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metric_row, ensure_ascii=False), flush=True)
    del model, x_train, x_valid, x_base, meta
    gc.collect()


def read_existing_fold_metrics(scheme: str) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        path = BASE_RESULTS / "nested_evaluation" / f"{scheme}_outer{fold}_metrics.csv"
        frame = pd.read_csv(path)
        row = frame.loc[frame["model"].eq("voting")].iloc[0].to_dict()
        rows.append(row)
    return pd.DataFrame(rows)


def read_new_metrics(condition: str) -> pd.DataFrame:
    paths = [TMP_DIR / f"{condition}_outer{fold}_metrics.csv" for fold in range(1, 6)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fold metrics: {missing}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def verify_prediction_set(condition: str) -> dict:
    frames = []
    aucs = []
    prs = []
    for fold in range(1, 6):
        path = TMP_DIR / f"{condition}_outer{fold}_predictions.csv.gz"
        frame = pd.read_csv(path)
        frames.append(frame)
        aucs.append(roc_auc_score(frame["y_true"], frame["p_voting"]))
        prs.append(average_precision_score(frame["y_true"], frame["p_voting"]))
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 944_767 or combined["row_index"].nunique() != 944_767:
        raise ValueError(f"{condition}: predictions do not cover each row exactly once")
    return {
        "condition": condition,
        "rows": int(len(combined)),
        "unique_rows": int(combined["row_index"].nunique()),
        "AUC_fold_mean_recomputed": float(np.mean(aucs)),
        "PR-AUC_fold_mean_recomputed": float(np.mean(prs)),
        "AUC_combined": float(roc_auc_score(combined["y_true"], combined["p_voting"])),
        "PR-AUC_combined": float(
            average_precision_score(combined["y_true"], combined["p_voting"])
        ),
    }


def summarize_frame(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in [
        "Score",
        "AUC",
        "PR-AUC",
        "Brier",
        "ECE_uniform_10",
        "ECE_adaptive_10",
    ]:
        result[metric] = float(frame[metric].mean())
        result[f"{metric}_std"] = float(frame[metric].std(ddof=1))
    return result


def summarize() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        "stratified_training_history": read_existing_fold_metrics("stratified"),
        "stratified_no_history": read_new_metrics("stratified_no_history"),
        "group_no_history": read_new_metrics("group_no_history"),
    }
    supplementary_paths = [
        TMP_DIR / f"group_prior_self_history_outer{fold}_metrics.csv"
        for fold in range(1, 6)
    ]
    if all(path.exists() for path in supplementary_paths):
        frames["group_prior_self_history"] = read_new_metrics(
            "group_prior_self_history"
        )
    labels = {
        "stratified_training_history": (
            "Individual overlap allowed",
            "Training-fold prior history",
            37,
        ),
        "stratified_no_history": (
            "Individual overlap allowed",
            "History excluded",
            21,
        ),
        "group_no_history": (
            "PrimaryKey-disjoint",
            "History excluded",
            21,
        ),
        "group_prior_self_history": (
            "PrimaryKey-disjoint",
            "Strictly prior history from the same person",
            37,
        ),
    }
    summary_rows = []
    for condition, frame in frames.items():
        split_label, history_label, n_features = labels[condition]
        valid_rate = (
            float(frame["valid_prior_history_rate"].mean())
            if "valid_prior_history_rate" in frame.columns
            else (0.0 if condition == "group_training_history_only" else np.nan)
        )
        summary_rows.append(
            {
                "Condition": condition,
                "Person split": split_label,
                "History condition": history_label,
                "Features": n_features,
                **summarize_frame(frame),
                "Validation prior-history rate": valid_rate,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        TABLE_DIR / "table_revision_primarykey_history_factorial.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.9f",
    )

    contrasts = [
        (
            "Prior-history contribution when individual overlap is allowed",
            "stratified_training_history",
            "stratified_no_history",
        ),
        (
            "PrimaryKey separation effect without history",
            "group_no_history",
            "stratified_no_history",
        ),
    ]
    if "group_prior_self_history" in frames:
        contrasts.append(
            (
                "Prior-history contribution among PrimaryKey-disjoint persons",
                "group_prior_self_history",
                "group_no_history",
            )
        )
    contrast_rows = []
    for label, left, right in contrasts:
        left_frame = frames[left].sort_values("outer_fold").reset_index(drop=True)
        right_frame = frames[right].sort_values("outer_fold").reset_index(drop=True)
        row = {"Contrast": label, "Left": left, "Right": right}
        for metric in ["Score", "AUC", "PR-AUC"]:
            delta = left_frame[metric].to_numpy() - right_frame[metric].to_numpy()
            row[f"Delta {metric}"] = float(delta.mean())
            row[f"Delta {metric} std"] = float(delta.std(ddof=1))
            row[f"Folds positive {metric}"] = int((delta > 0).sum())
        contrast_rows.append(row)
    pd.DataFrame(contrast_rows).to_csv(
        TABLE_DIR / "table_revision_primarykey_history_factorial_contrasts.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.9f",
    )

    proposed_conditions = [
        "stratified_training_history",
        "stratified_no_history",
        "group_no_history",
    ]
    proposed = summary.set_index("Condition").loc[proposed_conditions].reset_index()
    proposed.to_csv(
        TABLE_DIR / "table_revision_table7_factorial_proposed.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.9f",
    )
    paper_labels = {
        "stratified_training_history": ("Main nested CV", "Included"),
        "stratified_no_history": ("Main outer folds", "Excluded"),
        "group_no_history": ("PrimaryKey-disjoint CV", "Excluded"),
    }
    paper_table = proposed[["Condition", "Features", "Score", "AUC", "PR-AUC"]].copy()
    paper_table.insert(
        1,
        "History Features",
        paper_table["Condition"].map(lambda value: paper_labels[value][1]),
    )
    paper_table["Condition"] = paper_table["Condition"].map(
        lambda value: paper_labels[value][0]
    )
    public_table_dir = CODE_ROOT / "tables"
    public_table_dir.mkdir(parents=True, exist_ok=True)
    paper_table.to_csv(
        public_table_dir / "table7_primarykey_disjoint_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    verification = [verify_prediction_set(condition) for condition in TABLE7_NEW_CONDITIONS]
    verification_payload = {
        "status": "PASS",
        "conditions": verification,
        "checks": [
            "five prediction files per new condition",
            "944,767 unique row indices per condition",
            "AUC and PR-AUC recomputed directly from saved predictions",
            "PrimaryKey overlap rejected for group folds",
            "no-history missing count excludes history-column missingness",
        ],
    }
    (TMP_DIR / "primarykey_history_factorial_verification.json").write_text(
        json.dumps(verification_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(verification_payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.condition == "summarize":
        summarize()
    else:
        run_fold(args)


if __name__ == "__main__":
    main()
