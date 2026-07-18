"""Leakage-safe feature and evaluation utilities for the reviewer reanalysis.

The central rule is that a fold's validation rows are never used as history
sources for either training or validation features.  Every history statistic is
therefore rebuilt after splitting, using only source rows in the training fold
and only source months strictly earlier than the target month.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from copy import deepcopy
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE
RUNTIME_ROOT = Path(
    os.environ.get(
        "TRANSPORT_PAPER_RUNTIME",
        HERE / "runtime",
    )
).resolve()
REMAKE = RUNTIME_ROOT
STRICT_DIR = REMAKE / "strict_month_reanalysis"
STRICT_DATA = STRICT_DIR / "data"
OLD_OUTPUTS = STRICT_DIR / "outputs"
RESULTS = RUNTIME_ROOT / "results"
CACHE = RUNTIME_ROOT / "cache"
FIGURES = WORKSPACE_ROOT / "figures"
DATA = RUNTIME_ROOT / "data"
LOGS = RUNTIME_ROOT / "logs"

RANDOM_STATE = 42
N_BINS = 10
YMI_CENTER = 24000.0

HISTORY_COLS = [
    "prev_ab_YearMonthIndex_mean",
    "prev_ab_YearMonthIndex_sum",
    "prev_ab_YearMonthIndex_std",
    "prev_ab_all_label_mean",
    "prev_ab_all_label_sum",
    "prev_ab_all_label_std",
    "prev_YearMonthIndex_mean",
    "prev_YearMonthIndex_sum",
    "prev_YearMonthIndex_std",
    "prev_all_label_mean",
    "prev_all_label_sum",
    "prev_all_label_std",
    "other_test_Test_id_count",
    "other_test_YearMonthIndex_mean",
    "other_test_YearMonthIndex_std",
    "other_test_Label_mean",
]

ABSOLUTE_TIME_COLS = ["TestDate_year", "YearMonthIndex"]


def relative_time_matrix(x: pd.DataFrame) -> pd.DataFrame:
    """Replace absolute calendar indices with gaps from the target month.

    The standard deviations of prior months are retained because they are
    invariant to a calendar shift.  Prior-month means and sums are replaced by
    mean and summed month gaps; current year and raw YearMonthIndex are removed.
    """
    required = {
        "YearMonthIndex",
        "prev_ab_YearMonthIndex_mean",
        "prev_ab_YearMonthIndex_sum",
        "prev_YearMonthIndex_mean",
        "prev_YearMonthIndex_sum",
        "other_test_YearMonthIndex_mean",
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise ValueError(f"Relative-time conversion requires columns: {missing}")
    out = x.copy()
    current = out["YearMonthIndex"].astype(float)
    for prefix in ["prev_ab", "prev"]:
        mean_col = f"{prefix}_YearMonthIndex_mean"
        sum_col = f"{prefix}_YearMonthIndex_sum"
        count = out[sum_col].astype(float) / out[mean_col].astype(float)
        out[f"{prefix}_month_gap_mean"] = current - out[mean_col].astype(float)
        out[f"{prefix}_month_gap_sum"] = count * current - out[sum_col].astype(float)
    out["other_test_month_gap_mean"] = current - out["other_test_YearMonthIndex_mean"].astype(float)
    drop_cols = [
        "TestDate_year",
        "YearMonthIndex",
        "prev_ab_YearMonthIndex_mean",
        "prev_ab_YearMonthIndex_sum",
        "prev_YearMonthIndex_mean",
        "prev_YearMonthIndex_sum",
        "other_test_YearMonthIndex_mean",
    ]
    return out.drop(columns=drop_cols)


def ensure_dirs() -> None:
    for path in [RESULTS, CACHE, FIGURES, DATA, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def load_base() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the strict-month matrix that supplies non-history row features."""
    x = pd.read_pickle(STRICT_DATA / "features_past_only.pkl")
    y = np.load(STRICT_DATA / "target_past_only.npy")
    groups = np.load(STRICT_DATA / "groups_past_only.npy", allow_pickle=True)
    meta = pd.read_pickle(STRICT_DATA / "meta_past_only.pkl").reset_index(drop=True)
    if "YearMonthIndex" not in meta.columns:
        meta["YearMonthIndex"] = (meta["TestDate"] // 100) * 12 + (meta["TestDate"] % 100)
    if len(x) != len(y) or len(meta) != len(y):
        raise ValueError("Cached X, y, and metadata lengths do not match")
    if list(meta["Label"].to_numpy()) != list(y):
        raise ValueError("Metadata Label and target array do not match")
    if [c for c in HISTORY_COLS if c not in x.columns]:
        raise ValueError("One or more expected history columns are missing")
    return x, y, groups, meta


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def indices_sha256(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype=np.int64)
    return _sha256_bytes(values.tobytes(order="C"))


def metadata_sha256(meta: pd.DataFrame, indices: np.ndarray, include_label: bool) -> str:
    """Hash source/target fields used by history generation.

    Target hashes intentionally omit ``Label`` because target labels must not
    affect target features. Source hashes include ``Label`` because historical
    labels are legitimate source data.
    """
    cols = ["PrimaryKey", "Test", "TestDate", "YearMonthIndex"]
    if include_label:
        cols.append("Label")
    frame = meta.loc[np.asarray(indices, dtype=np.int64), cols]
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return _sha256_bytes(hashed.tobytes(order="C"))


def history_sha256(history: pd.DataFrame) -> str:
    frame = history[HISTORY_COLS]
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return _sha256_bytes(hashed.tobytes(order="C"))


def _safe_cache_key(cache_key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cache_key):
        raise ValueError(f"Unsafe cache key: {cache_key}")
    return cache_key


def history_cache_files(cache_key: str) -> tuple[Path, Path]:
    key = _safe_cache_key(cache_key)
    return CACHE / f"{key}_history.pkl", CACHE / f"{key}_history_manifest.json"


def build_or_load_history(
    meta: pd.DataFrame,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    cache_key: str,
    force: bool = False,
) -> pd.DataFrame:
    """Build one target history matrix and reject stale or mismatched caches."""
    ensure_dirs()
    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("source_indices contains duplicates")
    if len(np.unique(target_indices)) != len(target_indices):
        raise ValueError("target_indices contains duplicates")

    history_path, manifest_path = history_cache_files(cache_key)
    expected = {
        "cache_key": cache_key,
        "source_rows": int(len(source_indices)),
        "target_rows": int(len(target_indices)),
        "source_indices_sha256": indices_sha256(source_indices),
        "target_indices_sha256": indices_sha256(target_indices),
        "source_metadata_with_label_sha256": metadata_sha256(meta, source_indices, include_label=True),
        "target_metadata_without_label_sha256": metadata_sha256(meta, target_indices, include_label=False),
        "same_month_rule": "source.TestDate < target.TestDate",
        "history_columns": HISTORY_COLS,
    }

    if history_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"History cache manifest mismatch for {cache_key}: {key}")
        history = pd.read_pickle(history_path)
        if list(history.columns) != HISTORY_COLS:
            raise ValueError(f"History cache schema mismatch for {cache_key}")
        if not history.index.equals(pd.Index(target_indices)):
            raise ValueError(f"History cache target order mismatch for {cache_key}")
        if manifest.get("history_sha256") != history_sha256(history):
            raise ValueError(f"History cache content hash mismatch for {cache_key}")
        return history

    history = compute_fold_history(meta, source_indices, target_indices)
    if not history.index.equals(pd.Index(target_indices)):
        raise ValueError(f"Generated history target order mismatch for {cache_key}")
    manifest = {**expected, "history_sha256": history_sha256(history)}
    tmp_history = history_path.with_suffix(history_path.suffix + ".tmp")
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    history.to_pickle(tmp_history)
    tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_history.replace(history_path)
    tmp_manifest.replace(manifest_path)
    return history


def matrix_with_history(
    x_base: pd.DataFrame,
    history: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    """Replace only label-history columns while preserving original row features."""
    indices = np.asarray(indices, dtype=np.int64)
    x = x_base.iloc[indices].copy()
    if not history.index.equals(x.index):
        history = history.reindex(x.index)
    if not history.index.equals(x.index):
        raise ValueError("History and feature indices do not align")
    old_history_missing = x[HISTORY_COLS].isna().sum(axis=1)
    nonhistory_missing = x["isna_sum"].astype(float) - old_history_missing
    x.loc[:, HISTORY_COLS] = history[HISTORY_COLS].to_numpy()
    x["isna_sum"] = nonhistory_missing + x[HISTORY_COLS].isna().sum(axis=1)
    return x


def build_or_load_split_matrices(
    x_base: pd.DataFrame,
    meta: pd.DataFrame,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    cache_key: str,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create leakage-safe train/validation matrices for one declared split."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    if np.intersect1d(train_indices, valid_indices).size:
        raise ValueError(f"Train/validation overlap in {cache_key}")
    train_history = build_or_load_history(
        meta, train_indices, train_indices, f"{cache_key}.train", force=force
    )
    valid_history = build_or_load_history(
        meta, train_indices, valid_indices, f"{cache_key}.valid", force=force
    )
    return (
        matrix_with_history(x_base, train_history, train_indices),
        matrix_with_history(x_base, valid_history, valid_indices),
    )


def _cumulative_source_state(source: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Create one cumulative state per source person/month (and optional test)."""
    source = source.copy()
    source["_ymi_centered"] = source["YearMonthIndex"].astype(float) - YMI_CENTER
    source["_ymi_centered_sq"] = np.square(source["_ymi_centered"])
    source["_label_sq"] = np.square(source["Label"].astype(float))
    group_month = [*keys, "TestDate"]
    monthly = (
        source.groupby(group_month, sort=False)
        .agg(
            _count=("Label", "size"),
            _ymi_sum=("YearMonthIndex", "sum"),
            _ymi_centered_sum=("_ymi_centered", "sum"),
            _ymi_centered_sq_sum=("_ymi_centered_sq", "sum"),
            _label_sum=("Label", "sum"),
            _label_sq_sum=("_label_sq", "sum"),
        )
        .reset_index()
    )
    monthly = monthly.sort_values([*keys, "TestDate"], kind="mergesort").reset_index(drop=True)
    for col in [
        "_count",
        "_ymi_sum",
        "_ymi_centered_sum",
        "_ymi_centered_sq_sum",
        "_label_sum",
        "_label_sq_sum",
    ]:
        monthly[col] = monthly.groupby(keys, sort=False)[col].cumsum()
    # merge_asof requires global sorting by the on-key first.
    return monthly.sort_values(["TestDate", *keys], kind="mergesort").reset_index(drop=True)


def _merge_state(target: pd.DataFrame, state: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    trg = target[[*keys, "TestDate", "_target_index"]].sort_values(
        ["TestDate", *keys, "_target_index"], kind="mergesort"
    )
    merged = pd.merge_asof(
        trg,
        state,
        on="TestDate",
        by=keys,
        direction="backward",
        allow_exact_matches=False,
    )
    return merged.sort_values("_target_index", kind="mergesort").reset_index(drop=True)


def _sample_std(total: pd.Series, total_sq: pd.Series, count: pd.Series) -> pd.Series:
    variance = (total_sq - total * total / count) / (count - 1.0)
    return np.sqrt(variance.where(count > 1.0).clip(lower=0))


def _stats_from_merged(merged: pd.DataFrame, prefix: str, label_name: str) -> pd.DataFrame:
    count = merged["_count"].astype(float)
    ymi_sum = merged["_ymi_sum"].astype(float)
    label_sum = merged["_label_sum"].astype(float)
    return pd.DataFrame(
        {
            "_target_index": merged["_target_index"].to_numpy(),
            f"{prefix}_YearMonthIndex_mean": (ymi_sum / count).to_numpy(),
            f"{prefix}_YearMonthIndex_sum": ymi_sum.to_numpy(),
            f"{prefix}_YearMonthIndex_std": _sample_std(
                merged["_ymi_centered_sum"].astype(float),
                merged["_ymi_centered_sq_sum"].astype(float),
                count,
            ).to_numpy(),
            f"{label_name}_mean": (label_sum / count).to_numpy(),
            f"{label_name}_sum": label_sum.to_numpy(),
            f"{label_name}_std": _sample_std(
                label_sum, merged["_label_sq_sum"].astype(float), count
            ).to_numpy(),
        }
    )


def compute_fold_history(
    meta: pd.DataFrame,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> pd.DataFrame:
    """Compute all 16 history features from source rows only.

    A source row is eligible only if its TestDate is strictly lower than the
    target TestDate.  Consequently the target's own row, same-month rows,
    future rows, and every row outside source_indices are excluded.
    """
    source = meta.loc[source_indices, ["PrimaryKey", "Test", "TestDate", "YearMonthIndex", "Label"]].copy()
    target = meta.loc[target_indices, ["PrimaryKey", "Test", "TestDate"]].copy()
    target["_target_index"] = target_indices

    all_state = _cumulative_source_state(source, ["PrimaryKey"])
    all_merged = _merge_state(target, all_state, ["PrimaryKey"])
    all_stats = _stats_from_merged(all_merged, "prev_ab", "prev_ab_all_label")

    same_state = _cumulative_source_state(source, ["PrimaryKey", "Test"])
    same_merged = _merge_state(target, same_state, ["PrimaryKey", "Test"])
    same_stats = _stats_from_merged(same_merged, "prev", "prev_all_label")

    opposite_source = source.copy()
    opposite_source["TargetTest"] = opposite_source["Test"].map({"A": "B", "B": "A"})
    opposite_target = target.rename(columns={"Test": "TargetTest"})
    opposite_state = _cumulative_source_state(opposite_source, ["PrimaryKey", "TargetTest"])
    opposite_merged = _merge_state(opposite_target, opposite_state, ["PrimaryKey", "TargetTest"])
    opposite_count = opposite_merged["_count"].astype(float)
    opposite_y_sum = opposite_merged["_ymi_sum"].astype(float)
    opposite_label_sum = opposite_merged["_label_sum"].astype(float)
    opposite_stats = pd.DataFrame(
        {
            "_target_index": opposite_merged["_target_index"].to_numpy(),
            "other_test_Test_id_count": opposite_count.to_numpy(),
            "other_test_YearMonthIndex_mean": (opposite_y_sum / opposite_count).to_numpy(),
            "other_test_YearMonthIndex_std": _sample_std(
                opposite_merged["_ymi_centered_sum"].astype(float),
                opposite_merged["_ymi_centered_sq_sum"].astype(float),
                opposite_count,
            ).to_numpy(),
            "other_test_Label_mean": (opposite_label_sum / opposite_count).to_numpy(),
        }
    )

    out = all_stats.merge(same_stats, on="_target_index", validate="1:1").merge(
        opposite_stats, on="_target_index", validate="1:1"
    )
    out = out.set_index("_target_index").reindex(target_indices)
    return out[HISTORY_COLS]


def fold_history_cache_path(scheme: str, fold: int) -> Path:
    return CACHE / f"{scheme}_fold{fold}_history.pkl"


def build_or_load_fold_matrix(
    x_base: pd.DataFrame,
    meta: pd.DataFrame,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    scheme: str,
    fold: int,
    force: bool = False,
) -> pd.DataFrame:
    """Return a full-row matrix whose histories are isolated to one fold."""
    ensure_dirs()
    cache_path = fold_history_cache_path(scheme, fold)
    if cache_path.exists() and not force:
        new_history = pd.read_pickle(cache_path)
    else:
        train_history = compute_fold_history(meta, train_indices, train_indices)
        valid_history = compute_fold_history(meta, train_indices, valid_indices)
        new_history = pd.concat([train_history, valid_history]).sort_index()
        if len(new_history) != len(x_base) or not new_history.index.equals(x_base.index):
            raise ValueError("Fold history cache does not cover each row exactly once")
        new_history.to_pickle(cache_path)

    x_fold = x_base.copy()
    old_history_na = x_fold[HISTORY_COLS].isna().sum(axis=1)
    nonhistory_missing = x_fold["isna_sum"].astype(float) - old_history_na
    x_fold.loc[:, HISTORY_COLS] = new_history.loc[x_fold.index, HISTORY_COLS].to_numpy()
    x_fold["isna_sum"] = nonhistory_missing + x_fold[HISTORY_COLS].isna().sum(axis=1)
    return x_fold


def feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    columns = list(columns)
    return {
        "age": [c for c in columns if c == "Age"],
        "absolute_time": [c for c in columns if c in ABSOLUTE_TIME_COLS],
        "age_time": [c for c in columns if c in {"Age", "TestDate_year", "TestDate_month", "YearMonthIndex"}],
        "history": [c for c in columns if c in HISTORY_COLS],
        "accuracy_error": [c for c in columns if c.startswith("acc_stats") or c.startswith("err_stats")],
        "response_time": [c for c in columns if c.startswith("rt_mean_stats") or c.startswith("rt_std_stats")],
        "missingness": [c for c in columns if c == "isna_sum"],
    }


def load_hp() -> dict:
    hp_path = REMAKE / "outputs" / "optuna_retuned_past_only_best_params.json"
    payload = json.loads(hp_path.read_text(encoding="utf-8-sig"))
    return payload["hp_dict"]


def make_model(name: str, hp: dict):
    if name == "hgb":
        return HistGradientBoostingClassifier(**hp["hgb"])
    if name == "lgb":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**hp["lgb"])
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(**hp["xgb"])
    if name == "cb":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**hp["cb"])
    raise ValueError(name)


def make_voting_classifier(
    hp: dict,
    voting_jobs: int = -1,
    threads_per_estimator: int | None = None,
) -> VotingClassifier:
    """Return the manuscript's canonical final-model implementation."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    runtime_hp = deepcopy(hp)
    if threads_per_estimator is not None:
        runtime_hp["lgb"]["n_jobs"] = threads_per_estimator
        runtime_hp["xgb"]["n_jobs"] = threads_per_estimator
        runtime_hp["cb"]["thread_count"] = threads_per_estimator
    return VotingClassifier(
        estimators=[
            ("hgb", HistGradientBoostingClassifier(**runtime_hp["hgb"])),
            ("lgb", LGBMClassifier(**runtime_hp["lgb"])),
            ("xgb", XGBClassifier(**runtime_hp["xgb"])),
            ("cb", CatBoostClassifier(**runtime_hp["cb"])),
        ],
        n_jobs=voting_jobs,
        voting="soft",
    )


def uniform_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    weights = (totals / len(y_prob))[totals > 0]
    return float(np.sum(weights * np.abs(prob_true[: len(weights)] - prob_pred[: len(weights)])))


def adaptive_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> float:
    order = np.argsort(y_prob, kind="mergesort")
    total = len(y_prob)
    value = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        value += len(idx) / total * abs(float(np.mean(y_true[idx])) - float(np.mean(y_prob[idx])))
    return float(value)


def metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = uniform_ece(y_true, y_prob)
    aece = adaptive_ece(y_true, y_prob)
    score = 0.5 * (1.0 - auc) + 0.25 * brier + 0.25 * ece
    return {
        "Score": float(score),
        "AUC": auc,
        "PR-AUC": pr_auc,
        "Brier": brier,
        "ECE_uniform_10": ece,
        "ECE_adaptive_10": aece,
    }


def topk_table(y_true: np.ndarray, y_prob: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
    order = np.argsort(-y_prob, kind="mergesort")
    base = float(np.mean(y_true))
    positives = int(np.sum(y_true))
    rows = []
    for k in ks:
        n = int(np.ceil(len(order) * k / 100.0))
        idx = order[:n]
        label_count = int(np.sum(y_true[idx]))
        label_rate = float(np.mean(y_true[idx]))
        rows.append(
            {
                "Top-k": k,
                "n": n,
                "Label Count": label_count,
                "Label Rate": label_rate,
                "Lift": label_rate / base,
                "Cumulative Recall": label_count / positives,
                "Overall Label Rate": base,
            }
        )
    return pd.DataFrame(rows)


def reliability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    elif strategy == "adaptive":
        order = np.argsort(y_prob, kind="mergesort")
        bins = np.empty(len(y_prob), dtype=int)
        for bin_id, idx in enumerate(np.array_split(order, n_bins)):
            bins[idx] = bin_id
    else:
        raise ValueError(strategy)
    frame = pd.DataFrame({"bin": bins, "y": y_true, "p": y_prob})
    out = frame.groupby("bin", sort=True).agg(n=("y", "size"), mean_pred=("p", "mean"), observed_rate=("y", "mean")).reset_index()
    out["strategy"] = strategy
    return out[["strategy", "bin", "n", "mean_pred", "observed_rate"]]
