"""Five-outer-fold permutation-SHAP rank-stability analysis.

This analysis reads the leakage-safe outer-fold inputs, history caches, and
tuned hyperparameters created by the main reproduction pipeline. New artifacts
are written under ``revision``.
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


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
WORKSPACE_ROOT = CODE_ROOT

os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "5")
sys.path.insert(0, str(CODE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from fold_isolated_pipeline import (
    HISTORY_COLS,
    RANDOM_STATE,
    RUNTIME_ROOT,
    STRICT_DATA,
    build_or_load_split_matrices,
    make_voting_classifier,
)
from nested_protocol import outer_splits
from run_outer_evaluation import load_hp


INPUT_RUNTIME = RUNTIME_ROOT
STRICT_INPUT_DATA = STRICT_DATA

TMP_DIR = HERE / "tmp" / "shap_stability"
TABLE_DIR = HERE / "tables"
FIGURE_DIR = HERE / "figures"
MANIFEST_PATH = HERE / "tmp" / "shap_stability_manifest.json"

N_FOLDS = 5
N_EXPLAIN = 2_000
N_BACKGROUND = 100
VOTING_JOBS = 4
THREADS_PER_ESTIMATOR = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute fold artifacts even when a complete fold manifest exists.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_base() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load immutable base inputs from their archived sibling directory."""
    x = pd.read_pickle(STRICT_INPUT_DATA / "features_past_only.pkl")
    y = np.load(STRICT_INPUT_DATA / "target_past_only.npy")
    groups = np.load(STRICT_INPUT_DATA / "groups_past_only.npy", allow_pickle=True)
    meta = pd.read_pickle(STRICT_INPUT_DATA / "meta_past_only.pkl").reset_index(drop=True)
    if "YearMonthIndex" not in meta.columns:
        meta["YearMonthIndex"] = (meta["TestDate"] // 100) * 12 + (meta["TestDate"] % 100)
    if len(x) != len(y) or len(meta) != len(y) or len(groups) != len(y):
        raise ValueError("Frozen X, y, groups, and metadata lengths do not match")
    if not np.array_equal(meta["Label"].to_numpy(), y):
        raise ValueError("Frozen metadata Label and target array do not match")
    missing_history = [column for column in HISTORY_COLS if column not in x.columns]
    if missing_history:
        raise ValueError(f"Frozen base matrix is missing history columns: {missing_history}")
    return x, y, groups, meta


def fold_paths(fold: int) -> dict[str, Path]:
    stem = TMP_DIR / f"outer{fold}"
    return {
        "sample": stem.with_name(stem.name + "_validation_sample.csv.gz"),
        "background": stem.with_name(stem.name + "_training_background.csv.gz"),
        "values": stem.with_name(stem.name + "_shap_values.npz"),
        "importance": stem.with_name(stem.name + "_importance.csv"),
        "manifest": stem.with_name(stem.name + "_manifest.json"),
    }


def complete_fold(paths: dict[str, Path]) -> bool:
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "COMPLETE"


def load_cached_fold(fold: int, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    sample = pd.read_csv(paths["sample"])
    importance = pd.read_csv(paths["importance"])
    arrays = np.load(paths["values"])
    values = np.asarray(arrays["values"])
    base_values = np.asarray(arrays["base_values"])
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if len(sample) != N_EXPLAIN or values.shape != (N_EXPLAIN, len(importance)):
        raise ValueError(f"Cached fold {fold} SHAP shapes are invalid")
    return sample, importance, values, base_values, payload


def compute_fold(
    fold: int,
    x_base: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
    split: tuple[np.ndarray, np.ndarray],
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    paths = fold_paths(fold)
    if not force and complete_fold(paths):
        print(f"FOLD {fold}: cached", flush=True)
        return load_cached_fold(fold, paths)

    fold_started = time.perf_counter()
    train_idx, valid_idx = split
    if np.intersect1d(train_idx, valid_idx).size:
        raise ValueError(f"Outer fold {fold} train/validation overlap")

    matrix_started = time.perf_counter()
    hp = load_hp("stratified", fold, train_idx, valid_idx)
    x_train, x_valid = build_or_load_split_matrices(
        x_base,
        meta,
        train_idx,
        valid_idx,
        f"outer_stratified_fold{fold}",
        force=False,
    )
    matrix_seconds = time.perf_counter() - matrix_started
    if list(x_train.columns) != list(x_valid.columns):
        raise ValueError(f"Outer fold {fold} train/validation feature mismatch")

    fit_started = time.perf_counter()
    model = make_voting_classifier(
        hp,
        voting_jobs=VOTING_JOBS,
        threads_per_estimator=THREADS_PER_ESTIMATOR,
    )
    model.fit(x_train, y[train_idx])
    fit_seconds = time.perf_counter() - fit_started

    seed = RANDOM_STATE + fold - 1
    rng = np.random.default_rng(seed)
    sample_positions = np.sort(rng.choice(len(valid_idx), size=N_EXPLAIN, replace=False))
    background_positions = np.sort(rng.choice(len(train_idx), size=N_BACKGROUND, replace=False))
    x_sample = x_valid.iloc[sample_positions].copy()
    x_background = x_train.iloc[background_positions].copy()

    sample = x_sample.copy()
    sample.insert(0, "row_index", valid_idx[sample_positions])
    sample.insert(1, "Label", y[valid_idx[sample_positions]])
    background = x_background.copy()
    background.insert(0, "row_index", train_idx[background_positions])
    sample.to_csv(paths["sample"], index=False, compression="gzip")
    background.to_csv(paths["background"], index=False, compression="gzip")

    def predict_fn(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=x_train.columns)
        return model.predict_proba(frame)[:, 1]

    shap_started = time.perf_counter()
    np.random.seed(seed)
    masker = shap.maskers.Independent(x_background)
    explainer = shap.Explainer(
        predict_fn,
        masker,
        algorithm="permutation",
        feature_names=list(x_train.columns),
        seed=seed,
    )
    explanation = explainer(
        x_sample,
        max_evals=2 * x_train.shape[1] + 1,
        batch_size=64,
    )
    shap_seconds = time.perf_counter() - shap_started
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.shape != (N_EXPLAIN, x_train.shape[1]):
        raise ValueError(f"Outer fold {fold} SHAP shape {values.shape} is invalid")
    if not np.isfinite(values).all() or not np.isfinite(base_values).all():
        raise ValueError(f"Outer fold {fold} contains non-finite SHAP values")

    np.savez_compressed(paths["values"], values=values, base_values=base_values)
    importance = pd.DataFrame(
        {
            "fold": fold,
            "feature": x_train.columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_signed_shap": values.mean(axis=0),
        }
    )
    importance["rank"] = importance["mean_abs_shap"].rank(method="min", ascending=False).astype(int)
    importance = importance.sort_values(["rank", "feature"]).reset_index(drop=True)
    importance.to_csv(paths["importance"], index=False, encoding="utf-8-sig")

    total_seconds = time.perf_counter() - fold_started
    payload = {
        "status": "COMPLETE",
        "outer_fold": fold,
        "random_seed": seed,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(valid_idx)),
        "explained_rows": N_EXPLAIN,
        "background_rows": N_BACKGROUND,
        "feature_count": int(x_train.shape[1]),
        "algorithm": "permutation SHAP",
        "max_evals": int(2 * x_train.shape[1] + 1),
        "prediction_source": "fold-specific VotingClassifier.predict_proba",
        "matrix_seconds": matrix_seconds,
        "model_fit_seconds": fit_seconds,
        "shap_seconds": shap_seconds,
        "total_seconds": total_seconds,
    }
    paths["manifest"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"FOLD {fold}: complete total={total_seconds:.1f}s fit={fit_seconds:.1f}s shap={shap_seconds:.1f}s",
        flush=True,
    )

    del model, x_train, x_valid, x_sample, x_background, explanation, explainer, masker
    gc.collect()
    return sample, importance, values, base_values, payload


def pairwise_tables(importance_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    importance_wide = importance_long.pivot(index="feature", columns="fold", values="mean_abs_shap")
    importance_wide.columns = [f"fold_{int(col)}" for col in importance_wide.columns]
    spearman = importance_wide.corr(method="spearman")
    spearman.index.name = "fold"

    rows: list[dict] = []
    overlap_rows: list[dict] = []
    for left in range(1, N_FOLDS + 1):
        left_df = importance_long[importance_long["fold"] == left]
        for right in range(left + 1, N_FOLDS + 1):
            right_df = importance_long[importance_long["fold"] == right]
            rho = float(spearman.loc[f"fold_{left}", f"fold_{right}"])
            rows.append({"fold_left": left, "fold_right": right, "spearman_rho": rho})
            for k in (5, 10, 20):
                left_set = set(left_df.nsmallest(k, "rank")["feature"])
                right_set = set(right_df.nsmallest(k, "rank")["feature"])
                intersection = len(left_set & right_set)
                union = len(left_set | right_set)
                overlap_rows.append(
                    {
                        "fold_left": left,
                        "fold_right": right,
                        "k": k,
                        "intersection": intersection,
                        "overlap_rate": intersection / k,
                        "jaccard": intersection / union,
                    }
                )
    return spearman, pd.DataFrame(rows), pd.DataFrame(overlap_rows)


def aggregate_importance(importance_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_wide = importance_long.pivot(index="feature", columns="fold", values="rank")
    rank_wide.columns = [f"rank_fold_{int(col)}" for col in rank_wide.columns]
    importance_wide = importance_long.pivot(index="feature", columns="fold", values="mean_abs_shap")
    importance_wide.columns = [f"mean_abs_shap_fold_{int(col)}" for col in importance_wide.columns]

    grouped = importance_long.groupby("feature", sort=False)
    aggregate = grouped.agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        sd_abs_shap=("mean_abs_shap", "std"),
        mean_signed_shap=("mean_signed_shap", "mean"),
        sd_signed_shap=("mean_signed_shap", "std"),
        mean_rank=("rank", "mean"),
        sd_rank=("rank", "std"),
        min_rank=("rank", "min"),
        max_rank=("rank", "max"),
    )
    for k in (5, 10, 20):
        frequency = grouped["rank"].apply(lambda values, cutoff=k: int((values <= cutoff).sum()))
        aggregate[f"top{k}_fold_count"] = frequency
    aggregate = aggregate.join(rank_wide).join(importance_wide)
    aggregate = aggregate.sort_values(["mean_rank", "mean_abs_shap"], ascending=[True, False])
    aggregate.insert(0, "consensus_rank", np.arange(1, len(aggregate) + 1))
    aggregate = aggregate.reset_index()
    return aggregate, rank_wide.reset_index()


def make_figures(
    samples: list[pd.DataFrame],
    values_by_fold: list[np.ndarray],
    spearman: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> None:
    feature_columns = [column for column in samples[0].columns if column not in {"row_index", "Label"}]
    pooled_x = pd.concat([sample[feature_columns] for sample in samples], ignore_index=True)
    pooled_values = np.vstack(values_by_fold)
    if pooled_values.shape != (N_FOLDS * N_EXPLAIN, len(feature_columns)):
        raise ValueError("Pooled SHAP matrix has an invalid shape")

    np.random.seed(RANDOM_STATE)
    plt.figure(figsize=(8.5, 7.5), dpi=300)
    shap.summary_plot(pooled_values, pooled_x, max_display=20, show=False, plot_size=None)
    plt.title("(a) Pooled Five-Fold OOF SHAP Summary")
    plt.tight_layout()
    pooled_path = FIGURE_DIR / "Figure_revision_SHAP_5fold_OOF_summary_300dpi.png"
    plt.savefig(pooled_path, dpi=300, bbox_inches="tight")
    plt.close()

    fig, (ax_heat, ax_rank) = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.8),
        dpi=300,
        gridspec_kw={"width_ratios": [1.0, 1.55]},
    )
    image = ax_heat.imshow(spearman.to_numpy(), vmin=0.0, vmax=1.0, cmap="Blues")
    labels = [f"Fold {fold}" for fold in range(1, N_FOLDS + 1)]
    ax_heat.set_xticks(range(N_FOLDS), labels=labels, rotation=45, ha="right")
    ax_heat.set_yticks(range(N_FOLDS), labels=labels)
    ax_heat.set_title("(a) Pairwise Spearman rank correlation")
    for row in range(N_FOLDS):
        for col in range(N_FOLDS):
            value = float(spearman.iloc[row, col])
            ax_heat.text(col, row, f"{value:.3f}", ha="center", va="center", color="white" if value > 0.72 else "black")
    fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)

    shown = aggregate.head(12).copy()
    y_positions = np.arange(len(shown))
    fold_colors = plt.cm.tab10(np.linspace(0, 1, N_FOLDS))
    for y_pos, (_, row) in zip(y_positions, shown.iterrows()):
        ax_rank.hlines(y_pos, row["min_rank"], row["max_rank"], color="#b0b0b0", linewidth=2.0, zorder=1)
        for fold in range(1, N_FOLDS + 1):
            ax_rank.scatter(
                row[f"rank_fold_{fold}"],
                y_pos,
                s=36,
                color=fold_colors[fold - 1],
                label=f"Fold {fold}" if y_pos == 0 else None,
                zorder=2,
            )
        ax_rank.scatter(row["mean_rank"], y_pos, marker="D", s=46, color="black", label="Mean rank" if y_pos == 0 else None, zorder=3)
    ax_rank.set_yticks(y_positions, labels=shown["feature"])
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("SHAP importance rank (1 = highest)")
    ax_rank.set_title("(b) Fold-specific ranks of consensus top features")
    ax_rank.grid(axis="x", alpha=0.25)
    ax_rank.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    stability_path = FIGURE_DIR / "Figure_revision_SHAP_fold_rank_stability_300dpi.png"
    fig.savefig(stability_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # A direct arithmetic average of the five fold-wise mean absolute SHAP
    # values. Individual fold points are retained so the average is not shown
    # without its between-fold variation.
    average_shown = aggregate.nlargest(20, "mean_abs_shap").sort_values(
        "mean_abs_shap", ascending=True
    )
    fig, ax = plt.subplots(figsize=(9.2, 7.5), dpi=300)
    y_positions = np.arange(len(average_shown))
    ax.barh(
        y_positions,
        average_shown["mean_abs_shap"],
        xerr=average_shown["sd_abs_shap"],
        color="#4C78A8",
        alpha=0.82,
        error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capsize": 2.5},
        label="Five-fold average ± SD",
    )
    for fold in range(1, N_FOLDS + 1):
        ax.scatter(
            average_shown[f"mean_abs_shap_fold_{fold}"],
            y_positions,
            s=19,
            color="#F58518",
            alpha=0.75,
            label="Fold-specific value" if fold == 1 else None,
            zorder=3,
        )
    ax.set_yticks(y_positions, labels=average_shown["feature"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("(b) Average of Fold-Wise Mean Absolute SHAP Values")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    average_path = FIGURE_DIR / "Figure_revision_SHAP_fold_average_importance_300dpi.png"
    fig.savefig(average_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for directory in (TMP_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if not INPUT_RUNTIME.exists():
        raise FileNotFoundError(f"Frozen reproduction runtime is missing: {INPUT_RUNTIME}")
    if not STRICT_INPUT_DATA.exists():
        raise FileNotFoundError(f"Frozen strict-month inputs are missing: {STRICT_INPUT_DATA}")

    analysis_started = time.perf_counter()
    x_base, y, groups, meta = load_frozen_base()
    splits = outer_splits("stratified", y, groups, meta)
    if len(splits) != N_FOLDS:
        raise ValueError(f"Expected {N_FOLDS} outer folds, found {len(splits)}")

    samples: list[pd.DataFrame] = []
    values_by_fold: list[np.ndarray] = []
    importance_frames: list[pd.DataFrame] = []
    runtimes: list[dict] = []
    feature_columns: list[str] | None = None

    for fold, split in enumerate(splits, start=1):
        print(f"FOLD {fold}: starting", flush=True)
        sample, importance, values, _base_values, payload = compute_fold(
            fold,
            x_base,
            y,
            groups,
            meta,
            split,
            force=args.force,
        )
        current_features = list(importance.sort_values("feature")["feature"])
        if feature_columns is None:
            feature_columns = current_features
        elif current_features != feature_columns:
            raise ValueError(f"Feature set differs in outer fold {fold}")
        samples.append(sample)
        values_by_fold.append(values)
        importance_frames.append(importance)
        runtimes.append(payload)

    importance_long = pd.concat(importance_frames, ignore_index=True)
    if importance_long.groupby("fold")["feature"].nunique().ne(37).any():
        raise ValueError("Each fold must contain exactly 37 feature ranks")
    spearman, pairwise, overlaps = pairwise_tables(importance_long)
    aggregate, rank_wide = aggregate_importance(importance_long)
    if not np.allclose(np.diag(spearman), 1.0):
        raise ValueError("Spearman correlation diagonal is not one")

    table_paths = {
        "fold_importance": TABLE_DIR / "table_revision_shap_fold_importance.csv",
        "aggregate_importance": TABLE_DIR / "table_revision_shap_aggregate_importance.csv",
        "rank_matrix": TABLE_DIR / "table_revision_shap_rank_matrix.csv",
        "spearman_matrix": TABLE_DIR / "table_revision_shap_spearman_matrix.csv",
        "pairwise_spearman": TABLE_DIR / "table_revision_shap_pairwise_spearman.csv",
        "topk_overlap": TABLE_DIR / "table_revision_shap_topk_overlap.csv",
        "runtimes": TABLE_DIR / "table_revision_shap_fold_runtimes.csv",
    }
    importance_long.to_csv(table_paths["fold_importance"], index=False, encoding="utf-8-sig")
    aggregate.to_csv(table_paths["aggregate_importance"], index=False, encoding="utf-8-sig")
    rank_wide.to_csv(table_paths["rank_matrix"], index=False, encoding="utf-8-sig")
    spearman.to_csv(table_paths["spearman_matrix"], encoding="utf-8-sig")
    pairwise.to_csv(table_paths["pairwise_spearman"], index=False, encoding="utf-8-sig")
    overlaps.to_csv(table_paths["topk_overlap"], index=False, encoding="utf-8-sig")
    pd.DataFrame(runtimes).to_csv(table_paths["runtimes"], index=False, encoding="utf-8-sig")

    make_figures(samples, values_by_fold, spearman, aggregate)

    old_fold1_path = INPUT_RUNTIME / "results" / "shap_outer1_votingclassifier_importance.csv"
    fold1_reproduction_rho = None
    if old_fold1_path.exists():
        old_fold1 = pd.read_csv(old_fold1_path)
        new_fold1 = importance_long[importance_long["fold"] == 1][["feature", "mean_abs_shap"]]
        comparison = old_fold1.merge(new_fold1, on="feature", suffixes=("_original", "_new"))
        fold1_reproduction_rho = float(comparison[["mean_abs_shap_original", "mean_abs_shap_new"]].corr(method="spearman").iloc[0, 1])
        comparison.to_csv(
            TABLE_DIR / "table_revision_shap_fold1_reproduction.csv",
            index=False,
            encoding="utf-8-sig",
        )

    fold1_reproduction_path = TABLE_DIR / "table_revision_shap_fold1_reproduction.csv"

    pairwise_summary = {
        "mean": float(pairwise["spearman_rho"].mean()),
        "min": float(pairwise["spearman_rho"].min()),
        "max": float(pairwise["spearman_rho"].max()),
    }
    overlap_summary = {}
    for k in (5, 10, 20):
        subset = overlaps[overlaps["k"] == k]
        overlap_summary[str(k)] = {
            "mean_intersection": float(subset["intersection"].mean()),
            "min_intersection": int(subset["intersection"].min()),
            "max_intersection": int(subset["intersection"].max()),
            "mean_overlap_rate": float(subset["overlap_rate"].mean()),
            "mean_jaccard": float(subset["jaccard"].mean()),
        }

    artifacts = [*table_paths.values(), *FIGURE_DIR.glob("Figure_revision_SHAP_*300dpi.png")]
    if fold1_reproduction_path.exists():
        artifacts.append(fold1_reproduction_path)
    manifest = {
        "status": "COMPLETE",
        "analysis": "five-outer-fold permutation-SHAP rank stability",
        "input_runtime": str(INPUT_RUNTIME),
        "strict_input_data": str(STRICT_INPUT_DATA),
        "outer_split": "StratifiedKFold, 5 folds, random_state=42",
        "model": "fold-specific tuned soft-voting VotingClassifier",
        "explained_rows_per_fold": N_EXPLAIN,
        "background_rows_per_fold": N_BACKGROUND,
        "feature_count": 37,
        "pairwise_spearman": pairwise_summary,
        "topk_overlap": overlap_summary,
        "original_vs_recomputed_fold1_spearman": fold1_reproduction_rho,
        "total_seconds": time.perf_counter() - analysis_started,
        "artifacts": {
            str(path.relative_to(HERE)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(set(artifacts))
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
