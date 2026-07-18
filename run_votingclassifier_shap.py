"""Leakage-safe permutation SHAP for the direct outer-fold VotingClassifier."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from fold_isolated_pipeline import (
    DATA,
    FIGURES,
    RANDOM_STATE,
    RESULTS,
    build_or_load_split_matrices,
    load_base,
    make_voting_classifier,
)
from nested_protocol import outer_splits
from run_outer_evaluation import load_hp


def main() -> None:
    os.environ["OMP_NUM_THREADS"] = "5"
    os.environ["MKL_NUM_THREADS"] = "5"
    os.environ["OPENBLAS_NUM_THREADS"] = "5"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    x_base, y, groups, meta = load_base()
    train_idx, valid_idx = outer_splits("stratified", y, groups, meta)[0]
    hp = load_hp("stratified", 1, train_idx, valid_idx)
    x_train, x_valid = build_or_load_split_matrices(
        x_base, meta, train_idx, valid_idx, "outer_stratified_fold1", force=False
    )
    model = make_voting_classifier(hp, voting_jobs=4, threads_per_estimator=5)
    model.fit(x_train, y[train_idx])

    rng = np.random.default_rng(RANDOM_STATE)
    sample_positions = np.sort(rng.choice(len(valid_idx), size=min(2000, len(valid_idx)), replace=False))
    background_positions = np.sort(rng.choice(len(train_idx), size=min(100, len(train_idx)), replace=False))
    x_sample = x_valid.iloc[sample_positions].copy()
    x_background = x_train.iloc[background_positions].copy()
    sample = x_sample.copy()
    sample.insert(0, "row_index", valid_idx[sample_positions])
    sample.insert(1, "Label", y[valid_idx[sample_positions]])
    sample.to_csv(DATA / "shap_outer1_validation_sample.csv.gz", index=False, compression="gzip")
    background = x_background.copy()
    background.insert(0, "row_index", train_idx[background_positions])
    background.to_csv(DATA / "shap_outer1_training_background.csv.gz", index=False, compression="gzip")

    def predict_fn(values):
        frame = pd.DataFrame(values, columns=x_train.columns)
        return model.predict_proba(frame)[:, 1]

    masker = shap.maskers.Independent(x_background)
    explainer = shap.Explainer(
        predict_fn,
        masker,
        algorithm="permutation",
        feature_names=list(x_train.columns),
    )
    explanation = explainer(
        x_sample,
        max_evals=2 * x_train.shape[1] + 1,
        batch_size=64,
    )
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    np.save(DATA / "shap_outer1_votingclassifier_values.npy", values)
    np.save(DATA / "shap_outer1_votingclassifier_base_values.npy", base_values)
    importance = pd.DataFrame(
        {"feature": x_sample.columns, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(RESULTS / "shap_outer1_votingclassifier_importance.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(8, 7), dpi=300)
    shap.summary_plot(values, x_sample, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "Figure_3_SHAP_VotingClassifier_300dpi.png", dpi=300, bbox_inches="tight")
    plt.close()
    (RESULTS / "shap_outer1_votingclassifier_manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "random_seed": RANDOM_STATE,
                "training_partition": "stratified outer fold 1 training rows",
                "explained_partition": "2000 sampled outer fold 1 validation rows",
                "background": "100 sampled outer fold 1 training rows",
                "prediction_source": "VotingClassifier.predict_proba",
                "algorithm": "permutation SHAP",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
