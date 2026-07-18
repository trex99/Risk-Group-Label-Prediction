"""Two independent numerical passes over every manuscript-facing result.

Pass 1 uses the production metric helpers.  Pass 2 independently recomputes
AUC, average precision, Brier, uniform/adaptive ECE, Top-k, reliability bins,
summary means, bootstrap summaries, and subgroup metrics from saved prediction
files.  Manual component averaging is used only as an identity audit of
``VotingClassifier.predict_proba`` and never as a manuscript prediction source.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from fold_isolated_pipeline import RESULTS, load_base, metrics


METRICS = ["Score", "AUC", "PR-AUC", "Brier", "ECE_uniform_10", "ECE_adaptive_10"]
MODELS = ["hgb", "lgb", "xgb", "cb", "voting"]
SCHEME_FOLDS = {"stratified": range(1, 6), "group": range(1, 6), "temporal": range(1, 2)}
TASK_SCHEME = {
    "stepwise": "stratified",
    "ablation": "stratified",
    "stratified_time": "stratified",
    "temporal_time": "temporal",
}
TASK_CONDITIONS = {
    "stepwise": [
        "model_a_age_time",
        "model_b_age_time_history",
        "model_c_current_cognitive_response",
        "model_d_full",
    ],
    "ablation": [
        "full",
        "without_accuracy_error",
        "without_age_time",
        "without_history",
        "without_missingness",
        "without_response_time",
    ],
    "stratified_time": ["full", "without_absolute_time", "relative_time"],
    "temporal_time": ["full", "without_absolute_time", "relative_time"],
}
# Prediction CSV round-trips and XGBoost float32 probabilities can differ from
# the in-memory VotingClassifier calculation at the low 1e-9 level.  This
# tolerance remains two orders tighter than the manuscript's 1e-6 reporting
# precision while rejecting any numerically meaningful discrepancy.
TOL = 1e-8


def manual_auc(y: np.ndarray, p: np.ndarray) -> float:
    ranks = rankdata(p, method="average")
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = len(y) - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def manual_average_precision(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    ps = p[order]
    ends = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    tp = np.cumsum(ys)[ends]
    precision = tp / (ends + 1)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def manual_uniform_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    return float(
        sum(
            mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
            for bin_id in range(n_bins)
            if (mask := bins == bin_id).any()
        )
    )


def manual_adaptive_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    order = np.argsort(p, kind="mergesort")
    return float(
        sum(
            len(idx) / len(y) * abs(float(y[idx].mean()) - float(p[idx].mean()))
            for idx in np.array_split(order, n_bins)
            if len(idx)
        )
    )


def manual_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    auc = manual_auc(y, p)
    brier = float(np.mean(np.square(y - p)))
    ece = manual_uniform_ece(y, p)
    return {
        "Score": 0.5 * (1 - auc) + 0.25 * brier + 0.25 * ece,
        "AUC": auc,
        "PR-AUC": manual_average_precision(y, p),
        "Brier": brier,
        "ECE_uniform_10": ece,
        "ECE_adaptive_10": manual_adaptive_ece(y, p),
    }


def add_failure(failures: list[dict], source: str, check: str, left, right) -> None:
    failures.append({"source": source, "check": check, "left": left, "right": right})


def compare_value(failures: list[dict], source: str, check: str, left, right, tol: float = TOL) -> None:
    if not np.isclose(float(left), float(right), rtol=tol, atol=tol, equal_nan=True):
        add_failure(failures, source, check, float(left), float(right))


def verify_metrics(
    failures: list[dict], records: list[dict], source: str, y: np.ndarray, p: np.ndarray,
    reported: pd.Series | dict | None = None,
) -> dict[str, float]:
    pass1 = metrics(y, p)
    pass2 = manual_metrics(y, p)
    for key in METRICS:
        compare_value(failures, source, f"pass1_vs_pass2:{key}", pass1[key], pass2[key])
        if reported is not None:
            compare_value(failures, source, f"reported_vs_recomputed:{key}", reported[key], pass1[key])
    records.append(
        {"source": source, "n": len(y), **{f"pass1_{k}": pass1[k] for k in METRICS}, **{f"pass2_{k}": pass2[k] for k in METRICS}}
    )
    return pass1


def manual_topk(y: np.ndarray, p: np.ndarray, ks=(1, 5, 10)) -> pd.DataFrame:
    order = np.argsort(-p, kind="mergesort")
    overall = float(y.mean())
    positives = int(y.sum())
    rows = []
    for k in ks:
        n = int(np.ceil(len(y) * k / 100))
        idx = order[:n]
        count = int(y[idx].sum())
        rate = count / n
        rows.append({
            "Top-k": k, "n": n, "Label Count": count, "Label Rate": rate,
            "Lift": rate / overall, "Cumulative Recall": count / positives,
            "Overall Label Rate": overall,
        })
    return pd.DataFrame(rows)


def verify_topk(failures: list[dict], source: str, saved: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> None:
    expected = manual_topk(y, p)
    if list(saved["Top-k"].astype(int)) != list(expected["Top-k"]):
        add_failure(failures, source, "Top-k keys", saved["Top-k"].tolist(), expected["Top-k"].tolist())
        return
    for i in range(len(expected)):
        for key in ["n", "Label Count"]:
            if int(saved.iloc[i][key]) != int(expected.iloc[i][key]):
                add_failure(failures, source, f"Top-k:{key}", int(saved.iloc[i][key]), int(expected.iloc[i][key]))
        for key in ["Label Rate", "Lift", "Cumulative Recall", "Overall Label Rate"]:
            compare_value(failures, source, f"Top-k:{key}", saved.iloc[i][key], expected.iloc[i][key])


def reliability_from_predictions(y: np.ndarray, p: np.ndarray, strategy: str, n_bins: int = 10) -> pd.DataFrame:
    if strategy == "uniform":
        bins = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    else:
        order = np.argsort(p, kind="mergesort")
        bins = np.empty(len(p), dtype=int)
        for bin_id, idx in enumerate(np.array_split(order, n_bins)):
            bins[idx] = bin_id
    frame = pd.DataFrame({"bin": bins, "y": y, "p": p})
    out = frame.groupby("bin", sort=True).agg(n=("y", "size"), mean_pred=("p", "mean"), observed_rate=("y", "mean")).reset_index()
    out.insert(0, "strategy", strategy)
    return out


def verify_reliability(failures: list[dict], source: str, saved: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> None:
    expected = pd.concat(
        [reliability_from_predictions(y, p, "uniform"), reliability_from_predictions(y, p, "adaptive")],
        ignore_index=True,
    )
    saved = saved.sort_values(["strategy", "bin"]).reset_index(drop=True)
    expected = expected.sort_values(["strategy", "bin"]).reset_index(drop=True)
    if not saved[["strategy", "bin", "n"]].equals(expected[["strategy", "bin", "n"]]):
        add_failure(failures, source, "reliability bin identity", "saved", "recomputed")
        return
    for key in ["mean_pred", "observed_rate"]:
        for i in range(len(expected)):
            compare_value(failures, source, f"reliability:{key}:row{i}", saved.iloc[i][key], expected.iloc[i][key])


def sensitivity_prediction(task: str, fold: int, condition: str) -> pd.DataFrame:
    scheme = TASK_SCHEME[task]
    if condition in {"full", "model_d_full"}:
        return pd.read_csv(
            RESULTS / "nested_evaluation" / f"{scheme}_outer{fold}_predictions.csv.gz",
            usecols=["row_index", "y_true", "p_voting"],
        )
    return pd.read_csv(
        RESULTS / "sensitivity" / f"{task}_outer{fold}_{condition}_predictions.csv.gz",
        usecols=["row_index", "y_true", "p_voting"],
    )


def main() -> None:
    audit = json.loads((RESULTS / "leakage_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError("Leakage audit is not PASS")
    x_base, y_all, groups, meta = load_base()
    del x_base, y_all, groups
    failures: list[dict] = []
    records: list[dict] = []
    combined_outer: dict[str, pd.DataFrame] = {}

    # Outer predictions, direct VotingClassifier identity, summaries, Top-k, reliability.
    for scheme, folds in SCHEME_FOLDS.items():
        frames = []
        fold_metric_rows = []
        for fold in folds:
            path = RESULTS / "nested_evaluation" / f"{scheme}_outer{fold}_predictions.csv.gz"
            frame = pd.read_csv(path)
            frames.append(frame)
            reported = pd.read_csv(RESULTS / "nested_evaluation" / f"{scheme}_outer{fold}_metrics.csv")
            component_mean = frame[["p_hgb", "p_lgb", "p_xgb", "p_cb"]].mean(axis=1).to_numpy()
            direct = frame["p_voting"].to_numpy()
            max_diff = float(np.max(np.abs(component_mean - direct)))
            if max_diff > TOL:
                add_failure(failures, path.name, "VotingClassifier component identity", max_diff, 0.0)
            for model in MODELS:
                row = reported.loc[reported["model"].eq(model)].iloc[0]
                values = verify_metrics(
                    failures, records, f"{scheme}/outer{fold}/{model}",
                    frame["y_true"].to_numpy(dtype=np.int8), frame[f"p_{model}"].to_numpy(dtype=float), row,
                )
                fold_metric_rows.append({"model": model, **values})
        combined = pd.concat(frames, ignore_index=True).sort_values("row_index").reset_index(drop=True)
        combined_outer[scheme] = combined
        if combined["row_index"].duplicated().any():
            add_failure(failures, scheme, "outer row uniqueness", "duplicates", "none")
        summary = pd.read_csv(RESULTS / "nested_evaluation" / f"{scheme}_model_summary.csv")
        fold_df = pd.DataFrame(fold_metric_rows)
        for model in MODELS:
            row = summary.loc[summary["model"].eq(model)].iloc[0]
            subset = fold_df.loc[fold_df["model"].eq(model)]
            for key in METRICS:
                compare_value(failures, f"{scheme}/summary/{model}", f"{key}_fold_mean", row[f"{key}_fold_mean"], subset[key].mean())
                expected_std = subset[key].std(ddof=1) if len(subset) > 1 else 0.0
                compare_value(failures, f"{scheme}/summary/{model}", f"{key}_fold_std", row[f"{key}_fold_std"], expected_std)
            verify_metrics(
                failures, records, f"{scheme}/combined/{model}", combined["y_true"].to_numpy(dtype=np.int8),
                combined[f"p_{model}"].to_numpy(dtype=float),
                {key: row[f"{key}_combined"] for key in METRICS},
            )
        verify_topk(
            failures, f"{scheme}/Top-k",
            pd.read_csv(RESULTS / "nested_evaluation" / f"{scheme}_voting_topk.csv"),
            combined["y_true"].to_numpy(dtype=np.int8), combined["p_voting"].to_numpy(dtype=float),
        )
        verify_reliability(
            failures, f"{scheme}/reliability",
            pd.read_csv(RESULTS / "nested_evaluation" / f"{scheme}_voting_reliability.csv"),
            combined["y_true"].to_numpy(dtype=np.int8), combined["p_voting"].to_numpy(dtype=float),
        )

    # Sensitivity predictions and summaries.
    sensitivity_combined: dict[tuple[str, str], pd.DataFrame] = {}
    for task, scheme in TASK_SCHEME.items():
        folds = range(1, 2) if task == "temporal_time" else range(1, 6)
        per_fold: dict[str, list[dict]] = {condition: [] for condition in TASK_CONDITIONS[task]}
        for fold in folds:
            reported = pd.read_csv(RESULTS / "sensitivity" / f"{task}_outer{fold}_metrics.csv")
            for condition in TASK_CONDITIONS[task]:
                frame = sensitivity_prediction(task, fold, condition)
                row = reported.loc[reported["condition"].eq(condition)].iloc[0]
                values = verify_metrics(
                    failures, records, f"{task}/outer{fold}/{condition}",
                    frame["y_true"].to_numpy(dtype=np.int8), frame["p_voting"].to_numpy(dtype=float), row,
                )
                per_fold[condition].append(values)
        summary = pd.read_csv(RESULTS / "sensitivity" / f"{task}_summary.csv")
        for condition in TASK_CONDITIONS[task]:
            combined = pd.concat(
                [sensitivity_prediction(task, fold, condition) for fold in folds], ignore_index=True
            ).sort_values("row_index").reset_index(drop=True)
            sensitivity_combined[(task, condition)] = combined
            row = summary.loc[summary["condition"].eq(condition)].iloc[0]
            fold_df = pd.DataFrame(per_fold[condition])
            for key in METRICS:
                compare_value(failures, f"{task}/summary/{condition}", f"{key}_fold_mean", row[f"{key}_fold_mean"], fold_df[key].mean())
                expected_std = fold_df[key].std(ddof=1) if len(fold_df) > 1 else 0.0
                compare_value(failures, f"{task}/summary/{condition}", f"{key}_fold_std", row[f"{key}_fold_std"], expected_std)
            verify_metrics(
                failures, records, f"{task}/combined/{condition}",
                combined["y_true"].to_numpy(dtype=np.int8), combined["p_voting"].to_numpy(dtype=float),
                {key: row[f"{key}_combined"] for key in METRICS},
            )

    # Temporal Top-k for every time specification.
    for condition in TASK_CONDITIONS["temporal_time"]:
        frame = sensitivity_combined[("temporal_time", condition)]
        verify_topk(
            failures, f"temporal_time/Top-k/{condition}",
            pd.read_csv(RESULTS / "sensitivity" / f"temporal_time_{condition}_topk.csv"),
            frame["y_true"].to_numpy(dtype=np.int8), frame["p_voting"].to_numpy(dtype=float),
        )

    # Incremental-value delta and cluster-bootstrap summary.
    b = sensitivity_combined[("stepwise", "model_b_age_time_history")]
    d = sensitivity_combined[("stepwise", "model_d_full")]
    observed_delta: dict[str, float] = {}
    if not b[["row_index", "y_true"]].equals(d[["row_index", "y_true"]]):
        add_failure(failures, "incremental_value", "row alignment", "mismatch", "match")
    else:
        mb = manual_metrics(b["y_true"].to_numpy(dtype=np.int8), b["p_voting"].to_numpy(dtype=float))
        md = manual_metrics(d["y_true"].to_numpy(dtype=np.int8), d["p_voting"].to_numpy(dtype=float))
        delta = pd.read_csv(RESULTS / "sensitivity" / "stepwise_incremental_value.csv").iloc[0]
        for key in METRICS:
            compare_value(failures, "incremental_value", f"delta_{key}", delta[f"delta_{key}"], md[key] - mb[key])
        observed_delta = {"delta_AUC": md["AUC"] - mb["AUC"], "delta_PR_AUC": md["PR-AUC"] - mb["PR-AUC"]}
    distribution = pd.read_csv(RESULTS / "sensitivity" / "incremental_value_cluster_bootstrap_distribution.csv")
    bootstrap = pd.read_csv(RESULTS / "sensitivity" / "incremental_value_cluster_bootstrap_summary.csv")
    for metric in ["delta_AUC", "delta_PR_AUC"]:
        row = bootstrap.loc[bootstrap["metric"].eq(metric)].iloc[0]
        if metric in observed_delta:
            compare_value(failures, "bootstrap", f"{metric}:observed", row["observed"], observed_delta[metric])
        compare_value(failures, "bootstrap", f"{metric}:mean", row["bootstrap_mean"], distribution[metric].mean())
        compare_value(failures, "bootstrap", f"{metric}:ci_2.5", row["ci_2.5"], distribution[metric].quantile(0.025))
        compare_value(failures, "bootstrap", f"{metric}:ci_97.5", row["ci_97.5"], distribution[metric].quantile(0.975))

    # Subgroup metrics by 신규검사(A) and 자격유지검사(B).
    subgroup_saved = pd.read_csv(RESULTS / "subgroup" / "검사유형별_모델성능.csv")
    subgroup_sources = {
        ("nested outer-CV OOF", "전체모델"): combined_outer["stratified"][["row_index", "y_true", "p_voting"]],
        ("nested outer-CV OOF", "연령시점이력_Model_B"): b,
        ("nested outer-CV OOF", "현재인지반응_Model_C"): sensitivity_combined[("stepwise", "model_c_current_cognitive_response")],
        ("2021~2022 시간분할", "전체모델"): combined_outer["temporal"][["row_index", "y_true", "p_voting"]],
        ("2021~2022 시간분할", "절대시점제외"): sensitivity_combined[("temporal_time", "without_absolute_time")],
        ("2021~2022 시간분할", "상대시점"): sensitivity_combined[("temporal_time", "relative_time")],
    }
    for (evaluation, model), frame in subgroup_sources.items():
        tests = meta.loc[frame["row_index"].to_numpy(dtype=np.int64), "Test"].to_numpy()
        for test in ["A", "B"]:
            mask = tests == test
            row = subgroup_saved.loc[
                subgroup_saved["평가조건"].eq(evaluation) & subgroup_saved["모델"].eq(model) & subgroup_saved["Test"].eq(test)
            ].iloc[0]
            if int(row["검사건수"]) != int(mask.sum()):
                add_failure(failures, f"subgroup/{evaluation}/{model}/{test}", "n", int(row["검사건수"]), int(mask.sum()))
            compare_value(failures, f"subgroup/{evaluation}/{model}/{test}", "label_rate", row["위험군비율"], frame.loc[mask, "y_true"].mean())
            verify_metrics(
                failures, records, f"subgroup/{evaluation}/{model}/{test}",
                frame.loc[mask, "y_true"].to_numpy(dtype=np.int8), frame.loc[mask, "p_voting"].to_numpy(dtype=float), row,
            )

    subgroup_topk = pd.read_csv(RESULTS / "subgroup" / "검사유형별_OOF_Topk.csv")
    primary_frame = combined_outer["stratified"]
    primary_tests = meta.loc[primary_frame["row_index"].to_numpy(dtype=np.int64), "Test"].to_numpy()
    for test in ["A", "B"]:
        mask = primary_tests == test
        saved = subgroup_topk.loc[subgroup_topk["Test"].eq(test)].copy()
        verify_topk(
            failures, f"subgroup/OOF_Top-k/{test}", saved,
            primary_frame.loc[mask, "y_true"].to_numpy(dtype=np.int8),
            primary_frame.loc[mask, "p_voting"].to_numpy(dtype=float),
        )

    distribution_saved = pd.read_csv(RESULTS / "subgroup" / "연도별_검사유형별_라벨분포.csv")
    distribution_expected = meta.assign(연도=meta["TestDate"] // 100).groupby(
        ["연도", "Test"], sort=True
    )["Label"].agg(검사건수="size", 위험군건수="sum", 위험군비율="mean").reset_index()
    merged_distribution = distribution_saved.merge(
        distribution_expected, on=["연도", "Test"], suffixes=("_saved", "_expected"), validate="one_to_one"
    )
    if len(merged_distribution) != len(distribution_expected):
        add_failure(failures, "subgroup/distribution", "row coverage", len(merged_distribution), len(distribution_expected))
    for row in merged_distribution.itertuples(index=False):
        if int(row.검사건수_saved) != int(row.검사건수_expected) or int(row.위험군건수_saved) != int(row.위험군건수_expected):
            add_failure(failures, "subgroup/distribution", f"count:{row.연도}:{row.Test}", "saved", "expected")
        compare_value(failures, "subgroup/distribution", f"rate:{row.연도}:{row.Test}", row.위험군비율_saved, row.위험군비율_expected)

    pd.DataFrame(records).to_csv(RESULTS / "numeric_verification_two_passes.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "metric_vectors_checked": len(records),
        "outer_prediction_files_checked": 11,
        "sensitivity_tasks_checked": list(TASK_SCHEME),
        "topk_checked": ["stratified", "group", "temporal", "temporal_time:3 conditions"],
        "reliability_checked": ["stratified", "group", "temporal"],
        "bootstrap_checked": True,
        "subgroups_checked": True,
        "pass1": "production sklearn-based metric functions",
        "pass2": "independent rank/AP/bin/Top-k/summary arithmetic",
        "manual_probability_averaging_used_for_results": False,
        "failures": failures,
    }
    (RESULTS / "numeric_verification_two_passes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(payload["status"], flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
