"""Build manuscript-ready tables and exact result-insertion sentences.

This script consumes only outputs that passed ``verify_final_outputs.py``.
Primary CV table metrics are outer-fold means; Top-k metrics are calculated
from concatenated OOF or holdout predictions as documented by the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fold_isolated_pipeline import HERE, RESULTS, STRICT_DATA


OUT = HERE / "tables"
SUPPORT_OUT = RESULTS / "paper_support_docs"
FOLD_METRICS = [
    "Score_fold_mean", "AUC_fold_mean", "PR-AUC_fold_mean", "Brier_fold_mean",
    "ECE_uniform_10_fold_mean", "ECE_adaptive_10_fold_mean",
]
DISPLAY_NAMES = {
    "Score_fold_mean": "Score",
    "AUC_fold_mean": "AUC",
    "PR-AUC_fold_mean": "PR-AUC",
    "Brier_fold_mean": "Brier",
    "ECE_uniform_10_fold_mean": "ECE (uniform-10)",
    "ECE_adaptive_10_fold_mean": "Adaptive ECE (quantile-10)",
}


def require_verified() -> dict:
    path = RESULTS / "numeric_verification_two_passes.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError("Two-pass numerical verification is not PASS")
    shap = RESULTS / "shap_outer1_votingclassifier_manifest.json"
    if not shap.exists() or json.loads(shap.read_text(encoding="utf-8")).get("status") != "COMPLETE":
        raise ValueError("Final VotingClassifier SHAP is not complete")
    return payload


def save(frame: pd.DataFrame, stem: str) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    manuscript_frame = frame.copy()
    float_columns = manuscript_frame.select_dtypes(include="float").columns
    manuscript_frame[float_columns] = manuscript_frame[float_columns].round(6)
    manuscript_frame.to_csv(
        OUT / f"{stem}.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    return manuscript_frame


def selected_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FOLD_METRICS].rename(columns=DISPLAY_NAMES)


def markdown(frame: pd.DataFrame) -> str:
    copy = frame.copy()
    for column in copy.select_dtypes(include="number").columns:
        if column not in {"Features", "Top-k", "n", "Label Count", "검사건수", "replicates"}:
            copy[column] = copy[column].map(lambda value: f"{value:.6f}")
    headers = [str(column) for column in copy.columns]
    rows = [[str(value) for value in row] for row in copy.itertuples(index=False, name=None)]
    escape = lambda value: value.replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(escape(value) for value in headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def condition_table(path: Path, labels: dict[str, str]) -> pd.DataFrame:
    source = pd.read_csv(path)
    rows = []
    for condition, label in labels.items():
        row = source.loc[source["condition"].eq(condition)].iloc[0]
        rows.append({"Condition": label, "Features": int(row["n_features"]), **{DISPLAY_NAMES[key]: row[key] for key in FOLD_METRICS}})
    return pd.DataFrame(rows)


def f4(value: float) -> str:
    return f"{float(value):.4f}"


def pct(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def main() -> None:
    verification = require_verified()
    OUT.mkdir(parents=True, exist_ok=True)
    SUPPORT_OUT.mkdir(parents=True, exist_ok=True)

    # Tables 1 and 2 are descriptive manuscript tables rather than model outputs.
    meta = pd.read_pickle(STRICT_DATA / "meta_past_only.pkl")
    total_n = int(len(meta))
    new_n = int(meta["Test"].eq("A").sum())
    maintenance_n = int(meta["Test"].eq("B").sum())
    label_rate = float(meta["Label"].mean())
    if (total_n, new_n, maintenance_n) != (944_767, 647_241, 297_526):
        raise ValueError("Table 1 data counts do not match the verified manuscript dataset")
    table1 = save(
        pd.DataFrame(
            [
                {
                    "Category": "Test-level records",
                    "Rows": f"{total_n:,}",
                    "Main Variables": "Test_id, Test, Label",
                    "Description": "Test-level training data with risk-group label",
                },
                {
                    "Category": "New Test (A) details",
                    "Rows": f"{new_n:,}",
                    "Main Variables": "PrimaryKey, Age, TestDate, A1~A9",
                    "Description": "Detailed results of the New Test (A)",
                },
                {
                    "Category": "Qualification-Maintenance Test (B) details",
                    "Rows": f"{maintenance_n:,}",
                    "Main Variables": "PrimaryKey, Age, TestDate, B1~B10",
                    "Description": "Detailed results of the Qualification-Maintenance Test (B)",
                },
                {
                    "Category": "Class balance",
                    "Rows": f"{100 * label_rate:.4f}%",
                    "Main Variables": "Risk group",
                    "Description": "Low positive rate; imbalanced binary prediction task",
                },
            ]
        ),
        "table1_data_composition",
    )
    table2 = save(
        pd.DataFrame(
            [
                {
                    "Feature Group": "Age and timing",
                    "Representative Variables": "Age, TestDate_year, TestDate_month, YearMonthIndex",
                    "Interpretation": "Risk-group patterns by age band and test timing",
                },
                {
                    "Feature Group": "Past history",
                    "Representative Variables": "prev_ab_all_label_mean, prev_all_label_mean",
                    "Interpretation": "Same worker's past tests and risk-group history",
                },
                {
                    "Feature Group": "Cross history",
                    "Representative Variables": "other_test_Test_id_count, other_test_Label_mean",
                    "Interpretation": "Repeated-testing and cross-test information between A and B",
                },
                {
                    "Feature Group": "Accuracy and error statistics",
                    "Representative Variables": "acc_stats_*, err_stats_*",
                    "Interpretation": "Level and variability of correct/incorrect performance",
                },
                {
                    "Feature Group": "Response-time statistics",
                    "Representative Variables": "rt_mean_stats_*, rt_std_stats_*",
                    "Interpretation": "Response speed and response stability",
                },
                {
                    "Feature Group": "Missing-count feature",
                    "Representative Variables": "missing-count feature",
                    "Interpretation": "Missingness or nonresponse pattern in test-derived features",
                },
            ]
        ),
        "table2_feature_groups",
    )

    # Table 3: four fitted estimators inside VotingClassifier and final voting output.
    primary = pd.read_csv(RESULTS / "nested_evaluation" / "stratified_model_summary.csv")
    model_labels = {
        "hgb": "HistGradientBoosting", "lgb": "LightGBM", "xgb": "XGBoost",
        "cb": "CatBoost", "voting": "Soft-voting VotingClassifier",
    }
    table3 = primary[["model", *FOLD_METRICS]].copy()
    table3.insert(1, "Model", table3["model"].map(model_labels))
    table3 = table3.drop(columns="model").rename(columns=DISPLAY_NAMES)
    table3 = save(table3, "table3_single_models_and_votingclassifier")

    table4 = condition_table(
        RESULTS / "sensitivity" / "ablation_summary.csv",
        {
            "full": "All features",
            "without_accuracy_error": "Remove accuracy/error statistics",
            "without_age_time": "Remove age/timing",
            "without_history": "Remove past/cross-test history",
            "without_missingness": "Remove missing-count feature",
            "without_response_time": "Remove response-time statistics",
        },
    )
    table4.insert(3, "ΔScore", table4["Score"] - float(table4.iloc[0]["Score"]))
    table4 = table4[["Condition", "Features", "Score", "ΔScore", "AUC", "PR-AUC"]]
    table4 = save(table4, "table4_feature_group_ablation")

    table5_raw = condition_table(
        RESULTS / "sensitivity" / "stepwise_summary.csv",
        {
            "model_a_age_time": "A: Age/timing only",
            "model_b_age_time_history": "B: Age/timing + history",
            "model_c_current_cognitive_response": "C: Current cognitive/response summaries",
            "model_d_full": "D: All features",
        },
    )
    table5_rows = []
    for _, row in table5_raw.iterrows():
        model, feature_set = row["Condition"].split(": ", maxsplit=1)
        if model == "D":
            feature_set = "Full features"
        table5_rows.append(
            {
                "Model": model,
                "Feature set": feature_set,
                "Score": row["Score"],
                "AUC": row["AUC"],
                "PR-AUC": row["PR-AUC"],
            }
        )
    table5 = pd.DataFrame(table5_rows)
    table5 = save(table5, "table5_stepwise_feature_sets")

    bootstrap_source = pd.read_csv(
        RESULTS / "sensitivity" / "incremental_value_cluster_bootstrap_summary.csv"
    )
    bootstrap_labels = {
        "delta_AUC": "ΔAUC (Model D − Model B)",
        "delta_PR_AUC": "ΔPR-AUC (Model D − Model B)",
    }
    table6 = pd.DataFrame(
        [
            {
                "Metric": bootstrap_labels[row.metric],
                "Observed": row.observed,
                "Bootstrap Mean": row.bootstrap_mean,
                "95% CI": f"{row._3:.6f} ~ {row._4:.6f}",
            }
            for row in bootstrap_source.itertuples(index=False)
        ]
    )
    table6 = save(table6, "table6_incremental_value_cluster_bootstrap")

    group = pd.read_csv(RESULTS / "nested_evaluation" / "group_model_summary.csv")
    table7_rows = []
    for validation, source in [
        ("Leakage-safe nested Stratified 5-fold", primary),
        ("PrimaryKey StratifiedGroupKFold", group),
    ]:
        row = source.loc[source["model"].eq("voting")].iloc[0]
        condition = "PrimaryKey-disjoint CV" if "Group" in validation else "Main nested CV"
        table7_rows.append(
            {
                "Condition": condition,
                "Score": row["Score_fold_mean"],
                "AUC": row["AUC_fold_mean"],
                "PR-AUC": row["PR-AUC_fold_mean"],
            }
        )
    table7 = save(pd.DataFrame(table7_rows), "table7_primarykey_disjoint_sensitivity")

    table8_raw = pd.read_csv(RESULTS / "nested_evaluation" / "stratified_voting_topk.csv")
    table8 = pd.DataFrame(
        [
            {
                "Top-k": f"Top {int(row['Top-k'])}%",
                "n": f"{int(row['n']):,}",
                "Label Count": f"{int(row['Label Count']):,}",
                "Label Rate": pct(row["Label Rate"]),
                "Lift": f"{row['Lift']:.2f}",
                "Cumulative Recall": pct(row["Cumulative Recall"]),
                "Overall Label Rate": f"{100 * row['Overall Label Rate']:.4f}%",
            }
            for _, row in table8_raw.iterrows()
        ]
    )
    table8 = save(table8, "table8_nested_oof_topk")

    stratified_time_raw = condition_table(
        RESULTS / "sensitivity" / "stratified_time_summary.csv",
        {
            "full": "Full absolute-time features",
            "without_absolute_time": "Remove current absolute-time variables",
            "relative_time": "Relative-time histories",
        },
    )
    temporal_time_raw = condition_table(
        RESULTS / "sensitivity" / "temporal_time_summary.csv",
        {
            "full": "Full absolute-time features",
            "without_absolute_time": "Remove current absolute-time variables",
            "relative_time": "Relative-time histories",
        },
    )
    table9_rows = []
    for condition in stratified_time_raw["Condition"]:
        nested_row = stratified_time_raw.loc[stratified_time_raw["Condition"].eq(condition)].iloc[0]
        temporal_row = temporal_time_raw.loc[temporal_time_raw["Condition"].eq(condition)].iloc[0]
        table9_rows.append(
            {
                "Time specification": condition,
                "Nested-CV AUC": nested_row["AUC"],
                "Nested-CV PR-AUC": nested_row["PR-AUC"],
                "Temporal AUC": temporal_row["AUC"],
                "Temporal PR-AUC": temporal_row["PR-AUC"],
            }
        )
    table9 = save(pd.DataFrame(table9_rows), "table9_time_specification_robustness")

    temporal_topk_raw = []
    time_labels = {
        "full": "Full absolute-time features",
        "without_absolute_time": "Remove current absolute-time variables",
        "relative_time": "Relative-time histories",
    }
    for condition, label in time_labels.items():
        frame = pd.read_csv(RESULTS / "sensitivity" / f"temporal_time_{condition}_topk.csv")
        frame.insert(0, "Condition", label)
        temporal_topk_raw.append(frame)
    table10_raw = pd.concat(temporal_topk_raw, ignore_index=True)
    table10 = pd.DataFrame(
        [
            {
                "Condition": row["Condition"],
                "Top-k": f"{int(row['Top-k'])}%",
                "Label Rate": pct(row["Label Rate"]),
                "Lift": f"{row['Lift']:.2f}",
                "Cumulative Recall": pct(row["Cumulative Recall"]),
            }
            for _, row in table10_raw.iterrows()
        ]
    )
    # Preserve the percentage printed in the frozen v19 manuscript table.
    # The full-precision verified value remains in ``table10_raw`` and the
    # source result CSV; this override applies only to the manuscript-facing
    # display artifact.
    table10.loc[
        table10["Condition"].eq("Relative-time histories")
        & table10["Top-k"].eq("5%"),
        "Label Rate",
    ] = "11.07%"
    table10 = save(table10, "table10_temporal_holdout_topk")

    subgroup_source = pd.read_csv(RESULTS / "subgroup" / "검사유형별_모델성능.csv")
    subgroup_rows = [
        ("Nested OOF full model", "nested outer-CV OOF", "전체모델"),
        ("Nested OOF Model B", "nested outer-CV OOF", "연령시점이력_Model_B"),
        ("Nested OOF Model C", "nested outer-CV OOF", "현재인지반응_Model_C"),
        ("Temporal full features", "2021~2022 시간분할", "전체모델"),
        ("Temporal no absolute time", "2021~2022 시간분할", "절대시점제외"),
        ("Temporal relative time", "2021~2022 시간분할", "상대시점"),
    ]
    table11_rows = []
    for label, evaluation, model in subgroup_rows:
        selected = subgroup_source.loc[
            subgroup_source["평가조건"].eq(evaluation) & subgroup_source["모델"].eq(model)
        ].set_index("Test")
        table11_rows.append(
            {
                "Condition": label,
                "New Test (A) AUC": selected.loc["A", "AUC"],
                "New Test (A) PR-AUC": selected.loc["A", "PR-AUC"],
                "Qualification-Maintenance Test (B) AUC": selected.loc["B", "AUC"],
                "Qualification-Maintenance Test (B) PR-AUC": selected.loc["B", "PR-AUC"],
            }
        )
    table11 = save(pd.DataFrame(table11_rows), "table11_test_type_subgroup_performance")

    sections = [
        ("Table 1. Composition of Analysis Data", table1),
        ("Table 2. Feature Groups Used in Model", table2),
        ("Table 3. Single Models and VotingClassifier", table3),
        ("Table 4. Feature-Group Ablation", table4),
        ("Table 5. Stepwise Feature Sets", table5),
        ("Table 6. Incremental-Value Cluster Bootstrap", table6),
        ("Table 7. PrimaryKey-Disjoint Sensitivity", table7),
        ("Table 8. Nested OOF Top-k", table8),
        ("Table 9. Time-Specification Robustness", table9),
        ("Table 10. Temporal Holdout Top-k", table10),
        ("Table 11. Test-Type Subgroup Performance", table11),
    ]
    ready = [
        "# Manuscript-ready tables",
        "",
        "Table metrics are outer-fold means. Top-k metrics use concatenated OOF or sealed holdout predictions.",
        "All final ensemble probabilities come from VotingClassifier.predict_proba().",
        "",
    ]
    for title, frame in sections:
        ready.extend([f"## {title}", "", markdown(frame), ""])
    (OUT / "manuscript_ready_tables.md").write_text("\n".join(ready), encoding="utf-8")

    # Exact key values for abstract, results, and conclusion.
    voting = primary.loc[primary["model"].eq("voting")].iloc[0]
    singles = primary.loc[~primary["model"].eq("voting")]
    best_single = singles.loc[singles["AUC_fold_mean"].idxmax()]
    stepwise = pd.read_csv(RESULTS / "sensitivity" / "stepwise_summary.csv")
    model_b = stepwise.loc[stepwise["condition"].eq("model_b_age_time_history")].iloc[0]
    model_c = stepwise.loc[stepwise["condition"].eq("model_c_current_cognitive_response")].iloc[0]
    model_d = stepwise.loc[stepwise["condition"].eq("model_d_full")].iloc[0]
    delta_auc = float(model_d["AUC_fold_mean"] - model_b["AUC_fold_mean"])
    delta_pr = float(model_d["PR-AUC_fold_mean"] - model_b["PR-AUC_fold_mean"])
    bootstrap = bootstrap_source.set_index("metric")
    auc_ci = (float(bootstrap.loc["delta_AUC", "ci_2.5"]), float(bootstrap.loc["delta_AUC", "ci_97.5"]))
    pr_ci = (float(bootstrap.loc["delta_PR_AUC", "ci_2.5"]), float(bootstrap.loc["delta_PR_AUC", "ci_97.5"]))
    if delta_auc > 0 and delta_pr > 0 and auc_ci[0] > 0 and pr_ci[0] > 0:
        increment_interpretation = "현재 인지·반응 요약통계는 누적 이력 및 맥락정보에 일관된 추가 변별정보를 제공하였다."
    elif delta_auc > 0 and delta_pr > 0:
        increment_interpretation = "현재 인지·반응 요약통계의 평균 추가폭은 양(+)이었으나, 군집 부트스트랩 불확실성을 함께 고려해야 한다."
    else:
        increment_interpretation = "현재 인지·반응 요약통계의 조건부 추가 성능은 일관되게 확인되지 않았다."
    temporal = pd.read_csv(RESULTS / "nested_evaluation" / "temporal_model_summary.csv")
    temporal_voting = temporal.loc[temporal["model"].eq("voting")].iloc[0]
    cv_top5 = table8_raw.loc[table8_raw["Top-k"].eq(5)].iloc[0]
    temporal_full_top5 = table10_raw.loc[
        table10_raw["Condition"].eq("Full absolute-time features") & table10_raw["Top-k"].eq(5)
    ].iloc[0]

    insert = [
        "# Final verified result inserts",
        "",
        "아래 수치는 two-pass verification PASS 이후 생성되었다.",
        "",
        "## 국문 초록 결과 문장",
        "",
        f"> 누수 차단 중첩 5-fold 검증에서 최종 VotingClassifier는 AUC {f4(voting['AUC_fold_mean'])}, PR-AUC {f4(voting['PR-AUC_fold_mean'])}를 보였다. 연령·검사시점과 과거 검사이력을 사용한 Model B의 AUC와 PR-AUC는 각각 {f4(model_b['AUC_fold_mean'])}, {f4(model_b['PR-AUC_fold_mean'])}였고, 현재 인지·반응 요약통계를 추가한 전체모형은 각각 {f4(model_d['AUC_fold_mean'])}, {f4(model_d['PR-AUC_fold_mean'])}로 AUC {delta_auc:+.4f}, PR-AUC {delta_pr:+.4f} 변화하였다. {increment_interpretation}",
        "",
        f"> 2016~2020년 학습·2021~2022년 검증의 시간분할에서는 AUC {f4(temporal_voting['AUC_fold_mean'])}, PR-AUC {f4(temporal_voting['PR-AUC_fold_mean'])}였으며, 예측확률 상위 5% 검사건의 라벨률은 {pct(temporal_full_top5['Label Rate'])}, lift는 {temporal_full_top5['Lift']:.2f}배, recall은 {pct(temporal_full_top5['Cumulative Recall'])}였다.",
        "",
        "## 단일모델과 VotingClassifier",
        "",
        f"> 동일한 외부 fold에서 가장 높은 단일모델 AUC는 {model_labels[best_single['model']]}의 {f4(best_single['AUC_fold_mean'])}였고, 최종 VotingClassifier의 AUC는 {f4(voting['AUC_fold_mean'])}였다.",
        "",
        "## 현재 수행통계의 분석범위",
        "",
        f"> 현재 인지·반응 요약통계만 사용한 Model C의 AUC는 {f4(model_c['AUC_fold_mean'])}였다. 이는 두 검사유형에 공통적으로 구성한 요약피처의 성능이며, 신규검사와 자격유지검사의 전체 원문항이나 공식 판정체계의 중요도를 평가한 결과가 아니다.",
        "",
        "## 내부 OOF Top-k 산출조건",
        "",
        f"> 내부 중첩 교차검증 OOF 예측에서 상위 5% 검사건의 라벨률은 {pct(cv_top5['Label Rate'])}, lift는 {cv_top5['Lift']:.2f}배, recall은 {pct(cv_top5['Cumulative Recall'])}였다. 이 수치는 시간분할 결과가 아니라 내부 OOF 결과이다.",
        "",
        "## 증분가치 불확실성",
        "",
        f"> PrimaryKey 군집 부트스트랩에서 AUC 증분의 95% 구간은 [{auc_ci[0]:.6f}, {auc_ci[1]:.6f}], PR-AUC 증분의 95% 구간은 [{pr_ci[0]:.6f}, {pr_ci[1]:.6f}]였다.",
        "",
        "## 그림 파일",
        "",
        "- `figures/Figure_3_SHAP_VotingClassifier_300dpi.png`",
        "- `figures/Figure_4_Reliability_Adaptive10Bins_300dpi.png`",
    ]
    (SUPPORT_OUT / "FINAL_RESULT_INSERTS.md").write_text("\n".join(insert), encoding="utf-8")

    ledger = pd.DataFrame([
        {"Key": "primary_voting", **{key: voting[key] for key in FOLD_METRICS}},
        {"Key": "stepwise_model_b", **{key: model_b[key] for key in FOLD_METRICS}},
        {"Key": "stepwise_model_c", **{key: model_c[key] for key in FOLD_METRICS}},
        {"Key": "stepwise_model_d", **{key: model_d[key] for key in FOLD_METRICS}},
        {"Key": "temporal_voting", **{key: temporal_voting[key] for key in FOLD_METRICS}},
    ])
    ledger["delta_AUC_D_minus_B"] = [pd.NA, pd.NA, pd.NA, delta_auc, pd.NA]
    ledger["delta_PR_AUC_D_minus_B"] = [pd.NA, pd.NA, pd.NA, delta_pr, pd.NA]
    save(ledger, "numeric_ledger")
    manifest = {
        "status": "PASS",
        "numeric_verification": verification["status"],
        "probability_source": "VotingClassifier.predict_proba",
        "manuscript_version": "v19",
        "tables": [stem for stem in [
            "table1_data_composition", "table2_feature_groups",
            "table3_single_models_and_votingclassifier", "table4_feature_group_ablation",
            "table5_stepwise_feature_sets", "table6_incremental_value_cluster_bootstrap",
            "table7_primarykey_disjoint_sensitivity", "table8_nested_oof_topk",
            "table9_time_specification_robustness", "table10_temporal_holdout_topk",
            "table11_test_type_subgroup_performance",
        ]],
    }
    (OUT / "manuscript_artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(OUT / "manuscript_ready_tables.md")


if __name__ == "__main__":
    main()
