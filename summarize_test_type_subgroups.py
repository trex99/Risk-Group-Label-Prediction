"""Subgroup summaries for 신규검사(A) and 자격유지검사(B)."""

from __future__ import annotations

import json

import pandas as pd

from fold_isolated_pipeline import RESULTS, load_base, metrics, topk_table


TEST_NAMES = {"A": "신규검사", "B": "자격유지검사"}


def read_combined(pattern: str, folds=range(1, 6), usecols=None) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(RESULTS / pattern.format(fold=fold), usecols=usecols) for fold in folds],
        ignore_index=True,
    ).sort_values("row_index").reset_index(drop=True)


def add_test(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Test"] = meta.loc[out["row_index"].to_numpy(), "Test"].to_numpy()
    out["검사유형"] = out["Test"].map(TEST_NAMES)
    return out


def main() -> None:
    x, y, groups, meta = load_base()
    del x, y, groups
    out_dir = RESULTS / "subgroup"
    out_dir.mkdir(parents=True, exist_ok=True)
    distribution = meta.assign(연도=meta["TestDate"] // 100, 검사유형=meta["Test"].map(TEST_NAMES)).groupby(
        ["연도", "Test", "검사유형"], sort=True
    )["Label"].agg(검사건수="size", 위험군건수="sum", 위험군비율="mean").reset_index()
    distribution.to_csv(out_dir / "연도별_검사유형별_라벨분포.csv", index=False, encoding="utf-8-sig")

    conditions = {
        "전체모델": read_combined(
            "nested_evaluation/stratified_outer{fold}_predictions.csv.gz",
            usecols=["row_index", "y_true", "p_voting"],
        ),
        "연령시점이력_Model_B": read_combined(
            "sensitivity/stepwise_outer{fold}_model_b_age_time_history_predictions.csv.gz"
        ),
        "현재인지반응_Model_C": read_combined(
            "sensitivity/stepwise_outer{fold}_model_c_current_cognitive_response_predictions.csv.gz"
        ),
    }
    rows = []
    topk_rows = []
    for condition, frame in conditions.items():
        frame = add_test(frame, meta)
        for test, name in TEST_NAMES.items():
            subset = frame.loc[frame["Test"] == test]
            rows.append(
                {
                    "평가조건": "nested outer-CV OOF",
                    "모델": condition,
                    "Test": test,
                    "검사유형": name,
                    "검사건수": len(subset),
                    "위험군비율": float(subset["y_true"].mean()),
                    **metrics(subset["y_true"].to_numpy(), subset["p_voting"].to_numpy()),
                }
            )
            if condition == "전체모델":
                topk = topk_table(subset["y_true"].to_numpy(), subset["p_voting"].to_numpy())
                topk.insert(0, "검사유형", name)
                topk.insert(0, "Test", test)
                topk_rows.append(topk)

    temporal_conditions = {
        "전체모델": pd.read_csv(
            RESULTS / "nested_evaluation" / "temporal_outer1_predictions.csv.gz",
            usecols=["row_index", "y_true", "p_voting"],
        ),
        "절대시점제외": pd.read_csv(
            RESULTS / "sensitivity" / "temporal_time_outer1_without_absolute_time_predictions.csv.gz"
        ),
        "상대시점": pd.read_csv(
            RESULTS / "sensitivity" / "temporal_time_outer1_relative_time_predictions.csv.gz"
        ),
    }
    for condition, frame in temporal_conditions.items():
        frame = add_test(frame, meta)
        for test, name in TEST_NAMES.items():
            subset = frame.loc[frame["Test"] == test]
            rows.append(
                {
                    "평가조건": "2021~2022 시간분할",
                    "모델": condition,
                    "Test": test,
                    "검사유형": name,
                    "검사건수": len(subset),
                    "위험군비율": float(subset["y_true"].mean()),
                    **metrics(subset["y_true"].to_numpy(), subset["p_voting"].to_numpy()),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "검사유형별_모델성능.csv", index=False, encoding="utf-8-sig")
    pd.concat(topk_rows, ignore_index=True).to_csv(
        out_dir / "검사유형별_OOF_Topk.csv", index=False, encoding="utf-8-sig"
    )
    (out_dir / "검사유형별_분석_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "A": "신규검사, 2018~2022",
                "B": "자격유지검사, 2016~2022",
                "unit": "검사건(Test_id)",
                "retraining": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
