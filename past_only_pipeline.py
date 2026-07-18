# -*- coding: utf-8 -*-
"""Past-only feature pipeline for the remake experiments.

All cross-test ``other_test_*`` aggregates are computed from opposite-test
records satisfying ``source.TestDate < target.TestDate``. The rest of the
feature engineering intentionally follows the original training notebook so the
effect of time-available cross-test history can be evaluated directly.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import brier_score_loss, roc_auc_score


RANDOM_STATE = 42
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

A_SEQ_COLS = [
    ["A1-1", "A1-2", "A1-3"],
    ["A2-1", "A2-2", "A2-3"],
    ["A3-1", "A3-2", "A3-3", "A3-4", "A3-5", "A3-6"],
    ["A4-1", "A4-2", "A4-3", "A4-4"],
    ["A5-1", "A5-2", "A5-3"],
]
A_RT_COLS = ["A1-4", "A2-4", "A3-7", "A4-5"]
A_ADD_COLS = [
    "Age",
    "TestDate",
    "A6-1",
    "A7-1",
    "A8-1",
    "A8-2",
    "A9-1",
    "A9-2",
    "A9-3",
    "A9-4",
    "A9-5",
    "TestDate_year",
    "TestDate_month",
    "YearMonthIndex",
    "Test_id",
    "PrimaryKey",
]

B_SEQ_COLS = [["B1-1", "B1-3"], ["B2-1", "B2-3"], ["B3-1"], ["B4-1"], ["B5-1"], ["B6"], ["B7"], ["B8"]]
B_RT_COLS = ["B1-2", "B2-2", "B3-2", "B4-2", "B5-2"]
B_ADD_COLS = [
    "Age",
    "TestDate",
    "B9-1",
    "B9-2",
    "B9-3",
    "B9-4",
    "B9-5",
    "B10-1",
    "B10-2",
    "B10-3",
    "B10-4",
    "B10-5",
    "B10-6",
    "TestDate_year",
    "TestDate_month",
    "YearMonthIndex",
    "Test_id",
    "PrimaryKey",
]

OTHER_TEST_COLS = [
    "other_test_Test_id_count",
    "other_test_YearMonthIndex_mean",
    "other_test_YearMonthIndex_std",
    "other_test_Label_mean",
]


def default_data_path() -> Path:
    candidates = [
        PROJECT_DIR / "open_v2",
        PROJECT_DIR.parent / "open_v2",
        PROJECT_DIR.parent.parent / "open_v2",
        PROJECT_DIR.parent.parent.parent / "open_v2",
    ]
    for path in candidates:
        if (path / "train.csv").exists() and (path / "train" / "A.csv").exists() and (path / "train" / "B.csv").exists():
            return path
    raise FileNotFoundError("Could not find open_v2 with train.csv, train/A.csv, and train/B.csv.")


def load_data(data_path: str | Path):
    data_path = Path(data_path)
    train = pd.read_csv(data_path / "train.csv")
    train_a = pd.read_csv(data_path / "train" / "A.csv")
    train_b = pd.read_csv(data_path / "train" / "B.csv")
    return train, train_a, train_b


def merge_meta2ab(df: pd.DataFrame, df_a: pd.DataFrame, df_b: pd.DataFrame, cols: list[str]):
    if cols:
        df_a = df_a.merge(df[cols], how="left", on="Test_id", validate="1:1")
        df_b = df_b.merge(df[cols], how="left", on="Test_id", validate="1:1")

    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a.fillna("", inplace=True)
    df_b.fillna("", inplace=True)
    df_a["Age"] = df_a["Age"].map(lambda x: int(x[:-1]) + 5 if x[-1] == "b" else int(x[:-1]))
    df_b["Age"] = df_b["Age"].map(lambda x: int(x[:-1]) + 5 if x[-1] == "b" else int(x[:-1]))

    for frame in (df_a, df_b):
        frame["TestDate_year"] = frame["TestDate"] // 100
        frame["TestDate_month"] = frame["TestDate"] % 100
        frame["YearMonthIndex"] = frame["TestDate_year"] * 12 + frame["TestDate_month"]

    return df_a, df_b


def make_df_ab(train_a: pd.DataFrame, train_b: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.concat(
            [
                train_a[["Test_id", "Test", "PrimaryKey", "TestDate", "YearMonthIndex", "Label"]],
                train_b[["Test_id", "Test", "PrimaryKey", "TestDate", "YearMonthIndex", "Label"]],
            ],
            ignore_index=True,
        )
        .rename(columns={"Label": "all_label"})
        .reset_index(drop=True)
    )


def merge_seq(x: pd.Series, cols_list: list[list[str]]) -> str:
    merged = []
    for cols in cols_list:
        zip_args = [str(x[col]).split(",") for col in cols]
        merged.append(",".join([f'{cols[0][:2]}_{"".join(tup)}' for tup in zip(*zip_args)]))
    return ",".join(merged)


def expanding_prev_features(
    df_src: pd.DataFrame,
    df_trg: pd.DataFrame,
    col: str,
    prefix_name: str,
    is_train: bool = True,
) -> pd.DataFrame:
    """Add aggregates from records in months strictly before each target month.

    ``TestDate`` is available only at year-month resolution.  Records sharing a
    ``PrimaryKey`` and ``TestDate`` therefore have no observable ordering and
    must not be used as one another's history.  The same strict rule is applied
    to training rows and to later validation/target rows.
    """

    df_src = df_src.sort_values(["PrimaryKey", "TestDate"], kind="mergesort").reset_index(drop=True)
    df_trg = df_trg.copy()

    if is_train:
        e_tmp = df_src.groupby("PrimaryKey")[col].expanding()
        mean_tmp = e_tmp.mean().reset_index().groupby("PrimaryKey").shift(1)
        sum_tmp = e_tmp.sum().reset_index().groupby("PrimaryKey").shift(1)
        std_tmp = e_tmp.std().reset_index().groupby("PrimaryKey").shift(1)

        stat_cols = [
            f"{prefix_name}_{col}_mean",
            f"{prefix_name}_{col}_sum",
            f"{prefix_name}_{col}_std",
        ]
        row_stats = df_src[["Test_id", "PrimaryKey", "TestDate"]].copy()
        row_stats[stat_cols[0]] = mean_tmp[col]
        row_stats[stat_cols[1]] = sum_tmp[col]
        row_stats[stat_cols[2]] = std_tmp[col]

        # The first row in each person-month has history from strictly earlier
        # months.  Apply that same history to every row in the tied month.
        first_in_month = ~row_stats.duplicated(["PrimaryKey", "TestDate"], keep="first")
        month_stats = row_stats.loc[first_in_month, ["PrimaryKey", "TestDate", *stat_cols]]
        strict_stats = row_stats[["Test_id", "PrimaryKey", "TestDate"]].merge(
            month_stats,
            how="left",
            on=["PrimaryKey", "TestDate"],
            validate="m:1",
        )
        df_trg = df_trg.merge(strict_stats[["Test_id", *stat_cols]], how="left", on="Test_id", validate="1:1")
    else:
        # Temporal validation supplies only 2016-2020 source history for every
        # 2021-2022 target row, so all source records are strictly earlier.
        cols_dict = {
            "mean": f"{prefix_name}_{col}_mean",
            "sum": f"{prefix_name}_{col}_sum",
            "std": f"{prefix_name}_{col}_std",
        }
        features = df_src.groupby("PrimaryKey")[col].agg(["mean", "sum", "std"]).reset_index().rename(columns=cols_dict)
        df_trg = df_trg.merge(features, how="left", on="PrimaryKey")
    return df_trg


def preprocess_data(
    df: pd.DataFrame,
    df_ab: pd.DataFrame,
    cols_merge_seq_test: list[list[str]],
    ab_type: str,
    response_time_cols: list[str],
    add_cols: list[str],
    tdm: CountVectorizer | None = None,
    is_train: bool = True,
):
    df = df.copy()
    df["merge_seq_test"] = df.apply(merge_seq, axis=1, cols_list=cols_merge_seq_test)

    if tdm is None:
        tdm = CountVectorizer(max_features=None)
        tdm.fit(df["merge_seq_test"])
    features = pd.DataFrame(
        tdm.transform(df["merge_seq_test"]).toarray(),
        columns=tdm.get_feature_names_out(),
        index=df.index,
    )

    for col in response_time_cols:
        arr = df[col].str.split(",", expand=True).replace("", np.nan).to_numpy(dtype=np.float64)
        features[f"seq_rt_{col}_mean"] = np.mean(np.abs(arr), axis=1)
        features[f"seq_rt_{col}_std"] = np.std(np.abs(arr), axis=1)

    features = pd.concat([features.reset_index(drop=True), df[add_cols].reset_index(drop=True)], axis=1)

    for col in ["YearMonthIndex", "all_label"]:
        features = expanding_prev_features(df_ab, features, col, "prev_ab", is_train)

    df_src = df_ab.loc[df_ab["Test"] == ab_type].reset_index(drop=True)
    for col in ["YearMonthIndex", "all_label"]:
        features = expanding_prev_features(df_src, features, col, "prev", is_train)

    return features.drop(columns=["TestDate", "Test_id", "PrimaryKey"]), tdm


def cumulative_opposite_history(source: pd.DataFrame) -> pd.DataFrame:
    src = source[["PrimaryKey", "TestDate", "YearMonthIndex", "Label"]].copy()
    src["_source_order"] = np.arange(len(src), dtype=np.int64)
    src["_one"] = 1.0
    src["_ymi"] = src["YearMonthIndex"].astype(float)
    src["_ymi_sq"] = src["_ymi"] * src["_ymi"]
    src["_label"] = src["Label"].astype(float)

    src = src.sort_values(["PrimaryKey", "TestDate", "_source_order"]).reset_index(drop=True)
    grouped = src.groupby("PrimaryKey", sort=False)
    src["_count"] = grouped["_one"].cumsum()
    src["_ymi_sum"] = grouped["_ymi"].cumsum()
    src["_ymi_sq_sum"] = grouped["_ymi_sq"].cumsum()
    src["_label_sum"] = grouped["_label"].cumsum()

    return src[
        [
            "PrimaryKey",
            "TestDate",
            "_source_order",
            "_count",
            "_ymi_sum",
            "_ymi_sq_sum",
            "_label_sum",
        ]
    ].sort_values(["TestDate", "PrimaryKey", "_source_order"])


def add_past_only_other_test_features_one_side(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    source_history = cumulative_opposite_history(source)
    target_keys = target[["PrimaryKey", "TestDate"]].copy()
    target_keys["_row_id"] = np.arange(len(target_keys), dtype=np.int64)
    target_keys = target_keys.sort_values(["TestDate", "PrimaryKey", "_row_id"])

    merged = pd.merge_asof(
        target_keys,
        source_history,
        on="TestDate",
        by="PrimaryKey",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row_id")

    count = merged["_count"].astype(float)
    ymi_sum = merged["_ymi_sum"].astype(float)
    ymi_sq_sum = merged["_ymi_sq_sum"].astype(float)
    label_sum = merged["_label_sum"].astype(float)

    out = pd.DataFrame(index=target.index)
    out["other_test_Test_id_count"] = count.to_numpy()
    out["other_test_YearMonthIndex_mean"] = (ymi_sum / count).to_numpy()
    variance_num = ymi_sq_sum - (ymi_sum * ymi_sum / count)
    sample_var = (variance_num / (count - 1.0)).where(count > 1.0)
    out["other_test_YearMonthIndex_std"] = np.sqrt(sample_var.clip(lower=0)).to_numpy()
    out["other_test_Label_mean"] = (label_sum / count).to_numpy()
    return out


def add_past_only_other_test_features(train_a, train_b, train_ft_a, train_ft_b):
    other_a = add_past_only_other_test_features_one_side(train_b, train_a)
    other_b = add_past_only_other_test_features_one_side(train_a, train_b)

    train_ft_a = pd.concat([train_ft_a.reset_index(drop=True), other_a.reset_index(drop=True)], axis=1)
    train_ft_b = pd.concat([train_ft_b.reset_index(drop=True), other_b.reset_index(drop=True)], axis=1)
    train_ft_a["isna_sum"] = train_ft_a.isna().sum(axis=1)
    train_ft_b["isna_sum"] = train_ft_b.isna().sum(axis=1)
    return train_ft_a, train_ft_b


def add_past_only_other_test_features_from_sources(source_a, source_b, target_a, target_b, features_a, features_b):
    other_a = add_past_only_other_test_features_one_side(source_b, target_a)
    other_b = add_past_only_other_test_features_one_side(source_a, target_b)

    features_a = pd.concat([features_a.reset_index(drop=True), other_a.reset_index(drop=True)], axis=1)
    features_b = pd.concat([features_b.reset_index(drop=True), other_b.reset_index(drop=True)], axis=1)
    features_a["isna_sum"] = features_a.isna().sum(axis=1)
    features_b["isna_sum"] = features_b.isna().sum(axis=1)
    return features_a, features_b


def add_test_per_stats_features(train_ft_a: pd.DataFrame, train_ft_b: pd.DataFrame):
    cols = [
        col
        for col in train_ft_a
        if len(col) > 3
        and col.startswith("a")
        and ((col.startswith("a3") and col[7] in ["1", "3"]) or (col.startswith("a4") and col[5] in ["1"]) or (col.startswith("a5") and col[4] in ["1"]))
    ] + ["A6-1", "A7-1"]
    train_ft_a["acc_stats_std"] = train_ft_a[cols].std(axis=1)
    train_ft_a["acc_stats_mean"] = train_ft_a[cols].mean(axis=1)
    train_ft_a["acc_stats_skew"] = train_ft_a[cols].skew(axis=1)
    train_ft_a["acc_stats_kurt"] = train_ft_a[cols].kurt(axis=1)

    cols = [
        col
        for col in train_ft_a
        if len(col) > 3
        and col.startswith("a")
        and ((col.startswith("a3") and col[7] in ["2", "4"]) or (col.startswith("a4") and col[5] in ["2"]) or (col.startswith("a5") and col[4] in ["2"]))
    ]
    train_ft_a["err_stats_std"] = train_ft_a[cols].std(axis=1)
    train_ft_a["err_stats_mean"] = train_ft_a[cols].mean(axis=1)
    train_ft_a["err_stats_skew"] = train_ft_a[cols].skew(axis=1)
    train_ft_a["err_stats_kurt"] = train_ft_a[cols].kurt(axis=1)

    for prefix, suffix in [("rt_mean_stats", "mean"), ("rt_std_stats", "std")]:
        cols = [col for col in train_ft_a if col.startswith("seq_rt_") and col.endswith(suffix)]
        train_ft_a[f"{prefix}_std"] = train_ft_a[cols].std(axis=1)
        train_ft_a[f"{prefix}_mean"] = train_ft_a[cols].mean(axis=1)
        train_ft_a[f"{prefix}_skew"] = train_ft_a[cols].skew(axis=1)
        train_ft_a[f"{prefix}_kurt"] = train_ft_a[cols].kurt(axis=1)

    cols = [
        col
        for col in train_ft_b
        if len(col) > 3
        and col.startswith("b")
        and (
            (col.startswith("b1") and col[3] in ["1"])
            or (col.startswith("b2") and col[3] in ["1"])
            or (col.startswith("b3") and col[3] in ["1"])
            or (col.startswith("b4") and col[3] in ["1", "3", "5"])
            or (col.startswith("b5") and col[3] in ["1"])
            or (col.startswith("b6") and col[3] in ["1"])
            or (col.startswith("b7") and col[3] in ["1"])
            or (col.startswith("b8") and col[3] in ["1"])
        )
    ] + ["B9-1", "B9-4", "B10-1", "B10-4", "B10-6"]
    train_ft_b["acc_stats_std"] = train_ft_b[cols].std(axis=1)
    train_ft_b["acc_stats_mean"] = train_ft_b[cols].mean(axis=1)
    train_ft_b["acc_stats_skew"] = train_ft_b[cols].skew(axis=1)
    train_ft_b["acc_stats_kurt"] = train_ft_b[cols].kurt(axis=1)

    cols = [
        col
        for col in train_ft_b
        if len(col) > 3
        and col.startswith("b")
        and (
            (col.startswith("b1") and col[3] in ["2"])
            or (col.startswith("b2") and col[3] in ["2"])
            or (col.startswith("b3") and col[3] in ["2"])
            or (col.startswith("b4") and col[3] in ["2", "4", "6"])
            or (col.startswith("b5") and col[3] in ["2"])
            or (col.startswith("b6") and col[3] in ["2"])
            or (col.startswith("b7") and col[3] in ["2"])
            or (col.startswith("b8") and col[3] in ["2"])
        )
    ] + ["B9-2", "B9-3", "B9-5", "B10-2", "B10-3", "B10-5"]
    train_ft_b["err_stats_std"] = train_ft_b[cols].std(axis=1)
    train_ft_b["err_stats_mean"] = train_ft_b[cols].mean(axis=1)
    train_ft_b["err_stats_skew"] = train_ft_b[cols].skew(axis=1)
    train_ft_b["err_stats_kurt"] = train_ft_b[cols].kurt(axis=1)

    for prefix, suffix in [("rt_mean_stats", "mean"), ("rt_std_stats", "std")]:
        cols = [col for col in train_ft_b if col.startswith("seq_rt_") and col.endswith(suffix)]
        train_ft_b[f"{prefix}_std"] = train_ft_b[cols].std(axis=1)
        train_ft_b[f"{prefix}_mean"] = train_ft_b[cols].mean(axis=1)
        train_ft_b[f"{prefix}_skew"] = train_ft_b[cols].skew(axis=1)
        train_ft_b[f"{prefix}_kurt"] = train_ft_b[cols].kurt(axis=1)

    return train_ft_a, train_ft_b


def build_features(data_path: str | Path | None = None, max_rows: int | None = None):
    if data_path is None:
        data_path = default_data_path()
    train, train_a, train_b = load_data(data_path)

    if max_rows:
        train = train.sample(n=min(max_rows, len(train)), random_state=RANDOM_STATE).reset_index(drop=True)
        keep_ids = set(train["Test_id"])
        train_a = train_a[train_a["Test_id"].isin(keep_ids)].reset_index(drop=True)
        train_b = train_b[train_b["Test_id"].isin(keep_ids)].reset_index(drop=True)

    train_a, train_b = merge_meta2ab(train, train_a, train_b, ["Test_id", "Label"])
    df_ab = make_df_ab(train_a, train_b)

    train_ft_a, _ = preprocess_data(train_a, df_ab, A_SEQ_COLS, "A", A_RT_COLS, A_ADD_COLS)
    train_ft_b, _ = preprocess_data(train_b, df_ab, B_SEQ_COLS, "B", B_RT_COLS, B_ADD_COLS)

    target_a = train_a["Label"].to_numpy()
    target_b = train_b["Label"].to_numpy()
    groups = np.concatenate([train_a["PrimaryKey"].to_numpy(), train_b["PrimaryKey"].to_numpy()])
    meta = pd.concat(
        [
            train_a[["Test_id", "Test", "PrimaryKey", "TestDate", "TestDate_year", "Label"]],
            train_b[["Test_id", "Test", "PrimaryKey", "TestDate", "TestDate_year", "Label"]],
        ],
        ignore_index=True,
    )

    train_ft_a, train_ft_b = add_past_only_other_test_features(train_a, train_b, train_ft_a, train_ft_b)
    train_ft_a, train_ft_b = add_test_per_stats_features(train_ft_a, train_ft_b)

    common_cols = [col for col in train_ft_a.columns if col in train_ft_b.columns]
    x = pd.concat([train_ft_a[common_cols], train_ft_b[common_cols]], ignore_index=True)
    y = np.concatenate([target_a, target_b])

    availability = (
        pd.concat(
            [
                train_ft_a[OTHER_TEST_COLS].assign(Test="A"),
                train_ft_b[OTHER_TEST_COLS].assign(Test="B"),
            ],
            ignore_index=True,
        )
        .groupby("Test", sort=False)[OTHER_TEST_COLS]
        .agg(["count", "mean"])
    )
    availability.columns = [f"{col}_{stat}" for col, stat in availability.columns]
    availability = availability.reset_index()
    overall_availability = pd.DataFrame(
        [
            {
                "Test": "Integrated",
                **{f"{col}_count": int(x[col].notna().sum()) for col in OTHER_TEST_COLS},
                **{f"{col}_mean": float(x[col].mean()) for col in OTHER_TEST_COLS},
            }
        ]
    )
    availability = pd.concat([availability, overall_availability], ignore_index=True)

    del train, train_a, train_b, train_ft_a, train_ft_b, df_ab
    gc.collect()
    return x, y, groups, meta, availability


def split_by_year(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    return df.loc[df["TestDate_year"].between(start_year, end_year)].reset_index(drop=True)


def build_time_split_features(
    data_path: str | Path | None = None,
    train_start_year: int = 2016,
    train_end_year: int = 2020,
    valid_start_year: int = 2021,
    valid_end_year: int = 2022,
):
    if data_path is None:
        data_path = default_data_path()
    train_meta, all_a, all_b = load_data(data_path)
    all_a, all_b = merge_meta2ab(train_meta, all_a, all_b, ["Test_id", "Label"])

    train_a = split_by_year(all_a, train_start_year, train_end_year)
    train_b = split_by_year(all_b, train_start_year, train_end_year)
    valid_a = split_by_year(all_a, valid_start_year, valid_end_year)
    valid_b = split_by_year(all_b, valid_start_year, valid_end_year)
    train_history = make_df_ab(train_a, train_b)

    train_ft_a, tdm_a = preprocess_data(train_a, train_history, A_SEQ_COLS, "A", A_RT_COLS, A_ADD_COLS, is_train=True)
    train_ft_b, tdm_b = preprocess_data(train_b, train_history, B_SEQ_COLS, "B", B_RT_COLS, B_ADD_COLS, is_train=True)
    train_ft_a, train_ft_b = add_past_only_other_test_features(train_a, train_b, train_ft_a, train_ft_b)
    train_ft_a, train_ft_b = add_test_per_stats_features(train_ft_a, train_ft_b)

    valid_ft_a, _ = preprocess_data(valid_a, train_history, A_SEQ_COLS, "A", A_RT_COLS, A_ADD_COLS, tdm=tdm_a, is_train=False)
    valid_ft_b, _ = preprocess_data(valid_b, train_history, B_SEQ_COLS, "B", B_RT_COLS, B_ADD_COLS, tdm=tdm_b, is_train=False)
    valid_ft_a, valid_ft_b = add_past_only_other_test_features_from_sources(train_a, train_b, valid_a, valid_b, valid_ft_a, valid_ft_b)
    valid_ft_a, valid_ft_b = add_test_per_stats_features(valid_ft_a, valid_ft_b)

    common_cols = [col for col in train_ft_a.columns if col in train_ft_b.columns]
    x_train = pd.concat([train_ft_a[common_cols], train_ft_b[common_cols]], ignore_index=True)
    y_train = np.concatenate([train_a["Label"].to_numpy(), train_b["Label"].to_numpy()])
    x_valid = pd.concat([valid_ft_a[common_cols], valid_ft_b[common_cols]], ignore_index=True)
    y_valid = np.concatenate([valid_a["Label"].to_numpy(), valid_b["Label"].to_numpy()])

    valid_meta = pd.concat(
        [
            valid_a[["Test_id", "Test", "TestDate", "TestDate_year", "Label"]],
            valid_b[["Test_id", "Test", "TestDate", "TestDate_year", "Label"]],
        ],
        ignore_index=True,
    )
    year_distribution = (
        pd.concat([all_a[["TestDate_year", "Test", "Label"]], all_b[["TestDate_year", "Test", "Label"]]], ignore_index=True)
        .groupby(["TestDate_year", "Test"])["Label"]
        .agg(["size", "mean"])
        .reset_index()
        .rename(columns={"TestDate_year": "year", "size": "n", "mean": "label_rate"})
    )
    split_meta = {
        "train_rows": len(y_train),
        "valid_rows": len(y_valid),
        "train_label_rate": float(y_train.mean()),
        "valid_label_rate": float(y_valid.mean()),
        "n_features": len(common_cols),
    }

    del all_a, all_b, train_ft_a, train_ft_b, valid_ft_a, valid_ft_b, train_history
    gc.collect()
    return x_train, y_train, x_valid, y_valid, valid_meta, year_distribution, split_meta


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[: len(bin_weights)]
    prob_pred = prob_pred[: len(bin_weights)]
    return float(np.sum(bin_weights * np.abs(prob_true - prob_pred)))


def auc_brier_ece(y_true, y_prob, n_bins: int = 10):
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    score = 0.5 * (1 - auc) + 0.25 * brier + 0.25 * ece
    return float(score), float(auc), float(brier), float(ece)


def final_hp_dict():
    return {
        "hgb": {
            "random_state": RANDOM_STATE,
            "early_stopping": True,
            "learning_rate": 0.013083126286998413,
            "max_iter": 974,
            "max_depth": 19,
            "min_samples_leaf": 27,
            "l2_regularization": 2.3671420400911103,
            "max_bins": 124,
            "validation_fraction": 0.13775932137982771,
        },
        "lgb": {
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "learning_rate": 0.00566266560396128,
            "num_leaves": 90,
            "max_depth": 7,
            "min_child_samples": 78,
            "subsample": 0.5861448959666542,
            "colsample_bytree": 0.6294622514208068,
            "reg_alpha": 3.864865279991244e-06,
            "reg_lambda": 6.557569005815827,
            "n_estimators": 954,
        },
        "xgb": {
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
            "n_jobs": -1,
            "learning_rate": 0.011040891037651001,
            "max_depth": 7,
            "min_child_weight": 1.3167145256657375,
            "gamma": 0.002648080476999737,
            "subsample": 0.6355902293115955,
            "colsample_bytree": 0.7107606869220718,
            "reg_alpha": 2.492968196370563,
            "reg_lambda": 0.20890187656409528,
            "n_estimators": 440,
        },
        "cb": {
            "random_state": RANDOM_STATE,
            "verbose": 0,
            "n_estimators": 720,
            "learning_rate": 0.014186617324044443,
            "depth": 10,
            "l2_leaf_reg": 3.394142018601596,
            "bagging_temperature": 0.6984992670948069,
            "border_count": 240,
            "random_strength": 1.203082396603811,
            "grow_policy": "Depthwise",
        },
    }


def make_model(model_name: str):
    hp = final_hp_dict()
    if model_name == "hgb":
        return HistGradientBoostingClassifier(**hp["hgb"])
    if model_name == "lgb":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**hp["lgb"])
    if model_name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(**hp["xgb"])
    if model_name == "cb":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**hp["cb"])
    if model_name == "ensemble":
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier

        estimators = [
            ("hgb", HistGradientBoostingClassifier(**hp["hgb"])),
            ("lgb", LGBMClassifier(**hp["lgb"])),
            ("xgb", XGBClassifier(**hp["xgb"])),
            ("cb", CatBoostClassifier(**hp["cb"])),
        ]
        return VotingClassifier(estimators=estimators, n_jobs=-1, voting="soft")
    raise ValueError(f"Unknown model: {model_name}")


def feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    columns = list(columns)
    return {
        "age_time": [c for c in columns if c in {"Age", "TestDate_year", "TestDate_month", "YearMonthIndex"}],
        "history": [c for c in columns if c.startswith("prev") or c.startswith("other_test")],
        "accuracy_error_stats": [c for c in columns if c.startswith("acc_stats") or c.startswith("err_stats")],
        "response_time_stats": [c for c in columns if c.startswith("rt_mean_stats") or c.startswith("rt_std_stats") or c.startswith("seq_rt_")],
        "missingness": [c for c in columns if c == "isna_sum"],
    }


def summarize_mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_cols, sort=False)[metric_cols].agg(["mean", "std"]).reset_index()
