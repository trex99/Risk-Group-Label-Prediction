"""Deterministic outer/inner split definitions for the leakage-safe analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from fold_isolated_pipeline import RANDOM_STATE


def outer_splits(
    scheme: str,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    all_idx = np.arange(len(y), dtype=np.int64)
    if scheme == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return [(all_idx[tr], all_idx[va]) for tr, va in splitter.split(all_idx, y)]
    if scheme == "group":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return [(all_idx[tr], all_idx[va]) for tr, va in splitter.split(all_idx, y, groups)]
    if scheme == "temporal":
        years = meta["TestDate"] // 100
        train = np.flatnonzero(years.between(2016, 2020).to_numpy()).astype(np.int64)
        valid = np.flatnonzero(years.between(2021, 2022).to_numpy()).astype(np.int64)
        return [(train, valid)]
    raise ValueError(f"Unknown outer scheme: {scheme}")


def inner_splits(
    scheme: str,
    outer_train: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
    outer_fold: int,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    outer_train = np.asarray(outer_train, dtype=np.int64)
    if scheme == "stratified":
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        return [(outer_train[tr], outer_train[va]) for tr, va in splitter.split(outer_train, y[outer_train])]
    if scheme == "group":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        return [
            (outer_train[tr], outer_train[va])
            for tr, va in splitter.split(outer_train, y[outer_train], groups[outer_train])
        ]
    if scheme == "temporal":
        years = (meta["TestDate"] // 100).to_numpy()
        specs = [((2016, 2017), 2018), ((2016, 2018), 2019), ((2016, 2019), 2020)]
        result = []
        outer_set = np.zeros(len(y), dtype=bool)
        outer_set[outer_train] = True
        for (start, end), valid_year in specs:
            train = np.flatnonzero(outer_set & (years >= start) & (years <= end)).astype(np.int64)
            valid = np.flatnonzero(outer_set & (years == valid_year)).astype(np.int64)
            result.append((train, valid))
        return result
    raise ValueError(f"Unknown inner scheme: {scheme}")


def validate_splits(
    scheme: str,
    outer_train: np.ndarray,
    outer_valid: np.ndarray,
    inner: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    meta: pd.DataFrame,
) -> None:
    outer_train = np.asarray(outer_train, dtype=np.int64)
    outer_valid = np.asarray(outer_valid, dtype=np.int64)
    if np.intersect1d(outer_train, outer_valid).size:
        raise ValueError("Outer train/validation rows overlap")
    outer_train_set = set(outer_train.tolist())
    for fold, (train, valid) in enumerate(inner, start=1):
        if np.intersect1d(train, valid).size:
            raise ValueError(f"Inner fold {fold} train/validation rows overlap")
        if not set(train.tolist()).issubset(outer_train_set) or not set(valid.tolist()).issubset(outer_train_set):
            raise ValueError(f"Inner fold {fold} escapes the outer-training partition")
        if np.intersect1d(valid, outer_valid).size or np.intersect1d(train, outer_valid).size:
            raise ValueError(f"Outer validation rows reached inner fold {fold}")
        if scheme == "group" and np.intersect1d(groups[train], groups[valid]).size:
            raise ValueError(f"Inner group fold {fold} has PrimaryKey overlap")
        if scheme == "temporal":
            train_max = int(meta.loc[train, "TestDate"].max())
            valid_min = int(meta.loc[valid, "TestDate"].min())
            if train_max >= valid_min:
                raise ValueError(f"Temporal inner fold {fold} is not forward-only")
