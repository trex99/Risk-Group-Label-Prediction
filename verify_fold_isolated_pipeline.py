"""Deterministic smoke and brute-force checks for fold-isolated histories."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fold_isolated_pipeline import HISTORY_COLS, RESULTS, compute_fold_history, ensure_dirs, load_base


def expected_for_target(source: pd.DataFrame, row: pd.Series) -> dict[str, float]:
    earlier = source[(source.PrimaryKey == row.PrimaryKey) & (source.TestDate < row.TestDate)]
    same = earlier[earlier.Test == row.Test]
    opposite = earlier[earlier.Test != row.Test]

    def stats(frame: pd.DataFrame, prefix: str, label_prefix: str) -> dict[str, float]:
        if frame.empty:
            return {
                f"{prefix}_YearMonthIndex_mean": np.nan,
                f"{prefix}_YearMonthIndex_sum": np.nan,
                f"{prefix}_YearMonthIndex_std": np.nan,
                f"{label_prefix}_mean": np.nan,
                f"{label_prefix}_sum": np.nan,
                f"{label_prefix}_std": np.nan,
            }
        return {
            f"{prefix}_YearMonthIndex_mean": frame.YearMonthIndex.mean(),
            f"{prefix}_YearMonthIndex_sum": frame.YearMonthIndex.sum(),
            f"{prefix}_YearMonthIndex_std": frame.YearMonthIndex.std(ddof=1),
            f"{label_prefix}_mean": frame.Label.mean(),
            f"{label_prefix}_sum": frame.Label.sum(),
            f"{label_prefix}_std": frame.Label.std(ddof=1),
        }

    out = {}
    out.update(stats(earlier, "prev_ab", "prev_ab_all_label"))
    out.update(stats(same, "prev", "prev_all_label"))
    if opposite.empty:
        out.update(
            {
                "other_test_Test_id_count": np.nan,
                "other_test_YearMonthIndex_mean": np.nan,
                "other_test_YearMonthIndex_std": np.nan,
                "other_test_Label_mean": np.nan,
            }
        )
    else:
        out.update(
            {
                "other_test_Test_id_count": len(opposite),
                "other_test_YearMonthIndex_mean": opposite.YearMonthIndex.mean(),
                "other_test_YearMonthIndex_std": opposite.YearMonthIndex.std(ddof=1),
                "other_test_Label_mean": opposite.Label.mean(),
            }
        )
    return out


def main() -> None:
    ensure_dirs()
    _, _, _, meta = load_base()
    rng = np.random.default_rng(20260714)

    # Include complete histories for sampled people so tied-month and repeated
    # test cases are exercised rather than only single-record people.
    sampled_people = rng.choice(meta.PrimaryKey.unique(), size=5000, replace=False)
    pool = meta.index[meta.PrimaryKey.isin(sampled_people)].to_numpy()
    rng.shuffle(pool)
    source_indices = np.sort(pool[: int(len(pool) * 0.8)])
    target_indices = np.sort(pool[int(len(pool) * 0.8) :])

    actual = compute_fold_history(meta, source_indices, target_indices)
    source = meta.loc[source_indices]

    checked = 0
    for target_index in target_indices[: min(300, len(target_indices))]:
        expected = expected_for_target(source, meta.loc[target_index])
        for col in HISTORY_COLS:
            a = actual.loc[target_index, col]
            e = expected[col]
            if not (pd.isna(a) and pd.isna(e)) and not np.isclose(a, e, rtol=1e-10, atol=1e-10):
                raise AssertionError(f"Mismatch target={target_index} col={col}: actual={a}, expected={e}")
        checked += 1

    # Changing every non-source Label must not alter validation histories.
    mutated = meta.copy()
    non_source = np.ones(len(meta), dtype=bool)
    non_source[source_indices] = False
    mutated.loc[non_source, "Label"] = 1 - mutated.loc[non_source, "Label"]
    after_mutation = compute_fold_history(mutated, source_indices, target_indices)
    pd.testing.assert_frame_equal(actual, after_mutation, check_exact=True)

    payload = {
        "status": "PASS",
        "source_rows": int(len(source_indices)),
        "target_rows": int(len(target_indices)),
        "bruteforce_targets_checked": checked,
        "non_source_label_mutation_invariance": True,
        "same_month_rule": "source.TestDate < target.TestDate",
    }
    out = RESULTS / "fold_isolated_pipeline_verification.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
