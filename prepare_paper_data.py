"""Build the strict prior-month base dataset used by the v19 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fold_isolated_pipeline import OLD_OUTPUTS, STRICT_DATA
from past_only_pipeline import build_features, default_data_path


def output_paths() -> dict[str, Path]:
    return {
        "x": STRICT_DATA / "features_past_only.pkl",
        "y": STRICT_DATA / "target_past_only.npy",
        "groups": STRICT_DATA / "groups_past_only.npy",
        "meta": STRICT_DATA / "meta_past_only.pkl",
        "columns": STRICT_DATA / "feature_columns_past_only.txt",
        "availability": OLD_OUTPUTS / "feature_availability_past_only.csv",
        "manifest": OLD_OUTPUTS / "strict_month_data_manifest.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = output_paths()
    required = [paths[key] for key in ["x", "y", "groups", "meta"]]
    if not args.force and all(path.exists() for path in required):
        print(f"Existing strict dataset retained: {STRICT_DATA}", flush=True)
        return

    STRICT_DATA.mkdir(parents=True, exist_ok=True)
    OLD_OUTPUTS.mkdir(parents=True, exist_ok=True)
    data_path = (args.data_path or default_data_path()).resolve()
    x, y, groups, meta, availability = build_features(data_path=data_path)

    if x.shape != (944_767, 37):
        raise ValueError(f"Unexpected base feature shape: {x.shape}")
    if len(y) != len(groups) or len(y) != len(meta):
        raise ValueError("Base feature, label, group, and metadata lengths differ")

    x.to_pickle(paths["x"])
    np.save(paths["y"], y)
    np.save(paths["groups"], groups)
    meta.to_pickle(paths["meta"])
    paths["columns"].write_text("\n".join(x.columns) + "\n", encoding="utf-8")
    availability.to_csv(paths["availability"], index=False, encoding="utf-8-sig")
    paths["manifest"].write_text(
        json.dumps(
            {
                "rows": int(len(y)),
                "features": int(x.shape[1]),
                "label_rate": float(np.mean(y)),
                "random_seed": 42,
                "data_path": str(data_path),
                "history_rule": "source.TestDate < target.TestDate",
                "fold_rule": "history is rebuilt from each training partition after splitting",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Strict base dataset created: {STRICT_DATA}", flush=True)


if __name__ == "__main__":
    main()
