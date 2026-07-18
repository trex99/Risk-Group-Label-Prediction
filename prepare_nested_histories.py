"""Materialize manifest-validated inner history caches before parallel tuning."""

from __future__ import annotations

import argparse
import gc
import json

from fold_isolated_pipeline import RESULTS, ensure_dirs
from run_nested_optuna import prepare_inner_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["stratified", "group", "temporal"], required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--force-history", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    datasets, protocol = prepare_inner_data(
        args.scheme, args.outer_fold, args.inner_splits, args.force_history
    )
    del datasets
    gc.collect()
    out = RESULTS / "optuna" / f"{args.scheme}_outer{args.outer_fold}_prepared.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PREPARED", flush=True)


if __name__ == "__main__":
    main()
