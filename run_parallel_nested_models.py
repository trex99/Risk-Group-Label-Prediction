"""Tune pending models concurrently while sharing one in-memory inner dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

from fold_isolated_pipeline import RESULTS, ensure_dirs
from run_nested_optuna import MODELS, prepare_inner_data, tune_model


def done(scheme: str, fold: int, model: str, n_trials: int) -> bool:
    path = RESULTS / "optuna" / f"leakage_safe_{scheme}_outer{fold}_{model}_best.json"
    if not path.exists():
        return False
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["n_completed_trials"]) >= n_trials
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["stratified", "group", "temporal"], required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--total-threads", type=int, default=18)
    args = parser.parse_args()
    ensure_dirs()
    pending = [model for model in MODELS if not done(args.scheme, args.outer_fold, model, args.n_trials)]
    if not pending:
        print("ALREADY_COMPLETE", flush=True)
        return
    threads = max(1, args.total_threads // len(pending))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    datasets, protocol = prepare_inner_data(args.scheme, args.outer_fold, 3, False)
    results = {}
    with ThreadPoolExecutor(max_workers=len(pending)) as executor:
        futures = {
            executor.submit(tune_model, model, datasets, protocol, args.n_trials, threads): model
            for model in pending
        }
        for future in as_completed(futures):
            model = futures[future]
            results[model] = future.result()
    out = RESULTS / "optuna" / f"parallel_{args.scheme}_outer{args.outer_fold}_summary.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
