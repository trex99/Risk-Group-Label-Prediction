"""Resource-aware queue for all nested Optuna studies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from fold_isolated_pipeline import LOGS, RESULTS


HERE = Path(__file__).resolve().parent
MODELS = ("hgb", "lgb", "xgb", "cb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["stratified", "temporal", "group"],
        default=["stratified", "temporal", "group"],
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--threads", type=int, default=9)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    return parser.parse_args()


def folds_for(scheme: str) -> range:
    return range(1, 2) if scheme == "temporal" else range(1, 6)


def done(scheme: str, fold: int, model: str, n_trials: int) -> bool:
    path = RESULTS / "optuna" / f"leakage_safe_{scheme}_outer{fold}_{model}_best.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return int(payload["n_completed_trials"]) >= n_trials
    except Exception:
        return False


def wait_for_memory(min_free_gb: float) -> None:
    while psutil.virtual_memory().available / 1024**3 < min_free_gb:
        time.sleep(30)


def run_prepare(scheme: str, fold: int) -> None:
    log = LOGS / f"prepare_{scheme}_outer{fold}.log"
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "prepare_nested_histories.py"),
                "--scheme",
                scheme,
                "--outer-fold",
                str(fold),
            ],
            cwd=HERE,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"History preparation failed: {scheme} outer {fold}")


def run_shared_models(scheme: str, fold: int, args: argparse.Namespace) -> None:
    pending = [model for model in MODELS if not done(scheme, fold, model, args.n_trials)]
    if not pending:
        return
    wait_for_memory(args.min_free_gb)
    env = os.environ.copy()
    total_threads = min(18, max(1, len(pending) * args.threads))
    env["OMP_NUM_THREADS"] = str(max(1, total_threads // len(pending)))
    env["MKL_NUM_THREADS"] = env["OMP_NUM_THREADS"]
    env["OPENBLAS_NUM_THREADS"] = env["OMP_NUM_THREADS"]
    stdout = (LOGS / f"tune_{scheme}_outer{fold}_shared.out.log").open("a", encoding="utf-8")
    stderr = (LOGS / f"tune_{scheme}_outer{fold}_shared.err.log").open("a", encoding="utf-8")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "run_parallel_nested_models.py"),
                "--scheme",
                scheme,
                "--outer-fold",
                str(fold),
                "--n-trials",
                str(args.n_trials),
                "--total-threads",
                str(total_threads),
            ],
            cwd=HERE,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        stdout.close()
        stderr.close()
    if result.returncode:
        raise RuntimeError(f"Shared tuning failed: {scheme} outer {fold}")


def main() -> None:
    args = parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    for scheme in args.schemes:
        for fold in folds_for(scheme):
            if all(done(scheme, fold, model, args.n_trials) for model in MODELS):
                continue
            run_prepare(scheme, fold)
            run_shared_models(scheme, fold, args)
    payload = {
        "status": "COMPLETE",
        "schemes": args.schemes,
        "n_trials": args.n_trials,
        "random_seed": 42,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS / "tuning_queue_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
