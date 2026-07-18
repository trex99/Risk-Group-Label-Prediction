"""Sequential outer evaluation queue using direct VotingClassifier predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from fold_isolated_pipeline import LOGS, RESULTS

HERE = Path(__file__).resolve().parent


def folds_for(scheme: str) -> range:
    return range(1, 2) if scheme == "temporal" else range(1, 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["stratified", "temporal", "group"],
        default=["stratified", "temporal", "group"],
    )
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    for scheme in args.schemes:
        for fold in folds_for(scheme):
            log = LOGS / f"evaluate_{scheme}_outer{fold}.log"
            with log.open("a", encoding="utf-8") as stream:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "run_outer_evaluation.py"),
                        "--scheme",
                        scheme,
                        "--outer-fold",
                        str(fold),
                        "--threads-per-estimator",
                        "5",
                        "--voting-jobs",
                        "4",
                    ],
                    cwd=HERE,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                raise RuntimeError(f"Outer evaluation failed: {scheme} fold {fold}")
        result = subprocess.run(
            [sys.executable, str(HERE / "summarize_outer_evaluation.py"), "--scheme", scheme],
            cwd=HERE,
        )
        if result.returncode:
            raise RuntimeError(f"Outer summary failed: {scheme}")
    payload = {
        "status": "COMPLETE",
        "schemes": args.schemes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS / "outer_queue_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
