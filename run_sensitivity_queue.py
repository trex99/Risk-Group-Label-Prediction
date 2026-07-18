"""Sequential queue for all locked-hyperparameter sensitivity analyses."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from fold_isolated_pipeline import LOGS, RESULTS

HERE = Path(__file__).resolve().parent


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("stepwise", range(1, 6)),
        ("ablation", range(1, 6)),
        ("stratified_time", range(1, 6)),
        ("temporal_time", range(1, 2)),
    ]
    for task, folds in tasks:
        for fold in folds:
            log = LOGS / f"sensitivity_{task}_outer{fold}.log"
            with log.open("a", encoding="utf-8") as stream:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "run_sensitivity_evaluation.py"),
                        "--task",
                        task,
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
                raise RuntimeError(f"Sensitivity failed: {task} outer {fold}")
        result = subprocess.run(
            [sys.executable, str(HERE / "summarize_sensitivity.py"), "--task", task], cwd=HERE
        )
        if result.returncode:
            raise RuntimeError(f"Sensitivity summary failed: {task}")
    (RESULTS / "sensitivity_queue_status.json").write_text(
        json.dumps(
            {"status": "COMPLETE", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
