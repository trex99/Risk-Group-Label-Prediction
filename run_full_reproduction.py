"""Run the complete leakage-safe manuscript reproduction pipeline in stage order.

The default run is intentionally long: it performs 100 Optuna trials for each
model/fold before the outer, sensitivity, bootstrap, SHAP, table, and figure
stages. Use ``--dry-run`` to inspect the exact command sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def command(script: str, *args: str) -> list[str]:
    return [sys.executable, str(HERE / script), *args]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--bootstrap-jobs", type=int, default=10)
    parser.add_argument(
        "--start-at",
        choices=["data", "protocol", "tuning", "outer", "sensitivity", "post", "verification", "shap", "artifacts"],
        default="data",
    )
    parser.add_argument(
        "--stop-after",
        choices=["data", "protocol", "tuning", "outer", "sensitivity", "post", "verification", "shap", "artifacts"],
        default="artifacts",
    )
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_args: list[str] = []
    if args.data_path is not None:
        data_args.extend(["--data-path", str(args.data_path.resolve())])
    if args.force_data:
        data_args.append("--force")

    stages: list[tuple[str, list[list[str]]]] = [
        (
            "data",
            [
                command("prepare_paper_data.py", *data_args),
                command("create_figure1.py"),
            ],
        ),
        (
            "protocol",
            [
                command("verify_fold_isolated_pipeline.py"),
                command("audit_leakage_protocol.py"),
            ],
        ),
        (
            "tuning",
            [command("run_tuning_queue.py", "--n-trials", str(args.n_trials))],
        ),
        ("outer", [command("run_outer_queue.py")]),
        ("sensitivity", [command("run_sensitivity_queue.py")]),
        (
            "post",
            [
                command(
                    "bootstrap_incremental_value.py",
                    "--replicates",
                    "500",
                    "--jobs",
                    str(args.bootstrap_jobs),
                ),
                command("summarize_test_type_subgroups.py"),
                command("make_reliability_figure.py"),
            ],
        ),
        ("verification", [command("verify_final_outputs.py")]),
        ("shap", [command("run_votingclassifier_shap.py")]),
        ("artifacts", [command("build_manuscript_artifacts.py")]),
    ]

    names = [name for name, _ in stages]
    start = names.index(args.start_at)
    stop = names.index(args.stop_after)
    if start > stop:
        raise ValueError("--start-at must not come after --stop-after")

    for stage, commands in stages[start : stop + 1]:
        for argv in commands:
            print(f"[{stage}] {' '.join(argv)}", flush=True)
            if args.dry_run:
                continue
            subprocess.run(argv, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
