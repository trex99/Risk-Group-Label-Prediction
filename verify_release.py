"""Validate the public GitHub package without loading the research dataset."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md",
    "DATA.md",
    "REPRODUCIBILITY.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    ".gitignore",
    "run_full_reproduction.py",
    "fold_isolated_pipeline.py",
    "past_only_pipeline.py",
    "build_manuscript_artifacts.py",
    "revision/README.md",
    "revision/run_revision_analyses.py",
    "revision/tune_logistic_c.py",
    "revision/run_shap_fold_stability.py",
    "checksums.sha256",
]
FORBIDDEN_SUFFIXES = {
    ".pkl", ".npy", ".npz", ".sqlite", ".sqlite3", ".db", ".log", ".pid"
}
FORBIDDEN_NAMES = {"desktop.ini", ".DS_Store", ".env"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            failures.append(f"missing required file: {relative}")

    python_files = sorted(
        path
        for path in ROOT.rglob("*.py")
        if not any(part in {"runtime", ".git", "__pycache__"} for part in path.relative_to(ROOT).parts)
    )
    for path in python_files:
        try:
            source = path.read_text(encoding="utf-8-sig")
            ast.parse(source, filename=str(path))
        except Exception as exc:
            failures.append(f"python syntax: {path.name}: {exc}")
            continue
        # The validator contains the detection pattern itself, so only inspect
        # the public research and orchestration scripts for local paths.
        if path.resolve() != Path(__file__).resolve() and re.search(
            r"(?i)(?:[A-Z]:\\\\|/Users/|/home/[^/]+/)", source
        ):
            failures.append(f"hard-coded local path: {path.name}")

    excluded_roots = {"runtime", ".git", "__pycache__"}
    if (ROOT / ".git").exists():
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8").split("\0")
        tracked_candidates = [ROOT / relative for relative in listed if relative]
    else:
        tracked_candidates = [
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and not any(
                part in excluded_roots for part in path.relative_to(ROOT).parts
            )
        ]
    for path in tracked_candidates:
        relative = path.relative_to(ROOT)
        if path.name in FORBIDDEN_NAMES:
            failures.append(f"forbidden local/system file: {relative.as_posix()}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.endswith(".csv.gz"):
            failures.append(f"forbidden generated artifact: {relative.as_posix()}")
        if path.stat().st_size > 50 * 1024 * 1024:
            failures.append(f"file exceeds 50 MiB release policy: {relative.as_posix()}")

    checksum_path = ROOT / "checksums.sha256"
    checksums_checked = 0
    if checksum_path.exists():
        for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                expected, relative = line.split(maxsplit=1)
            except ValueError:
                failures.append(f"invalid checksum line {line_number}")
                continue
            target = ROOT / relative.strip()
            if not target.exists():
                failures.append(f"checksum target missing: {relative}")
                continue
            actual = sha256(target)
            checksums_checked += 1
            if actual.lower() != expected.lower():
                failures.append(f"checksum mismatch: {relative}")

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "python_files_checked": len(python_files),
        "release_files_checked": len(tracked_candidates),
        "checksums_checked": checksums_checked,
        "failures": failures,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
