#!/usr/bin/env python3
"""Select ./run check legs from the candidate diff.

The 2026-10-01 gl/adapter-lazy-config release changed two Python files and
still ran the full Go suite. Three cold `go` processes each downloaded the
same modules (18 `go: downloading` lines) and the check stage took 113.0s of
a 188.8s release. Go tests do not import the adapter package; the cloud-client
source scan they repeat already lives in tests/unit/test_no_cloud_video.py.
Skipping Go for a diff that cannot affect it is the impact cut.

Fails closed: an unknown path, a git error in the default (gate) mode, or an
empty diff selects every leg. Only paths proven not to affect a leg drop it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEGS = ("go", "python", "gpu-idle")

# Changing the harness can hide a failure. Select everything.
GLOBAL = {
    "run",
    "greenline.toml",
    "deploy-to-spark.sh",
    "pyproject.toml",
    "scripts/check_impact.py",
}

INERT_SUFFIXES = {".md", ".jpg", ".jpeg", ".png", ".html"}


def select(paths: list[str]) -> set[str]:
    """Map changed paths to check legs. Unknown or empty input selects all."""
    if not paths:
        return set(LEGS)
    selected: set[str] = set()
    for path in paths:
        legs = classify(path)
        if legs is None:
            return set(LEGS)
        selected |= legs
    return selected


def classify(path: str) -> set[str] | None:
    """Legs this one path can break, empty if inert, None if unknown."""
    name = Path(path).name
    if path in GLOBAL or name in GLOBAL:
        return set(LEGS)
    if path.startswith("docs/") or Path(path).suffix.lower() in INERT_SUFFIXES:
        return set()
    if path == "scripts/gpu_idle_hook.py" or path == "scripts/test_gpu_idle_hook.py":
        return {"gpu-idle"}
    if path.startswith("cmd/") and path.endswith(".sql"):
        return {"go"}
    if path.endswith(".go") or path in {"go.mod", "go.sum"}:
        return {"go"}
    if path.startswith(("src/", "tests/", "scripts/")) and path.endswith(".py"):
        return {"python"}
    if path.startswith("runtime/"):
        return {"python"}
    if path.startswith("config/"):
        return {"go", "python"}
    return None


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return result.stdout


def diff_paths(base: str, head: str) -> list[str] | None:
    """Files that differ between two commits, or None if git cannot tell us."""
    out = _git("diff", "--name-only", base, head)
    if out is None:
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def discover_paths() -> list[str] | None:
    """Candidate diff the gate sees, plus anything still dirty in this tree."""
    base = _git("rev-parse", "--verify", "refs/greenline/last-green")
    if base is None:
        base = _git("rev-parse", "--verify", "master")
    if base is None:
        return None
    base = base.strip()
    merge_base = _git("merge-base", "HEAD", base)
    if merge_base is None:
        return None
    committed = _git("diff", "--name-only", merge_base.strip(), "HEAD")
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    if committed is None or dirty is None:
        return None
    paths = {line.strip() for line in committed.splitlines() if line.strip()}
    for line in dirty.splitlines():
        entry = line[3:].strip()
        if not entry:
            continue
        paths.update(part.strip().strip('"') for part in entry.split(" -> "))
    return sorted(paths)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="diff base (with --head; ignores the worktree)")
    parser.add_argument("--head", help="diff head (with --base; ignores the worktree)")
    args = parser.parse_args(argv)
    if (args.base is None) != (args.head is None):
        print("pass both --base and --head, or neither", file=sys.stderr)
        return 1
    if args.base is not None:
        paths = diff_paths(args.base, args.head)
        if paths is None:
            print("impact selection could not read the diff", file=sys.stderr)
            return 1
    else:
        paths = discover_paths()
        if paths is None:
            for leg in LEGS:
                print(leg)
            return 0
    for leg in sorted(select(paths)):
        print(leg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
