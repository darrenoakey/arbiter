"""Impact selection for ./run check.

The gl/adapter-lazy-config release (98bba7d) changed two Python files. The Go
suite cannot see that diff; selecting it anyway was the 113s check.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_impact.py"

spec = importlib.util.spec_from_file_location("check_impact", SCRIPT)
assert spec is not None and spec.loader is not None
check_impact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_impact)

LAZY_CONFIG = [
    "src/arbiter/adapters/nimble_score.py",
    "tests/unit/test_adapter_imports.py",
]


def test_python_adapter_diff_skips_go():
    assert check_impact.select(LAZY_CONFIG) == {"python"}


def test_go_source_skips_python():
    assert check_impact.select(["cmd/arbiter/scheduler.go"]) == {"go"}


def test_embedded_sql_selects_go():
    assert check_impact.select(["cmd/arbiter/migrations/001.sql"]) == {"go"}


def test_config_selects_go_and_python():
    assert check_impact.select(["config/spark/ltx.json"]) == {"go", "python"}


def test_runtime_tree_selects_python_only():
    assert check_impact.select(["runtime/ltx25/foo.py"]) == {"python"}


def test_gpu_idle_hook_is_its_own_leg():
    assert check_impact.select(["scripts/gpu_idle_hook.py"]) == {"gpu-idle"}


def test_docs_select_nothing():
    assert check_impact.select(["docs/greenline.md"]) == set()


def test_unknown_or_empty_or_harness_selects_every_leg():
    every = set(check_impact.LEGS)
    assert check_impact.select(["mystery.bin"]) == every
    assert check_impact.select([]) == every
    assert check_impact.select(["run"]) == every
    assert check_impact.select(LAZY_CONFIG + ["run"]) == every


def test_recorded_lazy_config_commit_skips_go():
    result = subprocess.run(
        ["python3", str(SCRIPT), "--base", "98bba7d^", "--head", "98bba7d"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == ["python"]
    via_run = subprocess.run(
        ["./run", "check-impact", "--base", "98bba7d^", "--head", "98bba7d"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert via_run.stdout.split() == ["python"]
