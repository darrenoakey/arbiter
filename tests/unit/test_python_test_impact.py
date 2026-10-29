"""Release regression coverage uses the real repository and recorded candidate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.python_test_impact import (
    EXTRA_TESTS,
    ROOT as SELECTOR_ROOT,
    select_tests,
    test_files as suite_files,
)

ROOT = Path(__file__).resolve().parents[2]


def test_full_suite_preserves_the_previous_gate_contract():
    expected = {
        str(p.relative_to(ROOT)) for p in (ROOT / "tests/unit").glob("test_*.py")
    } | set(EXTRA_TESTS)
    assert set(suite_files()) == expected
    assert SELECTOR_ROOT == ROOT
    for paths in (
        None,
        [],
        ["run"],
        ["scripts/python_test_impact.py"],
        ["missing.py"],
        ["src/arbiter/gone.py"],
        ["runtime/ltx25/manifest.json"],
        ["config/spark/changed.json"],
    ):
        assert set(select_tests(paths)) == expected


def test_recorded_aesthetic_release_keeps_consumers_not_runtime_construction():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_impact.py",
            "--base",
            "06b3b9d^",
            "--head",
            "06b3b9d",
            "--python-tests",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    selected = set(result.stdout.splitlines())
    assert {
        "tests/unit/test_aesthetic_batch.py",
        "tests/unit/test_adapter_imports.py",
        "tests/unit/test_job_types.py",
        "src/arbiter/schemas_test.py",
        "src/arbiter/adapters/clef_score_test.py",
    } <= selected
    assert (
        not {
            "tests/unit/test_ltx25_runtime_builder.py",
            "tests/unit/test_ltx25_source_packaging.py",
            "tests/unit/test_photo_enhance_pipeline.py",
            "tests/unit/test_drain_wait.py",
        }
        & selected
    )


def test_shared_base_change_keeps_transitive_adapter_tests():
    selected = set(select_tests(["src/arbiter/adapters/base.py"]))
    assert {
        "tests/unit/test_aesthetic_batch.py",
        "tests/unit/test_photo_enhance_pipeline.py",
        "tests/unit/test_ltx2_dimensions.py",
        "tests/unit/test_adapter_imports.py",
    } <= selected


def test_deploy_and_test_only_changes_have_bounded_ownership():
    assert select_tests(["deploy-to-spark.sh", "docs/notes.md"]) == [
        "tests/unit/test_drain_wait.py"
    ]
    assert select_tests(["tests/unit/test_aesthetic_batch.py"]) == [
        "tests/unit/test_aesthetic_batch.py"
    ]


def test_invalid_source_fails_closed(tmp_path):
    # Parse real files in an isolated tree, with a real syntax error, not a mock.
    for path in suite_files() + ["src/arbiter/adapters/aesthetic_scorer.py"]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "def broken(:\n" if path.endswith("aesthetic_scorer.py") else ""
        )
    assert select_tests(
        ["src/arbiter/adapters/aesthetic_scorer.py"], tmp_path
    ) == suite_files(tmp_path)


def test_cli_produces_the_same_recorded_plan_through_run():
    command = ["--base", "06b3b9d^", "--head", "06b3b9d", "--python-tests"]
    direct = subprocess.run(
        [sys.executable, "scripts/check_impact.py", *command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    facade = subprocess.run(
        ["./run", "check-impact", *command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert facade.stdout == direct.stdout


def test_dynamic_loaders_and_source_readers_preserve_indirect_coverage(tmp_path):
    # A real miniature repository proves both direct and indirect consumers.
    contents = {
        "src/arbiter/changed.py": "value = 1\n",
        "src/arbiter/other.py": "value = 2\n",
        "tests/unit/test_direct.py": "from arbiter.changed import value\n",
        "tests/unit/test_unrelated.py": "from arbiter.other import value\n",
        "tests/unit/test_builtin_import.py": 'from arbiter.other import value\nloaded = __import__("arbiter.changed")\n',
        "tests/unit/test_computed_import.py": 'from importlib import import_module as load\nfrom arbiter.other import value\nname = "arbiter." + "changed"\nloaded = load(name)\n',
        "tests/unit/test_file_loader.py": 'from importlib import util\nfrom arbiter.other import value\nloaded = util.spec_from_file_location("x", "changed.py")\n',
        "tests/unit/test_open_reader.py": 'from arbiter.other import value\nsource = open("changed.py").read()\n',
        "tests/unit/test_path_reader.py": 'from pathlib import Path\nfrom arbiter.other import value\nsource = Path("changed.py").open().read()\n',
    }
    for path in set(suite_files(tmp_path)) | set(contents):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents.get(path, ""))
    selected = set(select_tests(["src/arbiter/changed.py"], tmp_path))
    assert {
        p for p in contents if p.startswith("tests/") and "unrelated" not in p
    } <= selected
    assert "tests/unit/test_unrelated.py" not in selected
