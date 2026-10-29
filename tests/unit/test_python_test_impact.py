"""Release regression coverage uses the real repository and recorded candidate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.python_test_impact import (
    EXTRA_TESTS,
    ROOT as SELECTOR_ROOT,
    imports,
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
        ["src/arbiter/adapters/photo_enhance_impl/__init__.py"],
        ["tests/conftest.py"],
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
            "tests/unit/test_ltx2_dimensions.py",
            "tests/unit/test_config.py",
            "tests/unit/test_python_test_impact.py",
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
    # The repository-wide cloud-video scan reads every changed text file.
    assert select_tests(["deploy-to-spark.sh", "docs/notes.md"]) == [
        "tests/unit/test_drain_wait.py",
        "tests/unit/test_no_cloud_video.py",
    ]
    assert select_tests(["tests/unit/test_aesthetic_batch.py"]) == [
        "tests/unit/test_aesthetic_batch.py",
        "tests/unit/test_no_cloud_video.py",
    ]


def test_audited_owners_and_fixtures_select_real_consumers():
    # conftest's store fixture is the only route from this test to the store.
    assert "tests/unit/test_job_lifecycle.py" in select_tests(["src/arbiter/store.py"])
    # Importing one adapter through the registry package does not make a test
    # a consumer of every adapter the registry computes; the smoke covers it.
    aesthetic = select_tests(["src/arbiter/adapters/aesthetic_scorer.py"])
    assert "tests/unit/test_adapter_imports.py" in aesthetic
    assert "tests/unit/test_latentsync_mouth_guard.py" not in aesthetic
    # worker_main is owned only through the audited subprocess declaration.
    assert "tests/unit/test_image_policy.py" in select_tests(
        ["src/arbiter/worker_main.py"]
    )
    assert set(select_tests(["src/arbiter/worker_main.py"])) != set(suite_files())


def test_invalid_source_fails_closed(tmp_path):
    # Parse real files in an isolated tree, with a real syntax error, not a mock.
    for path in suite_files() + ["src/arbiter/adapters/aesthetic_scorer.py"]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "def broken(:\n" if path.endswith("aesthetic_scorer.py") else ""
        )
    audited = {p: frozenset() for p in suite_files(tmp_path)}
    assert select_tests(
        ["src/arbiter/adapters/aesthetic_scorer.py"], tmp_path, audited
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


def build(root, contents):
    """Write a real miniature repository; extra tests are empty files."""
    for path in set(EXTRA_TESTS) | set(contents):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents.get(path, ""))
    return {p: frozenset() for p in EXTRA_TESTS}


MINIATURE = {
    "src/arbiter/changed.py": "value = 1\n",
    "src/arbiter/other.py": "value = 2\n",
    "src/arbiter/loader.py": 'import importlib\nname = "arbiter." + "x"\nloaded = importlib.import_module(name)\n',
    "tests/unit/test_direct.py": "from arbiter.changed import value\n",
    "tests/unit/test_unrelated.py": "from arbiter.other import value\n",
    "tests/unit/test_dynamic_source.py": "from arbiter.loader import loaded\n",
    "tests/unit/test_builtin_import.py": 'from arbiter.other import value\nloaded = __import__("arbiter.changed")\n',
    "tests/unit/test_computed_import.py": 'from importlib import import_module as load\nfrom arbiter.other import value\nname = "arbiter." + "changed"\nloaded = load(name)\n',
    "tests/unit/test_file_loader.py": 'from importlib import util\nfrom arbiter.other import value\nloaded = util.spec_from_file_location("x", "changed.py")\n',
    "tests/unit/test_open_reader.py": 'from arbiter.other import value\nsource = open("changed.py").read()\n',
    "tests/unit/test_aliased_file_loader.py": 'from importlib.util import spec_from_file_location as load\nfrom arbiter.other import value\nloaded = load("x", "changed.py")\n',
    "tests/unit/test_aliased_builtin.py": 'from builtins import __import__ as load\nfrom arbiter.other import value\nname = "arbiter." + "changed"\nloaded = load(name)\n',
    "tests/unit/test_path_reader.py": 'from pathlib import Path\nfrom arbiter.other import value\nsource = Path("changed.py").open().read()\n',
}
READERS = (
    "tests/unit/test_aliased_file_loader.py",
    "tests/unit/test_aliased_builtin.py",
    "tests/unit/test_computed_import.py",
    "tests/unit/test_file_loader.py",
    "tests/unit/test_open_reader.py",
    "tests/unit/test_path_reader.py",
)


def test_dynamic_loaders_and_source_readers_preserve_indirect_coverage(tmp_path):
    # A real miniature repository proves both direct and indirect consumers.
    audited = build(tmp_path, MINIATURE) | {
        reader: frozenset({"src/arbiter/changed.py"}) for reader in READERS
    }
    selected = set(select_tests(["src/arbiter/changed.py"], tmp_path, audited))
    assert {
        p for p in MINIATURE if p.startswith("tests/") and "unrelated" not in p
    } == selected
    # Each undeclared opaque test is unknown ownership: full suite.
    for reader in READERS:
        partial = {k: v for k, v in audited.items() if k != reader}
        assert select_tests(
            ["src/arbiter/changed.py"], tmp_path, partial
        ) == suite_files(tmp_path)
    # Stale declarations fail closed too.
    for stale in (
        {"tests/unit/test_gone.py": frozenset()},
        {READERS[0]: frozenset({"src/arbiter/gone.py"})},
    ):
        assert select_tests(
            ["src/arbiter/changed.py"], tmp_path, audited | stale
        ) == suite_files(tmp_path)


def test_unrelated_declared_readers_do_not_grow_the_suite(tmp_path):
    audited = build(tmp_path, MINIATURE) | {
        reader: frozenset({"src/arbiter/changed.py"}) for reader in READERS
    }
    before = select_tests(["src/arbiter/changed.py"], tmp_path, audited)
    added = {
        "tests/unit/test_other_reader.py": 'from arbiter.other import value\ndata = open("other.json").read()\n',
        "tests/unit/test_no_imports.py": 'import subprocess\nsubprocess.run(["true"])\n',
    }
    build(tmp_path, added)
    assert select_tests(["src/arbiter/changed.py"], tmp_path, audited) == suite_files(
        tmp_path
    )
    audited |= {p: frozenset({"src/arbiter/other.py"}) for p in added}
    assert select_tests(["src/arbiter/changed.py"], tmp_path, audited) == before
    # A declared exact owner proves ownership and selects the declared test.
    assert set(added) | {"tests/unit/test_unrelated.py"} <= set(
        select_tests(["src/arbiter/other.py"], tmp_path, audited)
    )


def test_dynamic_initializer_never_proves_ownership(tmp_path):
    contents = {
        "src/arbiter/pkg/__init__.py": 'import importlib\nfor name in ("owned", "other", "orphan"):\n    importlib.import_module(f".{name}", __name__)\n',
        "src/arbiter/pkg/owned.py": "value = 1\n",
        "src/arbiter/pkg/orphan.py": "value = 2\n",
        "src/arbiter/pkg/other.py": "value = 3\n",
        "tests/unit/test_registry.py": "import arbiter.pkg\n",
        "tests/unit/test_other.py": "from arbiter.pkg import other\n",
        "tests/unit/test_owned.py": "from arbiter.pkg.owned import value\n",
    }
    # Like test_adapter_imports, the smoke is selected by an explicit prefix.
    audited = build(tmp_path, contents) | {
        "tests/unit/test_registry.py": frozenset({"src/arbiter/pkg/"})
    }
    # The registry smoke reaches every source but owns none of them, and an
    # importer of the package is not a consumer of its computed imports.
    assert select_tests(
        ["src/arbiter/pkg/orphan.py"], tmp_path, audited
    ) == suite_files(tmp_path)
    assert select_tests(["src/arbiter/pkg/owned.py"], tmp_path, audited) == [
        "tests/unit/test_owned.py",
        "tests/unit/test_registry.py",
    ]


def test_package_initializer_relative_imports_and_fixtures(tmp_path):
    contents = {
        "src/arbiter/pkg/__init__.py": "from .helper import value\n",
        "src/arbiter/pkg/helper.py": "value = 1\n",
        "src/arbiter/pkg/sub.py": "value = 2\n",
        "src/arbiter/fixtured.py": "value = 3\n",
        "src/arbiter/shared.py": "value = 4\n",
        "tests/conftest.py": "import pytest\nfrom arbiter.shared import value\n\n@pytest.fixture\ndef base():\n    from arbiter import fixtured\n    return fixtured\n\n@pytest.fixture\ndef thing(base):\n    return base\n",
        "tests/unit/test_sub.py": "from arbiter.pkg.sub import value\n",
        "tests/unit/test_fixture.py": "def test_uses(thing):\n    assert thing\n",
        "tests/unit/test_plain.py": "from arbiter.pkg.sub import value\ndef test_plain(tmp_path):\n    assert tmp_path\n",
    }
    audited = build(tmp_path, contents)
    assert "arbiter.pkg.helper" in imports("src/arbiter/pkg/__init__.py", tmp_path)
    # Importing arbiter.pkg.sub executes the initializer, which imports helper.
    assert "tests/unit/test_sub.py" in select_tests(
        ["src/arbiter/pkg/helper.py"], tmp_path, audited
    )
    for fixture_source in ("src/arbiter/fixtured.py", "src/arbiter/shared.py"):
        assert select_tests([fixture_source], tmp_path, audited) == [
            "tests/unit/test_fixture.py"
        ]
    # A fixture requested by computed name is opaque until audited.
    build(
        tmp_path,
        {
            "tests/unit/test_by_name.py": 'def test_it(request):\n    request.getfixturevalue("th" + "ing")\n'
        },
    )
    assert select_tests(["src/arbiter/fixtured.py"], tmp_path, audited) == suite_files(
        tmp_path
    )
    (tmp_path / "tests/unit/test_by_name.py").unlink()
    # Nested conftest hierarchies are unsupported and fail closed.
    build(tmp_path, {"tests/unit/conftest.py": "import pytest\n"})
    assert select_tests(["src/arbiter/fixtured.py"], tmp_path, audited) == suite_files(
        tmp_path
    )
    for changed in ("src/arbiter/pkg/__init__.py", "tests/conftest.py"):
        assert select_tests([changed], tmp_path, audited) == suite_files(tmp_path)
