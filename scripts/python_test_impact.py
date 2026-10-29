"""Conservative Python dependency selection; unknown inputs retain the full suite.

The registry import smoke is always retained: Python imports execute package
initializers even when a test only names one adapter. Source-reading tests are
also retained unless their isolated infrastructure ownership is explicit below.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRA_TESTS = (
    "src/arbiter/schemas_test.py",
    "src/arbiter/adapters/nimble_score_test.py",
    "src/arbiter/adapters/clef_score_test.py",
    "src/arbiter/adapters/mini_agi_test.py",
    "src/arbiter/adapters/qlabs_noise_train_test.py",
    "src/arbiter/adapters/test_id_lora.py",
)
# These tests construct/activate the immutable runtime, not adapter parameters.
ISOLATED = {
    "tests/unit/test_ltx25_runtime_builder.py": {
        "scripts/build_ltx25_runtime.py",
        "runtime/",
        "deploy-to-spark.sh",
    },
    "tests/unit/test_ltx25_source_packaging.py": {
        "scripts/build_ltx25_runtime.py",
        "scripts/package_ltx25_source.py",
        "runtime/",
    },
    "tests/unit/test_drain_wait.py": {"deploy-to-spark.sh"},
    "tests/unit/test_check_impact.py": {
        "scripts/check_impact.py",
        "scripts/python_test_impact.py",
        "run",
    },
}


def test_files(root: Path = ROOT) -> list[str]:
    return sorted(
        {str(p.relative_to(root)) for p in (root / "tests/unit").glob("test_*.py")}
        | set(EXTRA_TESTS)
    )


def module_name(path: str) -> str:
    return (
        path.removeprefix("src/")
        .removesuffix(".py")
        .replace("/", ".")
        .removesuffix(".__init__")
    )


def imports(path: str, root: Path) -> set[str]:
    tree = ast.parse((root / path).read_text())
    package = module_name(path).split(".")[:-1]
    result: set[str] = set()
    loaders = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            loaders.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "import_module"
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            loader = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ) or (isinstance(node.func, ast.Name) and node.func.id in loaders)
            file_loader = isinstance(node.func, ast.Attribute) and node.func.attr in {
                "spec_from_file_location",
                "SourceFileLoader",
                "SourcelessFileLoader",
                "entry_points",
            }
            if loader:
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and not node.args[0].value.startswith(".")
                ):
                    result.add(node.args[0].value)
                else:
                    result.add("*")
            if file_loader:
                result.add("*")
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[: len(package) - node.level + 1] if node.level else []
            base = ".".join(prefix + ([node.module] if node.module else []))
            result.add(base)
            result.update(f"{base}.{alias.name}" for alias in node.names)
    return result


def select_tests(paths: list[str] | None, root: Path = ROOT) -> list[str]:
    full = test_files(root)
    # Harness/config/runtime changes need the complete existing Python contract.
    # Missing/deleted files and parse errors must never silently remove coverage.
    if not paths or any(
        p
        in {
            "run",
            "greenline.toml",
            "pyproject.toml",
            "scripts/check_impact.py",
            "scripts/python_test_impact.py",
            "src/arbiter/adapters/__init__.py",
            "src/arbiter/adapters/registry.py",
        }
        or p.startswith(("runtime/", "config/", "requirements/"))
        for p in paths
    ):
        return full
    relevant = [
        p
        for p in paths
        if not p.startswith(("docs/", "tests/integration/")) and not p.endswith(".md")
    ]
    if relevant and set(relevant) <= {"deploy-to-spark.sh"}:
        return ["tests/unit/test_drain_wait.py"]
    if relevant and all(p in full for p in relevant):
        return sorted(relevant)
    python_paths = [
        p for p in paths if p.endswith(".py") and not p.startswith("tests/integration/")
    ]
    if any(not (root / p).is_file() for p in python_paths):
        return full
    if any(not (p.startswith("src/arbiter/") or p in full) for p in python_paths):
        return full
    if any(
        not (p.endswith(".py") or p.endswith(".md") or p == "deploy-to-spark.sh")
        for p in paths
    ):
        return full
    try:
        sources = sorted(
            str(p.relative_to(root)) for p in (root / "src/arbiter").rglob("*.py")
        )
        modules = {module_name(p): p for p in sources}
        graph = {}
        for path in sources + full:
            dependencies = imports(path, root)
            graph[path] = (
                set(sources)
                if "*" in dependencies
                else {modules[name] for name in dependencies if name in modules}
            )
        opaque = {
            p
            for p in full
            # These two readers only inspect model JSON; config diffs already
            # select the full contract, and their code imports are in the graph.
            if p
            not in {
                "tests/unit/test_qwen_image_21.py",
                "tests/unit/test_flux3_klein.py",
                # Uses Image.open on generated images, not repository source.
                # Its literal dynamic adapter import is tracked above.
                "tests/unit/test_photo_enhance_pipeline.py",
            }
            and any(
                token in (root / p).read_text()
                for token in (
                    "read_text(",
                    "read_bytes(",
                    "subprocess.",
                    "inspect.",
                    "open(",
                    ".glob(",
                    ".rglob(",
                    "ast.parse(",
                )
            )
        }
    except (OSError, SyntaxError, UnicodeError):
        return full
    affected: set[str] = set()
    for changed in python_paths:
        consumers = {changed}
        while True:
            expanded = consumers | {
                p for p, dependencies in graph.items() if dependencies & consumers
            }
            if expanded == consumers:
                break
            consumers = expanded
        # Check EACH changed source, not the union: a covered adapter must not
        # hide an unrelated source whose test ownership is unknown.
        if changed.startswith("src/") and not any(t in consumers for t in full):
            return full
        affected |= consumers
    selected = set()
    for test in full:
        if test in ISOLATED:
            owners = ISOLATED[test]
            if test in paths or any(
                p == owner or (owner.endswith("/") and p.startswith(owner))
                for p in paths
                for owner in owners
            ):
                selected.add(test)
        elif test in affected or test in opaque or not graph[test]:
            # No static dependencies often means subprocess/source inspection.
            selected.add(test)
    if python_paths:
        selected.add("tests/unit/test_adapter_imports.py")
    return sorted(selected) or full
