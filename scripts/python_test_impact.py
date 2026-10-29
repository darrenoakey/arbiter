"""Conservative Python dependency selection; unknown inputs retain the full suite.

Two graphs are built from literal imports. The precise graph (literal static
imports, package initializers and conftest fixtures) is the only proof that a
changed source has a test owner. The coverage graph adds computed/file-loader
imports inside sources as arcs to every source, so dynamic consumers are still
selected; such wildcard arcs never prove ownership.

Tests whose dependencies static imports cannot show (source readers,
subprocesses, file loaders, no static source imports) must be audited in
AUDITED, otherwise the full suite runs.
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
GLOBAL = "*"
# Audited owners of opaque tests, ADDITIVE to their static imports. An exact
# path both selects the test and proves ownership (its own static imports are
# followed too); a "dir/" prefix or GLOBAL only selects. An empty set certifies
# that the test's I/O touches only tmp files, ffmpeg, config/fixture data
# (non-Python changes already select the full suite) or external mirrors, so
# its static imports are sufficient.
AUDITED: dict[str, frozenset[str]] = {
    k: frozenset(v)
    for k, v in {
        # Subprocess imports the whole adapter registry.
        "tests/unit/test_adapter_imports.py": {"src/arbiter/"},
        "tests/unit/test_check_impact.py": {
            "scripts/check_impact.py",
            "scripts/python_test_impact.py",
            "run",
        },
        "tests/unit/test_python_test_impact.py": {
            "scripts/check_impact.py",
            "scripts/python_test_impact.py",
            "run",
        },
        "tests/unit/test_config.py": set(),
        "tests/unit/test_drain_wait.py": {
            "scripts/drain_wait.py",
            "deploy-to-spark.sh",
        },
        "tests/unit/test_flux3_klein.py": set(),
        # __import__ of retained adapters; subprocesses import the registry
        # and run worker_main.
        "tests/unit/test_image_policy.py": {
            "src/arbiter/worker_main.py",
            "src/arbiter/adapters/flux.py",
            "src/arbiter/adapters/flux2.py",
            "src/arbiter/adapters/z_image.py",
            "src/arbiter/adapters/",
        },
        "tests/unit/test_latentsync_mouth_guard.py": set(),
        "tests/unit/test_ltx2_dimensions.py": set(),
        "tests/unit/test_ltx25_generated_keyframes.py": set(),
        "tests/unit/test_ltx25_runtime_builder.py": {
            "scripts/build_ltx25_runtime.py",
            "runtime/",
        },
        "tests/unit/test_ltx25_source_packaging.py": {
            "scripts/build_ltx25_runtime.py",
            "scripts/package_ltx25_source.py",
            "runtime/",
        },
        # Repository-wide text scan: a guard, never an ownership proof.
        "tests/unit/test_no_cloud_video.py": {GLOBAL},
        "tests/unit/test_photo_enhance_pipeline.py": set(),
        "tests/unit/test_qwen_image_21.py": set(),
        "tests/unit/test_remote_readiness.py": {"scripts/check_remote_ready.py"},
        "tests/unit/test_trellis2_download.py": {"scripts/download_trellis2.py"},
        "src/arbiter/adapters/nimble_score_test.py": set(),
        "src/arbiter/adapters/mini_agi_test.py": set(),
        "src/arbiter/adapters/qlabs_noise_train_test.py": set(),
        "src/arbiter/adapters/test_id_lora.py": set(),
    }.items()
}
OPAQUE_TOKENS = (
    "read_text(",
    "read_bytes(",
    "subprocess.",
    "inspect.",
    "open(",
    ".glob(",
    ".rglob(",
    "ast.parse(",
    # Fixtures requested by computed name are invisible to the resolver.
    "getfixturevalue(",
)
FULL_PATHS = {
    "run",
    "greenline.toml",
    "pyproject.toml",
    "scripts/check_impact.py",
    "scripts/python_test_impact.py",
    "src/arbiter/adapters/registry.py",
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


def node_imports(tree: ast.AST, package: list[str]) -> set[str]:
    """Imported module names; "*" marks a computed or file-based import."""
    result: set[str] = set()
    loaders = {"__import__"}
    file_loaders = {
        "spec_from_file_location",
        "SourceFileLoader",
        "SourcelessFileLoader",
        "entry_points",
    }
    # Imported aliases of loaders count exactly like their attribute forms.
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in {"import_module", "__import__"}:
                    loaders.add(alias.asname or alias.name)
                elif alias.name in file_loaders:
                    aliases.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            loader = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ) or (isinstance(node.func, ast.Name) and node.func.id in loaders)
            file_loader = (
                isinstance(node.func, ast.Attribute) and node.func.attr in file_loaders
            ) or (isinstance(node.func, ast.Name) and node.func.id in aliases)
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


def imports(path: str, root: Path) -> set[str]:
    module = module_name(path).split(".")
    # A package initializer's relative imports resolve against the package.
    package = module if path.endswith("__init__.py") else module[:-1]
    return node_imports(ast.parse((root / path).read_text()), package)


def fixture_imports(test: str, root: Path) -> set[str]:
    """Imports made by tests/conftest.py fixtures the test requests."""
    conftest = root / "tests/conftest.py"
    if not test.startswith("tests/") or not conftest.is_file():
        return set()
    fixtures = {}
    body = ast.parse(conftest.read_text()).body
    # Module-level imports may be returned by any fixture.
    shared = set().union(
        *(
            node_imports(node, ["tests"])
            for node in body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    )
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            autouse = any(
                isinstance(d, ast.Call)
                and any(
                    k.arg == "autouse"
                    and not (isinstance(k.value, ast.Constant) and not k.value.value)
                    for k in d.keywords
                )
                for d in node.decorator_list
            )
            fixtures[node.name] = (
                node_imports(node, ["tests"]),
                {a.arg for a in node.args.args},
                autouse,
            )
    tree = ast.parse((root / test).read_text())
    wanted = {name for name, (_, _, autouse) in fixtures.items() if autouse}
    wanted |= {
        a.arg
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for a in node.args.args + node.args.kwonlyargs
    }
    wanted |= {
        arg.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "usefixtures"
        for arg in n.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    pending = wanted & fixtures.keys()
    result: set[str] = set(shared) if pending else set()
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        seen.add(name)
        names, parameters, _ = fixtures[name]
        result |= names
        pending |= (parameters & fixtures.keys()) - seen
    return result


def owned(owner: str, path: str) -> bool:
    return (
        owner == GLOBAL
        or owner == path
        or (owner.endswith("/") and path.startswith(owner))
    )


def select_tests(
    paths: list[str] | None,
    root: Path = ROOT,
    audited: dict[str, frozenset[str]] = AUDITED,
) -> list[str]:
    full = test_files(root)
    # Harness/config/runtime changes need the complete existing Python contract.
    # Missing/deleted files and parse errors must never silently remove coverage.
    # Package initializers execute for every importer of the package.
    if not paths or any(
        p in FULL_PATHS
        or p.endswith("__init__.py")
        or p.startswith(("runtime/", "config/", "requirements/"))
        for p in paths
    ):
        return full
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
    # Only tests/conftest.py fixtures are resolved; any other conftest is
    # unsupported ownership.
    if any(
        str(p.relative_to(root)) != "tests/conftest.py"
        for base in ("tests", "src")
        for p in (root / base).rglob("conftest.py")
    ):
        return full
    # A stale or broken audit declaration is unknown ownership.
    if any(
        test not in full
        or any(owner != GLOBAL and not (root / owner).exists() for owner in owners)
        for test, owners in audited.items()
    ):
        return full
    try:
        sources = sorted(
            str(p.relative_to(root)) for p in (root / "src/arbiter").rglob("*.py")
        )
        modules = {module_name(p): p for p in sources}
        literal = {}
        parents = {}
        dynamic = set()
        for path in sources + full:
            names = imports(path, root)
            if path in full:
                names |= fixture_imports(path, root)
            dynamic |= {path} if "*" in names else set()
            literal[path] = {modules[n] for n in names if n in modules}
            # Importing a.b.c executes a/__init__ and a/b/__init__ first.
            parents[path] = {
                modules[n]
                for n in (
                    ".".join(name.split(".")[:i])
                    for name in names
                    for i in range(1, name.count(".") + 1)
                )
                if n in modules
            } - literal[path]
        for test, owners in audited.items():
            literal[test] |= {o for o in owners if o in modules.values()}
        # Initializer changes select the full suite, so an initializer is
        # replaced by its literal imports. Its computed registry imports are
        # wildcard smoke arcs: covered by test_adapter_imports, never proof.
        inits = {
            path: literal[path] - {path}
            for path in sources
            if path.endswith("__init__.py")
        }
        precise = {
            path: {d for d in deps if d not in inits}
            | {i for d in deps | parents[path] if d in inits for i in inits[d]}
            for path, deps in literal.items()
        }
        # Computed imports in other sources may reach any source; audited
        # tests replace their own computed imports with explicit owners.
        coverage = {
            path: set(sources)
            if path in dynamic and path not in audited and path not in inits
            else precise[path]
            for path in literal
        }
        opaque = {
            p
            for p in full
            if p in dynamic
            or not literal[p]
            or any(token in (root / p).read_text() for token in OPAQUE_TOKENS)
        }
    except (OSError, SyntaxError, UnicodeError):
        return full
    if opaque - audited.keys():
        return full

    def closure(graph: dict[str, set[str]], changed: str) -> set[str]:
        consumers = {changed}
        while True:
            expanded = consumers | {p for p, deps in graph.items() if deps & consumers}
            if expanded == consumers:
                return consumers
            consumers = expanded

    affected: set[str] = set()
    for changed in python_paths:
        # Check EACH changed source, not the union: a covered adapter must not
        # hide an unrelated source whose test ownership is unknown.
        if changed.startswith("src/") and not set(full) & closure(precise, changed):
            return full
        affected |= closure(coverage, changed)
    selected = {
        test
        for test in full
        if test in affected
        or any(owned(o, p) for o in audited.get(test, ()) for p in paths)
    }
    return sorted(selected) or full
