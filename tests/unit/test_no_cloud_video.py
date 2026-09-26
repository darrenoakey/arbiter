"""Cloud video renderers must not be reachable from this repository."""

from __future__ import annotations

from pathlib import Path


def test_cloud_minimax_client_and_host_are_absent():
    root = Path(__file__).resolve().parents[2]
    absent = [
        root / "src/arbiter/adapters/minimax_h3.py",
        root / "src/arbiter/adapters/minimax_h3_test.py",
        root / "config/spark/minimax-h3.model.json",
        root / "cmd/arbiter/staged_paths.go",
    ]
    present = [path for path in absent if path.exists()]
    assert present == []

    needles = ("minimax." + "io", 'get("' + "minimax" + '"')
    skip = {".git", ".venv", "venvs", "output", "local_output", "node_modules", "__pycache__"}
    hits = []
    for path in root.rglob("*"):
        if any(part in skip for part in path.parts):
            continue
        if not path.is_file() or path.suffix not in {
            ".py",
            ".go",
            ".md",
            ".json",
            ".sh",
            ".toml",
            ".yml",
            ".yaml",
            ".txt",
        }:
            continue
        if "\0" in path.read_bytes()[:8192].decode("latin1"):
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore").lower()
        for needle in needles:
            if needle in text:
                hits.append(f"{needle} in {path.relative_to(root)}")
    assert hits == []
