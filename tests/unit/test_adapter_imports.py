"""Every worker imports the whole adapter registry, including in per-model
venvs that only carry that model's dependencies (the birefnet, moondream and
echomimic venvs have no pydantic). A module-level import of pydantic-backed
code in any adapter therefore kills every one of those workers at load
("load failed: subprocess died"), as nimble_score did on 2026-09-30."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
PROBE = """
import sys

class BlockPydantic:
    def find_spec(self, name, path=None, target=None):
        if name == "pydantic" or name.startswith("pydantic."):
            raise ModuleNotFoundError("No module named 'pydantic'", name="pydantic")
        return None

sys.meta_path.insert(0, BlockPydantic())
import arbiter.adapters
print("adapters imported")
"""


def test_adapter_registry_imports_without_pydantic():
    result = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=SRC, env={"PYTHONPATH": str(SRC)}, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "adapters imported" in result.stdout
