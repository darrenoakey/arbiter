"""OrbitQuant MiniMax-H3-Image adapter: snapshot verification and bundle-loader
import against a real on-disk layout. No GPU, torch or model run; these are
static unit tests and prove no GPU route."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arbiter.adapters.base import LoadError
from arbiter.adapters.minimax_h3_image import EXPECTED_SAMPLE_SIGMAS
from arbiter.adapters.minimax_h3_image_orbitquant import (
    ORBITQUANT_DIFFUSERS_COMMIT,
    ORBITQUANT_VERSION,
    MiniMaxH3ImageOrbitQuantAdapter,
)

BINARY_FILES = (
    "LICENSE", "NOTICE",
    "transformer/config.json", "transformer/model.safetensors",
    "transformer/modulation.json", "transformer/orbitquant_manifest.json",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json", "tokenizer/tokenizer.json",
    "processor/preprocessor_config.json", "processor/tokenizer.json",
)


def make_bundle(root: Path, *, diffusers: str = ORBITQUANT_DIFFUSERS_COMMIT) -> Path:
    entry = root / "minimax-h3-image-orbitquant-w4a8"
    for rel in BINARY_FILES:
        target = entry / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"{}")
    (entry / "manifest.json").write_text(json.dumps({
        "format": "minimax-h3-image-orbitquant",
        "orbitquant": ORBITQUANT_VERSION,
        "diffusers": diffusers,
        "base_revision": "7b6142f879bf7a973c449e1dea5480289736cdbd",
    }))
    (entry / "modular_model_index.json").write_text(json.dumps({
        "_class_name": "MiniMaxH3ImageModularPipeline",
        "sample_sigmas": list(EXPECTED_SAMPLE_SIGMAS),
    }))
    (entry / "minimax_h3_image_orbitquant.py").write_text(
        "def load_pipeline(root='.', *, offload=False, attention='cudnn', device='cuda'):\n"
        "    return ('bundle', root, offload, attention, device)\n"
    )
    return entry


def test_verify_reads_manifest_file_inside_snapshot_directory(tmp_path):
    entry = make_bundle(tmp_path)
    info = MiniMaxH3ImageOrbitQuantAdapter(entry)._verify_orbitquant_snapshot()
    assert info["weights_dir"] == str(entry)
    assert info["manifest"]["diffusers"] == ORBITQUANT_DIFFUSERS_COMMIT
    assert info["precision"]["transformer"] == "orbitquant-w4a8"


def test_verify_rejects_foreign_diffusers_pin_and_missing_loader(tmp_path):
    entry = make_bundle(tmp_path / "a", diffusers="0" * 40)
    with pytest.raises(LoadError, match="revision mismatch"):
        MiniMaxH3ImageOrbitQuantAdapter(entry)._verify_orbitquant_snapshot()
    entry = make_bundle(tmp_path / "b")
    (entry / "minimax_h3_image_orbitquant.py").unlink()
    with pytest.raises(LoadError, match="minimax_h3_image_orbitquant.py"):
        MiniMaxH3ImageOrbitQuantAdapter(entry)._verify_orbitquant_snapshot()


def test_bundle_loader_imports_from_snapshot_file_not_sys_path(tmp_path):
    entry = make_bundle(tmp_path)
    module = MiniMaxH3ImageOrbitQuantAdapter(entry)._bundle_loader()
    assert Path(module.__file__) == entry / "minimax_h3_image_orbitquant.py"
    assert module.load_pipeline(entry, device="cuda")[0] == "bundle"


def test_adapter_sources_have_no_undefined_names():
    """Adapter load paths only run on the GPU host. A NameError there
    (2026-11-11: `verified` was never assigned) shows up only after a
    multi-minute load. Ruff's F821 finds it statically, before deploy."""
    import subprocess
    import sys

    src = Path(__file__).resolve().parents[2] / "src" / "arbiter"
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-cache", "--select", "F821",
         "--output-format", "concise", str(src)],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_result_metadata_reports_quantized_precision_not_bf16(tmp_path):
    adapter = MiniMaxH3ImageOrbitQuantAdapter(make_bundle(tmp_path))
    assert adapter.result_precision["transformer"] == "orbitquant-w4a8"
    assert adapter.result_precision["text_encoder"] == "orbitquant-w4a8"
    assert "bfloat16" not in adapter.result_precision.values()
    assert not any("staged loading" in item for item in adapter.result_limitations)
    assert adapter._verify_orbitquant_snapshot()["precision"] == adapter.result_precision
