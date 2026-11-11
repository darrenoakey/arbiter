"""Direct-CUDA OrbitQuant W4A8 MiniMax-H3-Image (never ComfyUI).

WaveCut's packed transformer and truncated Qwen encoder load directly into
CUDA. Fused INT8/Triton kernels avoid dequantized weight matrices. Both
components remain resident between jobs, eliminating BF16 component swaps.
"""
import os
import threading
import time
from pathlib import Path

from arbiter.adapters.minimax_h3_image import (
    EXPECTED_PIPELINE_CLASS,
    EXPECTED_SAMPLE_SIGMAS,
    EXPECTED_SCHEDULER_CLASS,
    EXPECTED_TRANSFORMER_CLASS,
    EXPECTED_VAE_CLASS,
    MiniMaxH3ImageAdapter,
)
from arbiter.adapters.base import HeapTrimGuard, LoadError
from arbiter.adapters.minimax_h3_image import log, verify_diffusers_runtime
from arbiter.adapters.registry import register

# WaveCut bundle manifest pins its own diffusers build (installed in venvs/orbitquant).
ORBITQUANT_DIFFUSERS_COMMIT = "875373ac7f5856071e345a3e394c74efccc6b9e6"
ORBITQUANT_VERSION = "0.18.0"


@register
class MiniMaxH3ImageOrbitQuantAdapter(MiniMaxH3ImageAdapter):
    """Resident direct-CUDA OrbitQuant MiniMax-H3-Image, never ComfyUI."""

    model_id = "minimax-h3-image-orbitquant-w4a8"
    hf_model_path = "/home/darren/nvme-models/minimax-h3-image-orbitquant-w4a8"

    def __init__(self, models_dir: Path | None = None) -> None:
        super().__init__()
        self.models_dir = Path(models_dir or self.hf_model_path)

    def _verify_orbitquant_snapshot(self) -> dict:
        entry = self.models_dir
        if entry.is_symlink() or not entry.is_dir():
            raise LoadError(f"{self.model_id} requires a real snapshot directory")
        manifest = self._read_manifest(entry / "manifest.json")
        if manifest.get("format") != "minimax-h3-image-orbitquant":
            raise LoadError("not an H3-Image OrbitQuant snapshot")
        if manifest.get("orbitquant") != ORBITQUANT_VERSION or manifest.get("diffusers") != ORBITQUANT_DIFFUSERS_COMMIT:
            raise LoadError("OrbitQuant/runtime revision mismatch")
        required = (
            "manifest.json", "modular_model_index.json", "LICENSE", "NOTICE",
            "minimax_h3_image_orbitquant.py",
            "transformer/config.json", "transformer/model.safetensors",
            "transformer/modulation.json", "transformer/orbitquant_manifest.json",
            "text_encoder/config.json", "text_encoder/model.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
            "scheduler/scheduler_config.json", "tokenizer/tokenizer.json",
            "processor/preprocessor_config.json", "processor/tokenizer.json",
        )
        for rel in required:
            target = entry / rel
            if target.is_symlink() or not target.is_file():
                raise LoadError(f"missing OrbitQuant file {rel}")
        sigmas = self._read_manifest(entry / "modular_model_index.json").get("sample_sigmas")
        if not isinstance(sigmas, list) or tuple(float(x) for x in sigmas) != EXPECTED_SAMPLE_SIGMAS:
            raise LoadError("OrbitQuant sigma grid differs from six-step checkpoint")
        return {
            "hf_repo": "WaveCut/MiniMax-H3-Image-OrbitQuant-W4A8",
            "revision": manifest.get("base_revision"),
            "weights_dir": str(entry),
            "manifest": manifest,
            "precision": {
                "transformer": "orbitquant-w4a8",
                "text_encoder": "orbitquant-w4a8",
                "vae": "original-bf16",
                "initial_noise": "float32",
            },
        }

    @staticmethod
    def _read_manifest(path: Path) -> dict:
        import json
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LoadError(f"cannot read OrbitQuant manifest {path}: {exc}") from exc

    def _bundle_loader(self):
        """Import the verified snapshot's own loader by file, never via sys.path."""
        import importlib.util

        source = self.models_dir / "minimax_h3_image_orbitquant.py"
        spec = importlib.util.spec_from_file_location("_orbitquant_h3_bundle_loader", source)
        if spec is None or spec.loader is None:
            raise LoadError(f"cannot import OrbitQuant bundle loader {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "load_pipeline", None)):
            raise LoadError("OrbitQuant bundle loader lacks load_pipeline")
        return module

    def load(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise LoadError(f"{self.model_id} requires cuda")
        declared = os.environ.get("ARBITER_MEMORY_GB")
        if not declared:
            raise LoadError(f"{self.model_id} requires Arbiter's CUDA memory cap")
        self._start_progress("verify OrbitQuant snapshot and direct-CUDA load")
        started = time.perf_counter()
        try:
            weights = self._verify_orbitquant_snapshot()
            runtime_pin = verify_diffusers_runtime(ORBITQUANT_DIFFUSERS_COMMIT)
            verified = time.perf_counter()
            import torch
            load_pipeline = self._bundle_loader().load_pipeline
            with HeapTrimGuard():
                pipe = load_pipeline(self.models_dir, offload=False, attention="cudnn", device=device)
            loaded = time.perf_counter()
        finally:
            self._stop_progress()
        if type(pipe).__name__ != EXPECTED_PIPELINE_CLASS:
            self._cleanup_gpu()
            raise LoadError(f"unexpected OrbitQuant pipeline class {type(pipe).__name__}")
        for attr, expected in (
            ("transformer", EXPECTED_TRANSFORMER_CLASS),
            ("vae", EXPECTED_VAE_CLASS),
            ("scheduler", EXPECTED_SCHEDULER_CLASS),
        ):
            if type(getattr(pipe, attr, None)).__name__ != expected:
                self._cleanup_gpu()
                raise LoadError(f"unexpected OrbitQuant {attr} class")
        self._pipe = pipe
        self._renders = 0
        self._load_info = {
            "weights": weights,
            "runtime": {
                **runtime_pin,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(),
                "loading": "direct-CUDA packed weights, fused INT8 kernels, no ComfyUI",
            },
            "declared_memory_gb": float(declared),
            "load_seconds": round(loaded - started, 3),
            "verify_seconds": round(verified - started, 3),
            "pipeline_load_seconds": round(loaded - verified, 3),
            "resident_allocated_bytes": torch.cuda.memory_allocated(),
            "resident_reserved_bytes": torch.cuda.memory_reserved(),
            "lifecycle": "resident-orbitquant-no-component-swap",
        }
        log.info("%s loaded: %s", self.model_id, self._load_info)

    def _encode_and_swap(self, pipe, prompt: str, cancel_flag: threading.Event):
        """Use the resident encoder and transformer; no component unload."""
        import torch
        from diffusers.modular_pipelines.minimax_h3_image.encoders import MiniMaxH3ImageTextEncoderStep
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        self._progress_phase = "prompt encoding (resident OrbitQuant)"
        state = PipelineState()
        state.set("prompt", prompt)
        started = time.perf_counter()
        with torch.inference_mode():
            _, state = MiniMaxH3ImageTextEncoderStep()(pipe, state)
            torch.cuda.synchronize()
            embeddings = state.get("prompt_embeds")
            tags = state.get("text_token_tags")
        encoded = time.perf_counter()
        state = PipelineState()
        state.set("prompt_embeds", embeddings)
        state.set("text_token_tags", tags)
        self._check_cancel(cancel_flag)
        return state, {
            "encode_seconds": round(encoded - started, 3),
            "encoder_free_seconds": 0.0,
            "image_stage_load_seconds": 0.0,
            "component_swaps": 0,
            "resident_orbitquant": True,
        }
