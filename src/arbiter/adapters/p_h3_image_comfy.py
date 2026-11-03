"""PrunaAI P-H3-Image W4A8 through official ComfyUI, benchmark-only.

The supported runtime is comfyanonymous/ComfyUI commit
576b642e77b4c42f84afe583c8e8f0eef7637618 (PH3Image support). This module
imports no ComfyUI code until an approved installed worker loads. Offline
unit tests exercise only our own validation, hash verification and graph
construction. The former diffusers scaffold was removed because upstream has
no PH3Image pipeline at the inspected revision.

Weights are the official three-file no-upsampler set, exactly 20,084,601,325
bytes. The existing Mac mini Arbiter share is the intended model location;
no existing MiniMax-H3 video component is accepted as a substitute.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from arbiter.adapters.base import CancelledException, InferenceError, LoadError, ModelAdapter
from arbiter.adapters.flux3_klein import write_png_durably
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

PH3_REPO = "PrunaAI/P-H3-Image-Comfy"
PH3_REVISION = "44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543"
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFY_REVISION = "576b642e77b4c42f84afe583c8e8f0eef7637618"
DEFAULT_MODELS_DIR = Path("/mnt/arbiter-store/models/p-h3-image")
EXPECTED_COMFY_GIT_BLOBS = {
    "comfy_extras/nodes_minimax_h3.py": "b0abbc717ba6af7da5169f44467f44364468cda9",
    "comfy/text_encoders/minimax.py": "3c9e8f7e45fc0f7700cc7aa9996399914007d87e",
    "comfy/ldm/minimax/vae.py": "9faf459198d1da9afd0a90240556dd0d1d0ecd3e",
    "comfy/sd.py": "ca61c01f77664dd94918297fc9f0898217798686",
    "execution.py": "a8918ada17af25941827763e0184f4dab87c46f9",
}

# oid is the official git blob SHA; lfs is the immutable SHA-256 content id.
WEIGHTS = {
    "diffusion_models/p-h3-image_w4a8.safetensors": {
        "size": 13362068864, "oid": "242cf5b5e5a39d7e521897f51f4d9e20dc2d37e4",
        "lfs": "12ea30d6fde736dd69ef16658ce9438823a06e351ac3fa335cc4a3ce6bb17a43",
    },
    "text_encoders/p-h3-image_text_encoder_w4a8.safetensors": {
        "size": 3575536021, "oid": "514569771d858351696704ac956567727a9ef9f2",
        "lfs": "d4ae62d48e1de846dd55fc975623434e9326df3449601c4e15bea8b3ccd67293",
    },
    "vae/p-h3-image_vae_int8_convrot.safetensors": {
        "size": 3146996440, "oid": "c1d12a83f659547cc0d2d78d829e3e2660be722f",
        "lfs": "583e0d43354b6b833aa9a0815a5fac831b1ef6d0c19a7eaa0616980a3dab5ffe",
    },
}
SAMPLE_SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)
SIGMA_STRING = ", ".join(str(x) for x in SAMPLE_SIGMAS)
MAX_SEED = 2**63 - 1
MAX_PROMPT_CHARS = 16000
MAX_TOTAL_PIXELS = 16_777_216


@dataclass(frozen=True)
class PH3Request:
    prompt: str
    seed: int
    width: int
    height: int


def parse_request(params: dict) -> PH3Request:
    """No prompt upsampling, CFG, LoRA, alternate sigma grid or image edits."""
    allowed = {"prompt", "seed", "width", "height", "steps", "prompt_upsampling", "_job_type"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise InferenceError(f"p-h3-image-comfy unsupported params: {unknown}")
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        raise InferenceError("prompt must be a non-empty string of at most 16000 characters")
    if params.get("prompt_upsampling") not in (None, False):
        raise InferenceError("prompt upsampling is excluded from the comparable arm")
    for key in ("seed", "width", "height", "steps"):
        if isinstance(params.get(key), bool):
            raise InferenceError(f"{key} must be an integer")
    seed = params.get("seed", 42)
    width = params.get("width", 1024)
    height = params.get("height", 1024)
    steps = params.get("steps", 8)
    for name, value in (("seed", seed), ("width", width), ("height", height), ("steps", steps)):
        if not isinstance(value, int):
            raise InferenceError(f"{name} must be an integer")
    if not 0 <= seed <= MAX_SEED:
        raise InferenceError("seed out of range")
    for side in (width, height):
        if not 256 <= side <= 4096 or side % 32:
            raise InferenceError("width and height must be 256-4096 and multiples of 32")
    if width * height > MAX_TOTAL_PIXELS:
        raise InferenceError("width*height exceeds the 16 MP model-card limit")
    if steps != 8:
        raise InferenceError("P-H3 is pinned to its official 8-step sigma grid")
    return PH3Request(prompt, seed, width, height)


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    if algorithm == "sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_weight_file(root: Path, relpath: str) -> str:
    meta = WEIGHTS[relpath]
    path = root / relpath
    if path.is_symlink() or not path.is_file():
        raise LoadError(f"p-h3-image-comfy missing real weight file: {path}")
    if path.stat().st_size != meta["size"]:
        raise LoadError(f"p-h3-image-comfy size mismatch: {relpath}")
    actual = _digest(path, "sha256")
    if actual != meta["lfs"]:
        raise LoadError(f"p-h3-image-comfy content hash mismatch: {relpath}")
    return actual


def verify_weights(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise LoadError(f"p-h3-image-comfy models dir is not a real directory: {root}")
    verified = {name: verify_weight_file(root, name) for name in sorted(WEIGHTS)}
    return {
        "repo": PH3_REPO, "revision": PH3_REVISION, "models_dir": str(root),
        "weight_bytes": sum(m["size"] for m in WEIGHTS.values()),
        "verified_files": verified, "precision": {"transformer": "w4a8", "text_encoder": "w4a8", "vae": "int8_convrot"},
    }


def verify_comfy_source(root: Path) -> dict:
    """Require a real checkout of the exact reviewed commit and PH3 file IDs."""
    if root.is_symlink() or not root.is_dir():
        raise LoadError(f"p-h3-image-comfy runtime dir is not a real directory: {root}")
    try:
        head = (root / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            revision = (root / ".git" / head[5:]).read_text().strip()
        else:
            revision = head
    except OSError as exc:
        raise LoadError(f"p-h3-image-comfy runtime has no readable immutable checkout: {exc}") from exc
    if revision != COMFY_REVISION:
        raise LoadError(f"p-h3-image-comfy requires ComfyUI {COMFY_REVISION}, got {revision}")
    for rel, expected in EXPECTED_COMFY_GIT_BLOBS.items():
        path = root / rel
        if path.is_symlink() or not path.is_file():
            raise LoadError(f"p-h3-image-comfy missing runtime source: {path}")
        if _digest(path, "sha1") != expected:
            raise LoadError(f"p-h3-image-comfy runtime source changed: {rel}")
    return {"repo": COMFY_REPO, "revision": revision, "verified_source_files": sorted(EXPECTED_COMFY_GIT_BLOBS)}


def build_graph(request: PH3Request, output_dir: Path) -> dict:
    """Seven-node prompt-as-written graph distilled from Pruna's official
    workflow id 9d43b9c5-ca8f-4a60-837d-d342773e8b8a, with its optional
    upsampler removed. The two audio latents required by PH3 conditioning
    are retained even for still images."""
    prefix = f"arbiter-ph3-{request.seed}-{time.time_ns()}"
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "p-h3-image_w4a8.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "p-h3-image_text_encoder_w4a8.safetensors", "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "p-h3-image_vae_int8_convrot.safetensors"}},
        "4": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": 1.0, "shift_audio": 1.0}},
        "5": {"class_type": "PH3ImageConditioning", "inputs": {"clip": ["2", 0], "prompt": request.prompt, "width": request.width, "height": request.height}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": request.seed}},
        "7": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMA_STRING}},
        "8": {"class_type": "BasicGuider", "inputs": {"model": ["4", 0], "conditioning": ["5", 0]}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0], "guider": ["8", 0], "sampler": ["9", 0], "sigmas": ["7", 0], "latent_image": ["5", 1]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
    }


class _NullServer:
    client_id = None
    last_node_id = None
    last_prompt_id = None

    def send_sync(self, event, data, sid=None):
        log.info("p-h3-image-comfy event %s node=%s", event, data.get("node"))


@register
class PH3ImageComfyAdapter(ModelAdapter):
    model_id = "p-h3-image-comfy-w4a8"

    def __init__(self, models_dir: Path = DEFAULT_MODELS_DIR, runtime_dir: Path | None = None,
                 executor_factory: Callable | None = None) -> None:
        self.models_dir = models_dir
        self.runtime_dir = runtime_dir
        self._executor_factory = executor_factory
        self._executor = None
        self._comfy = None
        self._provenance: dict | None = None
        self._load_seconds = 0.0
        self._renders = 0
        self._lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise LoadError(f"{self.model_id} requires cuda")
        if not os.environ.get("ARBITER_MEMORY_GB"):
            raise LoadError(f"{self.model_id} refuses to run outside Arbiter's CUDA-capped worker")
        started = time.perf_counter()
        weights = verify_weights(self.models_dir)
        runtime = verify_comfy_source(self.runtime_dir) if self.runtime_dir else {"repo": COMFY_REPO, "revision": COMFY_REVISION, "verification": "installed worker identity must be pinned"}
        self._provenance = {"weights": weights, "runtime": runtime}
        if self._executor_factory is None:
            # No HTTP listener, no Comfy main.py server and no direct GPU
            # execution outside Arbiter. This is main.py's in-process boundary.
            import execution
            self._executor_factory = execution.PromptExecutor
        self._executor = self._executor_factory(_NullServer())
        self._executor.output_dir = self.models_dir / "comfy-output"
        self._load_seconds = time.perf_counter() - started
        self._renders = 0

    def unload(self) -> None:
        self._executor = None
        self._provenance = None
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        if params.get("_job_type") != "p-h3-image-comfy":
            raise InferenceError("p-h3-image-comfy-w4a8 serves only p-h3-image-comfy")
        request = parse_request(params)
        with self._lock:
            if self._executor is None:
                raise InferenceError(f"{self.model_id} is not loaded")
            graph = build_graph(request, output_dir)
            torch = __import__("torch")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            self._check_cancel(cancel_flag)
            self._executor.execute(graph, f"arbiter-{time.time_ns()}", {}, ["12"])
            if cancel_flag.is_set():
                import nodes
                nodes.interrupt_processing()
                raise CancelledException(f"Job cancelled for model {self.model_id}")
            if not self._executor.success:
                raise InferenceError(f"Comfy execution failed: {self._executor.status_messages}")
            torch.cuda.synchronize()
            generated = time.perf_counter()
            prefix = graph["12"]["inputs"]["filename_prefix"]
            candidates = sorted(self._executor.output_dir.glob(f"{prefix}*.png"))
            if len(candidates) != 1:
                raise InferenceError(f"expected one Comfy PNG for {prefix} in {output_dir}, got {candidates}")
            shutil.copyfile(candidates[0], output_dir / "result.png")
            self._renders += 1
            return self._result(request, graph, started, generated, time.perf_counter(), torch)

    def _result(self, request, graph, started, generated, written, torch) -> dict:
        gib = 1024**3
        return {
            "format": "png", "file": "result.png", "mode": "t2i", "model": PH3_REPO,
            "model_id": self.model_id, "prompt": request.prompt, "seed": request.seed,
            "width": request.width, "height": request.height, "steps": 8,
            "sample_sigmas": list(SAMPLE_SIGMAS), "prompt_upsampling": False,
            "comparison": {"like_for_like_house_1mp": request.width == request.height == 1024,
                           "native_8mp": (request.width, request.height) == (2912, 2912)},
            "provenance": self._provenance,
            "timings": {"render_index": self._renders, "generate_seconds": round(generated-started, 3),
                        "png_write_seconds": round(written-generated, 3), "load_seconds": round(self._load_seconds, 3)},
            "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                       "peak_allocated_gb": round(torch.cuda.max_memory_allocated()/gib, 3),
                       "peak_reserved_gb": round(torch.cuda.max_memory_reserved()/gib, 3)},
        }

    def estimate_time(self, params: dict) -> float:
        r = parse_request(params)
        return 120.0 * r.width * r.height / (1024 * 1024)
