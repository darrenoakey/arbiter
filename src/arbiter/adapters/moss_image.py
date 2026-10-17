"""MOSS-Image-5B (OpenMOSS MOSS-Image-SFT) — owner-requested comparison adapter.

Owner request, 2026-10-17: the owner pointed at the newly released
``OpenMOSS-Team/MOSS-Image-SFT`` (~5B single-stream DiT + Qwen3.5 hybrid text
encoder, Apache 2.0) and asked for an Arbiter adapter to test it against the
other image models (Qwen-Image-2.1 family, FLUX 3 Image Klein, Iris-3B,
Ming-Image). This adapter exists only for that comparison. It is **not** wired
into ``generate_image`` / ``daz-agent-sdk`` / the Mac mini Codex IGS route and
is never a fallback when IGS is unavailable.

Scope, enforced in ``image_policy.py`` and the Go ``model_policy.go``: the
adapter serves ONLY the ``moss-image`` job type under the ``moss-image-sft``
model id. Every other ``moss-image-*`` / ``mossimage`` id (the Base checkpoint,
quantized community builds, LoRA variants, renamed copies) stays denied.

Runtime contract
----------------
* Weights are a ``snapshot_download`` local dir on Spark NVMe
  (``DEFAULT_WEIGHTS_DIR``) pinned to ``MOSS_REVISION``. Every file the
  pipeline reads — the remote-code ``pipeline_moss_image.py`` (pipeline,
  transformer and scheduler classes), ``model_index.json`` and every file
  under ``scheduler/``, ``transformer/``, ``text_encoder/``, ``tokenizer/``
  and ``vae/`` — is verified on load against the Hugging Face
  snapshot metadata (revision + content hash), the same fail-closed proof as
  the FLUX 3 Klein adapters. The model ships its own pipeline code
  (``custom_pipeline="pipeline_moss_image"`` + ``trust_remote_code``); the
  hash pin is what makes that code reviewed and immutable here.
* Sampling follows the model card exactly: 50 steps, CFG ``guidance_scale``
  3.0 with negative-branch skip-layer guidance ``uncond_skip_layers=[22]``
  (the card default; disabling it collapses detail per the user guide),
  transformer/text encoder bfloat16, VAE and initial noise float32, Euler
  flow-match scheduler with shift 5.0, CPU generator for cross-platform
  determinism. Text-to-image only: image/reference params fail.
* Load is fail-closed: it refuses to run without the Arbiter worker CUDA cap
  (``ARBITER_MEMORY_GB``), verifies every weight and code file against the
  pinned snapshot, and after loading checks the transformer config matches the
  MOSS-Image-5B architecture (dim 3072, 42 layers) and the scheduler/pipeline
  classes are the MOSS remote-code classes.
* Cancellation: checked before text encoding and before every transformer
  forward pass via a forward pre-hook (CFG runs cond + uncond forwards per
  step, so a cancel stops within one partial step).
* Results carry measured timings, peak allocated/reserved CUDA memory, and
  prompt/seed/revision provenance, matching the flux3-klein result shape so
  the comparison sheets stay uniform.

``torch`` and ``diffusers`` are imported only inside ``load``/``infer`` so the
main-venv registry smoke test stays clean. See ``requirements/moss-image.txt``
and ``docs/moss-image.md``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from arbiter.adapters.base import (
    CancelledException,
    HeapTrimGuard,
    InferenceError,
    LoadError,
    ModelAdapter,
)
from arbiter.adapters.flux3_klein import write_png_durably, verify_snapshot_file
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

MOSS_HF_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
MOSS_REVISION = "fc5de0d56fc910cbca6f6625a70f5f582b26deb1"
# Local snapshot on Spark NVMe (root disk had 3.5 GB free before the 2026-10-17
# cleanup; T9 is full — same placement rule as flux3 / qwen-image-turbo).
DEFAULT_WEIGHTS_DIR = Path("/home/darren/nvme-models/moss-image/MOSS-Image-SFT")

DEFAULT_STEPS = 50  # model card: best results at >= 50
MAX_STEPS = 100
DEFAULT_GUIDANCE = 3.0  # model card default; recommended range 3.0-6.0
MAX_GUIDANCE = 10.0
DEFAULT_SKIP_LAYERS = (22,)  # card/user-guide recommended uncond skip layer
DEFAULT_SIDE = 1024
MIN_SIDE = 256
MAX_SIDE = 2048  # 4M total pixels cap (guide: 0.25M-4M) ⇒ 2048 max side
SIDE_MULTIPLE = 16
MAX_TOTAL_PIXELS = 4 * 1024 * 1024
MAX_SEED = 2**63 - 1
MAX_PROMPT_CHARS = 16000

#: Architecture pins — a swapped or re-exported checkpoint can never silently
#: load as moss-image-sft.
EXPECTED_DIM = 3072
EXPECTED_LAYERS = 42
EXPECTED_PIPELINE_CLASS = "MOSSImagePipeline"
EXPECTED_SCHEDULER_MODULE = "pipeline_moss_image"

#: Top-level paths the pipeline reads; every regular file under them (plus
#: model_index.json) is hash-verified against the snapshot metadata on load.
_REQUIRED_PATHS = (
    "model_index.json",
    "pipeline_moss_image.py",
    "scheduler",
    "transformer",
    "text_encoder",
    "tokenizer",
    "vae",
)

#: Params the pipeline could honour but this comparison adapter must never
#: accept silently. CFG ambiguity (cfg_scale vs guidance_scale), image
#: conditioning, multi-image batches, and the second SLG variant all fail.
_UNSUPPORTED_PARAMS = (
    "image",
    "image_file",
    "images",
    "image_files",
    "references",
    "true_cfg_scale",
    "cfg_scale",
    "num_images_per_prompt",
    "slg_layers",
    "slg_scale",
    "output_resolution",
    "model_kind",
    "offload",
)


@dataclass(frozen=True)
class MossRequest:
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    steps: int
    guidance_scale: float
    uncond_skip_layers: tuple[int, ...]


def verify_moss_snapshot(entry: Path, revision: str = MOSS_REVISION) -> dict:
    """Fail-closed verification of everything the pipeline will read.
    Returns the weights provenance recorded in every result."""
    if entry.is_symlink() or not entry.is_dir():
        raise LoadError(f"moss-image weights dir is not a real directory: {entry}")
    verified: dict[str, str] = {}
    for rel in _REQUIRED_PATHS:
        target = entry / rel
        if not target.exists():
            raise LoadError(f"moss-image snapshot is missing required path: {target}")
        if target.is_file():
            verified[rel] = verify_snapshot_file(entry, rel, revision)
        else:
            for child in sorted(p for p in target.rglob("*") if p.is_file()):
                if ".cache" in child.parts:
                    continue
                verified[str(child.relative_to(entry))] = verify_snapshot_file(
                    entry, str(child.relative_to(entry)), revision
                )
    return {
        "hf_repo": MOSS_HF_REPO,
        "revision": revision,
        "weights_dir": str(entry),
        "verified_files": len(verified),
    }


def parse_request(params: dict) -> MossRequest:
    """Validate a job's params. Unsupported knobs fail instead of being
    silently ignored, so a benchmark row can never be mislabelled."""
    for key in _UNSUPPORTED_PARAMS:
        if params.get(key) not in (None, False):
            raise InferenceError(f"moss-image does not support {key!r} (t2i only, CFG via guidance_scale)")
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InferenceError("prompt is required")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise InferenceError(f"prompt exceeds {MAX_PROMPT_CHARS} characters")
    negative_prompt = params.get("negative_prompt") or ""

    def integer(key: str, default: int, low: int, high: int) -> int:
        value = params.get(key, default)
        if value is None:
            value = default
        if isinstance(value, bool) or not isinstance(value, int):
            raise InferenceError(f"{key} must be an integer")
        if not low <= value <= high:
            raise InferenceError(f"{key} must be between {low} and {high}")
        return value

    width = integer("width", DEFAULT_SIDE, MIN_SIDE, MAX_SIDE)
    height = integer("height", DEFAULT_SIDE, MIN_SIDE, MAX_SIDE)
    if width % SIDE_MULTIPLE or height % SIDE_MULTIPLE:
        raise InferenceError(f"width and height must be multiples of {SIDE_MULTIPLE}")
    if width * height > MAX_TOTAL_PIXELS:
        raise InferenceError(
            f"width*height must be <= {MAX_TOTAL_PIXELS} pixels (got {width * height})"
        )
    steps = integer("steps", DEFAULT_STEPS, 1, MAX_STEPS)

    guidance = params.get("guidance_scale", DEFAULT_GUIDANCE)
    if guidance is None:
        guidance = DEFAULT_GUIDANCE
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)):
        raise InferenceError("guidance_scale must be a number")
    guidance = float(guidance)
    if not 0.0 <= guidance <= MAX_GUIDANCE:
        raise InferenceError(f"guidance_scale must be between 0 and {MAX_GUIDANCE}")

    raw_skip = params.get("uncond_skip_layers")
    if raw_skip is None:
        skip_layers: tuple[int, ...] = DEFAULT_SKIP_LAYERS
    else:
        if not isinstance(raw_skip, list):
            raise InferenceError("uncond_skip_layers must be a JSON list of layer indices")
        skip_layers = tuple(_validate_layer(value, 0, EXPECTED_LAYERS - 1) for value in raw_skip)
    return MossRequest(
        prompt=prompt,
        negative_prompt=str(negative_prompt),
        seed=integer("seed", 42, 0, MAX_SEED),
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance,
        uncond_skip_layers=skip_layers,
    )


def _validate_layer(value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InferenceError("uncond_skip_layers entries must be integers")
    if not low <= value <= high:
        raise InferenceError(f"uncond_skip_layers entries must be between {low} and {high}")
    return value


@register
class MossImageSftAdapter(ModelAdapter):
    """Serves only the ``moss-image-sft`` model id via the ``moss-image``
    job type (owner comparison request, 2026-10-17)."""

    model_id = "moss-image-sft"

    def __init__(self) -> None:
        self._pipe = None
        self._load_info: dict | None = None
        self._renders = 0
        self._lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise LoadError(f"{self.model_id} requires cuda")
        declared = os.environ.get("ARBITER_MEMORY_GB")
        if not declared:
            raise LoadError(
                f"{self.model_id} refuses to load without the Arbiter worker CUDA cap "
                "(ARBITER_MEMORY_GB); GPU work runs only through the Arbiter queue"
            )
        started = time.perf_counter()
        weights = verify_moss_snapshot(DEFAULT_WEIGHTS_DIR)
        verified = time.perf_counter()

        import torch
        from diffusers import DiffusionPipeline

        with HeapTrimGuard():
            pipe = DiffusionPipeline.from_pretrained(
                str(DEFAULT_WEIGHTS_DIR),
                custom_pipeline="pipeline_moss_image",
                trust_remote_code=True,
                torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
            )
            # GB10 mmap->cuda workaround: clone each tensor off mmap-backed
            # storage before moving to device. See base._pipe_to_cuda_cloned.
            self._pipe_to_cuda_cloned(pipe, device)
        loaded = time.perf_counter()

        transformer_cfg = pipe.transformer.config
        if (
            getattr(transformer_cfg, "dim", None) != EXPECTED_DIM
            or getattr(transformer_cfg, "num_layers", None) != EXPECTED_LAYERS
        ):
            self._pipe = None
            self._cleanup_gpu()
            raise LoadError(
                f"{self.model_id} expected a {EXPECTED_LAYERS}-layer dim-{EXPECTED_DIM} "
                f"MOSS transformer, got {getattr(transformer_cfg, 'num_layers', '?')} layers "
                f"dim {getattr(transformer_cfg, 'dim', '?')}"
            )
        if type(pipe).__name__ != EXPECTED_PIPELINE_CLASS or EXPECTED_SCHEDULER_MODULE not in str(
            type(pipe.scheduler).__module__
        ):
            self._pipe = None
            self._cleanup_gpu()
            raise LoadError(
                f"{self.model_id} loaded unexpected pipeline/scheduler classes: "
                f"{type(pipe).__name__}/{type(pipe.scheduler).__name__}"
            )

        runtime = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(),
        }
        self._pipe = pipe
        self._renders = 0
        self._load_info = {
            "weights": weights,
            "runtime": runtime,
            "declared_memory_gb": float(declared),
            "verify_seconds": round(verified - started, 3),
            "pipeline_load_seconds": round(loaded - verified, 3),
            "load_seconds": round(loaded - started, 3),
            "resident_allocated_bytes": torch.cuda.memory_allocated(),
            "resident_reserved_bytes": torch.cuda.memory_reserved(),
        }
        log.info("%s loaded: %s", self.model_id, self._load_info)

    def unload(self) -> None:
        self._pipe = None
        self._load_info = None
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        job_type = params.get("_job_type") or self.model_id
        if job_type != "moss-image":
            raise InferenceError(f"{self.model_id} serves only the 'moss-image' job type, not {job_type!r}")
        request = parse_request(params)
        self._check_cancel(cancel_flag)
        with self._lock:
            if self._pipe is None or self._load_info is None:
                raise InferenceError(f"{self.model_id} is not loaded")
            return self._render(request, output_dir, cancel_flag)

    def _render(self, request: MossRequest, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import torch

        pipe = self._pipe
        transformer = pipe.transformer
        forwards = 0

        def cancel_between_forwards(_module, _args, _kwargs):
            nonlocal forwards
            if cancel_flag.is_set():
                raise CancelledException(f"Job cancelled for model {self.model_id}")
            forwards += 1

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        handle = transformer.register_forward_pre_hook(cancel_between_forwards, with_kwargs=True)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                image = pipe(
                    request.prompt,
                    negative_prompt=request.negative_prompt,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=request.steps,
                    guidance_scale=request.guidance_scale,
                    uncond_skip_layers=list(request.uncond_skip_layers),
                    generator=torch.Generator("cpu").manual_seed(request.seed),
                ).images[0]
            torch.cuda.synchronize()
        finally:
            handle.remove()
        generated = time.perf_counter()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        self._check_cancel(cancel_flag)

        write_png_durably(output_dir / "result.png", image)
        written = time.perf_counter()
        self._renders += 1
        info = self._load_info
        gib = float(1024**3)
        return {
            "format": "png",
            "file": "result.png",
            "mode": "t2i",
            "model": MOSS_HF_REPO,
            "model_id": self.model_id,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "seed": request.seed,
            "width": image.width,
            "height": image.height,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "uncond_skip_layers": list(request.uncond_skip_layers),
            "precision": {
                "transformer": "bfloat16",
                "text_encoder": "bfloat16",
                "vae": "float32",
                "initial_noise": "float32",
            },
            "provenance": {**info["weights"], **info["runtime"]},
            "timings": {
                "render_index": self._renders,
                "generate_seconds": round(generated - started, 3),
                "png_write_seconds": round(written - generated, 3),
                "total_seconds": round(written - started, 3),
                "transformer_forwards": forwards,
                "load_seconds": info["load_seconds"],
                "verify_seconds": info["verify_seconds"],
                "pipeline_load_seconds": info["pipeline_load_seconds"],
            },
            "memory": {
                "declared_memory_gb": info["declared_memory_gb"],
                "resident_allocated_bytes": info["resident_allocated_bytes"],
                "resident_reserved_bytes": info["resident_reserved_bytes"],
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "peak_allocated_gb": round(peak_allocated / gib, 3),
                "peak_reserved_gb": round(peak_reserved / gib, 3),
            },
        }

    def estimate_time(self, params: dict) -> float:
        steps = params.get("steps", DEFAULT_STEPS)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            steps = DEFAULT_STEPS
        # ~5B DiT, two forwards per step under CFG; load time is separate.
        return 12000.0 + 300.0 * steps
