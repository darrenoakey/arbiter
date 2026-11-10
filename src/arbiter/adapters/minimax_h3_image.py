"""MiniMaxAI/MiniMax-H3-Image (2026-11-10 still image) — P-H3 comparison adapter.

This is the *new* MiniMax-H3-Image checkpoint, a still-image model. It is NOT
the older MiniMax-H3 video model served by ``minimax_h3_local`` /
``minimax_fast_h3``, and it never reuses that model's weights: the snapshot
layout verified here (``transformer_diffusers/``, ``vae_diffusers/``,
``modular_model_index.json``) exists only in the new repo, and an NVFP4 /
quantized label anywhere in the transformer config fails the load.

Verified runtime API (source audit 2026-11-10, see docs/minimax-h3-image.md):
the official in-process runtime is Diffusers' modular pipeline at pinned
commit ``f598e6e6`` — ``MiniMaxH3ImageModularPipeline`` /
``MiniMaxH3ImageBlocks`` over ``transformer_diffusers`` + ``vae_diffusers``.
The checkpoint is distilled twice over: guidance is baked in (no
``negative_prompt``, no CFG — ``guidance_scale`` is not a pipeline input at
all), and sampling runs a fixed grid of six sigmas from
``modular_model_index.json`` (no ``num_inference_steps``). Usage follows the
class docstring: ``ModularPipeline.from_pretrained(...)`` then
``load_components(dtype=torch.bfloat16)``, called with ``prompt``,
``height``, ``width``, ``min_generation_area``, ``generator``,
``output_type="pil"``; the blocks declare a single ``images`` output.

The 4 MP trap, confirmed in the pinned runtime source
(``before_denoise.MiniMaxH3ImagePrepareCanvasStep`` /
``decoders.MiniMaxH3ImageDecodeStep``): ``min_generation_area`` defaults to
4194304, so a 1024x1024 request would be *rendered* on a 2048x2048 canvas and
resized back to 1024x1024 with LANCZOS by the image processor — a benchmark
row labelled "native 1024" would not be native. This adapter therefore sets
``min_generation_area=1024`` (the runtime's documented off switch) for the
controlled lane, records requested / rendered / delivered sizes separately,
and fails if the delivered image is not exactly the requested size.

Lanes (the only sizes this adapter serves, so rows can never be mislabelled):

* ``controlled-1024``: 1024x1024 requested, ``min_generation_area=1024``,
  rendered 1024x1024, delivered 1024x1024.
* ``recommended-4096``: 4096x4096 requested, ``min_generation_area=4194304``
  (the runtime default; at 16 MP it scales nothing), rendered and delivered
  4096x4096.

Licence gate: the MiniMax-H3-Image licence requires separate prior *written*
authorization from MiniMax if the annualized revenue of the user and/or its
affiliates exceeds USD 1M, or if the model is provided as a Hosted Service to
third parties. Coverage for this deployment is unverified, so every job must
carry an explicit local eligibility determination: ``license_confirmed``
exactly ``True`` plus a non-empty ``license_confirmation_ref`` identifying
the determination/authorization relied on. Anything else fails closed,
before any rendering. Third-party hosted-service flags are rejected outright.

Fail-closed load: refuses without the Arbiter worker CUDA cap
(``ARBITER_MEMORY_GB``), verifies every file the pipeline reads against the
pinned HF snapshot metadata at ``MINIMAX_H3_IMAGE_REVISION``, requires the
installed ``diffusers`` to be the pinned git commit, and checks the loaded
pipeline/transformer/VAE/scheduler class names and the fixed sigma grid.

No authoritative memory-capable GB10 route is proven for this ~135 GB bf16
snapshot; the adapter does not invent one — it loads bf16 in-process through
the Arbiter queue's declared cap and fails if it does not fit.

``torch``/``diffusers`` are imported only inside ``load``/``infer`` so module
import and all validation run without a GPU or the model runtime installed.
"""

from __future__ import annotations

import json
import logging
import math
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
from arbiter.adapters.flux3_klein import verify_snapshot_file, write_png_durably
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

MINIMAX_H3_IMAGE_HF_REPO = "MiniMaxAI/MiniMax-H3-Image"
MINIMAX_H3_IMAGE_REVISION = "492fd8c4eca666f1085f57e30738c1e0d86b3439"
DIFFUSERS_REPO = "https://github.com/huggingface/diffusers"
DIFFUSERS_COMMIT = "f598e6e6a99528858b962c1399c45b59a6c25484"
DIFFUSERS_VERSION_PREFIX = "0.42.0"

DEFAULT_WEIGHTS_DIR = Path("/home/darren/nvme-models/minimax-h3-image/MiniMax-H3-Image")

#: The fixed sampling grid declared by modular_model_index.json at the pin:
#: six denoising steps (seven sigma values including the terminal 0.0).
EXPECTED_SAMPLE_SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)
EXPECTED_STEPS = len(EXPECTED_SAMPLE_SIGMAS) - 1
EXPECTED_PIPELINE_CLASS = "MiniMaxH3ImageModularPipeline"
EXPECTED_TRANSFORMER_CLASS = "MiniMaxH3ImageTransformer2DModel"
EXPECTED_VAE_CLASS = "AutoencoderKLMiniMaxH3Image"
EXPECTED_SCHEDULER_CLASS = "MiniMaxH3Scheduler"

CANVAS_MULTIPLE = 32
DISABLE_MIN_GENERATION_AREA = 1024  # runtime-documented off switch
DEFAULT_MIN_GENERATION_AREA = 2048 * 2048  # 4194304, the runtime default
MAX_SEED = 2**63 - 1
MAX_PROMPT_CHARS = 16000

#: lane name -> (width, height, min_generation_area)
LANES: dict[str, tuple[int, int, int]] = {
    "controlled-1024": (1024, 1024, DISABLE_MIN_GENERATION_AREA),
    "recommended-4096": (4096, 4096, DEFAULT_MIN_GENERATION_AREA),
}

_REQUIRED_PATHS = (
    "LICENSE",
    "model_index.json",
    "modular_model_index.json",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "processor",
    "transformer_diffusers",
    "vae_diffusers",
)

#: Params this benchmark adapter must never accept silently: prompt
#: enhancement / API / cloud routes, image conditioning or editing, CFG and
#: negative prompts (unsupported by the distilled checkpoint), caller control
#: of the sampling grid or the canvas trap, and third-party hosted service.
_UNSUPPORTED_PARAMS = (
    "image",
    "image_file",
    "images",
    "image_files",
    "references",
    "negative_prompt",
    "true_cfg_scale",
    "cfg_scale",
    "num_inference_steps",
    "steps",
    "sigmas",
    "sample_sigmas",
    "min_generation_area",
    "prompt_enhancement",
    "prompt_enhancer",
    "enhance_prompt",
    "revised_prompt",
    "api",
    "api_key",
    "cloud",
    "hosted_service",
    "third_party_access",
    "num_images_per_prompt",
    "output_resolution",
    "model_kind",
    "offload",
)


@dataclass(frozen=True)
class MiniMaxH3ImageRequest:
    prompt: str
    seed: int
    lane: str
    width: int
    height: int
    min_generation_area: int
    license_confirmation_ref: str


def expected_canvas_size(height: int, width: int, min_generation_area: int) -> tuple[int, int]:
    """The canvas the pinned runtime renders on (mirrors
    ``before_denoise.get_image_canvas_size``): below ``min_generation_area``
    both sides scale by ``sqrt(min_area / area)``; either way sides round to
    the nearest multiple of 32."""
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def require_license_gate(params: dict) -> str:
    """Explicit local licence-eligibility gate. Returns the confirmation
    reference recorded in the result; raises otherwise. See module docstring:
    written MiniMax authorization is required above USD 1M annualized revenue
    or for third-party Hosted Service, and coverage here is unverified."""
    if params.get("license_confirmed") is not True:
        raise InferenceError(
            "minimax-h3-image licence gate: 'license_confirmed' must be exactly true, "
            "backed by a local eligibility determination (written MiniMax authorization "
            "is required above USD 1M annualized revenue or for third-party Hosted "
            "Service; coverage is otherwise unverified)"
        )
    ref = params.get("license_confirmation_ref")
    if not isinstance(ref, str) or not 8 <= len(ref.strip()) <= 200:
        raise InferenceError(
            "minimax-h3-image licence gate: 'license_confirmation_ref' must be an "
            "8-200 character reference to the eligibility determination or written "
            "authorization relied on"
        )
    return ref.strip()


def parse_request(params: dict) -> MiniMaxH3ImageRequest:
    """Validate a job's params. Unsupported knobs fail instead of being
    silently ignored, so a benchmark row can never be mislabelled."""
    for key in _UNSUPPORTED_PARAMS:
        if params.get(key) not in (None, False, ""):
            raise InferenceError(
                f"minimax-h3-image does not support {key!r} (t2i only, distilled: "
                "no CFG / negative prompt / prompt enhancement / API or cloud route)"
            )
    guidance = params.get("guidance_scale")
    if guidance is not None and guidance not in (1, 1.0):
        raise InferenceError(
            "minimax-h3-image is guidance-distilled: guidance_scale must be omitted or exactly 1"
        )
    license_ref = require_license_gate(params)
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InferenceError("prompt is required")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise InferenceError(f"prompt exceeds {MAX_PROMPT_CHARS} characters")
    seed = params.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise InferenceError(f"seed must be an integer between 0 and {MAX_SEED}")
    lane = params.get("lane", "controlled-1024")
    if lane not in LANES:
        raise InferenceError(f"lane must be one of {sorted(LANES)}, got {lane!r}")
    width, height, min_area = LANES[lane]
    for key, expected in (("width", width), ("height", height)):
        value = params.get(key)
        if value is not None and value != expected:
            raise InferenceError(
                f"lane {lane!r} is fixed at {width}x{height}; {key}={value!r} would mislabel the row"
            )
    return MiniMaxH3ImageRequest(
        prompt=prompt,
        seed=seed,
        lane=lane,
        width=width,
        height=height,
        min_generation_area=min_area,
        license_confirmation_ref=license_ref,
    )


def _read_json(entry: Path, relpath: str) -> dict:
    try:
        data = json.loads((entry / relpath).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LoadError(f"minimax-h3-image cannot read {relpath}: {error}") from error
    if not isinstance(data, dict):
        raise LoadError(f"minimax-h3-image {relpath} is not a JSON object")
    return data


def verify_h3_image_snapshot(entry: Path, revision: str = MINIMAX_H3_IMAGE_REVISION) -> dict:
    """Fail-closed verification of everything the pipeline will read.
    Returns the weights provenance recorded in every result."""
    if entry.is_symlink() or not entry.is_dir():
        raise LoadError(f"minimax-h3-image weights dir is not a real directory: {entry}")
    modular = _read_json(entry, "modular_model_index.json")
    if modular.get("_class_name") != EXPECTED_PIPELINE_CLASS:
        raise LoadError(
            f"minimax-h3-image modular_model_index declares {modular.get('_class_name')!r}, "
            f"expected {EXPECTED_PIPELINE_CLASS!r} (the old H3 video snapshot has no such index)"
        )
    sigmas = modular.get("sample_sigmas")
    if not isinstance(sigmas, list) or tuple(float(s) for s in sigmas) != EXPECTED_SAMPLE_SIGMAS:
        raise LoadError(f"minimax-h3-image sample_sigmas {sigmas!r} differ from the pinned six-step grid")
    for component, subfolder in (("transformer", "transformer_diffusers"), ("vae", "vae_diffusers")):
        declared = modular.get(component)
        if not isinstance(declared, list) or len(declared) < 3 or (declared[2] or {}).get("subfolder") != subfolder:
            raise LoadError(f"minimax-h3-image modular index does not pin {component} to {subfolder}/")
    transformer_cfg = _read_json(entry, "transformer_diffusers/config.json")
    quantization = json.dumps(transformer_cfg.get("quantization_config") or transformer_cfg.get("quantization") or "")
    if "nvfp4" in quantization.lower() or "fp4" in quantization.lower():
        raise LoadError(
            "minimax-h3-image transformer declares FP4/NVFP4 quantization; refusing to "
            "reuse or relabel quantized weights as the bf16 benchmark snapshot"
        )
    verified: dict[str, str] = {}
    for rel in _REQUIRED_PATHS:
        target = entry / rel
        if not target.exists():
            raise LoadError(f"minimax-h3-image snapshot is missing required path: {target}")
        if target.is_file():
            verified[rel] = verify_snapshot_file(entry, rel, revision)
        else:
            for child in sorted(p for p in target.rglob("*") if p.is_file()):
                if ".cache" in child.parts:
                    continue
                relpath = str(child.relative_to(entry))
                verified[relpath] = verify_snapshot_file(entry, relpath, revision)
    return {
        "hf_repo": MINIMAX_H3_IMAGE_HF_REPO,
        "revision": revision,
        "weights_dir": str(entry),
        "verified_files": len(verified),
    }


def verify_diffusers_runtime() -> dict:
    """The installed diffusers must be the pinned commit that ships
    ``MiniMaxH3ImageModularPipeline`` — a PyPI or other-commit install fails
    closed instead of silently running an unreviewed runtime."""
    from importlib import metadata

    try:
        dist = metadata.distribution("diffusers")
    except metadata.PackageNotFoundError as error:
        raise LoadError("diffusers is not installed in this worker venv") from error
    if not dist.version.startswith(DIFFUSERS_VERSION_PREFIX):
        raise LoadError(f"diffusers {dist.version} does not provide the pinned MiniMax-H3-Image runtime")
    raw = dist.read_text("direct_url.json")
    if not raw:
        raise LoadError("diffusers has no direct_url.json; install it from the pinned git commit")
    try:
        direct = json.loads(raw)
    except ValueError as error:
        raise LoadError("diffusers direct_url.json is malformed") from error
    commit = (direct.get("vcs_info") or {}).get("commit_id")
    if commit != DIFFUSERS_COMMIT:
        raise LoadError(
            f"diffusers was installed from {direct.get('url')} at {commit!r}; expected commit {DIFFUSERS_COMMIT}"
        )
    return {"source_repo": DIFFUSERS_REPO, "source_commit": commit, "package_version": dist.version}


@register
class MiniMaxH3ImageAdapter(ModelAdapter):
    """Serves only the ``minimax-h3-image`` model id. Registration with the
    shared registry/schemas is the parent task's wiring step; this module is
    import-safe without torch, diffusers, weights, or a GPU."""

    model_id = "minimax-h3-image"

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
        weights = verify_h3_image_snapshot(DEFAULT_WEIGHTS_DIR)
        runtime_pin = verify_diffusers_runtime()
        verified = time.perf_counter()

        import torch
        from diffusers import ModularPipeline

        with HeapTrimGuard():
            pipe = ModularPipeline.from_pretrained(
                str(DEFAULT_WEIGHTS_DIR),
                trust_remote_code=False,
            )
            pipe.load_components(dtype=torch.bfloat16)
            pipe.to(device)
        loaded = time.perf_counter()

        problems = []
        if type(pipe).__name__ != EXPECTED_PIPELINE_CLASS:
            problems.append(f"pipeline {type(pipe).__name__}")
        for attr, expected in (
            ("transformer", EXPECTED_TRANSFORMER_CLASS),
            ("vae", EXPECTED_VAE_CLASS),
            ("scheduler", EXPECTED_SCHEDULER_CLASS),
        ):
            actual = type(getattr(pipe, attr, None)).__name__
            if actual != expected:
                problems.append(f"{attr} {actual}")
        sigmas = getattr(pipe.config, "sample_sigmas", None)
        if not isinstance(sigmas, (list, tuple)) or tuple(float(s) for s in sigmas) != EXPECTED_SAMPLE_SIGMAS:
            problems.append(f"config.sample_sigmas {sigmas!r}")
        if problems:
            self._pipe = None
            self._cleanup_gpu()
            raise LoadError(f"{self.model_id} loaded unexpected runtime components: {', '.join(problems)}")

        runtime = {
            **runtime_pin,
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
        if job_type != self.model_id:
            raise InferenceError(f"{self.model_id} serves only the {self.model_id!r} job type, not {job_type!r}")
        request = parse_request(params)
        self._check_cancel(cancel_flag)
        with self._lock:
            if self._pipe is None or self._load_info is None:
                raise InferenceError(f"{self.model_id} is not loaded")
            return self._render(request, output_dir, cancel_flag)

    def _render(self, request: MiniMaxH3ImageRequest, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import torch

        pipe = self._pipe
        forwards = 0

        def cancel_between_forwards(_module, _args, _kwargs):
            nonlocal forwards
            if cancel_flag.is_set():
                raise CancelledException(f"Job cancelled for model {self.model_id}")
            forwards += 1

        rendered_height, rendered_width = expected_canvas_size(
            request.height, request.width, request.min_generation_area
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        handle = pipe.transformer.register_forward_pre_hook(cancel_between_forwards, with_kwargs=True)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                result = pipe(
                    prompt=request.prompt,
                    height=request.height,
                    width=request.width,
                    min_generation_area=request.min_generation_area,
                    generator=torch.Generator("cpu").manual_seed(request.seed),
                    output_type="pil",
                )
            torch.cuda.synchronize()
        finally:
            handle.remove()
        generated = time.perf_counter()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        self._check_cancel(cancel_flag)

        images = getattr(result, "images", None)
        if images is None and hasattr(result, "values"):
            values = result.values
            images = values.get("images") if isinstance(values, dict) else getattr(values, "images", None)
        if not images:
            raise InferenceError(f"{self.model_id} pipeline returned no images")
        image = images[0]
        delivered_width, delivered_height = image.size
        if (delivered_width, delivered_height) != (request.width, request.height):
            raise InferenceError(
                f"{self.model_id} delivered {delivered_width}x{delivered_height}, expected exactly "
                f"{request.width}x{request.height} for lane {request.lane!r} (rendered canvas "
                f"{rendered_width}x{rendered_height}); refusing a silently resized benchmark row"
            )

        write_png_durably(output_dir / "result.png", image)
        written = time.perf_counter()
        self._renders += 1
        info = self._load_info
        gib = float(1024**3)
        return {
            "format": "png",
            "file": "result.png",
            "mode": "t2i",
            "model": MINIMAX_H3_IMAGE_HF_REPO,
            "model_id": self.model_id,
            "prompt": request.prompt,
            "seed": request.seed,
            "lane": request.lane,
            "steps": EXPECTED_STEPS,
            "sample_sigmas": list(EXPECTED_SAMPLE_SIGMAS),
            "guidance_scale": 1,
            "negative_prompt": None,
            "min_generation_area": request.min_generation_area,
            "sizes": {
                "requested": {"width": request.width, "height": request.height},
                "rendered": {"width": rendered_width, "height": rendered_height},
                "delivered": {"width": delivered_width, "height": delivered_height},
            },
            "width": delivered_width,
            "height": delivered_height,
            "precision": {"components": "bfloat16", "initial_noise": "float32"},
            "license": {"confirmed": True, "confirmation_ref": request.license_confirmation_ref},
            "provenance": {**info["weights"], **info["runtime"]},
            "limitations": [
                "No authoritative memory-capable GB10 route is proven for this snapshot; "
                "load is in-process bf16 under the Arbiter worker CUDA cap and fails if it does not fit.",
            ],
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
        # Six fixed forwards; the 4096 lane renders 16x the pixels of the
        # 1024 lane. Load time is separate. Estimates, not measured timings.
        lane = params.get("lane", "controlled-1024")
        return 12000.0 if lane == "recommended-4096" else 3000.0
