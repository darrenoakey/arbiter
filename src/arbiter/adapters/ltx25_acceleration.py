"""Opt-in, capped LTX 2.5 acceleration A/B benchmark for ``ltx25-denoise1``.

Read docs/ltx25-acceleration.md before changing this file.

A job opts in with ``acceleration_benchmark`` set to one of three profiles.
An absent key never reaches this module, so the production denoise path is
unchanged.

- ``stock``: the exact sequence of the pinned runner's ``run_denoise_gpu``.
  Dev transformer, 30 LTX2Scheduler steps, GuidedDenoiser (CFG 3, STG 1 on
  block 28, rescale 0.7, modality 3), then dev + distilled LoRA-450 at 1.0
  on the runner's stage-2 sigmas, then the DiffVAE CHUNKED_EAGER decode.
- ``distilled``: the official Lightricks distilled transformer, one
  separately downloaded checkpoint used for both stages and no LoRA. Stage 1
  runs the installed ``DISTILLED_SIGMAS`` (8 steps) with SimpleDenoiser
  (CFG 1, STG 0, modality 1) and the installed 2.5 ancestral sampler
  (eta 1, s_noise 1, noise seed = seed + 10000). Stage 2 runs the installed
  ``STAGE_2_DISTILLED_SIGMAS`` with the default deterministic Euler step.
- ``distilled-conv``: ``distilled`` with only the decoder swapped for the
  installed conv VAE. Its encoder and per-channel statistics tensors are
  byte-identical to the DiffVAE file, so the latents are compatible.

All profiles keep the music-video contract: frozen original audio latent,
the encoded stage-1 image conditionings (first, last, repeated end, any
guides), the both-stage temporal attention ramp, exact boundary-pixel
replacement, and the NVENC mux of the original soundtrack. The official
distilled pipeline generates audio jointly. Freezing the audio here adapts
it to the music-video contract and is labelled as such; it is not stock
joint T2AV.

The runner is never imported at module scope. The laptop gate has no
torchaudio, so every ``ltx_core`` / ``ltx_pipelines`` / ``torch`` import is
inside a function that only runs in the audited worker.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from arbiter.adapters.base import CancelledException, InferenceError
from arbiter.adapters.ltx25_temporal_attention import (
    _MANIFEST_PATH,
    require_temporal_attention_bridge,
    resolve_denoise_temporal_attention_ramp,
)

log = logging.getLogger(__name__)

BENCHMARK_PARAM = "acceleration_benchmark"
PROFILE_STOCK = "stock"
PROFILE_DISTILLED = "distilled"
PROFILE_DISTILLED_CONV = "distilled-conv"
PROFILES = (PROFILE_STOCK, PROFILE_DISTILLED, PROFILE_DISTILLED_CONV)

# Hard caps. A benchmark job shares the GPU with production work, so it is
# limited to small clips. Frame counts follow 8n+1 and both sides must be
# divisible by 64 for the 2x latent upscale.
MAX_FRAMES = 49
MAX_WIDTH = 768
MAX_HEIGHT = 512

STOCK_STEPS = 30
STOCK_A2V_GUIDANCE = 3.0

# The stock profile replays the runner's run_denoise_gpu step by step so
# that model build and warm denoise can be timed apart. A different runner
# could diverge from this replay, so the benchmark refuses any other one.
PINNED_RUNNER_SHA256 = "1c8ea922378515ca891fda5d9af7b68a3ad945f0ba7138a44da9fda49267040f"

# Runner literals replayed for the stock profile (video_fast_gpu.py).
STOCK_VIDEO_GUIDER = {
    "cfg_scale": 3.0,
    "stg_scale": 1.0,
    "rescale_scale": 0.7,
    "modality_scale": STOCK_A2V_GUIDANCE,
    "skip_step": 0,
    "stg_blocks": [28],
}
# The runner's own constants.py rounds the official values to 3 places.
STOCK_STAGE2_SIGMAS = (0.909, 0.725, 0.422, 0.0)

# Installed ltx_pipelines/utils/constants.py and distilled.py. Checked
# against the imported modules at run time and against the pinned source
# archive in the unit tests.
DISTILLED_STAGE1_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
DISTILLED_STAGE2_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
ANCESTRAL_ETA = 1.0
ANCESTRAL_S_NOISE = 1.0
ANCESTRAL_NOISE_SEED_OFFSET = 10000

_SIZE_22B = 42018190584


@dataclass(frozen=True)
class WeightFile:
    """A fixed weight path with the size and SHA-256 it must have."""

    role: str
    path: str
    size: int
    sha256: str


DEV_TRANSFORMER = WeightFile(
    "dev_transformer",
    "/home/darren/models/ltx25/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    _SIZE_22B,
    "792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584",
)
# Same HF revision as the installed dev weights. The SHA is the Git LFS
# pointer oid at that revision; the HTTP etag is a Xet/CDN id, not this hash.
DISTILLED_TRANSFORMER = WeightFile(
    "distilled_transformer",
    "/mnt/arbiter-store/output/benchmarks/ltx25-acceleration/weights/"
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    _SIZE_22B,
    "31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4",
)
DISTILLED_REVISION = "6c7e5e573ac1667efc83407806fe9b0b93730e60"
DIFF_VAE = WeightFile(
    "diffvae_decoder",
    "/home/darren/models/ltx25/vae/ltx-2.5-video-vae-bf16.safetensors",
    1472223346,
    "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3",
)
CONV_VAE = WeightFile(
    "conv_decoder",
    "/home/darren/models/ltx25/vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    1452269922,
    "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d",
)


@dataclass(frozen=True)
class ProfileSpec:
    """Everything a profile fixes. Nothing here comes from job params."""

    name: str
    transformer: WeightFile
    decoder: WeightFile
    distilled: bool

    def weights(self) -> tuple[WeightFile, ...]:
        return (self.transformer, self.decoder)


_SPECS = {
    PROFILE_STOCK: ProfileSpec(PROFILE_STOCK, DEV_TRANSFORMER, DIFF_VAE, False),
    PROFILE_DISTILLED: ProfileSpec(PROFILE_DISTILLED, DISTILLED_TRANSFORMER, DIFF_VAE, True),
    PROFILE_DISTILLED_CONV: ProfileSpec(PROFILE_DISTILLED_CONV, DISTILLED_TRANSFORMER, CONV_VAE, True),
}


def profile_spec(profile: str) -> ProfileSpec:
    if profile not in _SPECS:
        raise InferenceError(f"unknown {BENCHMARK_PARAM} profile: {profile!r}")
    return _SPECS[profile]


# ---------------------------------------------------------------------------
# Request and bundle validation (pure, laptop-testable)
# ---------------------------------------------------------------------------


def parse_acceleration_benchmark(params: dict) -> str | None:
    """Return the requested profile. Absent means the production path."""
    if BENCHMARK_PARAM not in params:
        return None
    value = params[BENCHMARK_PARAM]
    if type(value) is not str or value not in PROFILES:
        raise InferenceError(
            f"{BENCHMARK_PARAM} must be one of {list(PROFILES)}, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value


# Other denoise opt-ins change the conditioning under test. They may be sent
# only at their no-op default alongside a benchmark.
_CONFLICTING_OPT_INS = {
    "stage1_guiding_keyframes": (False,),
    "generated_keyframes": (0,),
    "generated_keyframe_positions": ([],),
    "stage2_drop_end_image": (False,),
    "end_image_strength": (None,),
}


def _is_default(value: object, defaults: tuple) -> bool:
    # type() check so True is not accepted as 1 and 0.0 is not False.
    return any(type(value) is type(d) and value == d for d in defaults)


def require_benchmark_request(profile: str, params: dict) -> None:
    """Reject params that the fixed profile contract does not allow."""
    for key, defaults in _CONFLICTING_OPT_INS.items():
        if key in params and not _is_default(params[key], defaults):
            raise InferenceError(
                f"{BENCHMARK_PARAM} cannot be combined with {key}={params[key]!r}"
            )
    if profile == PROFILE_STOCK:
        if "num_inference_steps" in params:
            steps = params["num_inference_steps"]
            if type(steps) is not int or steps != STOCK_STEPS:
                raise InferenceError(
                    f"{BENCHMARK_PARAM}=stock fixes num_inference_steps at "
                    f"{STOCK_STEPS}, got {steps!r}"
                )
        if "a2v_guidance_scale" in params:
            scale = params["a2v_guidance_scale"]
            if type(scale) not in (int, float) or float(scale) != STOCK_A2V_GUIDANCE:
                raise InferenceError(
                    f"{BENCHMARK_PARAM}=stock fixes a2v_guidance_scale at "
                    f"{STOCK_A2V_GUIDANCE}, got {scale!r}"
                )
        return
    for key in ("num_inference_steps", "a2v_guidance_scale"):
        if key in params:
            raise InferenceError(
                f"{BENCHMARK_PARAM}={profile} uses the fixed official distilled "
                f"schedule and guidance; {key} must be absent"
            )


def require_benchmark_bundle(data: object) -> tuple[int, int, int]:
    """Enforce the clip caps on the encoded bundle. Returns (frames, h, w)."""
    if not isinstance(data, dict):
        raise InferenceError("denoise input must be a dict")
    values = {}
    for key in ("num_frames", "height", "width"):
        value = data.get(key)
        if type(value) is not int or value < 1:
            raise InferenceError(f"benchmark bundle {key} must be a positive int, got {value!r}")
        values[key] = value
    frames, height, width = values["num_frames"], values["height"], values["width"]
    if frames > MAX_FRAMES or frames % 8 != 1:
        raise InferenceError(
            f"benchmark num_frames must be 8n+1 and <= {MAX_FRAMES}, got {frames}"
        )
    if height > MAX_HEIGHT or height % 64 != 0:
        raise InferenceError(
            f"benchmark height must be a multiple of 64 and <= {MAX_HEIGHT}, got {height}"
        )
    if width > MAX_WIDTH or width % 64 != 0:
        raise InferenceError(
            f"benchmark width must be a multiple of 64 and <= {MAX_WIDTH}, got {width}"
        )
    return frames, height, width


def require_pinned_runner(manifest_path: Path = _MANIFEST_PATH) -> str:
    """Refuse a runtime whose runner is not the one the stock replay copies."""
    manifest = json.loads(manifest_path.read_text())
    runner_sha = manifest.get("patched_files", {}).get("video_fast_gpu.py")
    if runner_sha != PINNED_RUNNER_SHA256:
        raise InferenceError(
            "acceleration benchmark replays runner "
            f"{PINNED_RUNNER_SHA256[:12]}, manifest pins {str(runner_sha)[:12]}; "
            "re-audit the stock replay before running"
        )
    return runner_sha


# ---------------------------------------------------------------------------
# Content attestation
# ---------------------------------------------------------------------------

_ATTESTED: dict[str, tuple[tuple[int, int, int, int], str]] = {}
_ATTEST_LOCK = threading.Lock()


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


HASH_PROGRESS_BYTES = 1024**3
HASH_PROGRESS_SECONDS = 30.0


def sha256_file(
    path: Path,
    progress: Callable[[int, int, float], None] | None = None,
    progress_bytes: int = HASH_PROGRESS_BYTES,
    progress_seconds: float = HASH_PROGRESS_SECONDS,
    block_bytes: int = 16 * 1024 * 1024,
) -> str:
    """Hash a file. ``progress(bytes_hashed, total, elapsed)`` fires after
    every 1 GiB or 30 s of real hashing, whichever comes first, and once at
    the end. A 39 GiB file on the SMB store takes about six minutes; the
    worker's stderr lines are what the GPU idle watchdog reads as liveness.
    """
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    started = last_time = time.monotonic()
    last_bytes = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
            done += len(block)
            now = time.monotonic()
            if progress is not None and (
                done - last_bytes >= progress_bytes or now - last_time >= progress_seconds
            ):
                progress(done, total, now - started)
                last_bytes, last_time = done, now
    if progress is not None and done != last_bytes:
        progress(done, total, time.monotonic() - started)
    return digest.hexdigest()


def _log_hash_progress(role: str) -> Callable[[int, int, float], None]:
    def _emit(done: int, total: int, elapsed: float) -> None:
        log.info(
            "ltx25 benchmark attestation %s: hashed %d/%d bytes (%.1f%%) in %.1fs",
            role,
            done,
            total,
            100.0 * done / total if total else 100.0,
            elapsed,
        )

    return _emit


def attest_weight(weight: WeightFile) -> dict:
    """Check size and SHA-256 before any load.

    A full hash of a 39 GiB file takes minutes, so a match is remembered for
    this process keyed by (device, inode, size, mtime). Any change to the file
    forces a fresh hash. Mismatch fails closed.
    """
    path = Path(weight.path)
    if not path.is_file():
        raise InferenceError(f"{weight.role} weight is missing: {path}")
    resolved = path.resolve()
    identity = _file_identity(resolved)
    if identity[2] != weight.size:
        raise InferenceError(
            f"{weight.role} size {identity[2]} != expected {weight.size}: {resolved}"
        )
    key = str(resolved)
    started = time.perf_counter()
    with _ATTEST_LOCK:
        cached = _ATTESTED.get(key)
        if cached is not None and cached[0] == identity and cached[1] == weight.sha256:
            return {
                "role": weight.role,
                "path": weight.path,
                "resolved_path": key,
                "size": weight.size,
                "sha256": weight.sha256,
                "cached": True,
                "seconds": time.perf_counter() - started,
            }
        actual = sha256_file(resolved, _log_hash_progress(weight.role))
        if _file_identity(resolved) != identity:
            raise InferenceError(f"{weight.role} changed while hashing: {resolved}")
        if actual != weight.sha256:
            _ATTESTED.pop(key, None)
            raise InferenceError(
                f"{weight.role} sha256 {actual} != expected {weight.sha256}: {resolved}"
            )
        _ATTESTED[key] = (identity, actual)
    return {
        "role": weight.role,
        "path": weight.path,
        "resolved_path": key,
        "size": weight.size,
        "sha256": actual,
        "cached": False,
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class StageClock:
    """Wall-clock segments with CUDA synchronised at every boundary.

    Each segment resets the CUDA peak counters on entry and records the
    segment's peak allocated/reserved bytes on exit. The overall peak is the
    maximum over segments. Without CUDA (laptop tests) timings still record
    and peaks are omitted.
    """

    def __init__(self, torch_module=None):
        self._torch = torch_module
        self.timings: dict[str, float] = {}
        self.peaks: dict[str, dict[str, float]] = {}

    @property
    def _cuda(self) -> bool:
        return self._torch is not None and self._torch.cuda.is_available()

    def sync(self) -> None:
        if self._cuda:
            self._torch.cuda.synchronize()

    def now(self) -> float:
        self.sync()
        return time.perf_counter()

    def _reset_peak(self) -> None:
        if self._cuda:
            self._torch.cuda.reset_peak_memory_stats()

    def _read_peak(self, name: str) -> None:
        if self._cuda:
            gib = 1024**3
            self.peaks[name] = {
                "allocated_gib": self._torch.cuda.max_memory_allocated() / gib,
                "reserved_gib": self._torch.cuda.max_memory_reserved() / gib,
            }

    def record(self, name: str, seconds: float) -> None:
        if name in self.timings:
            raise InferenceError(f"benchmark timing {name} recorded twice")
        if not math.isfinite(seconds) or seconds < 0:
            raise InferenceError(f"benchmark timing {name} is invalid: {seconds}")
        self.timings[name] = seconds

    @contextmanager
    def segment(self, name: str):
        self._reset_peak()
        start = self.now()
        try:
            yield
        finally:
            end = self.now()
            self.record(name, end - start)
            self._read_peak(name)

    def run_stage(self, prefix: str, stage, loop, cleanup: Callable[[], None], **kwargs):
        """Run one DiffusionStage call split into build / denoise / release.

        DiffusionStage builds the transformer, then calls ``loop`` with
        keyword arguments, then frees it. Wrapping the public ``loop``
        argument marks the build/denoise and denoise/release boundaries
        without patching any library code.
        """
        marks: dict[str, float] = {}

        def timed_loop(**loop_kwargs):
            marks["loop_start"] = self.now()
            result = loop(**loop_kwargs)
            marks["loop_end"] = self.now()
            return result

        self._reset_peak()
        start = self.now()
        result = stage(loop=timed_loop, **kwargs)
        cleanup()
        end = self.now()
        if "loop_start" not in marks or "loop_end" not in marks:
            raise InferenceError(f"benchmark {prefix} never ran its denoising loop")
        self.record(f"{prefix}_build", marks["loop_start"] - start)
        self.record(f"{prefix}_denoise", marks["loop_end"] - marks["loop_start"])
        self.record(f"{prefix}_release", end - marks["loop_end"])
        self._read_peak(prefix)
        return result

    def overall_peak(self) -> dict[str, float] | None:
        if not self.peaks:
            return None
        return {
            "allocated_gib": max(p["allocated_gib"] for p in self.peaks.values()),
            "reserved_gib": max(p["reserved_gib"] for p in self.peaks.values()),
        }


# ---------------------------------------------------------------------------
# Static contract metadata
# ---------------------------------------------------------------------------


def profile_contract(profile: str) -> dict:
    """The sampler, sigma, guidance and weight contract of a profile."""
    spec = profile_spec(profile)
    audio = {
        "mode": "frozen_original_audio_latent",
        "note": (
            "music-video adaptation: the encoded original audio latent is frozen "
            "in both stages and the original soundtrack is muxed; not stock joint T2AV"
        ),
    }
    anchors = (
        "encoded stage-1 conditionings unchanged; stage-2 rebuilt from bundle "
        "images by the runner; temporal attention ramp on both stages; exact "
        "first/last boundary pixels copied after decode (pixel copy, not "
        "evidence of generative adherence)"
    )
    if not spec.distilled:
        return {
            "profile": profile,
            "label": "stock runner (dev transformer + distilled LoRA-450 stage 2)",
            "stage1": {
                "transformer": DEV_TRANSFORMER.role,
                "loras": [],
                "sigmas": f"LTX2Scheduler.execute(steps={STOCK_STEPS})",
                "steps": STOCK_STEPS,
                "denoiser": "GuidedDenoiser",
                "video_guider": dict(STOCK_VIDEO_GUIDER),
                "audio_guider": "MultiModalGuiderParams() defaults",
                "sampler": "EulerDiffusionStep + euler_denoising_loop (deterministic)",
            },
            "stage2": {
                "transformer": DEV_TRANSFORMER.role,
                "loras": ["ltx-2.5-22b-distilled-lora-450-bf16 @ 1.0"],
                "sigmas": list(STOCK_STAGE2_SIGMAS),
                "denoiser": "SimpleDenoiser",
                "sampler": "EulerDiffusionStep + euler_denoising_loop (deterministic)",
            },
            "decoder": {"weight": spec.decoder.role, "diffvae_optimization": "CHUNKED_EAGER"},
            "audio": audio,
            "anchors": anchors,
        }
    return {
        "profile": profile,
        "label": "official distilled transformer (separate checkpoint, no LoRA)",
        "distilled_revision": DISTILLED_REVISION,
        "stage1": {
            "transformer": DISTILLED_TRANSFORMER.role,
            "loras": [],
            "sigmas": list(DISTILLED_STAGE1_SIGMAS),
            "steps": len(DISTILLED_STAGE1_SIGMAS) - 1,
            "denoiser": "SimpleDenoiser (CFG 1, STG 0, modality 1)",
            "sampler": (
                f"EulerAncestralDiffusionStep(eta={ANCESTRAL_ETA}, s_noise={ANCESTRAL_S_NOISE}) "
                f"+ euler_ancestral_denoising_loop(noise_seed=seed+{ANCESTRAL_NOISE_SEED_OFFSET})"
            ),
        },
        "stage2": {
            "transformer": DISTILLED_TRANSFORMER.role,
            "loras": [],
            "sigmas": list(DISTILLED_STAGE2_SIGMAS),
            "noise_scale": DISTILLED_STAGE2_SIGMAS[0],
            "denoiser": "SimpleDenoiser (CFG 1, STG 0, modality 1)",
            "sampler": "EulerDiffusionStep + euler_denoising_loop (deterministic)",
        },
        "decoder": {
            "weight": spec.decoder.role,
            "diffvae_optimization": "CHUNKED_EAGER" if spec.decoder is DIFF_VAE else "n/a (conv VAE)",
        },
        "audio": audio,
        "anchors": anchors,
    }


# ---------------------------------------------------------------------------
# Worker-only GPU path
# ---------------------------------------------------------------------------


def _same_values(actual, expected: tuple[float, ...]) -> bool:
    values = [float(v) for v in actual]
    return len(values) == len(expected) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6) for a, b in zip(values, expected)
    )


def _require_installed_contract(runner, pipeline, spec: ProfileSpec) -> dict:
    """Check the imported runner and upstream modules match the pinned contract."""
    from ltx_pipelines import distilled as upstream_distilled
    from ltx_pipelines.utils import constants as upstream_constants

    if str(runner.CHECKPOINT) != DEV_TRANSFORMER.path:
        raise InferenceError(f"runner CHECKPOINT {runner.CHECKPOINT} is not the attested dev path")
    if str(runner.VIDEO_VAE) != DIFF_VAE.path:
        raise InferenceError(f"runner VIDEO_VAE {runner.VIDEO_VAE} is not the attested DiffVAE path")
    if str(pipeline.model_paths.transformer()) != DEV_TRANSFORMER.path:
        raise InferenceError("pipeline transformer path is not the attested dev path")
    if str(pipeline._video_decoder.checkpoint_path) != DIFF_VAE.path:
        raise InferenceError("resident decoder is not the attested DiffVAE")
    if pipeline._video_decoder.diffvae_optimization.name != "CHUNKED_EAGER":
        raise InferenceError("resident decoder is not DiffVAE CHUNKED_EAGER")
    if not _same_values(runner.STAGE_2_DISTILLED_SIGMAS, STOCK_STAGE2_SIGMAS):
        raise InferenceError("runner STAGE_2_DISTILLED_SIGMAS changed")
    if not _same_values(upstream_constants.DISTILLED_SIGMAS, DISTILLED_STAGE1_SIGMAS):
        raise InferenceError("installed DISTILLED_SIGMAS changed")
    if not _same_values(upstream_constants.STAGE_2_DISTILLED_SIGMAS, DISTILLED_STAGE2_SIGMAS):
        raise InferenceError("installed STAGE_2_DISTILLED_SIGMAS changed")
    installed = (
        upstream_distilled.ANCESTRAL_ETA,
        upstream_distilled.ANCESTRAL_S_NOISE,
        upstream_distilled.ANCESTRAL_NOISE_SEED_OFFSET,
    )
    if installed != (ANCESTRAL_ETA, ANCESTRAL_S_NOISE, ANCESTRAL_NOISE_SEED_OFFSET):
        raise InferenceError(f"installed ancestral constants changed: {installed}")
    version = upstream_constants.detect_model_version(spec.transformer.path)
    if spec.distilled and not upstream_distilled.should_use_ancestral_sampler(spec.transformer.path):
        raise InferenceError(
            f"distilled checkpoint model_version {version} does not select the ancestral sampler"
        )
    return {"transformer_model_version": ".".join(str(v) for v in version)}


def _run_gpu(pipeline, runner, data: dict, spec: ProfileSpec, strengths, clock: StageClock, progress_fn):
    """Replay the runner's denoise sequence with the profile's weights and sampler."""
    from functools import partial

    import numpy as np
    import torch
    from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
    from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
    from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.model.video_vae import AUTO_TILING
    from ltx_core.model.video_vae.transformer import DiffVAEMode
    from ltx_core.types import VideoPixelShape
    from ltx_pipelines.utils.blocks import DiffusionStage, VideoDecoder
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
    from ltx_pipelines.utils.helpers import ensure_tiling_config, tiling_scale_factors_for_vae
    from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop, euler_denoising_loop
    from ltx_pipelines.utils.types import ModalitySpec

    pipeline._ensure_denoise_models()
    device, dtype = pipeline.device, pipeline.dtype
    num_frames = data["num_frames"]
    height, width = data["height"], data["width"]
    fps = data.get("frame_rate", runner.DEFAULT_FPS)
    seed = data.get("seed", 42)
    chunk_index = data.get("chunk_index", 0)
    runner.validate_constraints(height, width, num_frames)
    images = runner.coerce_image_inputs(data.get("images", []))
    pipeline._validate_conditioning_images(images)
    temporal = runner._validate_attention_strengths(
        None if strengths is None else list(strengths), num_frames
    )
    seeds = {"noiser_seed": seed + chunk_index}
    if spec.distilled:
        seeds["ancestral_noise_seed"] = seed + chunk_index + ANCESTRAL_NOISE_SEED_OFFSET

    stages: dict[str, object] = {}
    decoder = None
    decoded_iter = None
    try:
        with clock.segment("context_to_device"):
            v_context_p = data["v_context_p"].to(device)
            a_context_p = data["a_context_p"].to(device)
            v_context_n = data["v_context_n"].to(device)
            audio_latent = data["encoded_audio_latent"].to(device)
            generator = torch.Generator(device=device).manual_seed(seeds["noiser_seed"])
            noiser = GaussianNoiser(generator=generator)
            stage_1_conditionings = runner.conditionings_to_device(
                data.get("stage_1_conditionings", []), device
            )
            stage_1_conditionings = runner._apply_endpoint_attention_strengths(
                stage_1_conditionings, temporal, num_frames
            )

        def frozen_audio():
            return ModalitySpec(
                context=a_context_p, frozen=True, noise_scale=0.0, initial_latent=audio_latent
            )

        def make_stage(loras):
            return DiffusionStage.from_checkpoint(
                spec.transformer.path,
                dtype,
                device,
                loras=loras,
                registry=pipeline.registry,
                alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
            )

        def release(name):
            def _release():
                stages.pop(name, None)
                runner._cleanup()
            return _release

        # Stage 1 at half resolution. One transformer is built inside the
        # call and freed on exit; nothing stays resident between stages.
        if spec.distilled:
            sigmas_1 = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
            denoiser_1 = SimpleDenoiser(v_context_p, a_context_p)
            stepper_1 = EulerAncestralDiffusionStep(eta=ANCESTRAL_ETA, s_noise=ANCESTRAL_S_NOISE)
            loop_1 = partial(
                euler_ancestral_denoising_loop,
                noise_seed=seeds["ancestral_noise_seed"],
                model_dtype=dtype,
            )
        else:
            sigmas_1 = pipeline._scheduler.execute(steps=STOCK_STEPS).to(
                dtype=torch.float32, device=device
            )
            denoiser_1 = GuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider=MultiModalGuider(
                    params=MultiModalGuiderParams(**STOCK_VIDEO_GUIDER),
                    negative_context=v_context_n,
                ),
                audio_guider=MultiModalGuider(params=MultiModalGuiderParams()),
            )
            stepper_1 = None
            loop_1 = euler_denoising_loop
        stages["stage1"] = make_stage(())
        video_state, _ = clock.run_stage(
            "stage1",
            stages["stage1"],
            loop_1,
            release("stage1"),
            denoiser=denoiser_1,
            sigmas=sigmas_1,
            noiser=noiser,
            width=width // 2,
            height=height // 2,
            frames=num_frames,
            fps=fps,
            video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
            audio=frozen_audio(),
            stepper=stepper_1,
            max_batch_size=1,
        )
        actual_sigmas_1 = [float(v) for v in sigmas_1.cpu()]
        if progress_fn:
            progress_fn("video", "stage1_done", chunk=chunk_index, total=1, idx=1, of=1)

        with clock.segment("upscale"):
            upscaled = pipeline._upsampler(video_state.latent[:1])
            del video_state

        with clock.segment("stage2_conditioning"):
            stage_2_conditionings = pipeline._build_stage_2_conditionings(
                images, height=height, width=width
            )
            stage_2_conditionings = runner._apply_endpoint_attention_strengths(
                stage_2_conditionings, temporal, num_frames
            )

        if spec.distilled:
            sigmas_2 = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
            stages["stage2"] = make_stage(())
        else:
            sigmas_2 = runner.STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
            stages["stage2"] = make_stage(pipeline.distilled_lora)
        video_state, _ = clock.run_stage(
            "stage2",
            stages["stage2"],
            euler_denoising_loop,
            release("stage2"),
            denoiser=SimpleDenoiser(v_context_p, a_context_p),
            sigmas=sigmas_2,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=fps,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_conditionings,
                noise_scale=sigmas_2[0].item(),
                initial_latent=upscaled,
            ),
            audio=frozen_audio(),
        )
        actual_sigmas_2 = [float(v) for v in sigmas_2.cpu()]
        del upscaled
        if progress_fn:
            progress_fn("video", "stage2_done", chunk=chunk_index, total=1, idx=1, of=1)

        if spec.decoder is DIFF_VAE:
            decoder = pipeline._video_decoder
        else:
            decoder = VideoDecoder(
                spec.decoder.path,
                dtype,
                device,
                registry=pipeline.registry,
                alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
                diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
            )
        with clock.segment("decode_build"):
            tiling_config = ensure_tiling_config(
                AUTO_TILING,
                scale_factors=tiling_scale_factors_for_vae(decoder.checkpoint_path),
                vae_checkpoint_path=decoder.checkpoint_path,
                video_shape=VideoPixelShape(
                    batch=1, frames=num_frames, height=height, width=width, fps=fps
                ),
                diffvae_optimization=decoder.diffvae_optimization,
                device=device,
            )
            decoded_iter = decoder(video_state.latent, tiling_config, generator, dtype=dtype)
        with clock.segment("decode"):
            tiles = [tile.cpu() for tile in decoded_iter]
            decoded_iter = None
            runner._cleanup()
        del video_state

        with clock.segment("postprocess"):
            frames = torch.cat(tiles, dim=0)
            del tiles
            if frames.dtype != torch.uint8:
                frames = (
                    (frames.clamp(0, 1) * 255.0).to(torch.uint8)
                    if frames.max() <= 1.0
                    else frames.clamp(0, 255).to(torch.uint8)
                )
            frames_np = frames.numpy()
            if frames_np.ndim == 4 and frames_np.shape[1] == 3:
                frames_np = np.transpose(frames_np, (0, 2, 3, 1))
            if frames_np.shape[1] == runner.DEFAULT_HEIGHT and frames_np.shape[2] == runner.DEFAULT_WIDTH:
                frames_np = runner.crop_frames_to_target(
                    frames_np,
                    target_width=runner.TARGET_WIDTH_1080P,
                    target_height=runner.TARGET_HEIGHT_1080P,
                )
            frames_np = pipeline._replace_conditioned_boundaries(
                frames_np,
                images,
                num_frames,
                conditioning_height=height,
                conditioning_width=width,
            )
        if progress_fn:
            progress_fn("video", "chunk_done", chunk=chunk_index, total=1, idx=1, of=1, frames=len(frames_np))
        actual = {
            "seeds": seeds,
            "stage1_sigmas": actual_sigmas_1,
            "stage2_sigmas": actual_sigmas_2,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "fps": fps,
            "images": [
                {"frame_idx": int(img.frame_idx), "strength": float(img.strength)} for img in images
            ],
            "stage1_conditionings": [type(c).__name__ for c in stage_1_conditionings],
            "stage2_conditionings": [type(c).__name__ for c in stage_2_conditionings],
            "temporal_attention_ramp": temporal is not None,
            "output_shape": list(frames_np.shape),
        }
        return frames_np, actual
    finally:
        # Free every benchmark-only object, on success and on error, before
        # the GPU lock is released.
        if decoded_iter is not None:
            decoded_iter.close()
        stages.clear()
        if decoder is not None and decoder is not pipeline._video_decoder:
            del decoder
        gc.collect()
        runner._cleanup()


WORKER_LOAD_KEYS = (
    "runtime_import_s",
    "pipeline_init_and_load_denoise_models_s",
    "load_total_s",
    "loaded_at_unix",
)


def worker_load_metadata(worker_load: dict | None) -> dict:
    """Validate and label the adapter's measured load() timing.

    ``timing`` is None only if load() never completed in this process, which
    cannot happen for a job that reached infer; it is reported, not invented.
    """
    worker_load = worker_load or {}
    timing = worker_load.get("timing")
    if timing is not None:
        if set(timing) != set(WORKER_LOAD_KEYS):
            raise InferenceError(f"worker load timing keys {sorted(timing)} are not {list(WORKER_LOAD_KEYS)}")
        for key in WORKER_LOAD_KEYS:
            value = timing[key]
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise InferenceError(f"worker load timing {key} is invalid: {value!r}")
    count = worker_load.get("benchmark_infers_since_load")
    if count is not None and (type(count) is not int or count < 1):
        raise InferenceError(f"benchmark_infers_since_load is invalid: {count!r}")
    return {
        "timing": None if timing is None else dict(timing),
        "benchmark_infers_since_load": count,
        "note": (
            "measured adapter load(): runtime import plus FastPipeline init and "
            "load_denoise_models (image encoder, upsampler, DiffVAE decoder blocks). "
            "Transformers are not part of worker load; each job builds them, see "
            "stage*_build. benchmark_infers_since_load counts benchmark jobs only; "
            "normal ltx25-denoise1 jobs on the same worker are not counted, so 1 means "
            "the first benchmark job since load, not proof of a cold process. Use "
            "worker_load.timing.loaded_at_unix and the scheduler job history to see "
            "what else ran on this worker."
        ),
    }


def run_acceleration_benchmark(
    *,
    pipeline,
    runner,
    runtime_provenance: dict,
    gpu_lock: threading.Lock,
    profile: str,
    params: dict,
    encoded_file: str,
    audio_file: str,
    start_time: float,
    fps: float,
    output_dir: Path,
    progress_fn: Callable,
    check_cancel: Callable[[], None],
    worker_load: dict | None = None,
) -> dict:
    """Run one benchmark denoise job and write result.mp4 plus benchmark.json."""
    import torch

    spec = profile_spec(profile)
    require_benchmark_request(profile, params)
    clock = StageClock(torch)
    total_start = time.perf_counter()
    try:
        with clock.segment("cpu_prep"):
            runner_sha = require_pinned_runner()
            data = pipeline.load_denoise_input(encoded_file)
            require_benchmark_bundle(data)
            strengths = resolve_denoise_temporal_attention_ramp(data, params, runtime_provenance)
            require_temporal_attention_bridge(runner, strengths)
        with clock.segment("attestation"):
            attestations = [attest_weight(w) for w in spec.weights()]
        check_cancel()
        with gpu_lock:
            check_cancel()
            gpu_start = clock.now()
            installed = _require_installed_contract(runner, pipeline, spec)
            with torch.inference_mode():
                frames_np, actual = _run_gpu(
                    pipeline, runner, data, spec, strengths, clock, progress_fn
                )
            clock.record("gpu_total", clock.now() - gpu_start)
        check_cancel()
        result_path = output_dir / "result.mp4"
        with clock.segment("encode_mux"):
            pipeline.save_denoise_output(
                frames_np, str(result_path), fps=fps, audio_path=audio_file, start_time=start_time
            )
        clock.record("total", time.perf_counter() - total_start)
    except (CancelledException, InferenceError):
        raise
    except Exception as exc:
        raise InferenceError(f"ltx25 acceleration benchmark ({profile}) failed: {exc}") from exc

    metadata = {
        "profile": profile,
        "contract": profile_contract(profile),
        "actual": actual,
        "weights": attestations,
        "runtime": {
            "provenance": dict(runtime_provenance),
            "runner_sha256": runner_sha,
            **installed,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "pid": os.getpid(),
        },
        "worker_load": worker_load_metadata(worker_load),
        "timings_s": dict(clock.timings),
        "peaks_gib": clock.peaks,
        "peak_gib": clock.overall_peak(),
        "timing_notes": (
            "torch.cuda.synchronize at every boundary; CUDA peaks reset per segment. "
            "*_build = transformer build/load, *_denoise = warm loop, *_release = free. "
            "Worker model load is outside infer and reported under worker_load. "
            "attestation is cached per process after the first hash."
        ),
    }
    (output_dir / "benchmark.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    log.info("ltx25 acceleration benchmark %s timings %s", profile, clock.timings)
    return {"file": "result.mp4", "format": "mp4", BENCHMARK_PARAM: metadata}
