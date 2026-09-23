"""LTX-2.5 denoise adapter — Stage B, the ONLY denoise stage of the 2-way
LTX 2.5 split pipeline.

*** There is no `ltx25-denoise2`, and there never will be. This is not a
*** truncated split — it is the correct one. Do NOT add a fake denoise2 stage.
Read this docstring before "completing the pattern" from the 2.3 lane.

Why 2-way, not 3-way like LTX 2.3:
    LTX 2.3 splits encode -> denoise1 -> denoise2 because it has TWO distinct
    transformer checkpoints (a stage-1 "dev" transformer and a stage-2
    distilled transformer) plus a spatial upsampler in between — three
    independently-scheduled weight sets. LTX 2.5 has exactly ONE 22B
    dual-stream DiT (`ltx-2.5-22b-dev-transformer-bf16.safetensors`); "stage 1"
    and "stage 2" are the SAME base transformer run twice — once bare at half
    resolution (544x960), once with the distilled LoRA applied at full
    resolution (1088x1920) after a 2x latent spatial upscale. Splitting that
    into two Arbiter models would force two workers to each hold the full
    ~39GB base transformer resident at once (~78GB just for the duplicated
    base weights) for zero benefit — the entire stage1 -> upscale -> stage2 ->
    decode chain already runs in-memory, back-to-back, inside one GPU-locked
    call. See LTX_CUSTOMIZATIONS.md's "LTX 2.5 (2-way split)" section and
    local/ltx25-stage-map.md (task t3) for the full inference-graph mapping
    and the explicit "why not 3-way" rejection.

    Renderer/orchestrator note (consumed by later tasks wiring the
    music_video_ltx25_full.json engine lane): the 2.3 renderer's
    `video-denoise2` step MUST be SKIPPED for the ltx25 lane — this adapter's
    `result.mp4` output IS the final per-chunk artifact, exactly like
    `ltx2-denoise2`'s output, just produced by one stage instead of two.
    There is no `denoise1_file`/`stage1_output.pt` hand-off to feed a second
    stage, and no `ltx25-denoise2` model exists to receive one.

Loads (resident, pre-loaded in `load()` via the runner's public
`FastPipeline.load_denoise_models()` hook — same "keep the big weights
resident across jobs" pattern as `ltx2_denoise2.py`):
    - 22B Dev Transformer (bf16)                    39.13 GB
    - Distilled LoRA (rank 450, bf16)                 8.29 GB
    - Latent Spatial Upscaler x2                      0.93 GB
    - CausalDiffusionVAE video decoder                1.37 GB
    Resident total: 49.72 GB. Peak during stage-2 1088x1920 refinement: ~80GB.

Driven by the dedicated `ltx25-spark` runner tree (`~/src/ltx25-spark`, its
own isolated venv via `worker_cmd` -> `venvs/ltx25/bin/python`) — deliberately
NOT `ltx2-spark`.

Expected params dict (README "Stage B: ltx25-denoise" contract):
    encoded_file        : str   — absolute path to encoded.pt from ltx25-encode
    audio_file          : str   — absolute path to the ORIGINAL Suno audio
                                   master on spark local disk. This is muxed
                                   into the final mp4 verbatim; any audio LTX
                                   2.5 itself decodes/generates internally is
                                   unconditionally discarded (see
                                   `video_fast_gpu.encode_video_nvenc`).
    start_time          : float — chunk start in seconds (audio mux slice offset)
    fps                 : float — output frame rate, default 25.0
    num_inference_steps : int   — stage-1 diffusion steps, default 30
    a2v_guidance_scale  : float — stage-1 audio-conditioning guidance scale,
                                  default 3.0; must be >= 1.0 else InferenceError.
                                  Forwarded verbatim to FastPipeline.run_denoise_gpu
                                  (ltx25-spark runner; the A/B lever that trades
                                  mouth fidelity against identity drift).
    stage1_guiding_keyframes : bool — opt-in, default false. When true, after
                                  load_denoise_input, the single stage-1 frame-0
                                  VideoConditionByLatentIndex is replaced by
                                  VideoConditionByKeyframeIndex(keyframes=the same
                                  latent tensor, frame_idx=0, strength=the same
                                  strength). Seed, images, audio, prompt contexts,
                                  and stage-2 inputs are not touched. This is not
                                  full official-pipeline equivalence: official
                                  keyframe interpolation guides every key in both
                                  stages; stage 2 here is still rebuilt from
                                  images by combined_image_conditionings. Absent
                                  or false keeps the historical path. Only a real
                                  boolean is accepted.
    generated_keyframes      : int  — opt-in, default 0. When positive, after
                                  load_denoise_input the adapter appends one
                                  VideoGeneratedKeyframeSlots item to
                                  stage_1_conditionings. Positions are the
                                  official evenly spaced interior frames
                                  (torch.linspace over the pixel span, endpoints
                                  excluded). DiffusionStage already applies those
                                  slots and strips them back out of the latent
                                  before stage 2, matching ti2vid_two_stages.
                                  0 or absent does not import the slot type and
                                  does not touch the bundle. Only a real int in
                                  [0, 8] is accepted; bools are rejected.

Output: `result.mp4` written directly into `output_dir` (the file, not the
directory, per `FastPipeline.save_denoise_output` / `encode_video_nvenc`'s
exact-path contract) — the FINAL chunk artifact.
"""

from __future__ import annotations

import gc
import importlib
import logging
import sys
import threading
from pathlib import Path

from arbiter.adapters.base import (
    CancelledException,
    GroupAdapter,
    HeapTrimGuard,
    InferenceError,
    LoadError,
)
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

# The dedicated LTX 2.5 runner tree — deliberately NOT ltx2-spark (the 2.3
# lane's runner). See ltx25-spark/README.md section 2 ("PYTHONPATH & Arbiter
# Worker Rule").
LTX25_SPARK_DIR = Path("/home/darren/src/ltx25-spark")
_STAGE1_CONDITIONINGS_KEY = "stage_1_conditionings"


def parse_stage1_guiding_keyframes(params: dict) -> bool:
    """Return the opt-in flag. Missing means false; only a real bool is valid."""
    if "stage1_guiding_keyframes" not in params:
        return False
    value = params["stage1_guiding_keyframes"]
    # Reject ints, strings, and null. JSON true/false are the only accepted
    # forms; "true" and 1 must not silently change the historical path.
    if type(value) is not bool:
        raise InferenceError(
            "stage1_guiding_keyframes must be a boolean, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value


def _guiding_conditioning_types():
    """Import the worker's real conditioning classes. Never called on the default path."""
    try:
        import torch
        from ltx_core.conditioning.types.keyframe_cond import (
            VideoConditionByKeyframeIndex,
        )
        from ltx_core.conditioning.types.latent_cond import (
            VideoConditionByLatentIndex,
        )
    except ImportError as exc:
        raise InferenceError(
            "stage1_guiding_keyframes requires ltx_core conditioning types "
            f"inside the audited worker: {exc}"
        ) from exc
    return torch, VideoConditionByLatentIndex, VideoConditionByKeyframeIndex


def _require_frame0_latent(item, torch_module) -> None:
    latent = getattr(item, "latent", None)
    if not isinstance(latent, torch_module.Tensor) or latent.ndim != 5:
        shape = tuple(getattr(latent, "shape", ()))
        raise InferenceError(
            "stage1 frame-0 latent must be a rank-5 torch.Tensor "
            f"[B, C, F, H, W], got {type(latent).__name__} shape {shape}"
        )
    if latent.shape[0] < 1 or latent.shape[1] < 1 or latent.shape[2] < 1:
        raise InferenceError(
            "stage1 frame-0 latent must have non-empty batch, channel, and "
            f"frame dims, got shape {tuple(latent.shape)}"
        )
    strength = getattr(item, "strength", None)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise InferenceError(
            "stage1 frame-0 strength must be a real number, "
            f"got {type(strength).__name__}: {strength!r}"
        )
    if strength != strength or strength in (float("inf"), float("-inf")):
        raise InferenceError(
            f"stage1 frame-0 strength must be finite, got {strength!r}"
        )


def apply_stage1_guiding_keyframes(data: dict) -> None:
    """Rewrite only stage-1 frame-0 latent-index conditioning to a guiding keyframe.

    Official Lightricks keyframe interpolation uses
    image_conditionings_by_adding_guiding_latent for every key. Spark's
    combined_image_conditionings uses VideoConditionByLatentIndex for frame 0
    and VideoConditionByKeyframeIndex for later keys. This isolates the stage-1
    frame-0 semantic change. It does not modify any other dict entry, so seed,
    images, audio, and prompt tensors stay identical and stage 2 — rebuilt
    later from images — stays on the historical combined path.
    """
    if not isinstance(data, dict):
        raise InferenceError("denoise input must be a dict")
    conditionings = data.get(_STAGE1_CONDITIONINGS_KEY)
    if not isinstance(conditionings, list):
        raise InferenceError(
            "stage1_guiding_keyframes requires stage_1_conditionings to be a list"
        )
    torch_module, latent_type, keyframe_type = _guiding_conditioning_types()
    allowed = (latent_type, keyframe_type)
    frame0_index = None
    for index, item in enumerate(conditionings):
        kind = type(item)
        if kind not in allowed:
            raise InferenceError(
                "stage1_guiding_keyframes expected VideoConditionByLatentIndex "
                "or VideoConditionByKeyframeIndex, got "
                f"{kind.__module__}.{kind.__name__}"
            )
        if kind is not latent_type:
            continue
        if item.latent_idx != 0:
            raise InferenceError(
                "stage1_guiding_keyframes only converts frame-0 "
                f"VideoConditionByLatentIndex, found latent_idx={item.latent_idx!r}"
            )
        if frame0_index is not None:
            raise InferenceError(
                "stage1_guiding_keyframes expected exactly one frame-0 "
                "VideoConditionByLatentIndex"
            )
        _require_frame0_latent(item, torch_module)
        frame0_index = index
    if frame0_index is None:
        raise InferenceError(
            "stage1_guiding_keyframes expected exactly one frame-0 "
            "VideoConditionByLatentIndex, found 0"
        )
    item = conditionings[frame0_index]
    # Same tensor object: guiding tokens must be the encoded frame, not a clone.
    conditionings[frame0_index] = keyframe_type(
        keyframes=item.latent,
        frame_idx=0,
        strength=item.strength,
    )
    log.info(
        "stage1_guiding_keyframes: converted 1 frame-0 "
        "VideoConditionByLatentIndex to VideoConditionByKeyframeIndex "
        "shape=%s strength=%s; stage2/images/seed/audio/prompt untouched",
        tuple(item.latent.shape),
        item.strength,
    )


def maybe_apply_stage1_guiding_keyframes(data: dict, enabled: bool) -> None:
    """Apply the opt-in. False returns without importing conditioning types."""
    if not enabled:
        return
    apply_stage1_guiding_keyframes(data)


# One interior slot is a full latent frame of extra tokens. Eight is already
# denser than the official first-and-last-frame examples and is the shared-GPU
# ceiling: a larger request would be a caller bug, not a quality setting.
GENERATED_KEYFRAMES_MAX = 8


def parse_generated_keyframes(params: dict) -> int:
    """Return the opt-in interior-slot count. Missing means 0.

    Only a real int is valid. ``True`` is an int subclass and must not become 1.
    """
    if "generated_keyframes" not in params:
        return 0
    value = params["generated_keyframes"]
    if type(value) is not int:
        raise InferenceError(
            "generated_keyframes must be an integer, "
            f"got {type(value).__name__}: {value!r}"
        )
    if value < 0 or value > GENERATED_KEYFRAMES_MAX:
        raise InferenceError(
            "generated_keyframes must be in "
            f"[0, {GENERATED_KEYFRAMES_MAX}], got {value}"
        )
    return value


def parse_generated_keyframe_positions(params: dict) -> tuple[int, ...]:
    """Return explicit interior positions. Missing means none.

    A JSON list of real ints only. Bools are rejected. Order must already be
    strictly increasing so a shuffled caller cannot silently change the slot.
    """
    if "generated_keyframe_positions" not in params:
        return ()
    value = params["generated_keyframe_positions"]
    if type(value) is not list:
        raise InferenceError(
            "generated_keyframe_positions must be a list of integers, "
            f"got {type(value).__name__}: {value!r}"
        )
    if len(value) > GENERATED_KEYFRAMES_MAX:
        raise InferenceError(
            "generated_keyframe_positions must contain at most "
            f"{GENERATED_KEYFRAMES_MAX} frames, got {len(value)}"
        )
    positions = []
    for item in value:
        if type(item) is not int:
            raise InferenceError(
                "generated_keyframe_positions entries must be integers, "
                f"got {type(item).__name__}: {item!r}"
            )
        if item < 0:
            raise InferenceError(
                f"generated_keyframe_positions must be non-negative, got {item}"
            )
        positions.append(item)
    if positions != sorted(set(positions)):
        raise InferenceError(
            "generated_keyframe_positions must be strictly increasing, "
            f"got {positions}"
        )
    return tuple(positions)


def resolve_keyframe_positions(
    count: int, positions: tuple[int, ...], num_frames: int
) -> list[int]:
    """Choose even spacing or the explicit interior list, never both."""
    if count and positions:
        raise InferenceError(
            "set generated_keyframes or generated_keyframe_positions, not both"
        )
    if not positions:
        return interior_keyframe_positions(count, num_frames)
    if positions[0] <= 0 or positions[-1] >= num_frames - 1:
        raise InferenceError(
            "generated_keyframe_positions must be interior frames, "
            f"excluding 0 and {num_frames - 1}, got {list(positions)}"
        )
    return list(positions)


# Locked to ltx_pipelines.utils.helpers.evenly_spaced_keyframe_positions.
# Imported as a function only inside the audited worker would also import
# ltx_pipelines.utils, which pulls torchaudio. The formula itself is the
# contract; the unit test fails if the upstream return line changes.
_OFFICIAL_KEYFRAME_POSITION_LINE = (
    "return torch.linspace(0, num_frames - 1, num_keyframes + 2)"
    ".round().to(torch.int64).tolist()[1:-1]"
)


def interior_keyframe_positions(count: int, num_frames: int) -> list[int]:
    """Evenly spaced interior pixel frames, endpoints excluded.

    This is the body of ``evenly_spaced_keyframe_positions``. A count that
    cannot leave both endpoints free is rejected, matching that helper.
    """
    if count < 0:
        raise InferenceError(
            f"generated_keyframes rejected: count must be non-negative, got {count}"
        )
    if count == 0:
        return []
    if num_frames < count + 2:
        raise InferenceError(
            "generated_keyframes rejected: need at least count + 2 frames, "
            f"got count={count}, num_frames={num_frames}"
        )
    try:
        import torch
    except ImportError as exc:
        raise InferenceError(
            f"generated_keyframes requires torch inside the audited worker: {exc}"
        ) from exc
    return (
        torch.linspace(0, num_frames - 1, count + 2)
        .round()
        .to(torch.int64)
        .tolist()[1:-1]
    )


def apply_generated_keyframes(
    data: dict, count: int, positions: tuple[int, ...] = ()
) -> None:
    """Append official interior keyframe slots to stage-1 conditionings.

    Does not rewrite existing items, images, seed, audio, prompt tensors, or
    stage 2. Stage 2 stays the historical combined-image refine of the
    upscaled stage-1 latent, matching ``ti2vid_two_stages``.
    """
    if count == 0 and not positions:
        return
    if not isinstance(data, dict):
        raise InferenceError("denoise input must be a dict")
    conditionings = data.get(_STAGE1_CONDITIONINGS_KEY)
    if not isinstance(conditionings, list):
        raise InferenceError(
            "generated_keyframes requires stage_1_conditionings to be a list"
        )
    num_frames = data.get("num_frames")
    if type(num_frames) is not int or num_frames < 1:
        raise InferenceError(
            "generated_keyframes requires num_frames to be a positive integer, "
            f"got {num_frames!r}"
        )
    chosen = resolve_keyframe_positions(count, positions, num_frames)
    try:
        from ltx_core.conditioning.types.keyframe_slots import (
            VideoGeneratedKeyframeSlots,
        )
    except ImportError as exc:
        raise InferenceError(
            "generated_keyframes requires VideoGeneratedKeyframeSlots "
            f"inside the audited worker: {exc}"
        ) from exc
    try:
        slot = VideoGeneratedKeyframeSlots(pixel_frame_indices=chosen)
    except ValueError as exc:
        raise InferenceError(f"generated_keyframes rejected: {exc}") from exc
    before = len(conditionings)
    existing = list(conditionings)
    conditionings.append(slot)
    if conditionings[:before] != existing:
        raise InferenceError(
            "generated_keyframes must append; existing stage-1 items changed"
        )
    log.info(
        "generated_keyframes: appended 1 VideoGeneratedKeyframeSlots "
        "count=%s positions=%s existing_stage1_items=%s; "
        "stage2/images/seed/audio/prompt untouched",
        count,
        tuple(slot.pixel_frame_indices),
        before,
    )


def maybe_apply_generated_keyframes(
    data: dict, count: int, positions: tuple[int, ...] = ()
) -> None:
    """Apply the opt-in. Zero and empty returns without importing slot types."""
    if count == 0 and not positions:
        return
    apply_generated_keyframes(data, count, positions)


@register
class LTX25Denoise1Adapter(GroupAdapter):
    """22B transformer + distilled LoRA + upscaler + VAE decoder, all
    resident. ~49.7GB resident, ~80GB peak during stage-2 refinement.

    Intentionally the ONLY denoise stage for LTX 2.5 — see module docstring.
    """

    model_id = "ltx25-denoise1"

    def __init__(self):
        self._pipeline = None
        self._device: str = "cuda"
        # Serialises the GPU phase only (run_denoise_gpu: stage1 + upscale +
        # stage2 + decode). save_denoise_output's NVENC encode/mux is CPU/
        # ffmpeg work that runs OUTSIDE the lock so it overlaps the next
        # job's GPU phase — same pipelining pattern as ltx2-denoise2.
        self._gpu_lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        self._device = device

        spark_str = str(LTX25_SPARK_DIR)
        if spark_str not in sys.path:
            sys.path.insert(0, spark_str)

        try:
            importlib.import_module("ltx_core")
            importlib.import_module("ltx_pipelines")
        except ImportError as e:
            raise LoadError(f"ltx_core / ltx_pipelines not importable: {e}")

        try:
            FastPipeline = importlib.import_module("video_fast_gpu").FastPipeline

            self._pipeline = FastPipeline()
            log.info(
                "LTX25-denoise1: pre-loading 22B transformer + LoRA + "
                "upscaler + VAE decoder (~49.7GB resident)"
            )
            with HeapTrimGuard():
                # Public preload hook — keeps the resident set loaded across
                # every subsequent chunk instead of re-loading it per job.
                self._pipeline.load_denoise_models()
            log.info("LTX25-denoise1: models resident")
        except Exception as e:
            self._pipeline = None
            raise LoadError(f"Failed to load LTX 2.5 denoise models: {e}") from e

    def unload(self) -> None:
        log.info("Unloading LTX25-denoise1")
        if self._pipeline is not None:
            self._pipeline.unload_denoise_models()
            del self._pipeline
            self._pipeline = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        # Validate params BEFORE the loaded-pipeline check so a malformed job
        # always reports the real defect, never "not loaded".
        raw_scale = params.get("a2v_guidance_scale", 3.0)
        try:
            a2v_guidance_scale = float(raw_scale)
        except (TypeError, ValueError) as e:
            raise InferenceError(
                f"a2v_guidance_scale must be a number, got {raw_scale!r}"
            ) from e
        if a2v_guidance_scale < 1.0:
            raise InferenceError(
                f"a2v_guidance_scale must be >= 1.0, got {a2v_guidance_scale}"
            )
        stage1_guiding_keyframes = parse_stage1_guiding_keyframes(params)
        generated_keyframes = parse_generated_keyframes(params)
        generated_keyframe_positions = parse_generated_keyframe_positions(params)
        if generated_keyframes and generated_keyframe_positions:
            raise InferenceError(
                "set generated_keyframes or generated_keyframe_positions, not both"
            )

        if self._pipeline is None:
            raise InferenceError("LTX 2.5 denoise pipeline not loaded")

        self._check_cancel(cancel_flag)

        encoded_file = params.get("encoded_file")
        if not encoded_file:
            raise InferenceError("encoded_file is required")
        if not Path(encoded_file).exists():
            raise InferenceError(f"encoded_file does not exist: {encoded_file}")

        audio_file = params.get("audio_file")
        if not audio_file or not Path(audio_file).exists():
            raise InferenceError(f"audio_file missing or does not exist: {audio_file}")

        start_time = float(params.get("start_time", 0.0))
        fps = float(params.get("fps", 25.0))
        num_inference_steps = int(params.get("num_inference_steps", 30))

        output_dir.mkdir(parents=True, exist_ok=True)

        def _progress(stage, status, **kw):
            log.info("ltx25 denoise progress: %s/%s %s", stage, status, kw)
            if cancel_flag.is_set():
                raise CancelledException(f"Cancelled during {stage}/{status}")

        try:
            # PHASE 1 (CPU/IO): torch.load encoded.pt, map_location="cpu" —
            # overlaps a concurrent call's GPU phase (same asymmetric
            # map_location rationale as the 2.3 lane's load_denoise1_input;
            # see LTX_CUSTOMIZATIONS.md §G).
            data = self._pipeline.load_denoise_input(encoded_file)
            # Opt-in only. Default false leaves the loaded bundle untouched,
            # including stage-1 latent-index conditioning.
            maybe_apply_stage1_guiding_keyframes(data, stage1_guiding_keyframes)
            maybe_apply_generated_keyframes(
                data, generated_keyframes, generated_keyframe_positions
            )
            self._check_cancel(cancel_flag)

            # PHASE 2 (GPU, locked): stage-1 diffusion (544x960) -> 2x latent
            # upscale -> stage-2 distilled-LoRA refine (1088x1920) -> tiled
            # VAE decode -> mandatory 1080p center-crop. This IS the "denoise1
            # -> denoise2" chain of the 2.3 lane, run in one call because
            # there is only one transformer to keep resident.
            with self._gpu_lock:
                self._check_cancel(cancel_flag)
                frames_np = self._pipeline.run_denoise_gpu(
                    data,
                    num_inference_steps=num_inference_steps,
                    a2v_guidance_scale=a2v_guidance_scale,
                    progress_fn=_progress,
                )

            self._check_cancel(cancel_flag)

            # PHASE 3 (CPU/NVENC, outside the lock): encode_video_nvenc
            # (inside save_denoise_output) strips any audio LTX 2.5 itself
            # produced and muxes the ORIGINAL Suno audio slice — never
            # generated audio. Overlaps the next job's GPU phase.
            result_path = output_dir / "result.mp4"
            self._pipeline.save_denoise_output(
                frames_np,
                str(result_path),
                fps=fps,
                audio_path=audio_file,
                start_time=start_time,
            )
        except (CancelledException, InferenceError):
            raise
        except Exception as e:
            raise InferenceError(f"ltx25 run_denoise failed: {e}") from e

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return {
            "file": "result.mp4",
            "format": "mp4",
        }

    def estimate_time(self, params: dict) -> float:
        # One call covers what the 2.3 lane splits across denoise1 (180s) +
        # denoise2 (120s); size at least as generously as their sum, with
        # margin for the larger 22B (vs split dev/distilled) forward passes.
        return 320_000.0
