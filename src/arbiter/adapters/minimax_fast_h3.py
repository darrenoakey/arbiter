"""MiniMax FastH3 local video generation adapter — GroupAdapter on the GB10 GPU.

FastH3 is FastVideo's 4-step DMD2 distillation of MiniMax-H3. It reuses the
base H3 modular pipeline, VAE, tokenizer, and Qwen3-VL text encoder; only the
denoiser transformer is the distilled student.

The published preview checkpoint distills the text-to-video-and-audio path
only. Keyframe conditioning is outside that contract, so the adapter rejects
first- or last-image parameters instead of silently running an unsupported
path. Audio-in is also unsupported; FastH3 emits its own audio and callers may
mux a different soundtrack.

The student was trained on four integer points of the shared 1000-step clock:
`[999, 749, 500, 250]`. Diffusers applies each scheduler's own shift and
appends its terminal zero. Passing normalized sigmas, or including zero in the
input ladder, changes the trained schedule and adds an invalid model forward.
This adapter therefore replaces both scheduler calls with those four integer
timesteps.

The NVFP4 text-encoder snapshot is reused from minimax-h3-local
(`/mnt/t9/models/h3-nvfp4-fl2va/text_encoder`). The FastH3 transformer is
quantized once and snapshotted under `/mnt/t9/models/fasth3-nvfp4-fl2va`.

Expected params match VideoGenerateH3Params except that keyframe fields are
rejected. `num_inference_steps` is accepted then ignored: FastH3 is 4-step
only.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError
from arbiter.adapters.minimax_h3_local import (
    H3_BASE_REPO,
    H3_FPS,
    H3_MAX_SECONDS,
    H3_MIN_SECONDS,
    H3_SNAPSHOT_DIR,
    H3_TEXT_ENCODER_FP_MODULES,
    H3_TRANSFORMER_FP_MODULES,
    H3_WORKFLOW,
    MinimaxH3LocalAdapter,
    _nvfp4_config,
    snap_frames,
)
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

FASTH3_REPO = "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"
FASTH3_SNAPSHOT_DIR = Path("/mnt/t9/models/fasth3-nvfp4-fl2va")
FASTH3_STEPS = 4
FASTH3_DMD_STEPS = (999, 749, 500, 250)


def fasth3_timesteps() -> list[int]:
    """Return the four trained points on FastH3's shared 1000-step clock."""
    return list(FASTH3_DMD_STEPS)


def fasth3_steps(_params: dict | None = None) -> int:
    """FastH3 is 4-step only; caller overrides are ignored."""
    return FASTH3_STEPS


def _fasth3_snapshot_ready() -> bool:
    """True when the FastH3 transformer snapshot and shared H3 text encoder exist."""
    return (FASTH3_SNAPSHOT_DIR / "transformer").is_dir() and (
        H3_SNAPSHOT_DIR / "text_encoder"
    ).is_dir()


def _install_trained_timestep_ladder(pipe) -> None:
    """Force both MiniMax schedulers onto the trained four-forward ladder.

    The upstream block always calls ``set_timesteps(num_inference_steps=N)``.
    Passing integer timesteps lets each scheduler apply its configured shift
    and append its own terminal zero exactly once.
    """
    timesteps = fasth3_timesteps()
    orig_video = pipe.scheduler.set_timesteps
    orig_audio = pipe.audio_scheduler.set_timesteps

    def video_set_timesteps(num_inference_steps=None, device=None, **kwargs):
        orig_video(device=device, timesteps=timesteps)

    def audio_set_timesteps(num_inference_steps=None, device=None, **kwargs):
        orig_audio(device=device, timesteps=timesteps)

    pipe.scheduler.set_timesteps = video_set_timesteps
    pipe.audio_scheduler.set_timesteps = audio_set_timesteps


def reject_keyframe_conditioning(params: dict) -> None:
    """Reject image conditioning unsupported by the preview checkpoint."""
    keyframe_fields = (
        "first_image_file",
        "first_image_b64",
        "last_image_file",
        "last_image_b64",
    )
    if any(params.get(field) for field in keyframe_fields):
        raise InferenceError(
            "FastH3 Preview is T2VA-only and does not support first or last keyframes"
        )


@register
class MinimaxFastH3Adapter(MinimaxH3LocalAdapter):
    """FastH3 4-step distilled MiniMax-H3 denoiser on GB10."""

    model_id = "minimax-fast-h3"

    def load(self, device: str = "cuda") -> None:
        """Load fl2va with the FastH3 transformer and the shared NVFP4 text encoder."""
        import torch
        from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
        from transformers import Qwen3VLForConditionalGeneration

        self._device = device
        try:
            self._pipe = ModularPipeline.from_pretrained(H3_BASE_REPO)
            transformer_ready = (FASTH3_SNAPSHOT_DIR / "transformer").is_dir()
            text_encoder_ready = (H3_SNAPSHOT_DIR / "text_encoder").is_dir()
            if transformer_ready:
                log.info("FastH3: loading NVFP4 transformer from %s", FASTH3_SNAPSHOT_DIR)
                transformer = MiniMaxH3Transformer3DModel.from_pretrained(
                    FASTH3_SNAPSHOT_DIR / "transformer",
                    dtype=torch.bfloat16,
                )
            else:
                log.info("FastH3: quantizing transformer from %s to NVFP4", FASTH3_REPO)
                transformer = MiniMaxH3Transformer3DModel.from_pretrained(
                    FASTH3_REPO,
                    subfolder="transformer",
                    dtype=torch.bfloat16,
                    quantization_config=_nvfp4_config(H3_TRANSFORMER_FP_MODULES),
                )
            if text_encoder_ready:
                log.info("FastH3: reusing H3 NVFP4 text encoder from %s", H3_SNAPSHOT_DIR)
                text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                    H3_SNAPSHOT_DIR / "text_encoder",
                    dtype=torch.bfloat16,
                )
            else:
                log.info("FastH3: quantizing H3 text encoder to NVFP4")
                text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                    H3_BASE_REPO,
                    subfolder="text_encoder",
                    dtype=torch.bfloat16,
                    quantization_config=_nvfp4_config(
                        H3_TEXT_ENCODER_FP_MODULES, transformers_flavour=True
                    ),
                )
            self._write_nvfp4_snapshot(transformer, text_encoder)
            self._pipe.update_components(
                transformer=transformer,
                text_encoder=text_encoder,
            )
            self._pipe.load_components(workflow=H3_WORKFLOW, dtype=torch.bfloat16)
            self._pipe.transformer.requires_grad_(False)
            self._pipe.text_encoder.requires_grad_(False)
            self._pipe.transformer.to(device)
            self._pipe.text_encoder.to(device)
            self._pipe.vae.to(device)
            self._pipe.audio_vae.to(device)
            _install_trained_timestep_ladder(self._pipe)
            torch.cuda.empty_cache()
            self._log_memory(torch, "post-load")
            log.info("MiniMax FastH3 pipeline loaded (NVFP4, 4-step ladder, on-device)")
        except LoadError:
            self.unload()
            raise
        except Exception as e:
            self.unload()
            raise LoadError(f"Failed to load FastH3 pipeline: {e}") from e

    def _write_nvfp4_snapshot(self, transformer, text_encoder) -> None:
        """Persist the FastH3 transformer snapshot. Text encoder stays on the H3 path."""
        try:
            FASTH3_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            dest = FASTH3_SNAPSHOT_DIR / "transformer"
            if dest.exists():
                return
            tmp = dest.with_name(dest.name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            log.info("FastH3: writing NVFP4 transformer snapshot")
            transformer.save_pretrained(tmp)
            tmp.rename(dest)
            log.info("FastH3: NVFP4 transformer snapshot complete at %s", dest)
        except Exception:
            log.exception(
                "FastH3: NVFP4 snapshot write failed; next load will retry "
                "on-the-fly quantization"
            )

    def infer(self, params: dict, output_dir: Path, cancel_flag) -> dict:
        """Generate one clip, always on the trained 4-step ladder."""
        reject_keyframe_conditioning(params)
        forced = dict(params)
        forced["num_inference_steps"] = fasth3_steps(params)
        return super().infer(forced, output_dir, cancel_flag)

    def estimate_time(self, params: dict) -> float:
        """4-step distilled denoise is ~2x the local 8-step H3 estimate."""
        duration = int(params.get("duration", 6))
        duration = max(H3_MIN_SECONDS, min(H3_MAX_SECONDS, duration))
        return snap_frames(duration * H3_FPS) * 1000.0
