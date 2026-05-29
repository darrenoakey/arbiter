"""Wan2.2-I2V-A14B first+last-frame adapter — start image + end image -> clip.

Wan-AI/Wan2.2-I2V-A14B-Diffusers (Apache-2.0): Alibaba's 27B-MoE (14B active)
image-to-video model. The diffusers WanImageToVideoPipeline accepts BOTH
`image` (first frame) and `last_image` (last frame), giving first-last-frame
interpolation — the exact contract the music-video pipeline's boundary
keyframe pairs need. Added as a NEW model (`wan-flf`); LTX and wan-s2v
untouched. This is the WAN renderer behind ltx2's `--renderer wan`.

RESIDENT, in-process (NOT subprocess): the ~80GB MoE takes ~25 min to load off
the CIFS share, so reloading per chunk is untenable for a 40-chunk render.
load() builds the pipeline once and infer() reuses it (mirrors
ltx2_denoise2). Runs in the flash-attn-free wans2v venv (torch 2.12 /
diffusers 0.38, last_image supported) — selected via the model's `worker_cmd`
in config (venvs/wans2v/bin/python -m arbiter.worker_main wan-flf). With one
resident worker, a whole song's chunks pay the load cost once.

Notes for this box (same constraints as wan_s2v):
- Weights live on the CIFS share (/mnt/arbiter-store) — spark root is full.
- Blackwell: no flash-attn wheel; Wan falls back to torch SDPA (same quality).
- GB10 mmap->cuda is pathologically slow; _pipe_to_cuda_cloned (base) clones
  each tensor before the device move (the documented fix).
- No audio: WAN has no audio conditioning. ltx2 muxes audio at assembly and
  gets lip-sync from the latentsync stage.

Invoked via {"type":"video-generate","model":"wan-flf","params":{...}} with
start_image_file + end_image_file + description + num_frames + width + height + fps.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

MODEL_DIR = Path("/mnt/arbiter-store/models/Wan2.2-I2V-A14B-Diffusers")
NEG_PROMPT = ("blurry, low quality, distorted, deformed, extra limbs, extra "
             "fingers, watermark, text, jpeg artifacts, oversaturated, flickering")


@register
class WanFlfAdapter(ModelAdapter):
    model_id = "wan-flf"

    def __init__(self):
        self._device = "cuda"
        self._pipe = None
        self._gpu_lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        if not (MODEL_DIR / "model_index.json").exists():
            raise LoadError(f"Wan2.2-I2V-A14B weights missing: {MODEL_DIR}")
        self._device = device
        try:
            import torch
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            log.info("wan-flf: loading Wan2.2-I2V-A14B (resident, ~80GB, slow off share)...")
            vae = AutoencoderKLWan.from_pretrained(
                str(MODEL_DIR), subfolder="vae", torch_dtype=torch.float32,
            )
            # Load with device_map="balanced": accelerate streams each shard
            # straight to the GPU, so we never materialise a full CPU copy.
            # A plain CPU-load-then-.to(cuda) doubles peak memory on the GB10's
            # UNIFIED 128GB (full CPU copy + GPU copy) and NVRM-OOMs at ~77GB
            # mid-move (observed crash-loop). "balanced" places the heavy
            # components — transformer, transformer_2, text_encoder — all on
            # cuda:0 (verified), giving full-speed resident inference.
            pipe = WanImageToVideoPipeline.from_pretrained(
                str(MODEL_DIR), vae=vae, torch_dtype=torch.bfloat16,
                device_map="balanced",
            )
            # device_map leaves the separately-loaded fp32 VAE on CPU; move it
            # to the GPU so VAE-decode matches the cuda latents (otherwise:
            # "Input type cuda vs weight cpu"). The fp32 VAE is small.
            pipe.vae.to(device)
            self._pipe = pipe
            log.info("wan-flf: model resident (balanced, vae->%s)", device)
        except Exception as e:
            self._pipe = None
            raise LoadError(f"Failed to load wan-flf: {e}") from e

    def unload(self) -> None:
        log.info("Unloading wan-flf.")
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        if self._pipe is None:
            raise InferenceError("wan-flf pipeline not loaded")
        self._check_cancel(cancel_flag)

        try:
            img0 = self._resolve_image_from(params, "start_image")
            img1 = self._resolve_image_from(params, "end_image")
        except Exception as e:
            raise InferenceError(f"Failed to resolve start/end images: {e}") from e

        prompt = params.get("description") or params.get("prompt") or "cinematic shot"
        width = int(params.get("width", 1280))
        height = int(params.get("height", 720))
        num_frames = int(params.get("num_frames", 81))
        if (num_frames - 1) % 4 != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
        seed = int(params.get("seed", 42))
        steps = int(params.get("steps", 40))
        guidance = float(params.get("guidance_scale", 5.0))
        target_fps = int(params.get("fps", 24))
        start_t = float(params.get("start_time", 0.0))
        end_t = float(params.get("end_time", 0.0))
        target_dur = max(0.1, end_t - start_t) if end_t > start_t else num_frames / 16.0

        img0 = img0.resize((width, height))
        img1 = img1.resize((width, height))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="wanflf_")
        raw_p = os.path.join(tmp, "raw.mp4")
        out_path = output_dir / "result.mp4"
        try:
            import torch
            from diffusers.utils import export_to_video

            self._check_cancel(cancel_flag)
            log.info("wan-flf: %dx%d nf=%d steps=%d -> %.2fs@%dfps",
                     width, height, num_frames, steps, target_dur, target_fps)
            with self._gpu_lock:
                self._check_cancel(cancel_flag)
                gen = torch.Generator(device=self._device).manual_seed(seed)
                frames = self._pipe(
                    image=img0, last_image=img1, prompt=prompt,
                    negative_prompt=NEG_PROMPT, height=height, width=width,
                    num_frames=num_frames, guidance_scale=guidance,
                    num_inference_steps=steps, generator=gen,
                ).frames[0]
            export_to_video(frames, raw_p, fps=16)

            # Normalise to the pipeline's exact chunk: retime to target_dur, set
            # target fps, scale to exact dims — interchangeable with an LTX chunk.
            wan_dur = num_frames / 16.0
            setpts = max(0.1, target_dur / wan_dur)
            vf = (f"setpts={setpts:.5f}*PTS,fps={target_fps},"
                  f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                  f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
            norm = subprocess.run(
                ["ffmpeg", "-y", "-i", raw_p, "-vf", vf,
                 "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                 "-an", str(out_path)],
                capture_output=True, text=True, timeout=300,
            )
            if norm.returncode != 0 or not out_path.is_file():
                raise InferenceError(f"ffmpeg normalise failed: {norm.stderr[-400:]}")
            with open(out_path, "rb") as fh:
                os.fsync(fh.fileno())
            log.info("wan-flf done: %dx%d %.2fs@%dfps", width, height, target_dur, target_fps)
            return {
                "format": "mp4", "file": "result.mp4",
                "width": width, "height": height, "fps": target_fps,
                "duration_seconds": round(target_dur, 2),
            }
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"wan-flf inference failed: {e}") from e
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _resolve_image_from(params: dict, key: str):
        """Resolve a PIL RGB image from params[key+'_file'] (or base64)."""
        import io
        from PIL import Image
        raw = ModelAdapter._resolve_media(params, key)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def estimate_time(self, params: dict) -> float:
        # ~40-step diffusion on a resident 27B-MoE (SDPA). Load is amortised
        # across a render; per-chunk gen dominates here.
        return 300_000.0
