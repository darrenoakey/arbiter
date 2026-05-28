"""Wan2.2-I2V-A14B first+last-frame adapter — start image + end image -> clip.

Wan-AI/Wan2.2-I2V-A14B-Diffusers (Apache-2.0): Alibaba's 27B-MoE (14B active)
image-to-video model. The diffusers WanImageToVideoPipeline accepts BOTH
`image` (first frame) and `last_image` (last frame), giving first-last-frame
interpolation — the exact contract the music-video pipeline's boundary
keyframe pairs need. Added as a NEW model (`wan-flf`); LTX and wan-s2v
untouched. This is the WAN renderer behind ltx2's `--renderer wan`.

Notes for this box (same constraints as wan_s2v):
- Weights live on the CIFS share (/mnt/arbiter-store, ~2.9TB) — spark root
  is full; do NOT relocate them locally.
- Runs in the flash-attn-free wans2v venv (torch 2.12 / diffusers 0.38);
  Wan falls back to torch SDPA on Blackwell — slower, identical quality.
- Subprocess wrapper around an embedded diffusers script (fresh process per
  job, ~80GB resident, frees cleanly) so the heavy model never fights LTX in
  the same process. Model reloads per job — fine for a heavy, low-QPS task.
- GB10 mmap->cuda is pathologically slow; the gen script clones each
  parameter before the .to("cuda") move (the documented base-adapter fix).
- No audio: WAN has no audio conditioning. The ltx2 pipeline muxes audio at
  assembly and gets lip-sync from the latentsync stage.

Invoked via {"type":"video-generate","model":"wan-flf","params":{...}} with
start_image_file + end_image_file + prompt + num_frames + width + height + fps.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

WAN_PY = Path("/home/darren/src/arbiter/venvs/wans2v/bin/python")
MODEL_DIR = Path("/mnt/arbiter-store/models/Wan2.2-I2V-A14B-Diffusers")
CACHE = Path("/mnt/arbiter-store/cache/wanflf")  # keep compile/HF cache off root

# Embedded diffusers generation script. Runs in the wans2v venv as a fresh
# subprocess. argv: model_dir start_img end_img prompt num_frames H W seed
# steps guidance out_path. Exports at WAN's native 16fps; the adapter
# ffmpeg-normalises fps/duration/dims afterwards.
_GEN_SCRIPT = r'''
import sys
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

(md, start_p, end_p, prompt, nf, H, W, seed, steps, guidance, out_p) = sys.argv[1:12]
nf, H, W, seed, steps, guidance = int(nf), int(H), int(W), int(seed), int(steps), float(guidance)

NEG = ("blurry, low quality, distorted, deformed, extra limbs, extra fingers, "
       "watermark, text, jpeg artifacts, oversaturated, flickering")


def _to_cuda_cloned(pipe):
    # GB10 mmap->cuda is ~170MB/s; cloning each tensor first restores normal
    # bandwidth (see arbiter base._pipe_to_cuda_cloned).
    import torch.nn as nn
    seen = set()
    for attr in dir(pipe):
        if attr.startswith("_"):
            continue
        try:
            obj = getattr(pipe, attr, None)
        except Exception:
            continue
        if not isinstance(obj, nn.Module) or id(obj) in seen:
            continue
        seen.add(id(obj))
        for p in obj.parameters(recurse=True):
            if p.data.device.type != "cuda":
                p.data = p.data.clone().to("cuda", non_blocking=False)
        for name, b in list(obj.named_buffers(recurse=True)):
            if b.device.type != "cuda":
                nb = b.clone().to("cuda", non_blocking=False)
                parts = name.split(".")
                mod = obj
                for pn in parts[:-1]:
                    mod = getattr(mod, pn)
                setattr(mod, parts[-1], nb)
    try:
        pipe.to("cuda")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()


vae = AutoencoderKLWan.from_pretrained(md, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(md, vae=vae, torch_dtype=torch.bfloat16)
_to_cuda_cloned(pipe)

img0 = Image.open(start_p).convert("RGB").resize((W, H))
img1 = Image.open(end_p).convert("RGB").resize((W, H))
gen = torch.Generator(device="cuda").manual_seed(seed)

frames = pipe(
    image=img0, last_image=img1, prompt=prompt, negative_prompt=NEG,
    height=H, width=W, num_frames=nf, guidance_scale=guidance,
    num_inference_steps=steps, generator=gen,
).frames[0]

export_to_video(frames, out_p, fps=16)
print("WAN_FLF_OK", out_p, len(frames))
'''


@register
class WanFlfAdapter(ModelAdapter):
    model_id = "wan-flf"

    def __init__(self):
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        for p in (WAN_PY, MODEL_DIR, MODEL_DIR / "model_index.json"):
            if not Path(p).exists():
                raise LoadError(f"Wan2.2-I2V-A14B install incomplete, missing: {p}")
        CACHE.mkdir(parents=True, exist_ok=True)
        self._device = device
        log.info("Wan2.2-I2V-A14B (wan-flf) ready (subprocess diffusers FLF).")

    def unload(self) -> None:
        log.info("Unloading wan-flf (no resident state).")
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        self._check_cancel(cancel_flag)
        try:
            start_bytes = self._resolve_media(params, "start_image")
            end_bytes = self._resolve_media(params, "end_image")
        except Exception as e:
            raise InferenceError(f"Failed to resolve start/end images: {e}") from e

        prompt = params.get("description") or params.get("prompt") or "cinematic shot"
        width = int(params.get("width", 1280))
        height = int(params.get("height", 720))
        # WAN wants 4n+1; default 81 (~5s @16fps). Caller passes a chunk-sized count.
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

        tmp = tempfile.mkdtemp(prefix="wanflf_")
        start_p = os.path.join(tmp, "start.png")
        end_p = os.path.join(tmp, "end.png")
        raw_p = os.path.join(tmp, "raw.mp4")
        script_p = os.path.join(tmp, "gen.py")
        try:
            with open(start_p, "wb") as f:
                f.write(start_bytes)
            with open(end_p, "wb") as f:
                f.write(end_bytes)
            with open(script_p, "w") as f:
                f.write(_GEN_SCRIPT)

            self._check_cancel(cancel_flag)

            cmd = [
                str(WAN_PY), script_p, str(MODEL_DIR), start_p, end_p, prompt,
                str(num_frames), str(height), str(width), str(seed),
                str(steps), str(guidance), raw_p,
            ]
            log.info("wan-flf: %dx%d nf=%d steps=%d -> %.2fs@%dfps",
                     width, height, num_frames, steps, target_dur, target_fps)

            env = dict(os.environ)
            env.pop("CLAUDECODE", None)
            env["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE / "inductor")
            env["HF_HOME"] = str(CACHE / "hf")
            env["TRITON_CACHE_DIR"] = str(CACHE / "triton")

            proc = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            out_lines: list[str] = []
            while True:
                if cancel_flag.is_set():
                    proc.kill()
                    raise InferenceError("wan-flf cancelled")
                line = proc.stdout.readline() if proc.stdout else ""
                if line:
                    out_lines.append(line.rstrip())
                    if len(out_lines) % 25 == 0:
                        log.info("wan-flf[..]: %s", line.rstrip()[:160])
                elif proc.poll() is not None:
                    break
            rc = proc.wait()
            tail = "\n".join(out_lines[-30:])
            if rc != 0 or not os.path.isfile(raw_p):
                raise InferenceError(f"wan-flf gen failed (rc={rc}). Tail:\n{tail}")

            # Normalise to the pipeline's exact chunk: retime to target_dur,
            # set target fps, scale to exact dims — so the chunk is
            # interchangeable with an LTX chunk for concat + audio mux.
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "result.mp4"
            wan_dur = num_frames / 16.0
            setpts = max(0.1, target_dur / wan_dur)
            vf = (f"setpts={setpts:.5f}*PTS,"
                  f"fps={target_fps},"
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
            shutil.rmtree(tmp, ignore_errors=True)

    def estimate_time(self, params: dict) -> float:
        # 27B-MoE, SDPA (no flash-attn), per-job model load + ~40-step diffusion.
        return 600_000.0
