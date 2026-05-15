"""EchoMimicV3 (Flash) talking-head adapter — single portrait + audio -> video.

EchoMimicV3 (antgroup, AAAI 2026, Apache-2.0): a 1.3B Wan2.1-Fun-based
audio-driven portrait animator. A modern, much higher-quality replacement
for the 2023 SadTalker — added as a NEW model (`echomimic`); SadTalker is
left untouched.

EchoMimicV3 ships only a CLI (`infer_flash.py`) with a large bespoke
pipeline (WanFunInpaintAudioPipeline + custom transformer/VAE/T5/CLIP +
wav2vec2). Rather than reverse-engineer load-once, this adapter shells out
to `infer_flash.py` in the isolated `venvs/echomimic` env (torch 2.12 /
diffusers 0.38 / transformers 5.8) with `cwd` = the vendored repo. The
model reloads per job (~1-2 min) — fine for a minutes-long, low-QPS
talking-head task and far more robust than re-implementing the pipeline.

Invoked via `{"type":"talking-head","model":"echomimic","params":{...}}`.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError, LoadError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

ECHO_DIR = Path("/home/darren/src/talking-head/echomimic_v3")
ECHO_PY = Path("/home/darren/src/arbiter/venvs/echomimic/bin/python")
W = ECHO_DIR / "weights" / "flash"
BASE_MODEL = W / "Wan2.1-Fun-V1.1-1.3B-InP"
TRANSFORMER = W / "transformer" / "diffusion_pytorch_model.safetensors"
WAV2VEC = W / "chinese-wav2vec2-base"
CONFIG = ECHO_DIR / "config" / "config.yaml"


@register
class EchoMimicV3Adapter(ModelAdapter):
    model_id = "echomimic"

    def __init__(self):
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        # Subprocess-per-job model: load() only validates the install.
        # The heavy ~1.3B+T5+CLIP load happens inside infer()'s subprocess;
        # the memory manager budgets memory_gb for the whole job duration.
        for p in (ECHO_PY, ECHO_DIR / "infer_flash.py", CONFIG, BASE_MODEL,
                  TRANSFORMER, WAV2VEC):
            if not Path(p).exists():
                raise LoadError(f"EchoMimicV3 install incomplete, missing: {p}")
        self._device = device
        log.info("EchoMimicV3 ready (subprocess infer_flash.py; cwd=%s).", ECHO_DIR)

    def unload(self) -> None:
        log.info("Unloading EchoMimicV3 (no resident state).")
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        self._check_cancel(cancel_flag)
        try:
            image_bytes = self._resolve_media(params, "image")
        except Exception as e:
            raise InferenceError(f"Failed to resolve image: {e}") from e
        try:
            audio_bytes = self._resolve_media(params, "audio")
        except Exception as e:
            raise InferenceError(f"Failed to resolve audio: {e}") from e

        prompt = params.get("prompt") or "A person is speaking, natural expression."
        steps = int(params.get("steps", 5))                    # README: 5 = talking-head
        seed = int(params.get("seed", 43))
        a_cfg = float(params.get("audio_guidance_scale", 2.0))  # 1.8-2 best lip-sync
        t_cfg = float(params.get("guidance_scale", 4.5))        # 3-6 optimal
        fps = int(params.get("fps", 25))
        # Cap; the script auto-trims to the audio duration (min(audio*fps, cap)).
        video_length = int(params.get("video_length", 1000))

        tmp = tempfile.mkdtemp(prefix="echomimic_")
        img_p = os.path.join(tmp, "input.png")
        aud_p = os.path.join(tmp, "input.wav")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)

        try:
            with open(img_p, "wb") as f:
                f.write(image_bytes)
            with open(aud_p, "wb") as f:
                f.write(audio_bytes)

            self._check_cancel(cancel_flag)

            cmd = [
                str(ECHO_PY), "infer_flash.py",
                "--image_path", img_p,
                "--audio_path", aud_p,
                "--prompt", prompt,
                "--num_inference_steps", str(steps),
                "--config_path", "config/config.yaml",
                "--model_name", str(BASE_MODEL),
                "--ckpt_idx", "50000",
                "--transformer_path", str(TRANSFORMER),
                "--wav2vec_model_dir", str(WAV2VEC),
                "--sampler_name", "Flow_Unipc",
                "--save_path", out_dir,
                "--use_un_ip_mask",
                "--audio_guidance_scale", str(a_cfg),
                "--guidance_scale", str(t_cfg),
                "--video_length", str(video_length),
                "--fps", str(fps),
                "--seed", str(seed),
            ]
            log.info("EchoMimicV3: %s", " ".join(cmd[2:]))
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ECHO_DIR)
            env.pop("CLAUDECODE", None)  # don't leak into the child

            proc = subprocess.Popen(
                cmd, cwd=str(ECHO_DIR), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            out_lines: list[str] = []
            while True:
                if cancel_flag.is_set():
                    proc.kill()
                    raise InferenceError("EchoMimicV3 cancelled")
                line = proc.stdout.readline() if proc.stdout else ""
                if line:
                    out_lines.append(line.rstrip())
                    if len(out_lines) % 25 == 0:
                        log.info("EchoMimicV3[..]: %s", line.rstrip()[:160])
                elif proc.poll() is not None:
                    break
            rc = proc.wait()
            tail = "\n".join(out_lines[-25:])
            if rc != 0:
                raise InferenceError(f"infer_flash.py exited {rc}. Tail:\n{tail}")

            mp4s = sorted(Path(out_dir).rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
            mp4s = [m for m in mp4s if not m.name.endswith("_tmp.mp4")] or mp4s
            if not mp4s:
                raise InferenceError(f"EchoMimicV3 produced no mp4. Tail:\n{tail}")
            result = str(mp4s[-1])

            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", result],
                capture_output=True, text=True, timeout=30,
            )
            dur = float(probe.stdout.strip()) if probe.stdout.strip() else 0.0
            if dur < 0.5:
                raise InferenceError(f"EchoMimicV3 output too short ({dur:.2f}s)")

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "result.mp4"
            shutil.copy2(result, str(out_path))
            with open(out_path, "rb") as fh:
                os.fsync(fh.fileno())
            w, h = self._probe_dims(str(out_path))
            log.info("EchoMimicV3 done: %.1fs %dx%d", dur, w, h)
            return {
                "format": "mp4", "file": "result.mp4",
                "width": w, "height": h, "duration_seconds": round(dur, 2),
            }
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"EchoMimicV3 inference failed: {e}") from e
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def estimate_time(self, params: dict) -> float:
        # ~5-step flash; dominated by per-job model load (~90s) + a few s/sec audio.
        dur = 6.0
        af = params.get("audio_file")
        if af and Path(af).is_file():
            try:
                dur = Path(af).stat().st_size / 32000
            except Exception:
                pass
        return 90_000.0 + dur * 4000.0

    @staticmethod
    def _probe_dims(path: str) -> tuple[int, int]:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
                capture_output=True, text=True, timeout=10,
            )
            p = r.stdout.strip().split("x")
            if len(p) == 2:
                return int(p[0]), int(p[1])
        except Exception:
            pass
        return 768, 768
