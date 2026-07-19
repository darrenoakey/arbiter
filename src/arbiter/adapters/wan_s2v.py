"""Wan2.2-S2V-14B speech-to-video adapter — single portrait + audio -> video.

Wan-AI/Wan2.2-S2V-14B (Apache-2.0, Aug 2025): Alibaba's 14B audio-driven
cinematic video model — the current best-quality open-weights option for
the SadTalker task. Added as a NEW model (`wan-s2v`); SadTalker untouched.
Highest quality, heavy (14B). Sibling of the lighter `echomimic`.

Notes for this box:
- Weights live on the CIFS share (/mnt/arbiter-store, 2.9TB) — the spark
  root is full; do NOT relocate them locally.
- The wans2v venv is intentionally flash-attn-free: flash-attn has no
  Blackwell/torch-2.12 wheel and source-building it is a doomed/wasteful
  long-pole. Wan2.2's wan/modules/attention.py guards the flash_attn
  import and falls back to torch SDPA — slower, identical quality.
- Subprocess wrapper around the vendored generate.py in venvs/wans2v
  (torch 2.12 / transformers 4.51.3 / numpy<2). Model reloads per job
  (~minutes) — fine for a heavy, low-QPS, minutes-long task.
- TORCHINDUCTOR/HF caches are forced onto the share so a compile cache
  can't fill the 98%-full root.

Invoked via {"type":"talking-head","model":"wan-s2v","params":{...}}.
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

WAN_DIR = Path("/home/darren/src/talking-head/Wan2.2")
WAN_PY = Path("/home/darren/src/arbiter/venvs/wans2v/bin/python")
CKPT = Path("/mnt/arbiter-store/models/Wan2.2-S2V-14B")
CACHE = Path("/mnt/arbiter-store/cache/wans2v")  # keep compile/HF cache off root


@register
class WanS2VAdapter(ModelAdapter):
    model_id = "wan-s2v"

    def __init__(self):
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        for p in (WAN_PY, WAN_DIR / "generate.py", CKPT):
            if not Path(p).exists():
                raise LoadError(f"Wan2.2-S2V install incomplete, missing: {p}")
        CACHE.mkdir(parents=True, exist_ok=True)
        self._device = device
        log.info("Wan2.2-S2V ready (subprocess generate.py; ckpt on share).")

    def unload(self) -> None:
        log.info("Unloading Wan2.2-S2V (no resident state).")
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        self._check_cancel(cancel_flag)
        try:
            image_bytes = self._resolve_media(params, "image")
        except Exception as e:
            raise InferenceError(f"Failed to resolve image: {e}") from e
        try:
            audio_bytes = self._resolve_media(params, "audio")
        except Exception as e:
            raise InferenceError(f"Failed to resolve audio: {e}") from e

        prompt = params.get("prompt") or (
            "A person speaking to camera, natural expression and lip movement, "
            "steady shot, realistic lighting."
        )
        size = params.get("size", "1024*704")
        seed = int(params.get("seed", 42))
        # Heavy 14B; bf16 conversion keeps it within budget. Offload only if asked.
        offload = str(params.get("offload_model", "false")).lower() == "true"

        tmp = tempfile.mkdtemp(prefix="wans2v_")
        img_p = os.path.join(tmp, "input.png")
        aud_p = os.path.join(tmp, "input.wav")
        save_file = os.path.join(tmp, "out.mp4")

        try:
            with open(img_p, "wb") as f:
                f.write(image_bytes)
            with open(aud_p, "wb") as f:
                f.write(audio_bytes)

            self._check_cancel(cancel_flag)

            cmd = [
                str(WAN_PY),
                "generate.py",
                "--task",
                "s2v-14B",
                "--ckpt_dir",
                str(CKPT),
                "--image",
                img_p,
                "--audio",
                aud_p,
                "--prompt",
                prompt,
                "--size",
                size,
                "--convert_model_dtype",
                "--base_seed",
                str(seed),
                "--save_file",
                save_file,
            ]
            if offload:
                cmd += ["--offload_model", "True", "--t5_cpu"]
            log.info("Wan2.2-S2V: %s", " ".join(cmd[2:]))

            env = dict(os.environ)
            env["PYTHONPATH"] = str(WAN_DIR)
            env.pop("CLAUDECODE", None)
            # Keep all caches OFF the 98%-full root.
            env["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE / "inductor")
            env["HF_HOME"] = str(CACHE / "hf")
            env["TRITON_CACHE_DIR"] = str(CACHE / "triton")

            proc = subprocess.Popen(
                cmd,
                cwd=str(WAN_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            out_lines: list[str] = []
            while True:
                if cancel_flag.is_set():
                    proc.kill()
                    raise InferenceError("Wan2.2-S2V cancelled")
                line = proc.stdout.readline() if proc.stdout else ""
                if line:
                    out_lines.append(line.rstrip())
                    if len(out_lines) % 25 == 0:
                        log.info("Wan2.2-S2V[..]: %s", line.rstrip()[:160])
                elif proc.poll() is not None:
                    break
            rc = proc.wait()
            tail = "\n".join(out_lines[-30:])
            if rc != 0:
                raise InferenceError(f"generate.py exited {rc}. Tail:\n{tail}")

            result = save_file
            if not os.path.isfile(result):
                mp4s = sorted(Path(tmp).rglob("*.mp4")) or sorted(
                    Path(WAN_DIR).glob("s2v-14B_*.mp4")
                )
                if not mp4s:
                    raise InferenceError(f"Wan2.2-S2V produced no mp4. Tail:\n{tail}")
                result = str(mp4s[-1])

            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    result,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            dur = float(probe.stdout.strip()) if probe.stdout.strip() else 0.0
            if dur < 0.5:
                raise InferenceError(f"Wan2.2-S2V output too short ({dur:.2f}s)")

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "result.mp4"
            shutil.copy2(result, str(out_path))
            with open(out_path, "rb") as fh:
                os.fsync(fh.fileno())
            w, h = self._probe_dims(str(out_path))
            log.info("Wan2.2-S2V done: %.1fs %dx%d", dur, w, h)
            return {
                "format": "mp4",
                "file": "result.mp4",
                "width": w,
                "height": h,
                "duration_seconds": round(dur, 2),
            }
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"Wan2.2-S2V inference failed: {e}") from e
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            for stray in Path(WAN_DIR).glob("s2v-14B_*.mp4"):
                try:
                    stray.unlink()
                except OSError:
                    pass

    def estimate_time(self, params: dict) -> float:
        # 14B, SDPA (no flash-attn) — heavy; per-job model load + slow diffusion.
        dur = 6.0
        af = params.get("audio_file")
        if af and Path(af).is_file():
            try:
                dur = Path(af).stat().st_size / 32000
            except Exception:
                pass
        return 240_000.0 + dur * 30000.0

    @staticmethod
    def _probe_dims(path: str) -> tuple[int, int]:
        try:
            r = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0:s=x",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            p = r.stdout.strip().split("x")
            if len(p) == 2:
                return int(p[0]), int(p[1])
        except Exception:
            pass
        return 1024, 704
