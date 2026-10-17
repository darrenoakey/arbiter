"""Voice-bench adapter for OpenBMB MiniCPM-o-4.5 (full-duplex omni model).

Loads the HF custom-code model on CUDA and drives the MiniCPMODuplex streaming
API (1-second slices: streaming_prefill + streaming_generate per slice). Input
audio 16 kHz mono float32; output audio 24 kHz mono float32 numpy.

Benchmark scenarios (see voice_bench_common): eou / interrupt / talkover.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

# Workers inherit HF_HOME=/mnt/t9/cache/huggingface from the arbiter service
# env (full disk, partial snapshot). huggingface_hub freezes HF_HOME at import
# time, so this MUST run at module import — before any hub import — not inside
# load(). The complete MiniCPM-o-4_5 snapshot lives in the root-disk
# voice-bench cache (the default ~/.cache/huggingface copy was wiped).
os.environ["HF_HOME"] = "/home/darren/hf-voicebench"

import numpy as np

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register
from arbiter.adapters import voice_bench_common as vc

log = logging.getLogger(__name__)

HF_REPO = "openbmb/MiniCPM-o-4_5"
SYSTEM_PROMPT = (
    "You are a helpful voice assistant in a latency benchmark. "
    "Answer in one or two short spoken sentences, then go silent."
)
SILENCE_1S = np.zeros(vc.SR_IN_MINICPMO, dtype=np.float32)


@register
class VoiceBenchMiniCPMOAdapter(ModelAdapter):
    model_id = "voice-bench-minicpmo"

    def __init__(self) -> None:
        self._model = None
        self._dm = None
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        import glob

        import torch
        from transformers import AutoModel

        log.info("Loading MiniCPM-o-4.5 (audio-only, sdpa, bf16) ...")
        # Load from the local root-disk snapshot: the full repo including
        # assets/token2wav is already there, so init_tts()'s _ensure_asset_dir
        # resolves locally and never touches the hub (the login-shell HF_HOME
        # points at the full /mnt/t9 volume).
        snaps = sorted(
            glob.glob(
                "/home/darren/hf-voicebench/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/"
            )
        )
        if not snaps:
            raise InferenceError("MiniCPM-o-4_5 snapshot not found on spark")
        model_path = snaps[-1]
        log.info("Using MiniCPM-o snapshot %s", model_path)
        self._model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=False,
            init_audio=True,
            init_tts=True,
        )
        self._model.eval()
        self._model.to(device)
        self._model.init_tts()
        self._dm = self._model.as_duplex()
        self._device = device
        # Reference voice for the duplex stream: without it prepare() never
        # initialises the token2wav voice cache and generate() stays silent.
        import soundfile as sf

        ref_candidates = sorted(
            glob.glob(f"{model_path}assets/*ref_audio*.wav")
            + glob.glob(f"{model_path}assets/*.wav")
        )
        self._ref_audio = None
        self._ref_wav_path = None
        if ref_candidates:
            ref, sr = sf.read(ref_candidates[0], dtype="float32")
            self._ref_audio = vc.resample_linear(
                np.asarray(ref, dtype=np.float32).reshape(-1), sr, vc.SR_IN_MINICPMO
            )
            self._ref_wav_path = ref_candidates[0]
            log.info("Using reference voice %s", ref_candidates[0])
        else:
            log.warning("No reference wav found in snapshot assets")
        log.info("MiniCPM-o-4.5 ready.")

    def unload(self) -> None:
        log.info("Unloading MiniCPM-o-4.5.")
        self._dm = None
        self._model = None
        self._cleanup_gpu()

    # ------------------------------------------------------------------ core

    def _reset_session(self) -> None:
        # ref_audio embeds the voice prompt; prompt_wav_path initialises the
        # token2wav cache — without the cache the duplex stream stays silent.
        self._dm.prepare(
            prefix_system_prompt=SYSTEM_PROMPT,
            ref_audio=self._ref_audio,
            prompt_wav_path=self._ref_wav_path,
        )

    def _slice(self, pcm16k: np.ndarray):
        """Feed one 1-second 16 kHz slice and pull one 1-second output slice.

        Returns (result_dict, t_slice_start, t_slice_end). Paced to real time:
        the slice represents 1.0 s of audio, so the wall clock advances 1 s
        per slice (plus any model overshoot — measured as cost_* and RTF).
        """
        t0 = time.monotonic()
        self._dm.streaming_prefill(audio_waveform=pcm16k)
        res = self._dm.streaming_generate(temperature=0.7)
        vc.pace(t0, 1.0)
        return res, t0, t0 + 1.0

    def _res_audio(self, res) -> np.ndarray | None:
        wav = res.get("audio_waveform")
        if wav is None:
            return None
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------- scenarios

    def _run_eou(self, utts: dict, ev: vc.Events, cancel: threading.Event) -> dict:
        self._reset_session()
        u1 = vc.resample_linear(utts["u1"], 24000, vc.SR_IN_MINICPMO)
        chunks = [vc.pad_to(c, vc.SR_IN_MINICPMO) for c in vc.frames_of(u1, vc.SR_IN_MINICPMO)]

        t_user_end = None
        for c in chunks:
            self._check_cancel(cancel)
            res, t0, t1 = self._slice(c)
            ev.add("user_slice", dur_s=round(len(c) / vc.SR_IN_MINICPMO, 3))
            t_user_end = t0 + len(c) / vc.SR_IN_MINICPMO
            if res.get("end_of_turn"):
                break

        # Silence until the model speaks (or gives up).
        t_first_audio = None
        audio_parts = []
        transcript = ""
        cost_all = []
        max_silence = 15
        for i in range(max_silence):
            self._check_cancel(cancel)
            res, t0, t1 = self._slice(SILENCE_1S)
            audio = self._res_audio(res)
            cost_all.append(res.get("cost_all", 0.0))
            if audio is not None:
                audio_parts.append(audio)
                if vc.is_active(audio) and t_first_audio is None:
                    t_first_audio = t0
                    ev.add("first_audio")
            transcript += res.get("text") or ""
            if res.get("end_of_turn") and t_first_audio is not None:
                break
        else:
            if t_first_audio is None:
                return {
                    "error": "model never produced audio after EOU",
                    "transcript": transcript.strip(),
                    "slices_watched": max_silence,
                }

        # Let the response finish (bounded).
        for i in range(40):
            self._check_cancel(cancel)
            res, t0, t1 = self._slice(SILENCE_1S)
            audio = self._res_audio(res)
            if audio is not None:
                audio_parts.append(audio)
            transcript += res.get("text") or ""
            if res.get("end_of_turn"):
                break

        audio = np.concatenate(audio_parts) if audio_parts else None
        return {
            "eou_to_first_audio_ms": round((t_first_audio - t_user_end) * 1000.0, 1),
            "transcript": transcript.strip(),
            "model_cost_all_avg_ms": round(
                float(np.mean(cost_all)) * 1000.0 if cost_all else 0.0, 1
            ),
            "_audio": audio,
            "_sr": 24000,
        }

    def _run_interrupt(self, utts: dict, ev: vc.Events, cancel: threading.Event) -> dict:
        self._reset_session()
        u2 = vc.resample_linear(utts["u2"], 24000, vc.SR_IN_MINICPMO)
        u3 = vc.resample_linear(utts["u3"], 24000, vc.SR_IN_MINICPMO)
        u2c = [vc.pad_to(c, vc.SR_IN_MINICPMO) for c in vc.frames_of(u2, vc.SR_IN_MINICPMO)]
        u3c = [vc.pad_to(c, vc.SR_IN_MINICPMO) for c in vc.frames_of(u3, vc.SR_IN_MINICPMO)]

        # User asks u2; wait for the model to start answering.
        t_user_end = None
        started_idx = 0
        response_started = False
        for i, c in enumerate(u2c):
            self._check_cancel(cancel)
            res, t0, t1 = self._slice(c)
            t_user_end = t0 + len(c) / vc.SR_IN_MINICPMO
            audio = self._res_audio(res)
            if audio is not None and vc.is_active(audio):
                response_started = True
                started_idx = i
                ev.add("response_start")
                break
        # Feed remaining u2 audio while the model speaks (real duplex), then
        # keep speaking into silence until the response is ~2 slices old.
        spoken_slices = 1
        for c in u2c[started_idx + 1 :]:
            self._check_cancel(cancel)
            res, t0, t1 = self._slice(c)
            audio = self._res_audio(res)
            if audio is not None and vc.is_active(audio):
                spoken_slices += 1
        for _wait in range(30):  # 30 s cap; model may stay silent
            self._check_cancel(cancel)
            if spoken_slices >= 3:
                break
            res, t0, t1 = self._slice(SILENCE_1S)
            audio = self._res_audio(res)
            if audio is not None and vc.is_active(audio):
                spoken_slices += 1

        # Barge in with u3.
        t_barge = None
        for c in u3c:
            self._check_cancel(cancel)
            res, t0, t1 = self._slice(c)
            if t_barge is None:
                t_barge = t0
                ev.add("barge_in")
            audio = self._res_audio(res)
        # Silence; watch for the model to stop, then respond again.
        t_stop = None
        t_new_audio = None
        transcript_after = ""
        audio_parts = []
        post_barge_active_ends = []
        for i in range(40):
            self._check_cancel(cancel)
            res, t0, t1 = self._slice(SILENCE_1S)
            audio = self._res_audio(res)
            transcript_after += res.get("text") or ""
            active = audio is not None and vc.is_active(audio)
            if audio is not None:
                audio_parts.append(audio)
            if active:
                if t_stop is not None:
                    t_new_audio = t0
                    ev.add("new_response_start")
                    break
                post_barge_active_ends.append(t1)
            else:
                if t_stop is None and post_barge_active_ends and t1 > t_barge + 0.5:
                    t_stop = post_barge_active_ends[-1]
                    ev.add("assistant_stopped")
            if res.get("end_of_turn") and t_stop is not None:
                break

        audio = np.concatenate(audio_parts) if audio_parts else None
        return {
            "interrupt_latency_ms": round((t_stop - t_barge) * 1000.0, 1)
            if t_stop
            else None,
            "respond_latency_ms": round((t_new_audio - t_barge) * 1000.0, 1)
            if t_new_audio
            else None,
            "transcript_after": transcript_after.strip(),
            "_audio": audio,
            "_sr": 24000,
        }

    def _run_talkover(self, utts: dict, ev: vc.Events, cancel: threading.Event) -> dict:
        self._reset_session()
        u2 = vc.resample_linear(utts["u2"], 24000, vc.SR_IN_MINICPMO)
        u4 = vc.resample_linear(utts["u4"], 24000, vc.SR_IN_MINICPMO)
        u2c = [vc.pad_to(c, vc.SR_IN_MINICPMO) for c in vc.frames_of(u2, vc.SR_IN_MINICPMO)]
        u4c = [vc.pad_to(c, vc.SR_IN_MINICPMO) for c in vc.frames_of(u4, vc.SR_IN_MINICPMO)]

        t_user_end = None
        started_idx = 0
        for i, c in enumerate(u2c):
            self._check_cancel(cancel)
            res, t0, t1 = self._slice(c)
            t_user_end = t0 + len(c) / vc.SR_IN_MINICPMO
            audio = self._res_audio(res)
            if audio is not None and vc.is_active(audio):
                started_idx = i
                break
        spoken = 1
        for c in u2c[started_idx + 1 :]:
            self._check_cancel(cancel)
            res, t0, t1 = self._slice(c)
            audio = self._res_audio(res)
            if audio is not None and vc.is_active(audio):
                spoken += 1
        for _wait in range(30):  # 30 s cap; model may stay silent
            self._check_cancel(cancel)
            if spoken >= 3:
                break
            res, t0, t1 = self._slice(SILENCE_1S)
            audio = self._res_audio(res)
            if audio is not None and vc.is_active(audio):
                spoken += 1

        # Talk over the model with u4 (no stop intent).
        t_probe = None
        t_probe_end = None
        for c in u4c:
            self._check_cancel(cancel)
            res, t0, t1 = self._slice(c)
            if t_probe is None:
                t_probe = t0
                ev.add("probe_start")
            t_probe_end = t0 + len(c) / vc.SR_IN_MINICPMO
            audio = self._res_audio(res)
        # Observe behaviour after the probe.
        transcript_after = ""
        audio_after = []
        audio_during = []
        paused = False
        for i in range(25):
            self._check_cancel(cancel)
            res, t0, t1 = self._slice(SILENCE_1S)
            audio = self._res_audio(res)
            transcript_after += res.get("text") or ""
            if audio is not None:
                if t0 < t_probe_end:
                    audio_during.append(audio)
                elif t0 >= t_probe_end:
                    audio_after.append(audio)
                    if not vc.is_active(audio):
                        paused = True
            if res.get("end_of_turn") and t1 > t_probe_end + 3:
                break

        during = np.concatenate(audio_during) if audio_during else np.zeros(1, np.float32)
        after = np.concatenate(audio_after) if audio_after else np.zeros(1, np.float32)
        return {
            "mentions_probe": "purple" in transcript_after.lower(),
            "assistant_audio_during_probe_s": round(
                sum(len(a) for a in audio_during) / 24000.0, 3
            ),
            "assistant_audio_after_probe_s": round(
                sum(len(a) for a in audio_after) / 24000.0, 3
            ),
            "paused_after_probe": paused,
            "transcript_after": transcript_after.strip(),
            "_audio": after,
            "_sr": 24000,
        }

    # ---------------------------------------------------------------- arbiter

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import torch

        self._check_cancel(cancel_flag)
        scenario = params.get("scenario", "eou")
        run = int(params.get("run", 0))
        utts = vc.resolve_utterances(params)
        ev = vc.Events()
        ev.add("scenario_start", scenario=scenario, run=run)

        with torch.no_grad():
            if scenario == "eou":
                out = self._run_eou(utts, ev, cancel_flag)
            elif scenario == "interrupt":
                out = self._run_interrupt(utts, ev, cancel_flag)
            elif scenario == "talkover":
                out = self._run_talkover(utts, ev, cancel_flag)
            else:
                raise InferenceError(f"unknown scenario: {scenario}")

        audio = out.pop("_audio", None)
        sr = out.pop("_sr", 24000)
        files = {}
        if "error" not in out:
            files = vc.save_artifacts(
                output_dir, f"minicpmo_{scenario}_r{run}", audio, sr,
                out.get("transcript") or out.get("transcript_after") or "",
            )
        result = {
            "system": "minicpm-o-4.5",
            "scenario": scenario,
            "run": run,
            "files": files,
            **{k: v for k, v in out.items()},
        }
        return result

    def estimate_time(self, params: dict) -> float:
        return {"eou": 45000.0, "interrupt": 90000.0, "talkover": 90000.0}.get(
            params.get("scenario", "eou"), 60000.0
        )
