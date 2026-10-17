"""Voice-bench adapter for Kyutai Moshi (full-duplex speech-to-speech, PyTorch).

Loads mimi codec + Moshi 7B LM in-process (kyutai/moshiko-pytorch-bf16) and runs
the canonical LMGen streaming loop at real time (80 ms mimi frames @ 24 kHz).
Barge-in is implicit: user audio codes enter the LM at 12.5 Hz and steer
generation, exactly as in production serving.

Benchmark scenarios (see voice_bench_common): eou / interrupt / talkover.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register
from arbiter.adapters import voice_bench_common as vc

log = logging.getLogger(__name__)

FRAME_DUR_S = vc.FRAME_24K / 24000.0  # 80 ms
PRIME_FRAMES = 13  # ~1 s silence prefix to prime the stream
TEXT_PAD_TOKENS = (0, 3)


@register
class VoiceBenchMoshiAdapter(ModelAdapter):
    model_id = "voice-bench-moshi"

    def __init__(self) -> None:
        self._mimi = None
        self._lm_gen = None
        self._sp = None
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        import os

        # Weights for this stack were pre-downloaded to the root-disk cache;
        # the login-shell HF_HOME points at the (full) T9 cache, so force ours.
        os.environ["HF_HOME"] = "/home/darren/hf-voicebench"
        import torch
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders, LMGen
        import sentencepiece as spm

        log.info("Loading Moshi (moshiko pytorch bf16) + mimi ...")
        repo = loaders.DEFAULT_REPO  # kyutai/moshiko-pytorch-bf16
        mimi_path = hf_hub_download(repo, loaders.MIMI_NAME)
        lm_path = hf_hub_download(repo, loaders.MOSHI_NAME)
        tok_path = hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME)

        self._mimi = loaders.get_mimi(mimi_path, device=device)
        self._mimi.set_num_codebooks(8)
        lm = loaders.get_moshi_lm(lm_path, device=device)
        self._lm_gen = LMGen(lm, temp=0.8, temp_text=0.7)
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(tok_path)
        self._device = device
        log.info("Moshi ready.")

    def unload(self) -> None:
        log.info("Unloading Moshi.")
        self._lm_gen = None
        self._mimi = None
        self._sp = None
        self._cleanup_gpu()

    # ------------------------------------------------------------------ core

    @contextmanager
    def _session(self):
        import torch

        # reset_streaming() asserts a live streaming state, so reset inside
        # the context (freshly loaded model has no state yet).
        with torch.no_grad(), self._mimi.streaming(1), self._lm_gen.streaming(1):
            self._mimi.reset_streaming()
            self._lm_gen.reset_streaming()
            yield

    def _step(self, frame: np.ndarray):
        """One 80 ms real-time step. Returns (text_piece, wav_frame or None)."""
        import torch

        t0 = time.monotonic()
        x = torch.from_numpy(frame).to(self._device)[None, None, :]
        codes = self._mimi.encode(x)
        tokens = self._lm_gen.step(codes[:, :, 0:1])
        wav = None
        text_piece = ""
        if tokens is not None:
            wav_t = self._mimi.decode(tokens[:, 1:])
            wav = wav_t[0, 0].float().cpu().numpy()
            tid = int(tokens[0, 0, 0].item())
            if tid not in TEXT_PAD_TOKENS:
                text_piece = self._sp.id_to_piece(tid).replace("▁", " ")
        vc.pace(t0, FRAME_DUR_S)
        return text_piece, wav

    def _prime(self) -> None:
        sil = np.zeros(vc.FRAME_24K, dtype=np.float32)
        for _ in range(PRIME_FRAMES):
            self._step(sil)

    # ------------------------------------------------------------- scenarios

    def _run_eou(self, utts: dict, ev: vc.Events, cancel) -> dict:
        u1 = vc.pad_to(utts["u1"], int(np.ceil(len(utts["u1"]) / vc.FRAME_24K)) * vc.FRAME_24K)
        with self._session():
            self._prime()
            t_user_end = None
            for f in vc.frames_of(u1, vc.FRAME_24K):
                self._check_cancel(cancel)
                t_start = time.monotonic()
                piece, wav = self._step(f)
                t_user_end = t_start + FRAME_DUR_S
            ev.add("user_end")
            # Listen until the model speaks.
            t_first_audio = None
            audio_parts = []
            transcript = ""
            silent_run = 0
            n_watch = int(20.0 / FRAME_DUR_S)
            for i in range(n_watch):
                self._check_cancel(cancel)
                t_start = time.monotonic()
                piece, wav = self._step(np.zeros(vc.FRAME_24K, dtype=np.float32))
                transcript += piece
                if wav is not None:
                    audio_parts.append(wav)
                    if vc.is_active(wav):
                        if t_first_audio is None:
                            t_first_audio = t_start
                            ev.add("first_audio")
                        silent_run = 0
                    else:
                        silent_run += 1
                else:
                    silent_run += 1
                if t_first_audio is not None and silent_run >= int(4.0 / FRAME_DUR_S):
                    break
        audio = np.concatenate(audio_parts) if audio_parts else None
        return {
            "eou_to_first_audio_ms": round((t_first_audio - t_user_end) * 1000.0, 1),
            "transcript": transcript.strip(),
            "_audio": audio,
            "_sr": 24000,
        }

    def _run_interrupt(self, utts: dict, ev: vc.Events, cancel) -> dict:
        u2 = utts["u2"]
        u3 = utts["u3"]
        with self._session():
            self._prime()
            # Feed u2, catch first response frame.
            t_user_end = None
            t_resp = None
            for f in vc.frames_of(u2, vc.FRAME_24K):
                self._check_cancel(cancel)
                t_start = time.monotonic()
                piece, wav = self._step(vc.pad_to(f, vc.FRAME_24K))
                t_user_end = t_start + FRAME_DUR_S
                if wav is not None and vc.is_active(wav) and t_resp is None:
                    t_resp = t_start
                    ev.add("response_start")
            # Let the response run ~2 s.
            spoken_frames = 1
            while spoken_frames < 25:
                self._check_cancel(cancel)
                piece, wav = self._step(np.zeros(vc.FRAME_24K, dtype=np.float32))
                if wav is not None and vc.is_active(wav):
                    spoken_frames += 1
            # Barge in with u3.
            t_barge = None
            for f in vc.frames_of(u3, vc.FRAME_24K):
                self._check_cancel(cancel)
                t_start = time.monotonic()
                piece, wav = self._step(vc.pad_to(f, vc.FRAME_24K))
                if t_barge is None:
                    t_barge = t_start
                    ev.add("barge_in")
            # Watch for stop + new response.
            t_stop = None
            t_new_audio = None
            transcript_after = ""
            audio_parts = []
            post_barge_active_ends = []
            for i in range(int(40.0 / FRAME_DUR_S)):
                self._check_cancel(cancel)
                t_start = time.monotonic()
                piece, wav = self._step(np.zeros(vc.FRAME_24K, dtype=np.float32))
                transcript_after += piece
                if wav is not None:
                    audio_parts.append(wav)
                active = wav is not None and vc.is_active(wav)
                if active:
                    if t_stop is not None:
                        t_new_audio = t_start
                        ev.add("new_response_start")
                        break
                    post_barge_active_ends.append(t_start + FRAME_DUR_S)
                else:
                    if t_stop is None and post_barge_active_ends and t_start > t_barge + 0.5:
                        t_stop = post_barge_active_ends[-1]
                        ev.add("assistant_stopped")
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

    def _run_talkover(self, utts: dict, ev: vc.Events, cancel) -> dict:
        u2 = utts["u2"]
        u4 = utts["u4"]
        with self._session():
            self._prime()
            t_resp = None
            for f in vc.frames_of(u2, vc.FRAME_24K):
                self._check_cancel(cancel)
                t_start = time.monotonic()
                piece, wav = self._step(vc.pad_to(f, vc.FRAME_24K))
                if wav is not None and vc.is_active(wav) and t_resp is None:
                    t_resp = t_start
                    break
            spoken_frames = 1
            while spoken_frames < 25:
                self._check_cancel(cancel)
                piece, wav = self._step(np.zeros(vc.FRAME_24K, dtype=np.float32))
                if wav is not None and vc.is_active(wav):
                    spoken_frames += 1
            # Talk over with u4.
            t_probe = None
            t_probe_end = None
            for f in vc.frames_of(u4, vc.FRAME_24K):
                self._check_cancel(cancel)
                t_start = time.monotonic()
                piece, wav = self._step(vc.pad_to(f, vc.FRAME_24K))
                if t_probe is None:
                    t_probe = t_start
                    ev.add("probe_start")
                t_probe_end = t_start + FRAME_DUR_S
            # Observe.
            transcript_after = ""
            audio_during = []
            audio_after = []
            paused = False
            for i in range(int(25.0 / FRAME_DUR_S)):
                self._check_cancel(cancel)
                t_start = time.monotonic()
                piece, wav = self._step(np.zeros(vc.FRAME_24K, dtype=np.float32))
                transcript_after += piece
                if wav is not None:
                    if t_start < t_probe_end:
                        audio_during.append(wav)
                    else:
                        audio_after.append(wav)
                        if not vc.is_active(wav):
                            paused = True
        during_s = sum(len(a) for a in audio_during) / 24000.0
        after_s = sum(len(a) for a in audio_after) / 24000.0
        return {
            "mentions_probe": "purple" in transcript_after.lower(),
            "assistant_audio_during_probe_s": round(during_s, 3),
            "assistant_audio_after_probe_s": round(after_s, 3),
            "paused_after_probe": paused,
            "transcript_after": transcript_after.strip(),
            "_audio": np.concatenate(audio_after) if audio_after else None,
            "_sr": 24000,
        }

    # ---------------------------------------------------------------- arbiter

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        scenario = params.get("scenario", "eou")
        run = int(params.get("run", 0))
        utts = vc.resolve_utterances(params)
        ev = vc.Events()
        ev.add("scenario_start", scenario=scenario, run=run)

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
                output_dir, f"moshi_{scenario}_r{run}", audio, sr,
                out.get("transcript") or out.get("transcript_after") or "",
            )
        return {
            "system": "moshi",
            "scenario": scenario,
            "run": run,
            "files": files,
            **out,
        }

    def estimate_time(self, params: dict) -> float:
        return {"eou": 45000.0, "interrupt": 90000.0, "talkover": 90000.0}.get(
            params.get("scenario", "eou"), 60000.0
        )
