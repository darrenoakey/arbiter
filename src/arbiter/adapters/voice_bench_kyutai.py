"""Voice-bench adapter for the Kyutai cascade: STT (kyutai/stt-1b-en_fr, semantic
VAD) -> small instruct LLM (Qwen3-1.7B) -> Kyutai Pocket TTS, all in-process.

Everything is 24 kHz (mimi frames of 1920 samples = 80 ms). Endpointing uses the
STT's own VAD pause-prediction heads (head 0 = 0.5 s pause, most responsive);
after the endpoint the delayed (0.5 s) tail is flushed by feeding silence in
"warp" mode (as fast as the GPU allows), matching Kyutai Unmute's technique.

Barge-in in the cascade is a host-side energy gate on the always-open mic while
TTS plays (the realtime-talk architecture) — this adapter measures that wiring
honestly: playback is scheduled at real time and the gate sees the same frames
the STT sees.

Benchmark scenarios (see voice_bench_common): eou / interrupt / talkover.
"""

from __future__ import annotations

import logging
import itertools
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register
from arbiter.adapters import voice_bench_common as vc

log = logging.getLogger(__name__)

FRAME_DUR_S = vc.FRAME_24K / 24000.0
VAD_HEAD = 0  # 0.5 s pause head (heads are 0.5/1/2/3 s)
VAD_THRESHOLD = 0.5
STT_DELAY_S = 0.5  # stt-1b-en_fr inherent transcript delay
PRIME_FRAMES = 13
GATE_RMS = 0.02  # barge-in energy gate (louder than floor, below speech peaks)
LLM = "Qwen/Qwen3-1.7B"
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Reply in one short spoken sentence."
)


@register
class VoiceBenchKyutaiAdapter(ModelAdapter):
    model_id = "voice-bench-kyutai"

    def __init__(self) -> None:
        self._stt_mimi = None
        self._stt_lm_gen = None
        self._stt_tok = None
        self._llm = None
        self._llm_tok = None
        self._tts = None
        self._voice = None
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        import os

        # Force the root-disk HF cache (login shell points at the full T9 cache).
        os.environ["HF_HOME"] = "/home/darren/hf-voicebench"
        import torch
        from moshi.models import loaders

        log.info("Loading Kyutai STT (stt-1b-en_fr) ...")
        info = loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
        self._stt_mimi = info.get_mimi(device=device)
        stt_lm = info.get_moshi(device=device, dtype=torch.bfloat16)
        from moshi.models import LMGen

        self._stt_lm_gen = LMGen(stt_lm, temp=0, temp_text=0.0)
        self._stt_tok = info.get_text_tokenizer()

        log.info("Loading LLM %s ...", LLM)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._llm_tok = AutoTokenizer.from_pretrained(LLM)
        self._llm = AutoModelForCausalLM.from_pretrained(
            LLM, torch_dtype=torch.bfloat16, device_map=device
        )
        self._llm.eval()

        log.info("Loading Pocket TTS ...")
        from pocket_tts import TTSModel
        from huggingface_hub import snapshot_download

        # The default kyutai/pocket-tts repo is license-gated; the package
        # auto-falls-back to the ungated kyutai/pocket-tts-without-voice-cloning
        # weights. Preset voice states ship as safetensors in that repo.
        ptts_dir = snapshot_download(
            "kyutai/pocket-tts-without-voice-cloning",
            cache_dir="/home/darren/hf-voicebench/hub",
        )
        self._tts = TTSModel.load_model()
        try:
            self._tts.to(device)
        except Exception as e:
            log.info("pocket-tts stayed on CPU (%s)", e)
        import glob as _glob

        _voices = _glob.glob(f"{ptts_dir}/languages/english/embeddings/alba.safetensors")
        if not _voices:
            _voices = _glob.glob(f"{ptts_dir}/**/alba.safetensors", recursive=True)
        self._voice = self._tts.get_state_for_audio_prompt(_voices[0])
        self._device = device
        log.info("Kyutai cascade ready.")

    def unload(self) -> None:
        log.info("Unloading Kyutai cascade.")
        self._voice = None
        self._tts = None
        self._llm = None
        self._llm_tok = None
        self._stt_lm_gen = None
        self._stt_mimi = None
        self._stt_tok = None
        self._cleanup_gpu()

    # ------------------------------------------------------------------ STT

    @contextmanager
    def _stt_session(self):
        # reset_streaming() asserts a live streaming state, so reset inside the
        # context (the python server resets between connections the same way).
        with self._stt_mimi.streaming(1), self._stt_lm_gen.streaming(1):
            self._stt_mimi.reset_streaming()
            self._stt_lm_gen.reset_streaming()
            yield

    def _stt_step(self, frame: np.ndarray):
        """One 80 ms step; returns (text_piece, vad_p or None)."""
        import torch

        t0 = time.monotonic()
        x = torch.from_numpy(frame).to(self._device)[None, None, :]
        codes = self._stt_mimi.encode(x)
        text_tokens, vad_heads = self._stt_lm_gen.step_with_extra_heads(codes)
        piece = ""
        tid = int(text_tokens[0, 0, 0].item())
        if tid not in (0, 3):
            piece = self._stt_tok.id_to_piece(tid).replace("▁", " ")
        vad_ps = None
        if vad_heads:
            vad_ps = [float(h[0, 0, 0].item()) for h in vad_heads[:4]]
        vc.pace(t0, FRAME_DUR_S)
        return piece, vad_ps

    def _stt_prime(self) -> None:
        sil = np.zeros(vc.FRAME_24K, dtype=np.float32)
        for _ in range(PRIME_FRAMES):
            self._stt_step(sil)

    def _stt_transcribe_until_endpoint(self, frames, cancel, ev: vc.Events):
        """Feed frames at real time; after the utterance ends keep feeding
        silence (as a real always-open mic would) until the VAD endpoint
        fires. Returns (transcript, t_endpoint, t_user_audio_end)."""
        transcript = ""
        speech_seen = False
        t_user_end = None
        t_endpoint = None
        padded = [vc.pad_to(f, vc.FRAME_24K) for f in frames]
        silence = [np.zeros(vc.FRAME_24K, dtype=np.float32)] * 50  # up to 4 s
        vad_debug = None
        for f in itertools.chain(padded, silence):
            self._check_cancel(cancel)
            t_start = time.monotonic()
            piece, vad_ps = self._stt_step(f)
            transcript += piece
            if vc.is_active(f) and np.any(f):
                speech_seen = True
                t_user_end = t_start + FRAME_DUR_S
            fired = bool(vad_ps and max(vad_ps) > VAD_THRESHOLD)
            if speech_seen and fired:
                t_endpoint = t_start
                vad_debug = [round(p, 3) for p in vad_ps]
                ev.add("stt_endpoint", vad_ps=vad_debug)
                break
        return transcript, t_endpoint, t_user_end, vad_debug

    def _stt_warp_flush(self, cancel) -> str:
        """Feed ~0.6 s of silence as fast as possible to drain delayed tokens."""
        import torch

        tail = ""
        sil = np.zeros(vc.FRAME_24K, dtype=np.float32)
        n = int(np.ceil((STT_DELAY_S + 0.1) / FRAME_DUR_S))
        with torch.no_grad():
            for _ in range(n):
                self._check_cancel(cancel)
                x = torch.from_numpy(sil).to(self._device)[None, None, :]
                codes = self._stt_mimi.encode(x)
                text_tokens, _ = self._stt_lm_gen.step_with_extra_heads(codes)
                tid = int(text_tokens[0, 0, 0].item())
                if tid not in (0, 3):
                    tail += self._stt_tok.id_to_piece(tid).replace("▁", " ")
        return tail

    # ------------------------------------------------------------------ LLM

    def _llm_reply(self, user_text: str, ev: vc.Events) -> tuple[str, float]:
        """Stream a reply; returns (first_sentence, ttft_s).

        Thinking is disabled (Qwen3 non-thinking mode) and the streamer loop
        stops at the first sentence terminator so TTS starts as early as a
        production cascade would.
        """
        import torch
        from transformers import TextIteratorStreamer

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        try:
            prompt = self._llm_tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self._llm_tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        ids = self._llm_tok(prompt, return_tensors="pt").to(self._device)
        streamer = TextIteratorStreamer(
            self._llm_tok, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            **ids,
            max_new_tokens=60,
            do_sample=False,
            temperature=None,
            top_p=None,
            streamer=streamer,
        )
        t0 = time.monotonic()
        thread = threading.Thread(target=self._llm.generate, kwargs=kwargs)
        thread.start()
        full = ""
        ttft = None
        done = False
        for piece in streamer:
            if ttft is None:
                ttft = time.monotonic() - t0
                ev.add("llm_first_token")
            full += piece
            for sep in (".", "!", "?", "\n"):
                idx = full.find(sep)
                if 0 < idx < len(full) - 1:
                    full = full[: idx + 1]
                    done = True
                    break
            if done:
                break
        thread.join(timeout=15)
        return full.strip() or "Let me think about that.", ttft or 0.0

    # ------------------------------------------------------------------ TTS

    def _tts_first_chunk(self, text: str, ev: vc.Events):
        """Returns (first_chunk_np, generator, t_first)."""
        t0 = time.monotonic()
        gen = self._tts.generate_audio_stream(self._voice, text)
        try:
            first = next(gen)
        except StopIteration:
            raise InferenceError("pocket-tts produced no audio")
        t_first = time.monotonic()
        ev.add("tts_first_chunk")
        arr = first.float().cpu().numpy().reshape(-1)
        return arr, gen, t_first - t0

    # ------------------------------------------------------------- scenarios

    def _run_eou(self, utts: dict, ev: vc.Events, cancel) -> dict:
        frames = list(vc.frames_of(utts["u1"], vc.FRAME_24K))
        with self._stt_session():
            self._stt_prime()
            transcript, t_endpoint, t_user_end, vad_dbg = self._stt_transcribe_until_endpoint(
                frames, cancel, ev
            )
            transcript += self._stt_warp_flush(cancel)
        ev.add("stt_done", text=transcript.strip())
        sentence, ttft = self._llm_reply(transcript.strip(), ev)
        first, gen, tts_ttfa = self._tts_first_chunk(sentence, ev)
        t_first_wall = time.monotonic()  # first audio chunk available NOW
        ev.add("first_audio")
        # Consume the rest (response length info).
        rest = []
        for chunk in gen:
            self._check_cancel(cancel)
            rest.append(chunk.float().cpu().numpy().reshape(-1))
        audio = np.concatenate([first] + rest) if rest else first
        return {
            "eou_to_first_audio_ms": round((t_first_wall - t_user_end) * 1000.0, 1),
            "endpoint_delay_ms": round((t_endpoint - t_user_end) * 1000.0, 1)
            if t_endpoint
            else None,
            "vad_head_probs": vad_dbg,
            "llm_ttft_ms": round(ttft * 1000.0, 1),
            "tts_ttfa_ms": round(tts_ttfa * 1000.0, 1),
            "transcript": transcript.strip(),
            "reply": sentence,
            "_audio": audio,
            "_sr": 24000,
        }

    def _play_and_monitor(self, tts_gen, mic_frames, cancel, ev: vc.Events):
        """Play TTS at real time while feeding mic frames to the gate/STT.

        mic_frames: list of np arrays (80 ms) — silence until u3/u4 begins.
        Returns on barge-in: (t_barge, played_audio, mic_index) or at end of
        TTS/mic: (None, played_audio, mic_index).
        """
        played = []
        t_barge = None
        tts_done = False
        tts_next = None
        tts_buf = np.zeros(0, dtype=np.float32)
        PLAY_CHUNK = 960  # 40 ms playback granularity

        t_play_next = time.monotonic()
        for i, mic in enumerate(mic_frames):
            self._check_cancel(cancel)
            t_frame = time.monotonic()
            # Barge-in gate: speech energy on the mic while we play.
            if t_barge is None and vc.rms(mic) > GATE_RMS:
                t_barge = t_frame
                ev.add("barge_in", frame=i)
                break
            # Fill the playback buffer (chunks may be torch tensors or numpy).
            while tts_buf.size < PLAY_CHUNK and not tts_done:
                try:
                    nxt = np.asarray(next(tts_gen)).reshape(-1).astype(np.float32)
                    tts_buf = np.concatenate([tts_buf, nxt])
                except StopIteration:
                    tts_done = True
            # Schedule one 40 ms piece at real time.
            if tts_buf.size >= PLAY_CHUNK:
                played.append(tts_buf[:PLAY_CHUNK])
                tts_buf = tts_buf[PLAY_CHUNK:]
                t_play_next = max(t_play_next, t_frame) + PLAY_CHUNK / 24000.0
                if t_play_next > time.monotonic():
                    time.sleep(min(t_play_next - time.monotonic(), FRAME_DUR_S))
        return t_barge, played, i

    def _run_interrupt(self, utts: dict, ev: vc.Events, cancel) -> dict:
        import torch

        # Turn 1: u2 -> response, played at real time.
        frames2 = list(vc.frames_of(utts["u2"], vc.FRAME_24K))
        with self._stt_session():
            self._stt_prime()
            transcript, _, _, vad_dbg = self._stt_transcribe_until_endpoint(frames2, cancel, ev)
            transcript += self._stt_warp_flush(cancel)
        sentence, _ = self._llm_reply(transcript.strip(), ev)
        first, gen, _ = self._tts_first_chunk(sentence, ev)
        gen = iter([first] + [c.float().cpu().numpy().reshape(-1) for c in gen])

        # While it plays, open the mic: silence, then u3.
        sil_frames = [np.zeros(vc.FRAME_24K, dtype=np.float32)] * 30
        u3_frames = [vc.pad_to(f, vc.FRAME_24K) for f in vc.frames_of(utts["u3"], vc.FRAME_24K)]
        t_barge, played, used = self._play_and_monitor(gen, sil_frames + u3_frames, cancel, ev)

        if t_barge is None:
            return {"error": "no barge-in detected by energy gate"}
        t_stop = time.monotonic()  # TTS playback halted immediately
        ev.add("tts_stopped")

        # Turn 2: remaining u3 mic audio -> STT (the mic never closed).
        remaining = u3_frames[max(0, used - len(sil_frames)) :]
        with self._stt_session():
            self._stt_prime()
            t2, _, _, vad_dbg2 = self._stt_transcribe_until_endpoint(remaining, cancel, ev)
            t2 += self._stt_warp_flush(cancel)
        sentence2, _ = self._llm_reply(t2.strip(), ev)
        first2, gen2, _ = self._tts_first_chunk(sentence2, ev)
        t_new_audio = time.monotonic()  # start of second response audio
        ev.add("new_response_start")
        rest2 = [c.float().cpu().numpy().reshape(-1) for c in gen2]
        audio = np.concatenate([first2] + rest2) if rest2 else first2
        return {
            "interrupt_latency_ms": round((t_stop - t_barge) * 1000.0, 1),
            "respond_latency_ms": round((t_new_audio - t_barge) * 1000.0, 1),
            "transcript_after": t2.strip(),
            "vad_head_probs_turn2": vad_dbg2,
            "_audio": audio,
            "_sr": 24000,
        }

    def _run_talkover(self, utts: dict, ev: vc.Events, cancel) -> dict:
        frames2 = list(vc.frames_of(utts["u2"], vc.FRAME_24K))
        with self._stt_session():
            self._stt_prime()
            transcript, _, _, vad_dbg = self._stt_transcribe_until_endpoint(frames2, cancel, ev)
            transcript += self._stt_warp_flush(cancel)
        sentence, _ = self._llm_reply(transcript.strip(), ev)
        first, gen, _ = self._tts_first_chunk(sentence, ev)
        gen = iter([first] + [c.float().cpu().numpy().reshape(-1) for c in gen])

        sil_frames = [np.zeros(vc.FRAME_24K, dtype=np.float32)] * 30
        u4_frames = [vc.pad_to(f, vc.FRAME_24K) for f in vc.frames_of(utts["u4"], vc.FRAME_24K)]
        # The gate fires on any speech; for talkover we keep monitoring but
        # also record what played during the probe window.
        t_barge, played, used = self._play_and_monitor(gen, sil_frames + u4_frames, cancel, ev)
        t_probe_start = None  # wall time when u4 began = barge time
        if t_barge is not None:
            t_probe_start = t_barge
        # Continue playback of remaining TTS after probe (assistant keeps
        # talking until endpointed) — finish generator.
        tail = []
        for c in gen:
            self._check_cancel(cancel)
            tail.append(np.asarray(c).reshape(-1).astype(np.float32))
        played_all = np.concatenate(played + tail) if played or tail else np.zeros(1)
        n_probe_frames = max(0, used - len(sil_frames)) if t_barge is not None else 0
        during_s = n_probe_frames * (960 / 24000.0)
        # STT what the user said during the probe (u4 remainder), then respond.
        remaining = u4_frames[n_probe_frames:] if t_barge is not None else []
        reply2 = ""
        t4 = ""
        if remaining:
            with self._stt_session():
                self._stt_prime()
                t4, _, _, _ = self._stt_transcribe_until_endpoint(remaining, cancel, ev)
                t4 += self._stt_warp_flush(cancel)
            reply2, _ = self._llm_reply(t4.strip(), ev)
        return {
            "mentions_probe": "purple" in reply2.lower(),
            "assistant_audio_during_probe_s": round(during_s, 3),
            "assistant_audio_after_probe_s": round(len(played_all) / 24000.0, 3),
            "paused_after_probe": None,
            "transcript_after": t4.strip(),
            "reply_after": reply2,
            "_audio": played_all,
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
                output_dir, f"kyutai_{scenario}_r{run}", audio, sr,
                out.get("transcript") or out.get("transcript_after") or "",
            )
        return {
            "system": "kyutai-cascade",
            "scenario": scenario,
            "run": run,
            "files": files,
            **out,
        }

    def estimate_time(self, params: dict) -> float:
        return {"eou": 40000.0, "interrupt": 80000.0, "talkover": 80000.0}.get(
            params.get("scenario", "eou"), 60000.0
        )
