"""Breeze TTS 2 adapter (BreezeBlue/Breeze-TTS-2).

Breeze TTS 2 is an open-weight bilingual (en/zh) TTS model ranked #1 among
open-weight models on the Artificial Analysis TTS leaderboard. One model
supports three prompting modes, selected automatically from the params:

- **Voice design** — ``text`` + ``instruction`` (natural-language voice
  description), no reference audio. Use ``cfg_scale`` ~4 to strengthen
  instruction-following.
- **Voice clone** — ``text`` + ``ref_audio``/``ref_audio_file`` +
  ``ref_text`` (exact transcript of the reference). Preserves timbre,
  rhythm, emotion, and style of the reference speaker.
- **Voice direction** — clone + ``instruction`` together: keep the
  reference identity while steering tone, emotion, pace, and delivery.
- **Plain** — ``text`` only.

There is no native multi-speaker dialogue mode: callers synthesize one
line at a time (or one batched ``items`` job) and stitch per-speaker
segments themselves.

The inference code lives in the breeze-tts repo clone on spark
(``/home/darren/src/breeze-tts``, packages ``breeze_infer`` and ``models``);
the checkpoint lives at ``/mnt/t9/models/breeze-tts-2``. The adapter runs in
its own venv (``venvs/breeze``) because of its torch 2.12/transformers pin
set. Eager attention is used: the ``--fast-all`` CUDA-graph path targets
Hopper (sm90) and roughly doubles VRAM for no need here.

Batch mode mirrors the kokoro adapter: ``items`` is a list of per-line
param dicts (same fields as single mode). The whole batch is synthesized
inside one job and returned as one concatenated ``result.wav`` plus
``item_samples`` so the caller can slice it back into per-line WAVs. This
amortises the scheduler's per-job dispatch overhead, which otherwise
dominates thousands of tiny TTS lines.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from .base import ModelAdapter, InferenceError, LoadError
from .registry import register

# Spark-local paths (adapter runs only on spark via its own venv).
BREEZE_REPO = Path("/home/darren/src/breeze-tts")
BREEZE_CHECKPOINT = Path("/mnt/t9/models/breeze-tts-2")

# Generation defaults, matching the upstream infer.py.
_MAX_NEW_TOKENS = 1500
_MAX_SEQ_LEN = 2048
_REPETITION_PENALTY = 1.1


def _ensure_breeze_importable() -> None:
    repo = str(BREEZE_REPO)
    if not BREEZE_REPO.is_dir():
        raise LoadError(f"breeze-tts repo clone not found at {repo}")
    if repo not in sys.path:
        sys.path.insert(0, repo)


@register
class TTSBreezeAdapter(ModelAdapter):
    model_id = "tts-breeze"

    def __init__(self):
        self._runtime = None  # FastBreezeStreamingRuntime
        self._tokenizer = None
        self._model = None
        self._audio_tokenizer = None

    # ----------------------------------------------------------------
    # load
    # load tokenizer + model + audio tokenizer once; eager attention
    def load(self, device: str = "cuda") -> None:
        _ensure_breeze_importable()
        from breeze_infer.runtime import (  # type: ignore[import-not-found]
            load_runtime,
            update_generation_config_for_breeze,
        )
        from models.fast_streaming import (  # type: ignore[import-not-found]
            FastBreezeStreamingRuntime,
            FastStreamingConfig,
        )

        if not BREEZE_CHECKPOINT.is_dir():
            raise LoadError(f"breeze checkpoint not found at {BREEZE_CHECKPOINT}")
        tokenizer, model, audio_tokenizer = load_runtime(
            BREEZE_CHECKPOINT,
            device=device,
            attn_implementation="eager",
        )
        update_generation_config_for_breeze(model)
        config = FastStreamingConfig(
            max_new_tokens=_MAX_NEW_TOKENS,
            max_seq_len=_MAX_SEQ_LEN,
            fast_all=False,
            fast_text_encoder=False,
            fast_backbone_prefill=False,
            fast_backbone_decode=False,
            fast_depth_decoder=False,
            fast_codec=False,
            repetition_penalty=_REPETITION_PENALTY,
        )
        self._runtime = FastBreezeStreamingRuntime(
            model, audio_tokenizer, config, tokenizer=tokenizer
        )
        self._tokenizer = tokenizer
        self._model = model
        self._audio_tokenizer = audio_tokenizer

    def unload(self) -> None:
        self._runtime = None
        self._tokenizer = None
        self._audio_tokenizer = None
        del self._model
        self._model = None
        self._cleanup_gpu()

    # ----------------------------------------------------------------
    # synth one
    # synthesize one request dict → float32 mono samples at the model rate
    def _synth_one(
        self,
        item: dict,
        work_dir: Path,
        index: int,
        cancel_flag: threading.Event,
    ):
        import numpy as np
        from breeze_infer.runtime import set_all_seeds  # type: ignore[import-not-found]
        from breeze_infer.templates import (  # type: ignore[import-not-found]
            get_template,
            prepare_inputs,
            select_template_name,
        )

        text = str(item.get("text", "") or "")
        # An empty line is a valid beat in a script — emit a short silence
        # rather than sinking the whole batch on one blank item.
        if not text.strip():
            return np.zeros(int(0.2 * self._runtime.sample_rate), dtype=np.float32)

        request: dict = {
            "id": f"item-{index}",
            "text": text,
            "speaker": str(item.get("speaker", "S0") or "S0"),
        }
        instruction = item.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            request["instruction"] = instruction.strip()

        # Reference audio (clone / direction): staged path or inline base64,
        # always with the exact transcript as ref_text.
        ref_text = item.get("ref_text")
        has_ref = bool(
            item.get("ref_audio") or item.get("ref_audio_file") or ref_text
        )
        if has_ref:
            if not (isinstance(ref_text, str) and ref_text.strip()):
                raise InferenceError(
                    "ref_text (exact transcript) is required with reference audio"
                )
            raw = self._resolve_media(item, "ref_audio")
            ref_path = work_dir / f"ref_{index}.wav"
            ref_path.write_bytes(raw)
            request["ref_audio_path"] = str(ref_path)
            request["ref_text"] = ref_text.strip()

        cfg_scale = float(item.get("cfg_scale", 1.0))
        if cfg_scale <= 0:
            raise InferenceError("cfg_scale must be greater than 0")
        seed = int(item.get("seed", 42))

        template = get_template(select_template_name(request))
        set_all_seeds(seed)
        inputs = prepare_inputs(
            self._tokenizer,
            self._audio_tokenizer,
            self._model,
            [request],
            template,
            guidance_scale=cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        chunks: list = []
        for chunk in self._runtime.iter_audio_chunks(
            inputs, request_id=request["id"], seed=seed
        ):
            self._check_cancel(cancel_flag)
            audio = chunk.audio
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise InferenceError(f"breeze produced no audio for text: {text[:80]!r}")
        return (
            np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        )

    # ----------------------------------------------------------------
    # infer
    # single: params{text, instruction?/ref_audio+ref_text?, ...} → result.wav
    # batch:  params{items:[{...}], gap_seconds?} → concatenated result.wav
    #         + item_samples for client-side slicing
    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        import numpy as np
        import soundfile as sf

        self._check_cancel(cancel_flag)
        if self._runtime is None:
            raise InferenceError("tts-breeze runtime is not loaded")

        sr = int(self._runtime.sample_rate)
        out_path = output_dir / "result.wav"

        items = params.get("items")
        if items:
            gap = float(params.get("gap_seconds", 0.0))
            gap_samples = int(gap * sr)
            silence = (
                np.zeros(gap_samples, dtype=np.float32) if gap_samples > 0 else None
            )
            segments: list = []
            item_samples: list[int] = []
            for index, item in enumerate(items):
                self._check_cancel(cancel_flag)
                wav = self._synth_one(item, output_dir, index, cancel_flag)
                item_samples.append(int(wav.shape[0]))
                segments.append(wav)
                if silence is not None:
                    segments.append(silence)
            full = (
                np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
            )
            sf.write(str(out_path), full, sr)
            return {
                "format": "wav",
                "sample_rate": sr,
                "item_samples": item_samples,
                "gap_samples": gap_samples,
            }

        wav = self._synth_one(params, output_dir, 0, cancel_flag)
        sf.write(str(out_path), wav, sr)
        return {"format": "wav", "sample_rate": sr}

    def estimate_time(self, params: dict) -> float:
        # Eager Breeze on the GB10 runs at roughly realtime to 2x realtime;
        # budget ~0.5s per word plus a fixed per-item floor.
        items = params.get("items")
        if items:
            words = sum(len(str(it.get("text", "")).split()) for it in items)
            return max(2000, 1000 * len(items) + words * 500)
        return max(2000, len(params.get("text", "").split()) * 500)
