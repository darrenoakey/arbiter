from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from arbiter.adapters.base import GroupAdapter, HeapTrimGuard, LoadError
from arbiter.adapters.registry import register

SOURCE = Path("/home/darren/src/id-lora-runtime")
REVISION = "13dc9c286a2d3c1a21a7781f9ac0fe7727ec56cd"
MODELS = Path("/mnt/t9/models/id-lora")
CHECKPOINT = Path("/mnt/t9/models/ltx2/ltx-2.3-22b-dev.safetensors")
FAST_SOURCE = Path("/home/darren/src/ltx2-spark/video_fast_gpu.py")
LORA_SHA256 = "e5af73441743b4852f228b03e444888dff3da80d2666033af2367ab7bda6d8b9"


def verify_runtime() -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise LoadError(f"ID-LoRA source revision mismatch: {revision}")
    manifest = json.loads(Path(__file__).with_name("id_lora_runtime.json").read_text())
    for relative, expected in manifest.items():
        path = (
            FAST_SOURCE if relative == "__spark_fast_pipeline__" else SOURCE / relative
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise LoadError(f"ID-LoRA optimized source checksum mismatch: {path}")
    if CHECKPOINT.stat().st_size != 46149344974:
        raise LoadError(f"Incomplete LTX2.3 checkpoint: {CHECKPOINT}")
    with (MODELS / "talkvid.safetensors").open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != LORA_SHA256:
        raise LoadError(f"ID-LoRA TalkVid checksum mismatch: {digest}")


# Verify runtime: fail closed on pinned upstream, optimization or identity-weight drift.


def load_runtime():
    verify_runtime()
    script = SOURCE / "ID-LoRA-2.3/scripts/inference_one_stage.py"
    spec = importlib.util.spec_from_file_location("id_lora_one_stage", script)
    if spec is None or spec.loader is None:
        raise LoadError(f"Cannot import ID-LoRA pipeline: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if str(FAST_SOURCE.parent) not in sys.path:
        sys.path.insert(0, str(FAST_SOURCE.parent))
    return module, importlib.import_module("video_fast_gpu").FastPipeline


# Runtime: executed only inside the audited worker after CUDA containment.


def parse_request(params, schema, job_type):
    supplied = params.get("_job_type", job_type)
    if supplied != job_type:
        raise ValueError(f"ID-LoRA worker requires {job_type}, received {supplied}")
    return schema.model_validate(
        {key: value for key, value in params.items() if key != "_job_type"}
    )


# Request: validate dispatcher identity and remove only its known private metadata.


@register
class IDLoraEncodeAdapter(GroupAdapter):
    model_id = "id-lora-encode"

    def __init__(self):
        self._pipeline = None
        self._negative = None

    def load(self, device="cuda"):
        with contextlib.redirect_stdout(sys.stderr), HeapTrimGuard():
            module, fast = load_runtime()
            self._module = module
            self._pipeline = fast.__new__(fast)
            self._pipeline.device = module.torch.device(device)
            self._pipeline.dtype = module.torch.bfloat16
            self._pipeline.stage_1_ledger = module.ModelLedger(
                dtype=module.torch.bfloat16,
                device=self._pipeline.device,
                checkpoint_path=str(CHECKPOINT),
                gemma_root_path=str(MODELS / "gemma"),
            )
            self._pipeline._ensure_encode_models()

    # Encode load: reuse the optimized LTX2 FastPipeline resident encoder lifecycle.

    def unload(self):
        if self._pipeline is not None:
            self._pipeline.unload_encode_models()
        self._pipeline = None
        self._negative = None
        self._cleanup_gpu()

    # Unload: encode and denoise remain separately budgeted, evictable workers.

    def infer(self, params, output_dir, cancel_flag):
        from arbiter.schemas import IDLoraEncodeParams

        values = parse_request(params, IDLoraEncodeParams, "video-id-lora-encode")
        self._check_cancel(cancel_flag)
        with (
            contextlib.redirect_stdout(sys.stderr),
            self._module.torch.inference_mode(),
        ):
            pipe = self._pipeline
            if self._negative is None:
                self._negative = self._context(self._module.DEFAULT_NEGATIVE_PROMPT)
            positive = self._context(values.prompt)
            result = dict(
                zip(
                    ("v_ctx_p", "a_ctx_p", "v_ctx_n", "a_ctx_n"),
                    (*positive, *self._negative),
                    strict=True,
                )
            )
            result.update(prompt=values.prompt, upstream_revision=REVISION)
            self._check_cancel(cancel_flag)
            return pipe.save_encode_output(result, str(output_dir))

    # Encode: reuse cached models and negative context, save CPU-only tensor bundle.

    def _context(self, prompt):
        pipe = self._pipeline
        raw = pipe._enc_text_encoder.encode(prompt)
        context = pipe._enc_embeddings_proc.process_hidden_states(*raw)
        return context.video_encoding.cpu(), context.audio_encoding.cpu()

    # Context: the existing FastPipeline text path, without freezing target audio.

    def estimate_time(self, params):
        return 15000

    # Estimate: same prompt-only encoding cost as the existing staged path.


@register
class IDLoraAdapter(GroupAdapter):
    model_id = "id-lora"

    def __init__(self):
        self._pipeline = None

    def load(self, device="cuda"):
        with contextlib.redirect_stdout(sys.stderr), HeapTrimGuard():
            module, self._fast = load_runtime()
            self._module = module
            lora = module.LoraPathStrengthAndSDOps(
                path=str(MODELS / "talkvid.safetensors"),
                strength=1.0,
                sd_ops=module.LTXV_LORA_COMFY_RENAMING_MAP,
            )
            self._pipeline = module.IDLoraOneStagePipeline(
                checkpoint_path=str(CHECKPOINT),
                gemma_root=str(MODELS / "gemma"),
                loras=[lora],
                device=module.torch.device(device),
            )
            self._pipeline.load_models()

    # Load: cache the dev transformer and reference encoders across shots.

    def unload(self):
        self._pipeline = None
        self._cleanup_gpu()

    # Unload: scheduler eviction releases every resident denoise/decode component.

    def infer(self, params, output_dir, cancel_flag):
        from arbiter.schemas import IDLoraParams

        values = parse_request(params, IDLoraParams, "video-id-lora")
        self._check_cancel(cancel_flag)
        output_dir = Path(output_dir)
        image = output_dir / "first-frame.png"
        audio = output_dir / "reference.wav"
        self._resolve_image(params).save(image)
        audio.write_bytes(self._resolve_media(params, "ref_audio"))
        with (
            contextlib.redirect_stdout(sys.stderr),
            HeapTrimGuard(),
            self._module.torch.inference_mode(),
        ):
            self._generate(values, image, audio, output_dir, cancel_flag)
        self._check_cancel(cancel_flag)
        return {
            "file": "result.mp4",
            "format": "mp4",
            "width": values.width,
            "height": values.height,
            "num_frames": values.num_frames,
            "fps": 25,
            "seed": values.seed,
            "upstream_revision": REVISION,
            "audio_source": "jointly generated with reference-speaker conditioning",
        }

    # Infer: first-frame and identity audio condition newly generated AV, never TTS.

    def _generate(self, values, image, audio, output_dir, cancel_flag):
        module, pipeline = self._module, self._pipeline
        contexts = self._fast.load_denoise1_input(pipeline, values.encoded_file)
        if (
            contexts.get("prompt") != values.prompt
            or contexts.get("upstream_revision") != REVISION
        ):
            raise ValueError(
                f"ID-LoRA encoded prompt or revision mismatch: {values.encoded_file}"
            )
        context = {
            key: contexts[key].to(pipeline.device)
            for key in ("v_ctx_p", "a_ctx_p", "v_ctx_n", "a_ctx_n")
        }
        from ltx_pipelines.utils.media_io import decode_audio_from_file

        decoded = decode_audio_from_file(
            str(audio), module.torch.device("cpu"), 0.0, None
        )
        waveform, sample_rate = decoded.waveform, decoded.sampling_rate
        if not 4.5 <= waveform.shape[-1] / sample_rate <= 5.5:
            raise ValueError(
                f"ID-LoRA expects approximately five seconds of reference audio: {audio}"
            )
        self._check_cancel(cancel_flag)
        pipeline.cancel_check = lambda: self._check_cancel(cancel_flag)
        video, sound = pipeline(
            v_context_p=context["v_ctx_p"],
            a_context_p=context["a_ctx_p"],
            v_context_n=context["v_ctx_n"],
            a_context_n=context["a_ctx_n"],
            seed=values.seed,
            height=values.height,
            width=values.width,
            num_frames=values.num_frames,
            frame_rate=25.0,
            num_inference_steps=values.steps,
            video_guidance_scale=3.0,
            audio_guidance_scale=7.0,
            reference_audio=waveform,
            reference_audio_sample_rate=sample_rate,
            condition_image=module.load_first_frame_image(image),
        )
        self._check_cancel(cancel_flag)
        module.save_video(
            video_tensor=video,
            output_path=output_dir / "result.mp4",
            fps=25.0,
            audio=sound,
            audio_sample_rate=pipeline._vocoder_sr,
        )

    # Generate: ID reference-token joint denoising plus cached tiled decode.

    def estimate_time(self, params):
        return 1800000

    # Estimate: provisional conservative cost pending audited calibration.
