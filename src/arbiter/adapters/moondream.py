"""Moondream 3 vision-language adapter.

FlexAttention is fused via torch.compile for speed. moondream3's text
decode calls torch.nn.attention.flex_attention; uncompiled it falls back
to an unfused kernel that materialises the full scores matrix (slow).
MoondreamModel.compile() torch.compile()s the vision encoder (fullgraph)
and the decode-one-token path (fullgraph, mode="reduce-overhead" → CUDA
graphs) and warms them up.

The catch: mode="reduce-overhead" uses CUDA graph trees, which are bound
to the thread that captured them. arbiter's worker_main runs infer() in
an 8-thread ThreadPoolExecutor and load() on the main thread, so a graph
captured anywhere is replayed on a different thread → AssertionError in
torch/_inductor/cudagraph_trees.py. (Measured: no-cudagraph compile gives
no net gain; the speedup *is* the CUDA graphs.)

Fix: this adapter owns a private single-thread executor. The compile +
warmup (graph capture) AND every subsequent model call (graph replay) all
run on that one thread, regardless of which worker_main pool thread called
infer(). Verified thread-safe driven from the 8-pool, query ~1.3-1.6x
faster, outputs unchanged. One-time ~50-115s compile/warmup on first call.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

MODEL_HF_ID = "moondream/moondream3-preview"


@register
class MoondreamAdapter(ModelAdapter):
    model_id = "moondream"

    def __init__(self):
        self._model = None
        self._device = "cuda"
        # Private single-thread executor: pins CUDA-graph capture + every
        # replay to ONE thread so reduce-overhead compile is safe under
        # worker_main's 8-thread infer pool.
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moondream-gpu")
        self._compiled = False
        self._compile_failed = False
        self._compile_lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM

        log.info("Loading %s on %s with bfloat16 ...", MODEL_HF_ID, device)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_HF_ID,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": device},
        )
        self._device = device
        # KV caches are plain tensors (not graph-captured) — fine to set up
        # here on the main thread. compile()/warmup is deferred to the first
        # inference so it runs on the private GPU thread (graph affinity).
        self._model._setup_caches()
        log.info("Moondream3 ready (flex_attention compiles on first call).")

    def unload(self) -> None:
        log.info("Unloading Moondream3.")
        self._gpu.shutdown(wait=False, cancel_futures=True)
        del self._model
        self._model = None
        self._cleanup_gpu()

    def _compile_on_gpu_thread(self) -> None:
        """Replicate MoondreamModel.compile() — runs ON the private GPU
        thread so the reduce-overhead CUDA graphs are captured there and
        every later replay (also routed through this thread) is valid."""
        import torch

        m = self._model.model
        t0 = time.time()
        for mod in m.modules():
            if type(mod).__name__ == "QuantizedLinear" and hasattr(mod, "unpack"):
                mod.unpack()
        # Materialise lazy props before capture (avoids first-call overhead).
        m.causal_block_mask
        m.point_gen_indices

        m._vis_enc = torch.compile(m._vis_enc, fullgraph=True)
        m._decode_one_tok = torch.compile(
            m._decode_one_tok, fullgraph=True, mode="reduce-overhead"
        )

        device = m.device
        dtype = m.vision.pos_emb.dtype
        with torch.no_grad():
            m._vis_enc(torch.randn(1, 3, 378, 378, device=device, dtype=dtype))
            dummy_emb = torch.randn(1, 1, m.config.text.dim, device=device, dtype=dtype)
            dummy_mask = torch.ones(
                1, 1, m.config.text.max_context, device=device, dtype=torch.bool
            )
            dummy_pos = torch.tensor([100], device=device, dtype=torch.long)
            m._decode_one_tok(dummy_emb, dummy_mask, dummy_pos, None)
            m._decode_one_tok(
                dummy_emb, dummy_mask, dummy_pos, None,
                lm_head_indices=m.point_gen_indices,
            )
        log.info("Moondream3 flex_attention compiled+warmed in %.0fs.", time.time() - t0)

    def _run(self, fn):
        """Run fn on the private GPU thread. First call compiles there.
        If compile ever fails, fall back to UNCOMPILED (still on this
        thread for consistency) rather than taking the vision path down."""
        def task():
            if not self._compiled and not self._compile_failed:
                with self._compile_lock:
                    if not self._compiled and not self._compile_failed:
                        try:
                            self._compile_on_gpu_thread()
                            self._compiled = True
                        except Exception as e:  # noqa: BLE001
                            self._compile_failed = True
                            log.error(
                                "Moondream3 compile() failed, serving UNCOMPILED: %s", e
                            )
            return fn()

        return self._gpu.submit(task).result()

    def _decode_image(self, params: dict):
        return self._resolve_image(params)

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        self._check_cancel(cancel_flag)

        image = self._decode_image(params)
        # Determine task from _job_type (injected by worker) or explicit "task" param
        job_type = params.get("_job_type", "")
        task = params.get("task") or job_type or "caption"

        self._check_cancel(cancel_flag)

        if task == "caption":
            result = self._caption(image, params)
        elif task == "query":
            result = self._query(image, params)
        elif task == "detect":
            result = self._detect(image, params)
        elif task == "point":
            result = self._point(image, params)
        else:
            raise InferenceError(f"Unknown task: {task}")

        self._check_cancel(cancel_flag)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "result.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))

        return {
            "format": "json",
            "file": "result.json",
            "task": task,
            **result,
        }

    def _sampling_kwargs(self, params: dict) -> dict:
        kw = {}
        if "temperature" in params:
            kw["temperature"] = float(params["temperature"])
        if "max_tokens" in params:
            kw["max_new_tokens"] = int(params["max_tokens"])
        if "top_p" in params:
            kw["top_p"] = float(params["top_p"])
        return kw

    def _caption(self, image, params: dict) -> dict:
        length = params.get("length", "normal")
        skw = self._sampling_kwargs(params)
        result = self._run(lambda: self._model.caption(image, length=length, **skw))
        return {"caption": result["caption"]}

    def _query(self, image, params: dict) -> dict:
        question = params.get("question", "")
        if not question:
            raise InferenceError("question is required for query task")
        reasoning = str(params.get("reasoning", "false")).lower() == "true"
        skw = self._sampling_kwargs(params)
        result = self._run(
            lambda: self._model.query(
                image=image, question=question, reasoning=reasoning, **skw
            )
        )
        return {"answer": result["answer"]}

    def _detect(self, image, params: dict) -> dict:
        obj = params.get("object") or params.get("obj", "")
        if not obj:
            raise InferenceError("object is required for detect task")
        w, h = image.size
        result = self._run(lambda: self._model.detect(image, obj))
        objects = []
        for det in result.get("objects", []):
            objects.append({
                "bbox": [
                    round(det["x_min"] * w),
                    round(det["y_min"] * h),
                    round(det["x_max"] * w),
                    round(det["y_max"] * h),
                ],
                "confidence": det.get("confidence", 1.0),
            })
        return {"objects": objects}

    def _point(self, image, params: dict) -> dict:
        obj = params.get("object") or params.get("obj", "")
        if not obj:
            raise InferenceError("object is required for point task")
        w, h = image.size
        result = self._run(lambda: self._model.point(image, obj))
        points = [{"x": round(p["x"] * w), "y": round(p["y"] * h)} for p in result.get("points", [])]
        return {"points": points, "count": len(points)}

    def estimate_time(self, params: dict) -> float:
        return 2000.0
