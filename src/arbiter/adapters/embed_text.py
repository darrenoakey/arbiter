"""Text embedding adapter — nomic-embed-text-v1.5 via HuggingFace transformers.

Exposes a single job type `embed-text` that takes a list of strings and
returns a list of 768-dim float embeddings (mean-pooled + L2-normalized,
matching the nomic-embed-text convention the rest of the monorepo uses).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

_MODEL_REPOSITORY = "nomic-ai/nomic-embed-text-v1.5"
_MAX_SEQ_LENGTH = 8192
_EMBEDDING_DIM = 768


def _mean_pool(last_hidden_state, attention_mask):
    """Mean-pool the last hidden state weighted by the attention mask."""

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@register
class EmbedTextAdapter(ModelAdapter):
    model_id = "embed-text"

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        log.info("Loading %s on %s ...", _MODEL_REPOSITORY, device)
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_REPOSITORY)
        model = AutoModel.from_pretrained(
            _MODEL_REPOSITORY,
            trust_remote_code=True,
            safe_serialization=True,
            torch_dtype=torch.float16,
        )
        model.eval()
        self._model = model.to(device)
        self._device = device
        log.info("embed-text ready (dim=%d).", _EMBEDDING_DIM)

    def unload(self) -> None:
        log.info("Unloading embed-text.")
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        import torch
        import torch.nn.functional as F

        self._check_cancel(cancel_flag)

        raw_texts = params.get("texts")
        if raw_texts is None:
            single = params.get("text")
            if single is None:
                raise InferenceError(
                    "embed-text requires 'texts' (list[str]) or 'text' (str)"
                )
            raw_texts = [single]
        if not isinstance(raw_texts, list) or not raw_texts:
            raise InferenceError("'texts' must be a non-empty list of strings")
        for i, t in enumerate(raw_texts):
            if not isinstance(t, str):
                raise InferenceError(f"texts[{i}] is not a string: {type(t).__name__}")

        # nomic-embed-text-v1.5 expects a task prefix. Default to
        # search_document which is what omniscience-style ingestion uses;
        # callers can override via params["task"] ("search_query" for
        # querying, "classification", "clustering", "search_document").
        task = str(params.get("task", "search_document"))
        valid_tasks = {
            "search_document",
            "search_query",
            "classification",
            "clustering",
        }
        if task not in valid_tasks:
            raise InferenceError(f"invalid task '{task}'; valid: {sorted(valid_tasks)}")
        prefix = f"{task}: "
        prefixed = [prefix + t for t in raw_texts]

        start = time.perf_counter()

        # Batch through the model. Small batches avoid VRAM spikes on
        # long inputs while still getting most of the throughput benefit.
        batch_size = int(params.get("batch_size", 16))
        all_embeddings: list[list[float]] = []

        for i in range(0, len(prefixed), batch_size):
            self._check_cancel(cancel_flag)
            chunk = prefixed[i : i + batch_size]
            encoded = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=_MAX_SEQ_LENGTH,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                output = self._model(**encoded)

            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
            all_embeddings.extend(normalized.cpu().tolist())

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        result = {
            "embeddings": all_embeddings,
            "dimension": _EMBEDDING_DIM,
            "count": len(all_embeddings),
            "task": task,
            "model_repository": _MODEL_REPOSITORY,
            "dtype": "float16",
            "elapsed_ms": round(elapsed_ms, 1),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(json.dumps(result))

        return result

    def estimate_time(self, params: dict) -> float:
        texts = params.get("texts") or [params.get("text", "")]
        # ~5ms per short text on GPU, ~20ms for long ones. Budget 10ms
        # per text as a reasonable default.
        n = len(texts) if isinstance(texts, list) else 1
        return max(50.0, 10.0 * n)
