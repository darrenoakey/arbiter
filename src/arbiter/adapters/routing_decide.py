"""Train and score GLiNER2.5-Decide for AgentD3 routing labels.

This is the only sanctioned place that fine-tunes that checkpoint. The
worker loads the pinned local weights, adapts them on the staged training
split, and returns held-out labels. It never logs prompt text.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from arbiter.adapters.base import InferenceError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

CHECKPOINT = Path("/home/darren/models/gliner2.5-decide")
INBOX_ROOT = Path("/mnt/arbiter-store/inbox")
TASK = "route_model"


def _read_dataset(path_text: str) -> dict:
    path = Path(path_text)
    if not path.is_file():
        raise InferenceError("dataset file is missing: " + str(path))
    try:
        path.resolve().relative_to(INBOX_ROOT.resolve())
    except ValueError as exc:
        raise InferenceError("dataset file must be staged in the arbiter inbox") from exc
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise InferenceError("dataset file is not json") from exc
    labels = payload.get("labels")
    train = payload.get("train")
    test = payload.get("test")
    descriptions = payload.get("descriptions")
    if not isinstance(labels, list) or not labels or not all(isinstance(item, str) and item for item in labels):
        raise InferenceError("dataset labels must be a non-empty list of strings")
    if len(set(labels)) != len(labels):
        raise InferenceError("dataset labels must be unique")
    if not isinstance(descriptions, dict):
        raise InferenceError("dataset descriptions must be an object")
    for label in labels:
        if not isinstance(descriptions.get(label), str) or not descriptions[label]:
            raise InferenceError("every label needs a description")
    _rows(train, set(labels), "train")
    _rows(test, set(labels), "test")
    return payload


def _rows(rows: object, labels: set[str], name: str) -> None:
    if not isinstance(rows, list) or not rows:
        raise InferenceError(name + " split is empty")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise InferenceError(name + " row is missing text")
        if row.get("label") not in labels:
            raise InferenceError(name + " row has a label outside the label list")


def _label_of(result: object, labels: list[str]) -> str:
    value = result.get(TASK) if isinstance(result, dict) else result
    if isinstance(value, dict):
        value = value.get("label") or value.get("choice") or value.get("classification")
    if value not in labels:
        raise InferenceError("Decide returned an unknown label")
    return str(value)


@register
class RoutingDecideAdapter(ModelAdapter):
    """LoRA-adapt the pinned Decide checkpoint, then score the held-out split."""

    model_id = "routing-decide"

    def __init__(self) -> None:
        self._loaded = False
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        if not (CHECKPOINT / "model.safetensors").is_file():
            raise InferenceError("pinned Decide weights missing at " + str(CHECKPOINT))
        self._device = device
        self._loaded = True
        log.info("routing-decide ready to train from %s", CHECKPOINT)

    def unload(self) -> None:
        self._loaded = False

    def estimate_time(self, params: dict) -> float:
        epochs = int(params.get("num_epochs", 1))
        return float(epochs) * 3_600_000.0

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        if not self._loaded:
            raise InferenceError("routing-decide is not loaded")
        self._check_cancel(cancel_flag)
        dataset = _read_dataset(str(params["dataset_file"]))
        labels = list(dataset["labels"])
        descriptions = dict(dataset["descriptions"])
        log.info(
            "routing-decide train=%d test=%d labels=%d max_len=%s",
            len(dataset["train"]),
            len(dataset["test"]),
            len(labels),
            params.get("max_len", 256),
        )
        model, train_seconds = self._train(dataset, labels, descriptions, params, output_dir, cancel_flag)
        predicted, score_seconds, call_seconds = self._score(
            model, dataset["train"][0]["text"], dataset["test"], labels, descriptions, cancel_flag
        )
        adapter_dir = output_dir / "adapter"
        adapter_bytes = sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file()) if adapter_dir.is_dir() else 0
        result = {
            "predicted": predicted,
            "actual": [row["label"] for row in dataset["test"]],
            "train_count": len(dataset["train"]),
            "test_count": len(dataset["test"]),
            "label_count": len(labels),
            "train_seconds": train_seconds,
            "score_seconds": score_seconds,
            "mean_seconds_per_headline": sum(call_seconds) / len(call_seconds),
            "p50_seconds_per_headline": _percentile(call_seconds, 0.50),
            "p95_seconds_per_headline": _percentile(call_seconds, 0.95),
            "score_batch_size": 1,
            "adapter_bytes": adapter_bytes,
            "checkpoint": str(CHECKPOINT),
            "training": "LoRA on the training split only; max_len capped; bf16 on CUDA",
            "evaluation": "sequential one-headline classify_text calls on the untouched test split; model load excluded",
            "max_len": int(params.get("max_len", 256)),
            "num_epochs": int(params.get("num_epochs", 1)),
            "batch_size": int(params.get("batch_size", 4)),
            "lora_r": int(params.get("lora_r", 8)),
            "seed": int(params.get("seed", 42)),
        }
        (output_dir / "routing-decide.json").write_text(json.dumps(result))
        return result

    def _train(self, dataset: dict, labels: list[str], descriptions: dict, params: dict, output_dir: Path, cancel_flag: threading.Event):
        from gliner2 import AutoExtractor
        from gliner2.training.data import create_classification_example
        from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

        self._check_cancel(cancel_flag)
        model = AutoExtractor.from_pretrained(str(CHECKPOINT))
        examples = [
            create_classification_example(row["text"], TASK, labels, row["label"], label_descriptions=descriptions)
            for row in dataset["train"]
        ]
        config = TrainingConfig(
            output_dir=str(output_dir / "adapter"),
            experiment_name="agentd3-routing-decide",
            num_epochs=int(params.get("num_epochs", 1)),
            batch_size=int(params.get("batch_size", 4)),
            eval_batch_size=int(params.get("batch_size", 4)),
            gradient_accumulation_steps=1,
            encoder_lr=1e-5,
            task_lr=5e-4,
            eval_strategy="no",
            report_to_wandb=False,
            use_lora=True,
            lora_r=int(params.get("lora_r", 8)),
            lora_alpha=int(params.get("lora_alpha", 16)),
            gradient_checkpointing=False,
            num_workers=0,
            seed=int(params.get("seed", 42)),
            logging_steps=20,
            fp16=False,
            bf16=True,
            max_len=int(params.get("max_len", 256)),
        )
        trainer = GLiNER2Trainer(model, config)
        started = time.perf_counter()
        trainer.train(train_data=examples)
        self._check_cancel(cancel_flag)
        return model, time.perf_counter() - started

    def _score(self, model, warmup_text: str, rows: list[dict], labels: list[str], descriptions: dict,
               cancel_flag: threading.Event) -> tuple[list[str], float, list[float]]:
        import torch

        schema = {TASK: {"labels": descriptions}}
        model.eval()
        with torch.inference_mode():
            self._check_cancel(cancel_flag)
            _label_of(model.classify_text(warmup_text, schema), labels)
            started = time.perf_counter()
            predicted: list[str] = []
            call_seconds: list[float] = []
            for row in rows:
                self._check_cancel(cancel_flag)
                call_started = time.perf_counter()
                result = model.classify_text(row["text"], schema)
                call_seconds.append(time.perf_counter() - call_started)
                predicted.append(_label_of(result, labels))
            score_seconds = time.perf_counter() - started
        return predicted, score_seconds, call_seconds


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
