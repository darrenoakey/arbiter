# Exact-revision Cloudflare Clef / Clef-flash SystemOne scoring through an Arbiter worker.
#
# Clef is Cloudflare's open-weight, Jev-compatible decision model: a Qwen3.5
# backbone run prefill-only plus a joint schema head that emits one logit per
# allowed option. The worker loads the pinned Hugging Face release, verifies
# every weight and the release's custom code by SHA-256, and scores a staged
# JSONL of label-free SystemOne requests one record at a time.

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import statistics
import threading
import time
import zipfile
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

_INBOX_ROOT = Path("/mnt/arbiter-store/inbox")
_MAX_DATASET_BYTES = 64 * 1024 * 1024
_MAX_RECORDS = 50_000
_MAX_LENGTH = 16384
# Same file in both releases; it is executed, so it is pinned like a weight.
_CODE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
_QUESTION_KEYS = {"type", "instructions", "criteria"}
_RECORD_KEYS = {"id", "state", "questions"}

RELEASES: dict[str, dict] = {
    "clef-scorer": {
        "repo": "Cloudflare/clef",
        "revision": "2f3de3dd85f379784083b0814d997ab627200f0c",
        "base": "Qwen/Qwen3.8-27B",
        "weights": {
            "joint_head.safetensors": "a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953",
            "model-00001-of-00012.safetensors": "54d83c1d36631de231876217a8e0c2483eccee8746369a482b79442bdfc5d958",
            "model-00002-of-00012.safetensors": "464086af08be8e2ec14960a4dcff083ebc39974ade00d79d35497385f960ab3a",
            "model-00003-of-00012.safetensors": "092212d3a02fafacd6424723eda59d60e5282d2068d68f0e37cb891f63bbb658",
            "model-00004-of-00012.safetensors": "d06ff197668c782145fafa74bba61bbc296fb27e39afb15fd522918ce3514dc5",
            "model-00005-of-00012.safetensors": "cc693b8829614a72e0c2be303fb290cf05dbb4d7ded872a817b97bb222a78427",
            "model-00006-of-00012.safetensors": "e7cce15da2443cb8b84aaed66a9a71f0c87dc9d043f83b5f58f4a89f64ba60ad",
            "model-00007-of-00012.safetensors": "75fc7e76b57d5d17a5d85fff3e879d07dd33edc885a8ee04ad437a899bcd5307",
            "model-00008-of-00012.safetensors": "189b15cb6b1af48d5f118951446e15639bfeaf76081d5f20aed1f1b4253afe1d",
            "model-00009-of-00012.safetensors": "8101e2664bb14684fc7051f2e1f84903dbdf5489b17cae3212ac08a0af744a60",
            "model-00010-of-00012.safetensors": "a8e69016a1a8dab1c9ce8dd151c0d3224412ab864cf06d3a5e2b8a5775cfdfb8",
            "model-00011-of-00012.safetensors": "b328d21c36ab384696e40a30ac86a95aaf6dd82da89438ba009cb97d87c1d6b9",
            "model-00012-of-00012.safetensors": "7505eed910a84ea18e66953572476f8a8a6a6ef54192a6643d6e9cd21a3bb978",
        },
    },
    "clef-flash-scorer": {
        "repo": "Cloudflare/clef-flash",
        "revision": "17f0b0ad64efb65d273590632833508766b2aae6",
        "base": "Qwen/Qwen3.5-9B",
        "weights": {
            "joint_head.safetensors": "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
            "model-00001-of-00004.safetensors": "8b45a8e968141cdcc58fb71c9adfc258e2c77b5f062bc636c1fd5bc5d916b565",
            "model-00002-of-00004.safetensors": "7590856c713eed844a2dcf48e6c43c4de165b788bc3f80e328311183cdbc7db8",
            "model-00003-of-00004.safetensors": "e6eac2467952c33361ed7dcb3c7959d1086bbe57201cd3749c3d769fdc17fe63",
            "model-00004-of-00004.safetensors": "9fcecc6556b39171238373a465f409794b7f821fb4cd1e6459e3a9c0fe317af7",
        },
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: object, seen: set[str]) -> dict:
    """Accept only a label-free SystemOne request: {id, state, questions}."""
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        raise InferenceError("each record must have exactly the keys id, state, questions")
    record_id = record["id"]
    if not isinstance(record_id, str) or not record_id or record_id in seen:
        raise InferenceError("record ids must be unique non-empty strings")
    seen.add(record_id)
    questions = record["questions"]
    if not isinstance(questions, dict) or not questions:
        raise InferenceError(f"{record_id}: questions must be a non-empty object")
    for name, question in questions.items():
        if not isinstance(question, dict) or not set(question) <= _QUESTION_KEYS or "type" not in question:
            raise InferenceError(f"{record_id}/{name}: questions may only carry type, instructions, criteria")
        kind, criteria = question["type"], question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise InferenceError(f"{record_id}/{name}: choice needs a non-empty criteria object")
        elif kind == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise InferenceError(f"{record_id}/{name}: score needs at least two criteria levels")
        elif kind == "noul":
            if criteria is not None and (not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}):
                raise InferenceError(f"{record_id}/{name}: noul criteria may only define true/false")
        else:
            raise InferenceError(f"{record_id}/{name}: type must be noul, choice, or score")
    return record


def read_records(params: dict) -> tuple[str, list[dict]]:
    path = Path(str(params.get("dataset_file", "")))
    try:
        path.resolve().relative_to(_INBOX_ROOT.resolve())
    except ValueError as error:
        raise InferenceError("dataset must be staged in the Arbiter inbox") from error
    if not path.is_file():
        raise InferenceError("staged dataset is missing: " + str(path))
    if path.stat().st_size > _MAX_DATASET_BYTES:
        raise InferenceError("staged dataset exceeds the 64 MiB safety limit")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != params.get("dataset_sha256"):
        raise InferenceError("staged dataset checksum differs from the submitted dataset_sha256")
    seen: set[str] = set()
    records = [validate_record(json.loads(line), seen) for line in payload.splitlines() if line.strip()]
    if not records or len(records) > _MAX_RECORDS:
        raise InferenceError(f"dataset must hold 1..{_MAX_RECORDS} records")
    return digest, records


def probabilities_for(option_ids, logits) -> dict[str, float]:
    values = logits.float().softmax(-1).tolist()
    if len(values) != len(option_ids):
        raise InferenceError("Clef returned a different number of logits than options")
    return {str(option): float(value) for option, value in zip(option_ids, values)}


def latency_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"mean_ms": statistics.fmean(ordered), "p50_ms": ordered[len(ordered) // 2],
            "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]}


def _load_release_code(snapshot: Path, model_id: str):
    code = snapshot / "joint_schema_model.py"
    if file_sha256(code) != _CODE_SHA256:
        raise LoadError("Clef joint_schema_model.py differs from the pinned release code")
    spec = importlib.util.spec_from_file_location(f"arbiter_clef_release_{model_id.replace('-', '_')}", code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verified_snapshot(cache_dir: Path, release: dict) -> Path:
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(release["repo"], revision=release["revision"],
                                      cache_dir=str(cache_dir), token=False))
    if snapshot.name != release["revision"]:
        raise LoadError("Hugging Face snapshot path differs from the exact Clef revision pin")
    marker = cache_dir / f"VERIFIED-{release['revision']}.json"
    stats = {name: (snapshot / name).stat().st_size for name in release["weights"] if (snapshot / name).is_file()}
    if marker.is_file() and json.loads(marker.read_text()).get("sizes") == stats and len(stats) == len(release["weights"]):
        return snapshot
    for name, expected in release["weights"].items():
        path = snapshot / name
        if not path.is_file() or file_sha256(path) != expected:
            raise LoadError("Clef weight hash mismatch: " + name)
    marker.write_text(json.dumps({"repo": release["repo"], "revision": release["revision"],
                                  "sizes": stats, "verified_unix": time.time()}, indent=2) + "\n")
    return snapshot


class _ClefScoreBase(ModelAdapter):
    model_id = ""

    def __init__(self):
        self._model = None
        self._processor = None
        self._module = None
        self._runtime: dict = {}

    @property
    def release(self) -> dict:
        return RELEASES[self.model_id]

    def load(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise LoadError("Clef scoring requires the Arbiter CUDA worker")
        # imported lazily: every worker imports every adapter module and some
        # per-model venvs have neither pydantic nor torch
        import torch
        import transformers

        from arbiter.config import load_config

        model = load_config().models.get(self.model_id)
        if model is None or not model.model_path:
            raise LoadError(f"{self.model_id} requires model_path in Arbiter local config")
        snapshot = _verified_snapshot(Path(model.model_path).expanduser(), self.release)
        self._module = _load_release_code(snapshot, self.model_id)
        self._model, self._processor = self._module.load_release_model(snapshot, device="cuda",
                                                                       dtype=torch.bfloat16)
        self._runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
                         "device": torch.cuda.get_device_name(0), "dtype": "bfloat16"}
        log.info("%s loaded %s@%s", self.model_id, self.release["repo"], self.release["revision"])

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._module = None
        self._cleanup_gpu()

    def estimate_time(self, params: dict) -> float:
        return 600_000.0

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import torch

        if self._model is None:
            raise InferenceError(f"{self.model_id} is not loaded")
        digest, records = read_records(params)
        module, tokenizer = self._module, self._processor.tokenizer
        device = next(self._model.parameters()).device
        max_length = int(params.get("max_length") or _MAX_LENGTH)
        rows_path = output_dir / "rows.jsonl"
        latencies: list[float] = []
        truncated = 0
        started = time.perf_counter()
        with rows_path.open("w", encoding="utf-8") as stream, torch.inference_mode():
            for index, record in enumerate(records):
                self._check_cancel(cancel_flag)
                record_started = time.perf_counter()
                encoded = module.encode_record(tokenizer, record, max_length=max_length,
                                               processor=self._processor)
                batch = module.collate_records([encoded], tokenizer.pad_token_id, device)
                logits = self._model(batch)[0]
                answers = {question.question_id: probabilities_for(question.option_ids, question_logits)
                           for question, question_logits in zip(encoded.questions, logits)}
                torch.cuda.synchronize()
                latency_ms = 1000 * (time.perf_counter() - record_started)
                latencies.append(latency_ms)
                if set(answers) != set(record["questions"]):
                    raise InferenceError("Clef answered different question ids than the request")
                # encode_record trims only the state, and only to exactly max_length tokens
                was_truncated = len(encoded.input_ids) >= max_length
                truncated += int(was_truncated)
                stream.write(json.dumps({"index": index, "id": record["id"], "probabilities": answers,
                                         "input_tokens": len(encoded.input_ids), "state_truncated": was_truncated,
                                         "latency_ms": latency_ms}, ensure_ascii=False, allow_nan=False) + "\n")
                if (index + 1) % 100 == 0:
                    stream.flush()
                    log.info("%s records=%d/%d", self.model_id, index + 1, len(records))
        report = {
            "model_id": self.model_id, "repo": self.release["repo"], "revision": self.release["revision"],
            "base_model": self.release["base"], "code_sha256": _CODE_SHA256,
            "weight_sha256": self.release["weights"], "runtime": self._runtime,
            "records": len(records), "dataset_sha256": digest, "max_length": max_length,
            "state_truncated_records": truncated, "batch_size": 1,
            "score_seconds": time.perf_counter() - started, "latency": latency_summary(latencies),
            "latency_note": "per record: tokenize + single bf16 forward + softmax, CUDA synchronized",
        }
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        archive_path = output_dir / "clef-results.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.write(rows_path, "rows.jsonl")
            archive.write(report_path, "report.json")
        return {"file": archive_path.name, "format": "zip", "records": len(records),
                "score_seconds": report["score_seconds"], "mean_latency_ms": report["latency"]["mean_ms"]}


@register
class ClefScoreAdapter(_ClefScoreBase):
    model_id = "clef-scorer"


@register
class ClefFlashScoreAdapter(_ClefScoreBase):
    model_id = "clef-flash-scorer"
