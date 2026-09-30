# Exact-revision Bespoke Nimble candidate scoring through an Arbiter worker.

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError, ModelAdapter
from arbiter.adapters.registry import register
from arbiter.config import load_config

log = logging.getLogger(__name__)

_MODEL_ID = "bespokelabs/Bespoke-Nimble-9B"
_MODEL_REVISION = "bd792f44ec8e265be861bfcdf4e05967ffe0e858"
_BASE_ID = "Qwen/Qwen3.5-9B"
_BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
_SOURCE_COMMIT = "62076b4f2d365b5879dafcf7f6dd072a1fe76df7"
_ADAPTER_SHA256 = "29ef39b072dee97287947455337879c1e916705c2f727287922a2d81f5e2f20a"
_INBOX_ROOT = Path("/mnt/arbiter-store/inbox")
_DEV_SUITES = {
    "decision-v4": {
        "sha256": "8d5765d7aec4d08c61854f4664ca79ec2ba44ed092e9b967eaf61ed86c496d9c",
        "records": 1204,
    },
    "transfer-v4": {
        "sha256": "ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e",
        "records": 764,
    },
}


def _serialize(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


def _schema_for(record: dict) -> tuple[str, dict]:
    state = record["state"]
    questions = record["questions"]
    if not isinstance(questions, dict) or not questions:
        raise InferenceError("development record must contain questions")
    schema = {}
    for name, question in questions.items():
        kind = question.get("type")
        criteria = question.get("criteria")
        field = {"description": _serialize(question["instructions"])}
        if kind == "choice" and isinstance(criteria, dict):
            field.update(type="enum", choices=list(criteria), choice_descriptions={
                str(key): _serialize(value) if value is not None else str(key)
                for key, value in criteria.items()
            })
        elif kind == "noul":
            field.update(type="boolean", choices=[False, True])
            if criteria is not None:
                if not isinstance(criteria, dict):
                    raise InferenceError(f"Nimble boolean criteria must be an object or null: {name}")
                descriptions = {str(key).lower(): value for key, value in criteria.items()}
                if set(descriptions) != {"false", "true"}:
                    raise InferenceError(f"Nimble boolean criteria must define false and true: {name}")
                field["choice_descriptions"] = {
                    key: _serialize(value) if value is not None else key
                    for key, value in descriptions.items()
                }
        elif kind == "score" and isinstance(criteria, list):
            field.update(type="enum", choices=[str(index) for index in range(len(criteria))],
                         choice_descriptions={str(index): _serialize(value)
                                              for index, value in enumerate(criteria)})
        else:
            raise InferenceError(f"unsupported Nimble question schema: {name}")
        schema[name] = field
    return _serialize(state), schema


def _target(question: dict) -> tuple[list[str], int]:
    kind = question["type"]
    criteria = question.get("criteria")
    target = question["label"]
    if kind == "choice":
        if not isinstance(criteria, dict):
            raise InferenceError("Nimble choice question is missing its criteria object")
        keys = list(criteria)
        return keys, keys.index(target)
    if kind == "noul":
        selected = int(target) if isinstance(target, (bool, int, float)) else int(str(target).lower() == "true")
        return ["false", "true"], selected
    if kind == "score":
        if not isinstance(criteria, list):
            raise InferenceError("Nimble score question is missing its criteria list")
        return [str(index) for index in range(len(criteria))], int(target)
    raise InferenceError(f"unsupported Nimble label type: {kind}")


def _safe_probability(scores: dict, keys: list[str]) -> list[float]:
    if set(scores) != set(keys):
        raise InferenceError("Nimble candidate keys differ from the frozen development row")
    values = [float(scores[key]) for key in keys]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise InferenceError("Nimble returned a non-finite or out-of-range probability")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-5):
        raise InferenceError(f"Nimble candidate probabilities sum to {total}")
    return values


def _read_records(params: dict) -> tuple[bytes, list[dict], dict]:
    suite = params.get("suite")
    expected = _DEV_SUITES.get(suite)
    if expected is None:
        raise InferenceError("only decision-v4/development and transfer-v4/development are accepted")
    path = Path(str(params.get("dataset_file", "")))
    try:
        path.resolve().relative_to(_INBOX_ROOT.resolve())
    except ValueError as error:
        raise InferenceError("development partition must be staged in the Arbiter inbox") from error
    if not path.is_file():
        raise InferenceError("staged development partition is missing: " + str(path))
    if path.stat().st_size > 100 * 1024 * 1024:
        raise InferenceError("staged development partition exceeds the 100 MiB safety limit")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected["sha256"]:
        raise InferenceError("development partition checksum differs from its frozen manifest")
    records = [json.loads(line) for line in payload.splitlines() if line]
    if len(records) != expected["records"]:
        raise InferenceError("development partition record count differs from its frozen manifest")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("_meta"), dict):
            raise InferenceError("development record metadata is missing")
        _schema_for(record)
    return payload, records, expected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_checkpoint(model_path: Path) -> None:
    from huggingface_hub import snapshot_download

    model_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = model_path.parent / "nimble-source"
    _ensure_nimble_source(source_path)
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    cache_path = model_path.parent / "huggingface-cache"
    adapter_path = Path(snapshot_download(_MODEL_ID, revision=_MODEL_REVISION,
                                         cache_dir=str(cache_path), token=False))
    base_path = Path(snapshot_download(_BASE_ID, revision=_BASE_REVISION,
                                       cache_dir=str(cache_path), token=False))
    if adapter_path.name != _MODEL_REVISION or base_path.name != _BASE_REVISION:
        raise LoadError("Hugging Face snapshot paths differ from the exact model pins")
    if _file_sha256(adapter_path / "adapter_model.safetensors") != _ADAPTER_SHA256:
        raise LoadError("downloaded Nimble adapter weight hash differs from the exact checkpoint pin")
    contract = json.loads((adapter_path / "schema_config.json").read_text())
    if contract.get("model") != _BASE_ID or contract.get("revision") != _BASE_REVISION:
        raise LoadError("pinned Nimble adapter contract does not match the exact Qwen base revision")
    from nimble.training.candidate_schema import validate_contract
    from transformers import AutoTokenizer

    validate_contract(contract, AutoTokenizer.from_pretrained(adapter_path, local_files_only=True))
    if (model_path / "READY.json").is_file():
        return
    if model_path.exists() and any(model_path.iterdir()):
        raise LoadError("Nimble model path exists without a verified READY.json")
    if model_path.exists():
        model_path.rmdir()
    subprocess.run([
        sys.executable, "-m", "nimble.scoring.merge_local_adapter", "--adapter", str(adapter_path),
        "--base", str(base_path), "--output", str(model_path),
    ], cwd=source_path, check=True, text=True)


def _manifest_digest(model_path: Path) -> str:
    path = model_path / "READY.json"
    if not path.is_file():
        raise LoadError("pinned merged Nimble checkpoint is missing READY.json")
    ready = json.loads(path.read_text())
    if (ready.get("base_model") != _BASE_ID or ready.get("base_revision") != _BASE_REVISION
            or ready.get("adapter_sha256") != _ADAPTER_SHA256
            or ready.get("precision") != "bfloat16" or ready.get("quantized") is not False):
        raise LoadError("merged Nimble READY.json does not match the exact BF16 checkpoint pin")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_weight_hashes(model_path: Path) -> None:
    ready = json.loads((model_path / "READY.json").read_text())
    expected = ready.get("weight_sha256")
    if not isinstance(expected, dict) or not expected:
        raise LoadError("merged Nimble READY.json has no weight hashes")
    for name, digest in expected.items():
        path = model_path / name
        if not path.is_file() or _file_sha256(path) != digest:
            raise LoadError("merged Nimble weight hash mismatch: " + name)


# Score only manifest-matched frozen development partitions.
@register
class NimbleScoreAdapter(ModelAdapter):
    model_id = "nimble-scorer"

    def __init__(self):
        self._scorer = None
        self._model_path = None
        self._ready_sha256 = ""

    def load(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise LoadError("Nimble scoring requires the Arbiter CUDA worker")
        model = load_config().models.get(self.model_id)
        if model is None or not model.model_path:
            raise LoadError("nimble-scorer requires model_path in Arbiter local config")
        model_path = Path(model.model_path).expanduser()
        _ensure_checkpoint(model_path)
        self._ready_sha256 = _manifest_digest(model_path)
        _verify_weight_hashes(model_path)
        source_path = model_path.parent / "nimble-source"
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        try:
            from nimble.scoring.cuda_scorer import CudaCandidateScorer
        except ImportError as error:
            raise LoadError("pinned Nimble CUDA scorer dependencies are unavailable") from error
        self._scorer = CudaCandidateScorer(
            model_path=str(model_path), model_id=_MODEL_ID, revision=_MODEL_REVISION,
            max_input_tokens=8192, temperature=1.0, device_map="auto",
            max_gpu_memory="20GiB", dtype="bfloat16", allow_uncalibrated=True,
        )
        self._model_path = model_path
        log.info("nimble-scorer loaded revision=%s dtype=bfloat16 ready_sha256=%s",
                 _MODEL_REVISION, self._ready_sha256)

    def unload(self) -> None:
        self._scorer = None
        self._model_path = None
        self._cleanup_gpu()

    def estimate_time(self, params: dict) -> float:
        expected = _DEV_SUITES.get(params.get("suite"), {"records": 1204})
        return float(expected["records"] * 1000)

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        scorer = self._scorer
        if scorer is None or self._model_path is None:
            raise InferenceError("nimble-scorer is not loaded")
        payload, records, expected = _read_records(params)
        started = time.perf_counter()
        rows_path = output_dir / "rows.jsonl"
        with rows_path.open("w", encoding="utf-8") as stream:
            for index, record in enumerate(records):
                self._check_cancel(cancel_flag)
                context, schema = _schema_for(record)
                score_started = time.perf_counter()
                response = scorer.score(context, schema, mode="independent")
                elapsed_ms = 1000 * (time.perf_counter() - score_started)
                prediction = _result_rows(record, response, elapsed_ms)
                stream.write(json.dumps({"index": index, "record_id": record["_meta"]["id"],
                                         "rows": prediction}, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if (index + 1) % 50 == 0:
                    log.info("nimble-scorer suite=%s records=%d/%d", params["suite"], index + 1,
                             len(records))
        checkpoint_manifest = json.loads((self._model_path / "READY.json").read_text())
        checkpoint_manifest.pop("base_path", None)
        report = {
            "suite": params["suite"], "partition": "development", "records": len(records),
            "dataset_sha256": hashlib.sha256(payload).hexdigest(),
            "model": _MODEL_ID, "model_revision": _MODEL_REVISION,
            "base_model": _BASE_ID, "base_revision": _BASE_REVISION,
            "source_commit": _SOURCE_COMMIT, "adapter_sha256": _ADAPTER_SHA256,
            "ready_sha256": self._ready_sha256, "checkpoint_manifest": checkpoint_manifest,
            "dtype": "bfloat16", "temperature": 1.0,
            "temperature_fitted": False, "backend": response["backend"],
            "runtime": response["runtime"], "score_seconds": time.perf_counter() - started,
            "full_vocabulary_projection": False,
        }
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        archive_path = output_dir / "nimble-results.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.write(rows_path, "rows.jsonl")
            archive.write(report_path, "report.json")
        return {"file": archive_path.name, "format": "zip", "suite": params["suite"],
                "records": len(records), "score_seconds": report["score_seconds"]}


def _ensure_nimble_source(source_path: Path) -> None:
    if source_path.exists():
        completed = subprocess.run(["git", "-C", str(source_path), "rev-parse", "HEAD"],
                                   check=True, capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(source_path), "status", "--porcelain"],
                               check=True, capture_output=True, text=True)
        if completed.stdout.strip() != _SOURCE_COMMIT or dirty.stdout.strip():
            raise LoadError("cached Nimble source is not the exact clean pinned commit")
        return
    source_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                    "https://github.com/bespokelabsai/nimble.git", str(source_path)],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(source_path), "fetch", "--depth=1", "origin", _SOURCE_COMMIT],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(source_path), "checkout", "--detach", _SOURCE_COMMIT],
                   check=True, capture_output=True, text=True)


def _result_rows(record: dict, response: dict, latency_ms: float) -> list[dict]:
    fields = response.get("fields")
    questions = record["questions"]
    if not isinstance(fields, dict) or set(fields) != set(questions):
        raise InferenceError("Nimble returned different question IDs than the request")
    metadata = record["_meta"]
    result = []
    for question_id, question in questions.items():
        keys, target = _target(question)
        probabilities = _safe_probability(fields[question_id]["scores"], keys)
        result.append({
            "id": metadata["id"], "group": metadata.get("group_id", metadata["id"]),
            "question": question_id, "source": metadata["source"], "task": question["src"],
            "type": question["type"], "variant": metadata.get("variant", "clean"),
            "keys": keys, "label": target, "p": probabilities,
            "pair_id": metadata.get("pair_id"), "sibling": metadata.get("sibling"),
            "prompt_token_count": fields[question_id].get("prompt_token_count"),
            "prompt_token_sha256": fields[question_id].get("prompt_token_sha256"),
            "latency_ms": latency_ms,
        })
    return result
