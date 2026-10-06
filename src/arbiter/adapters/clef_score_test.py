from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from arbiter.adapters import clef_score
from arbiter.adapters.base import InferenceError
from arbiter.adapters.clef_score import (
    RELEASES,
    ClefFlashScoreAdapter,
    ClefScoreAdapter,
    latency_summary,
    read_records,
    validate_record,
)
from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, ClefScoreParams

_QUESTION = {
    "type": "choice",
    "instructions": "Which editorial-interest category best fits this news headline?",
    "criteria": {"great": "Exceptional.", "good": "Worthwhile.", "other": "Not a priority."},
}


def test_clef_jobs_and_release_pins_are_exact():
    assert JOB_TYPE_TO_MODEL["clef-score"] == "clef-scorer"
    assert JOB_TYPE_TO_MODEL["clef-flash-score"] == "clef-flash-scorer"
    assert JOB_TYPE_PARAMS["clef-score"] is ClefScoreParams
    assert ClefScoreAdapter.model_id == "clef-scorer"
    assert ClefFlashScoreAdapter.model_id == "clef-flash-scorer"
    assert RELEASES["clef-scorer"]["revision"] == "2f3de3dd85f379784083b0814d997ab627200f0c"
    assert RELEASES["clef-flash-scorer"]["revision"] == "17f0b0ad64efb65d273590632833508766b2aae6"
    assert len([n for n in RELEASES["clef-scorer"]["weights"] if n.startswith("model-")]) == 12
    assert len([n for n in RELEASES["clef-flash-scorer"]["weights"] if n.startswith("model-")]) == 4
    for release in RELEASES.values():
        assert "joint_head.safetensors" in release["weights"]
        assert all(len(digest) == 64 for digest in release["weights"].values())


def test_label_free_systemone_record_is_accepted():
    record = {"id": "news-0001", "state": "Council approves bike lanes", "questions": {"label": _QUESTION}}
    assert validate_record(record, set()) is record


@pytest.mark.parametrize("record", [
    {"id": "a", "state": "x", "questions": {"label": _QUESTION}, "label": "great"},
    {"id": "a", "state": "x", "questions": {"label": {**_QUESTION, "label": "great"}}},
    {"id": "a", "state": "x", "questions": {"label": {"type": "choice", "criteria": {}}}},
    {"id": "a", "state": "x", "questions": {"label": {"type": "score", "criteria": ["only"]}}},
    {"id": "a", "state": "x", "questions": {"label": {"type": "free-text"}}},
    {"id": "", "state": "x", "questions": {"label": _QUESTION}},
])
def test_records_carrying_labels_or_bad_schemas_are_rejected(record):
    with pytest.raises(InferenceError):
        validate_record(record, set())


def test_duplicate_record_ids_are_rejected():
    seen: set[str] = set()
    validate_record({"id": "a", "state": "x", "questions": {"q": _QUESTION}}, seen)
    with pytest.raises(InferenceError):
        validate_record({"id": "a", "state": "y", "questions": {"q": _QUESTION}}, seen)


def test_staged_dataset_must_be_in_inbox_and_match_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(clef_score, "_INBOX_ROOT", tmp_path)
    payload = (json.dumps({"id": "n1", "state": "Headline", "questions": {"label": _QUESTION}}) + "\n").encode()
    path = tmp_path / "news.jsonl"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert read_records({"dataset_file": str(path), "dataset_sha256": digest}) == (
        digest, [json.loads(payload)])
    with pytest.raises(InferenceError, match="checksum"):
        read_records({"dataset_file": str(path), "dataset_sha256": "0" * 64})
    outside = Path(__file__)
    with pytest.raises(InferenceError, match="inbox"):
        read_records({"dataset_file": str(outside), "dataset_sha256": digest})


def test_schema_rejects_malformed_checksum():
    with pytest.raises(ValueError):
        ClefScoreParams(dataset_file="/mnt/arbiter-store/inbox/x.jsonl", dataset_sha256="abc")


def test_latency_summary_uses_sorted_percentiles():
    summary = latency_summary([30.0, 10.0, 20.0, 40.0])
    assert summary["mean_ms"] == 25.0
    assert summary["p50_ms"] == 30.0
    assert summary["p95_ms"] == 40.0


def test_release_code_with_dataclasses_loads_as_a_registered_module(tmp_path, monkeypatch):
    code = tmp_path / "joint_schema_model.py"
    code.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class EncodedQuestion:\n"
        "    question_id: str\n"
    )
    monkeypatch.setattr(clef_score, "_CODE_SHA256", clef_score.file_sha256(code))
    module = clef_score._load_release_code(tmp_path, "clef-test-scorer")
    assert module.EncodedQuestion("label").question_id == "label"


def test_release_code_hash_mismatch_is_refused(tmp_path):
    (tmp_path / "joint_schema_model.py").write_text("x = 1\n")
    with pytest.raises(clef_score.LoadError):
        clef_score._load_release_code(tmp_path, "clef-test-scorer")
