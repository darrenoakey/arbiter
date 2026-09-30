from __future__ import annotations

import json
from pathlib import Path

import pytest

from arbiter.adapters.base import InferenceError
from arbiter.adapters.nimble_score import (
    NimbleScoreAdapter,
    _DEV_SUITES,
    _MODEL_REVISION,
    _SOURCE_COMMIT,
    _safe_probability,
    _schema_for,
    _target,
    _read_records,
)
from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, NimbleScoreParams


_FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "nimble_score" / "decision-v4-development.json"


def test_nimble_score_job_and_checkpoint_pins_are_exact():
    assert JOB_TYPE_TO_MODEL["nimble-score"] == "nimble-scorer"
    assert JOB_TYPE_PARAMS["nimble-score"] is NimbleScoreParams
    assert NimbleScoreAdapter.model_id == "nimble-scorer"
    assert _MODEL_REVISION == "bd792f44ec8e265be861bfcdf4e05967ffe0e858"
    assert _SOURCE_COMMIT == "62076b4f2d365b5879dafcf7f6dd072a1fe76df7"
    assert _DEV_SUITES["decision-v4"]["records"] == 1204
    assert _DEV_SUITES["transfer-v4"]["records"] == 764


def test_development_fixture_builds_seventy_seven_candidate_prompt_without_label():
    record = json.loads(_FIXTURE.read_text())
    context, schema = _schema_for(record)
    question = record["questions"]["intent"]
    keys, target = _target(question)
    assert context == record["state"]
    assert schema["intent"]["choices"] == list(question["criteria"])
    assert len(keys) == 77
    assert keys[target] == question["label"]
    assert "label" not in schema["intent"]


def test_development_schema_accepts_valid_candidate_probabilities():
    keys = ["false", "true"]
    probabilities = _safe_probability({"false": 0.25, "true": 0.75}, keys)
    assert probabilities == [0.25, 0.75]


def test_development_schema_rejects_invalid_candidate_probabilities():
    with pytest.raises(InferenceError, match="sum"):
        _safe_probability({"false": 0.25, "true": 0.5}, ["false", "true"])


def test_locked_partition_name_is_rejected_before_file_access():
    with pytest.raises(InferenceError, match="only decision-v4/development"):
        _read_records({"suite": "decision-v4/test", "dataset_file": "/not/read"})


def test_job_params_accept_only_the_two_development_suite_names():
    params = NimbleScoreParams(suite="decision-v4", dataset_file="/mnt/arbiter-store/inbox/dev.jsonl")
    assert params.suite == "decision-v4"
    with pytest.raises(ValueError):
        NimbleScoreParams(suite="decision-v4/test", dataset_file="/not/read")


def test_spark_config_uses_the_arbiter_worker_and_pinned_cache_path():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "config/spark/nimble-scorer.model.json").read_text())
    assert config["memory_gb"] == 24
    assert config["max_runtime_seconds"] == 14400
    assert config["model_path"] == "/mnt/t9/models/nimble-scorer-bd792/merged"
    assert config["worker_cmd"] == [
        "/home/darren/src/arbiter/venvs/nimble-scorer/bin/python",
        "-m", "arbiter.worker_main", "nimble-scorer",
    ]
