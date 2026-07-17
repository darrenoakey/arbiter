"""Parity tests for job-type registration.

A job type is only fully wired when it appears in BOTH the Python maps
(JobType enum, JOB_TYPE_TO_MODEL, JOB_TYPE_PARAMS) and the Go JobTypeToModel
map (cmd/arbiter/config.go — validated separately by `go test`). These tests
guard the Python side and specifically the voice-pipeline additions so a
half-registered type can't ship.
"""
from __future__ import annotations

from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, JobType


def test_every_jobtype_has_a_model_and_params():
    for jt in JobType:
        assert jt.value in JOB_TYPE_TO_MODEL, f"{jt.value} missing from JOB_TYPE_TO_MODEL"
        assert jt.value in JOB_TYPE_PARAMS, f"{jt.value} missing from JOB_TYPE_PARAMS"


def test_no_orphan_map_entries():
    values = {jt.value for jt in JobType}
    assert set(JOB_TYPE_TO_MODEL) == values
    assert set(JOB_TYPE_PARAMS) == values


def test_voice_pipeline_job_types_registered():
    expected = {
        "demucs": "demucs",
        "rvc-train": "rvc-train",
        "rvc-convert": "rvc-convert",
    }
    for job_type, model in expected.items():
        assert JOB_TYPE_TO_MODEL[job_type] == model
        assert job_type in JOB_TYPE_PARAMS


def test_voice_param_schema_shapes():
    from arbiter.schemas import DemucsParams, RvcConvertParams, RvcTrainParams

    # demucs accepts either b64 or a staged file, and opts into inline b64.
    d = DemucsParams(audio_file="/x.wav")
    assert d.return_b64 is False

    # rvc-train requires a name and defaults to 40k / 300 epochs / rmvpe.
    t = RvcTrainParams(name="leo-laporte", dataset_file="/data")
    assert (t.sample_rate, t.epochs, t.f0_method) == (40000, 300, "rmvpe")

    # rvc-convert requires a model reference; transpose defaults to 0.
    c = RvcConvertParams(model="leo-laporte", audio_file="/in.wav")
    assert c.transpose == 0 and c.f0_method == "rmvpe"
