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
        assert jt.value in JOB_TYPE_TO_MODEL, (
            f"{jt.value} missing from JOB_TYPE_TO_MODEL"
        )
        assert jt.value in JOB_TYPE_PARAMS, f"{jt.value} missing from JOB_TYPE_PARAMS"


def test_no_orphan_map_entries():
    values = {jt.value for jt in JobType}
    assert set(JOB_TYPE_TO_MODEL) == values
    assert set(JOB_TYPE_PARAMS) == values


def test_voice_pipeline_job_types_registered():
    expected = {
        "demucs": "demucs",
        "vocal-stem": "vocal-stem",
        "rvc-train": "rvc-train",
        "rvc-convert": "rvc-convert",
    }
    for job_type, model in expected.items():
        assert JOB_TYPE_TO_MODEL[job_type] == model
        assert job_type in JOB_TYPE_PARAMS


def test_voice_param_schema_shapes():
    from arbiter.schemas import (
        DemucsParams,
        LTX25Denoise1Params,
        RvcConvertParams,
        RvcTrainParams,
        VocalStemParams,
    )

    # demucs accepts either b64 or a staged file, and opts into inline b64.
    d = DemucsParams(audio_file="/x.wav")
    assert d.return_b64 is False

    # vocal-stem requires an on-disk audio_file; htdemucs/-14 LUFS defaults.
    v = VocalStemParams(audio_file="/x.mp3")
    assert v.model == "htdemucs"
    assert v.target_lufs == -14.0

    # ltx25-denoise1 defaults the a2v guidance lever to 3.0.
    p = LTX25Denoise1Params(encoded_file="/e.pt", audio_file="/x.mp3")
    assert p.a2v_guidance_scale == 3.0

    # rvc-train requires a name and defaults to 40k / 300 epochs / rmvpe.
    t = RvcTrainParams(name="leo-laporte", dataset_file="/data")
    assert (t.sample_rate, t.epochs, t.f0_method) == (40000, 300, "rmvpe")

    # rvc-convert requires a model reference; transpose defaults to 0.
    c = RvcConvertParams(model="leo-laporte", audio_file="/in.wav")
    assert c.transpose == 0 and c.f0_method == "rmvpe"


def test_minimax_h3_local_job_type_registered():
    from arbiter.schemas import VideoGenerateH3Params

    assert JOB_TYPE_TO_MODEL["video-generate-h3"] == "minimax-h3-local"
    params = VideoGenerateH3Params(
        prompt="A singer performs beneath falling snow",
        first_image_file="/shared/first.jpg",
        last_image_file="/shared/last.jpg",
        duration=5,
        width=960,
        height=544,
        seed=42,
        num_inference_steps=8,
    )
    assert params.duration == 5
    assert params.first_image_file == "/shared/first.jpg"
    assert params.last_image_file == "/shared/last.jpg"



def test_minimax_fast_h3_job_type_registered():
    from arbiter.schemas import VideoGenerateH3Params

    assert JOB_TYPE_TO_MODEL["video-generate-fast-h3"] == "minimax-fast-h3"
    params = VideoGenerateH3Params(
        prompt="Corinne and Arlene try pickleball",
        duration=5,
        num_inference_steps=4,
    )
    assert params.duration == 5
    assert params.num_inference_steps == 4


def test_music_generate_job_type_registered():
    from arbiter.schemas import MusicGenerateParams

    assert JOB_TYPE_TO_MODEL["music-generate"] == "music-generate"
    params = MusicGenerateParams(
        prompt="Epic cinematic orchestral music with soaring brass",
        lyrics="[verse]\nRise up now\n[chorus]\nFeel the power",
        audio_duration=60.0,
        num_inference_steps=50,
        guidance_scale=7.0,
        shift=3.0,
        seed=1234,
    )
    assert params.prompt == "Epic cinematic orchestral music with soaring brass"
    assert params.audio_duration == 60.0
    assert params.num_inference_steps == 50
    assert params.guidance_scale == 7.0
    assert params.shift == 3.0
    assert params.seed == 1234
    assert params.format == "mp3"


def test_music_generate_format_defaults_to_mp3_but_wav_is_selectable():
    from arbiter.schemas import MusicGenerateParams

    assert MusicGenerateParams(prompt="test").format == "mp3"
    assert MusicGenerateParams(prompt="test", format="wav").format == "wav"
