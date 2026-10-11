"""Param-validation and registration tests for the YuE2 music adapter
(job type `music-generate-yue2`, model `yue2`, weights m-a-p/YuE2-3B).

The adapter validates its params before touching the pipeline once a pipe
handle exists, so these checks exercise the real validation code on any
machine — the failure mode under test is a malformed job being rejected
with a clear error, not the model running. Real generation runs on spark
via arbiter jobs (the worker venv `venvs/yue2` carries the model package).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from arbiter.adapters.yue2 import Yue2Adapter
from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, Yue2MusicParams

OUT = Path("/tmp/yue2-unit-out")


class _Sentinel(Exception):
    """Raised by the fake pipe so a 'reached the pipeline' call is observable."""


class _FakePipe:
    def __call__(self, **kwargs: Any) -> Any:
        raise _Sentinel(kwargs)


def _yue2(params: dict) -> None:
    adapter = Yue2Adapter()
    adapter._pipe = _FakePipe()  # validation gate only; never a real model
    adapter.infer(params, OUT, threading.Event())


class TestYue2Registration:
    def test_job_type_registered(self):
        assert JOB_TYPE_TO_MODEL["music-generate-yue2"] == "yue2"
        assert "music-generate-yue2" in JOB_TYPE_PARAMS

    def test_params_defaults(self):
        p = Yue2MusicParams(style="funk, upbeat, horns", lyrics="la la la")
        assert p.cot == "full"
        assert p.format == "mp3"
        assert p.seed is None
        assert p.cfg_scale is None
        assert p.abc is None


class TestYue2ParamValidation:
    def test_missing_style_rejected(self):
        with pytest.raises(ValueError, match="style is required"):
            _yue2({"lyrics": "la la la"})

    def test_empty_style_rejected(self):
        with pytest.raises(ValueError, match="style is required"):
            _yue2({"style": "   ", "lyrics": "la la la"})

    def test_missing_lyrics_rejected(self):
        with pytest.raises(ValueError, match="lyrics are required"):
            _yue2({"style": "funk, upbeat, horns"})

    def test_invalid_cot_rejected(self):
        with pytest.raises(ValueError, match="cot must be one of"):
            _yue2({"style": "funk", "lyrics": "la la la", "cot": "sideways"})

    def test_valid_job_reaches_pipeline_with_defaults(self):
        with pytest.raises(_Sentinel) as sent:
            _yue2({"style": "funk, upbeat, horns", "lyrics": "la la la"})
        kwargs = sent.value.args[0]
        assert kwargs["cot"] == "full"

    def test_all_cot_modes_accepted(self):
        for cot in ("full", "melody", "off"):
            with pytest.raises(_Sentinel) as sent:
                _yue2({"style": "funk", "lyrics": "la la la", "cot": cot})
            assert sent.value.args[0]["cot"] == cot
