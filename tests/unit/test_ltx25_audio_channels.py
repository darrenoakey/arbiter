"""Audio channel contract for the LTX 2.5 encode lane.

The audio VAE's first convolution is stereo, so a mono waveform used to
fail inside the GPU phase with a bare tensor-shape error and trip the
per-model circuit breaker after ten identical failures. These tests pin
the coercion that makes mono callers work, using real arrays and the real
functions — the channel gather is index-based so numpy here behaves
exactly as torch does inside the spark runner.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from arbiter.adapters.base import InferenceError
from arbiter.adapters.ltx25_encode import (
    AUDIO_VAE_CHANNELS,
    _audio_channel_index,
    _coerce_audio_channels,
    _stereo_audio,
)


@dataclasses.dataclass(frozen=True)
class _FrozenAudio:
    """Stands in for `ltx_core.types.Audio`, which only exists on spark.

    Same shape of value: a frozen dataclass carrying a waveform and its
    sampling rate. `_stereo_audio` reaches for exactly those two fields.
    """

    waveform: np.ndarray
    sampling_rate: int


def test_mono_waveform_is_duplicated_into_both_stereo_channels() -> None:
    mono = np.arange(6, dtype=np.float32).reshape(1, 1, 6)

    stereo = _coerce_audio_channels(mono)

    assert stereo.shape == (1, 2, 6)
    assert np.array_equal(stereo[0, 0], mono[0, 0])
    assert np.array_equal(stereo[0, 1], mono[0, 0])


def test_stereo_waveform_is_returned_untouched() -> None:
    already = np.arange(12, dtype=np.float32).reshape(1, 2, 6)

    assert _coerce_audio_channels(already) is already


def test_surplus_channels_are_dropped_to_the_vae_width() -> None:
    surround = np.arange(24, dtype=np.float32).reshape(1, 4, 6)

    coerced = _coerce_audio_channels(surround)

    assert coerced.shape == (1, AUDIO_VAE_CHANNELS, 6)
    assert np.array_equal(coerced[0, 0], surround[0, 0])
    assert np.array_equal(coerced[0, 1], surround[0, 1])


def test_mono_source_cycles_to_fill_a_wider_target() -> None:
    assert _audio_channel_index(1, 4) == [0, 0, 0, 0]
    assert _audio_channel_index(2, 4) == [0, 1, 0, 1]
    assert _audio_channel_index(3, 3) == [0, 1, 2]


def test_channelless_audio_is_refused_rather_than_gathered() -> None:
    with pytest.raises(InferenceError, match="no channels"):
        _audio_channel_index(0, AUDIO_VAE_CHANNELS)


def test_stereo_audio_rebuilds_the_frozen_container_and_keeps_rate() -> None:
    mono = _FrozenAudio(waveform=np.ones((1, 1, 4), dtype=np.float32), sampling_rate=24_000)

    fixed = _stereo_audio(mono)

    assert fixed.waveform.shape == (1, AUDIO_VAE_CHANNELS, 4)
    assert fixed.sampling_rate == 24_000
    assert mono.waveform.shape == (1, 1, 4), "the caller's audio must not be mutated"


def test_stereo_audio_returns_the_same_object_when_no_work_is_needed() -> None:
    already = _FrozenAudio(waveform=np.ones((1, 2, 4), dtype=np.float32), sampling_rate=48_000)

    assert _stereo_audio(already) is already
