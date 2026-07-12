"""Spatial-lattice contracts for the split LTX-2 dev pipeline."""

import numpy as np
import pytest

from arbiter.adapters.base import InferenceError
from arbiter.adapters.ltx2_denoise2 import _crop_frames_to_target


def test_model_only_spatial_padding_is_cropped_to_exact_1080p() -> None:
    frames = np.arange(2 * 1088 * 1920 * 3, dtype=np.uint8).reshape(2, 1088, 1920, 3)

    cropped = _crop_frames_to_target(frames, 1920, 1080)

    assert cropped.shape == (2, 1080, 1920, 3)
    assert np.array_equal(cropped[:, 0], frames[:, 4])
    assert np.array_equal(cropped[:, -1], frames[:, 1083])


def test_model_crop_rejects_target_larger_than_decoded_frames() -> None:
    frames = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)

    with pytest.raises(InferenceError, match="exceeds decoded model frame"):
        _crop_frames_to_target(frames, 1920, 1088)
