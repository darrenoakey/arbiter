import pytest

from arbiter.adapters.ltx2_clock import lattice_frame_count, native_frame_rate


def test_native_frame_rate_accepts_only_twenty_five() -> None:
    assert native_frame_rate({}) == 25
    assert native_frame_rate({"fps": 25}) == 25
    with pytest.raises(ValueError, match="requires 25 fps"):
        native_frame_rate({"fps": 24})


def test_lattice_frame_count_preserves_explicit_valid_count() -> None:
    assert lattice_frame_count({"num_frames": 121}, 25, 0.0, 4.84) == 121
    assert lattice_frame_count({"num_frames": 129}, 25, 4.84, 10.0) == 129
    with pytest.raises(ValueError, match=r"8n\+1"):
        lattice_frame_count({"num_frames": 125}, 25, 0.0, 5.0)
