"""LatentSync mouth guard against real recorded frames.

Fixtures are crops of real LatentSync source/result frame pairs with the real
insightface 106-point landmarks detected on the source frame (see
tests/fixtures/latentsync_mouth_guard/README.json for provenance):

* jaw_bad     — I Will Be Back chunk 0 frame 24: closed mouth painted over an open jaw.
* outside_bad — Patrick ragtime chunk 3 frame 51: mouth smeared off the side of the face.
* clean       — 2026-09-07 digest chunk 0 frame 10: a correct sync.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from arbiter.adapters import latentsync_mouth_guard as guard

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "latentsync_mouth_guard"


def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = cv2.imread(str(FIXTURES / f"{name}_source.png"))
    synced = cv2.imread(str(FIXTURES / f"{name}_synced.png"))
    landmarks = np.load(FIXTURES / f"{name}_landmarks.npy")
    return source, synced, landmarks


@pytest.mark.parametrize(
    ("name", "reasons"),
    [("jaw_bad", ["closed_mouth_on_open_jaw"]), ("outside_bad", ["mouth_outside_face"]), ("clean", [])],
)
def test_score_frame_flags_real_defects_and_passes_a_clean_sync(name: str, reasons: list[str]) -> None:
    source, synced, landmarks = load(name)
    assert guard.score_frame(0, source, synced, landmarks).reasons == reasons


def test_identical_frames_are_never_flagged() -> None:
    source, _, landmarks = load("jaw_bad")
    score = guard.score_frame(0, source, source.copy(), landmarks)
    assert score.face and not score.bad and score.outside == 0.0


def test_missing_or_tiny_face_is_skipped() -> None:
    source, synced, landmarks = load("jaw_bad")
    assert not guard.score_frame(0, source, synced, None).face
    assert not guard.score_frame(0, source, synced, landmarks * 0.05).face


def test_repair_weights_dilate_and_ramp_around_flagged_frames() -> None:
    weights = guard.repair_weights([False] * 10 + [True] + [False] * 10)
    touched = np.nonzero(weights)[0]
    assert touched.min() == 6 and touched.max() == 14
    assert weights[10] == pytest.approx(1.0)
    assert weights[6] < weights[8] < weights[10]
    assert not guard.repair_weights([False] * 5).any()


def test_guard_frames_restores_a_sustained_jaw_defect_and_leaves_far_frames_untouched() -> None:
    bad_source, bad_synced, bad_landmarks = load("jaw_bad")
    sources = [bad_source] * 16
    synced = [bad_source.copy() for _ in range(16)]
    for i in range(6, 10):
        synced[i] = bad_synced
    landmarks = [bad_landmarks] * 16
    out, scores, _ = guard.guard_frames(sources, synced, landmarks)
    assert [s.index for s in scores if s.bad] == [6, 7, 8, 9]
    assert out[0] is synced[0] and out[15] is synced[15]
    repaired = guard.score_frame(7, bad_source, out[7], bad_landmarks)
    assert not repaired.bad
    eye_top = int(bad_landmarks[guard.LEFT_EYE][:, 1].min())
    assert np.array_equal(out[7][:eye_top], bad_synced[:eye_top])


def test_isolated_jaw_frames_are_treated_as_a_legitimate_viseme() -> None:
    bad_source, bad_synced, bad_landmarks = load("jaw_bad")
    synced = [bad_source.copy() for _ in range(16)]
    synced[3] = synced[4] = synced[11] = bad_synced
    out, scores, weights = guard.guard_frames([bad_source] * 16, synced, [bad_landmarks] * 16)
    assert not any(s.bad for s in scores) and not weights.any()
    assert all(a is b for a, b in zip(out, synced))


def test_one_frame_gap_is_bridged_inside_a_jaw_run() -> None:
    scores = [guard.FrameScore(i, face=True) for i in range(10)]
    for i in (2, 3, 5):
        scores[i].reasons.append("closed_mouth_on_open_jaw")
    scores[8].reasons.append("mouth_outside_face")
    guard.require_persistent_jaw_flags(scores)
    assert [s.index for s in scores if s.bad] == [2, 3, 5, 8]


def test_outside_face_defect_is_repaired_on_a_single_frame() -> None:
    source, synced, landmarks = load("outside_bad")
    frames = [source.copy() for _ in range(9)]
    frames[4] = synced
    out, scores, _ = guard.guard_frames([source] * 9, frames, [landmarks] * 9)
    assert [s.index for s in scores if s.bad] == [4]
    assert not guard.score_frame(4, source, out[4], landmarks).bad


def write_clip(path: Path, frames: list[np.ndarray]) -> None:
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
         "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv444p", str(path)],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    assert process.wait() == 0


def test_run_repairs_a_bad_clip_and_reports_it(tmp_path: Path) -> None:
    source, synced, landmarks = load("jaw_bad")
    even = (source.shape[1] // 2 * 2, source.shape[0] // 2 * 2)
    scale = np.array([even[0] / source.shape[1], even[1] / source.shape[0]], dtype=np.float32)
    src_frames = [cv2.resize(source, even)] * 10
    syn_frames = [cv2.resize(source, even)] * 3 + [cv2.resize(synced, even)] * 4 + [cv2.resize(source, even)] * 3
    src_path, syn_path, out_path = tmp_path / "src.mp4", tmp_path / "syn.mp4", tmp_path / "out.mp4"
    write_clip(src_path, src_frames)
    write_clip(syn_path, syn_frames)
    report = guard.run(src_path, syn_path, out_path, landmarks=[landmarks * scale] * 10)
    assert [item["frame"] for item in report["flagged"]] == [3, 4, 5, 6]
    assert all(item["reasons"] == ["closed_mouth_on_open_jaw"] for item in report["flagged"])
    assert report["repaired_frames"] >= 4
    repaired, _ = guard._read_frames(out_path)
    assert len(repaired) == 10
    assert not guard.score_frame(4, src_frames[4], repaired[4], landmarks * scale).bad


def test_cli_rejects_wrong_arguments() -> None:
    result = subprocess.run([sys.executable, guard.__file__], capture_output=True, text=True)
    assert result.returncode == 2


def test_run_copies_a_clean_clip_unchanged(tmp_path: Path) -> None:
    source, synced, landmarks = load("clean")
    even = (source.shape[1] // 2 * 2, source.shape[0] // 2 * 2)
    frames_src = [cv2.resize(source, even)] * 4
    frames_syn = [cv2.resize(synced, even)] * 4
    src_path, syn_path, out_path = tmp_path / "src.mp4", tmp_path / "syn.mp4", tmp_path / "out.mp4"
    write_clip(src_path, frames_src)
    write_clip(syn_path, frames_syn)
    scale = np.array([even[0] / source.shape[1], even[1] / source.shape[0]], dtype=np.float32)
    # Detection needs insightface, which only LatentSync's venv on spark carries;
    # pass the fixture's real detected landmarks instead.
    report = guard.run(src_path, syn_path, out_path, landmarks=[landmarks * scale] * 4)
    assert report["frames"] == 4 and report["repaired_frames"] == 0 and report["flagged"] == []
    assert out_path.read_bytes() == syn_path.read_bytes()
    assert json.loads(json.dumps(report)) == report
