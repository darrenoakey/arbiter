"""Mouth-placement guard for LatentSync output.

LatentSync re-inpaints a mouth inside an affine-aligned face crop. Two failure
modes recur in production and are both visible only by comparing the result
with the clip it was given:

* **Stray mouth outside the face.** On side-on heads the frontal mouth crop is
  pasted partly beyond the face outline, so pink lips float in front of the
  profile (Security Now, 2026-10).
* **Closed mouth on a dropped jaw.** LatentSync only repaints the mouth, so
  when the source jaw is open it paints a small closed mouth high on the face
  above a long empty chin (I Will Be Back, 2026-10).

This module scores every frame against the SOURCE frame's 106-point face
landmarks, flags frames showing either failure, and blends the source lower
face back into just those frames (dilated and ramped in time so the repair
never pops). Frames that pass are left byte-for-byte untouched; when nothing
is flagged the LatentSync file is returned as is.

It runs as a subprocess under LatentSync's own venv (which ships insightface
and the buffalo_l models it already uses), so the module imports only cv2 and
numpy at top level and loads insightface lazily for detection. The scoring and
blending functions are pure so they are unit-tested against recorded real
landmarks without a detector.

CLI: ``python latentsync_mouth_guard.py SOURCE SYNCED OUTPUT REPORT_JSON``
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# 106-point (insightface 2d106det) index groups.
CONTOUR = slice(0, 33)
LEFT_EYE = slice(33, 43)
RIGHT_EYE = slice(87, 97)
MOUTH = slice(52, 72)
NOSE = slice(72, 87)
NOSE_TIP = 86

MIN_FACE_WIDTH_PX = 40.0  # smaller faces are too blurry to judge or repair
DIFF_LEVEL = 22  # blurred grey absdiff that counts as "LatentSync changed this pixel"
OUTSIDE_LIMIT = 0.06  # changed fraction of the just-outside-the-face band that marks a stray mouth
JAW_OPEN = 0.15  # source mouth-box dark fraction that means the jaw is dropped
JAW_CLOSED_RATIO = 0.6  # result darker than this share of the source counts as closed
JAW_MARGIN = 0.08  # and at least this much less dark in absolute terms
JAW_MIN_RUN = 4  # frames a closed-mouth-on-open-jaw run must span to count
JAW_RUN_GAP = 1  # single unflagged frames inside a run are bridged
DILATE_FRAMES = 2
RAMP = np.array([1, 2, 3, 2, 1], dtype=np.float64) / 9.0


@dataclass
class FrameScore:
    index: int
    face: bool
    outside: float = 0.0
    src_dark: float = 0.0
    syn_dark: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def bad(self) -> bool:
        return bool(self.reasons)


def face_width(landmarks: np.ndarray) -> float:
    contour = landmarks[CONTOUR]
    return float(contour[:, 0].max() - contour[:, 0].min())


def _lower_face_points(landmarks: np.ndarray) -> np.ndarray:
    contour = landmarks[CONTOUR]
    nose = landmarks[NOSE]
    nose_mid = nose[:, 1].mean()
    return np.vstack([contour[contour[:, 1] > nose_mid], nose[nose[:, 1] > nose_mid], landmarks[MOUTH]])


def lower_face_mask(shape: tuple[int, ...], landmarks: np.ndarray, grow: float) -> np.ndarray:
    """Filled hull of the lower face (below mid-nose), grown by ``grow`` face widths."""
    height, width = shape[:2]
    mask = np.zeros((height, width), np.uint8)
    hull = cv2.convexHull(_lower_face_points(landmarks).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    radius = int(round(grow * face_width(landmarks)))
    if radius > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    return mask


def changed_pixels(source: np.ndarray, synced: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), cv2.cvtColor(synced, cv2.COLOR_BGR2GRAY))
    return cv2.GaussianBlur(diff, (9, 9), 0) > DIFF_LEVEL


def outside_change(source: np.ndarray, synced: np.ndarray, landmarks: np.ndarray) -> float:
    """Changed fraction of a band just outside the lower-face outline.

    LatentSync should only alter pixels inside the face; a frontal mouth pasted
    beyond a profile shows up as change in this band.
    """
    band = (lower_face_mask(source.shape, landmarks, 0.30) > 0) & ~(lower_face_mask(source.shape, landmarks, 0.06) > 0)
    nose_y = landmarks[NOSE][:, 1].mean()
    band[: max(int(nose_y), 0)] = False
    area = int(band.sum())
    if area == 0:
        return 0.0
    return float((changed_pixels(source, synced) & band).sum()) / area


def mouth_darkness(image: np.ndarray, landmarks: np.ndarray) -> float:
    """Fraction of the mouth box much darker than cheek skin: an open mouth."""
    width = face_width(landmarks)
    centre = landmarks[MOUTH].mean(0)
    nose = landmarks[NOSE].mean(0)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows, cols = grey.shape
    eyes = np.linalg.norm(landmarks[LEFT_EYE].mean(0) - landmarks[RIGHT_EYE].mean(0))
    scale = max(eyes, 0.3 * width)
    cheek = grey[max(int(nose[1] - 0.2 * scale), 0):max(int(nose[1] + 0.1 * scale), 1),
                 max(int(nose[0] - 0.5 * scale), 0):max(int(nose[0] - 0.25 * scale), 1)]
    skin = float(np.median(cheek)) if cheek.size else float(np.median(grey))
    box = grey[max(int(centre[1] - 0.35 * scale), 0):min(int(centre[1] + 0.45 * scale), rows),
               max(int(centre[0] - 0.55 * scale), 0):min(int(centre[0] + 0.55 * scale), cols)]
    return float((box < 0.45 * skin).mean()) if box.size else 0.0


def score_frame(index: int, source: np.ndarray, synced: np.ndarray, landmarks: np.ndarray | None) -> FrameScore:
    if landmarks is None or face_width(landmarks) < MIN_FACE_WIDTH_PX:
        return FrameScore(index, face=False)
    score = FrameScore(index, face=True)
    score.outside = outside_change(source, synced, landmarks)
    if score.outside > OUTSIDE_LIMIT:
        score.reasons.append("mouth_outside_face")
    score.src_dark = mouth_darkness(source, landmarks)
    score.syn_dark = mouth_darkness(synced, landmarks)
    if (score.src_dark >= JAW_OPEN and score.syn_dark < JAW_CLOSED_RATIO * score.src_dark
            and score.src_dark - score.syn_dark > JAW_MARGIN):
        score.reasons.append("closed_mouth_on_open_jaw")
    return score


def repair_weights(bad: list[bool]) -> np.ndarray:
    """Dilate flagged frames and ramp their edges so restored frames fade in and out."""
    flags = np.array(bad, dtype=np.float64)
    if not flags.any():
        return np.zeros(len(flags))
    width = 2 * DILATE_FRAMES + 1
    dilated = np.convolve(flags, np.ones(width), "same").clip(0, 1)
    return np.convolve(dilated, RAMP, "same").clip(0, 1)


def blend_source(source: np.ndarray, synced: np.ndarray, landmarks: np.ndarray, weight: float) -> np.ndarray:
    """Feathered lower face (plus the band where stray lips float) from source."""
    width = face_width(landmarks)
    nose_mid = float(landmarks[NOSE][:, 1].mean())
    mask = lower_face_mask(source.shape, landmarks, 0.30).astype(np.float32)
    mask[: max(int(nose_mid - 0.10 * width), 0)] = 0.0  # never restore the eyes or above
    sigma = max(0.08 * width, 2.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    mask[mask < 0.01] = 0.0  # the blur tail must not nudge pixels far from the mouth
    mask[: max(int(nose_mid - 0.25 * width), 0)] = 0.0
    mask = mask[..., None] * weight
    mixed = synced.astype(np.float32) * (1 - mask) + source.astype(np.float32) * mask
    return mixed.round().clip(0, 255).astype(np.uint8)


def _read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"mouth guard: no frames decoded from {path}")
    return frames, fps


def detect_landmarks(frames: list[np.ndarray]) -> list[np.ndarray | None]:
    """106-point landmarks of the largest face per frame, via LatentSync's detector stack."""
    # This file lives beside arbiter's own ``insightface.py`` adapter; running it
    # as a script puts that directory first on sys.path and shadows the real
    # insightface package, so drop it before importing.
    here = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != here]
    sys.modules.pop("insightface", None)
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    found: list[np.ndarray | None] = []
    for frame in frames:
        faces = app.get(frame)
        if not faces:
            found.append(None)
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        found.append(np.asarray(face.landmark_2d_106, dtype=np.float32))
    return found


def require_persistent_jaw_flags(scores: list[FrameScore]) -> None:
    """Keep ``closed_mouth_on_open_jaw`` only inside sustained runs.

    LatentSync legitimately draws a smaller, correctly placed mouth on odd
    frames of an open-mouthed source (a different viseme), which scores the
    same as the defect frame by frame. The real defect, a closed mouth floating
    above a dropped jaw, persists for several frames, so the flag survives only
    in runs of at least JAW_MIN_RUN frames (gaps of JAW_RUN_GAP frames bridged).
    ``mouth_outside_face`` is a strong per-frame signal and is never filtered.
    """
    reason = "closed_mouth_on_open_jaw"
    flagged = [i for i, s in enumerate(scores) if reason in s.reasons]
    runs: list[list[int]] = []
    for i in flagged:
        if runs and i - runs[-1][-1] <= JAW_RUN_GAP + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    for run in runs:
        if run[-1] - run[0] + 1 < JAW_MIN_RUN:
            for i in run:
                scores[i].reasons.remove(reason)


def guard_frames(sources: list[np.ndarray], synced: list[np.ndarray],
                 landmarks: list[np.ndarray | None]) -> tuple[list[np.ndarray], list[FrameScore], np.ndarray]:
    """Score paired frames and return repaired frames; unpaired tail frames pass through."""
    count = min(len(sources), len(synced), len(landmarks))
    scores = [score_frame(i, sources[i], synced[i], landmarks[i]) for i in range(count)]
    require_persistent_jaw_flags(scores)
    weights = repair_weights([s.bad for s in scores])
    out = list(synced)
    for i in range(count):
        if weights[i] > 0 and landmarks[i] is not None and scores[i].face:
            out[i] = blend_source(sources[i], synced[i], landmarks[i], float(weights[i]))
    return out, scores, weights


def _encode(frames: list[np.ndarray], fps: float, audio_from: Path, output: Path) -> None:
    height, width = frames[0].shape[:2]
    video_only = output.with_suffix(".guard-video.mp4")
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
         "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
         "-pix_fmt", "yuv420p", str(video_only)],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    for frame in frames:
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("mouth guard: video encode failed")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video_only), "-i", str(audio_from),
                    "-map", "0:v", "-map", "1:a?", "-c", "copy", str(output)], check=True)
    video_only.unlink()


def run(source: Path, synced: Path, output: Path, landmarks: list[np.ndarray | None] | None = None) -> dict:
    """Guard one LatentSync result. ``landmarks`` (per source frame) are detected when not given."""
    source_frames, _ = _read_frames(source)
    synced_frames, fps = _read_frames(synced)
    if landmarks is None:
        landmarks = detect_landmarks(source_frames[: len(synced_frames)])
    repaired, scores, weights = guard_frames(source_frames, synced_frames, landmarks)
    touched = [i for i, w in enumerate(weights) if w > 0]
    if touched:
        _encode(repaired, fps, synced, output)
    else:
        shutil.copyfile(synced, output)
    return {
        "frames": len(synced_frames),
        "faces": sum(s.face for s in scores),
        "flagged": [{"frame": s.index, "reasons": s.reasons, "outside": round(s.outside, 3),
                     "src_dark": round(s.src_dark, 3), "syn_dark": round(s.syn_dark, 3)}
                    for s in scores if s.bad],
        "repaired_frames": len(touched),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    report = run(Path(argv[1]), Path(argv[2]), Path(argv[3]))
    Path(argv[4]).write_text(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
