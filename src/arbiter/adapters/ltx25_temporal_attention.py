"""Whole-clip temporal attention-ramp contract and pinned LTX runtime loader."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from arbiter.adapters.base import InferenceError

END_IMAGE_ATTENTION_STRENGTHS = "end_image_attention_strengths"
RUNTIME_PROVENANCE = "ltx25_runtime_provenance"
CAPABILITY_VERSION = "ltx25-temporal-endpoint-v1"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _REPOSITORY_ROOT / "runtime" / "ltx25" / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_provenance(manifest: dict) -> dict:
    return {
        "capability_version": manifest["capability_version"],
        "release_id": manifest["release_id"],
        "source_archive_sha256": manifest["source"]["archive_sha256"],
        "source_commit": manifest["source"]["commit"],
        "source_tree": manifest["source"]["tree"],
        "source_runner_sha256": manifest["source"]["runner_sha256"],
        "source_runner_test_sha256": manifest["source"]["runner_test_sha256"],
        "patch_sha256": manifest["patch"]["sha256"],
    }


def load_ltx25_runtime(config_path: Path | None = None):
    """Load only the configured immutable runtime and attest every patched file."""
    config = config_path or (_REPOSITORY_ROOT / "local" / "config.toml")
    if not config.is_file():
        raise InferenceError(f"LTX 2.5 runtime config is missing: {config}")
    settings = tomllib.loads(config.read_text()).get("ltx25", {})
    runtime_value = settings.get("runtime_path")
    if not isinstance(runtime_value, str) or not runtime_value:
        raise InferenceError(f"LTX 2.5 runtime_path is missing from {config}")
    runtime = Path(runtime_value)
    if not runtime.is_absolute() or not runtime.is_dir():
        raise InferenceError(f"LTX 2.5 runtime_path is invalid: {runtime}")

    manifest = json.loads(_MANIFEST_PATH.read_text())
    expected = _expected_provenance(manifest)
    provenance_path = runtime / "arbiter-ltx25-runtime.json"
    if not provenance_path.is_file() or json.loads(provenance_path.read_text()) != expected:
        raise InferenceError(f"LTX 2.5 runtime provenance mismatch: {provenance_path}")
    for key in ("release_id", "capability_version", "patch_sha256"):
        if settings.get(key) != expected[key]:
            raise InferenceError(f"LTX 2.5 config {key} does not match runtime provenance")
    for relative, checksum in manifest["patched_files"].items():
        path = runtime / relative
        if not path.is_file() or _sha256(path) != checksum:
            raise InferenceError(f"LTX 2.5 runtime file checksum mismatch: {path}")

    roots = [runtime, runtime / "packages" / "ltx-core" / "src", runtime / "packages" / "ltx-pipelines" / "src"]
    for name, module in tuple(sys.modules.items()):
        if name != "video_fast_gpu" and not name.startswith(("ltx_core", "ltx_pipelines")):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None and not Path(module_file).resolve().is_relative_to(runtime.resolve()):
            raise InferenceError(f"mixed LTX 2.5 runtime module loaded before activation: {name} from {module_file}")
    for root in reversed(roots):
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    runner = importlib.import_module("video_fast_gpu")
    capability = runner.temporal_attention_capability()
    if capability.get("version") != CAPABILITY_VERSION or capability.get("both_stages") is not True:
        raise InferenceError(f"LTX 2.5 temporal-attention capability mismatch: {capability!r}")
    return runner, expected


def normalize_end_image_attention_strengths(
    value: object,
    num_frames: int,
) -> tuple[float, ...] | None:
    """Validate and expand an optional frame-space attention schedule."""
    if value is None:
        return None
    if type(num_frames) is not int or num_frames < 1:
        raise InferenceError(
            f"num_frames must be a positive integer, got {num_frames!r}"
        )
    if type(value) is not list:
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} must be a JSON list or null, "
            f"got {type(value).__name__}"
        )
    if num_frames < 2:
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} requires at least 2 frames"
        )
    strengths = tuple(_attention_strength(item) for item in value)
    if len(strengths) == num_frames:
        return strengths
    if len(strengths) != 2:
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} must contain 2 or {num_frames} "
            f"values, got {len(strengths)}"
        )
    first, last = strengths
    return tuple(
        first + (last - first) * frame / (num_frames - 1)
        for frame in range(num_frames)
    )


def _attention_strength(value: object) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} values must be JSON numbers, "
            f"got {type(value).__name__}: {value!r}"
        )
    strength = float(value)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} values must be finite and in "
            f"[0, 1], got {value!r}"
        )
    return strength


def _frame_index(item: object) -> int:
    if hasattr(item, "frame_idx"):
        value = item.frame_idx
    elif isinstance(item, dict):
        value = item.get("frame_idx", item.get("frame"))
    elif isinstance(item, (tuple, list)) and len(item) >= 2:
        value = item[1]
    else:
        raise InferenceError(
            f"ramp image has no frame_idx: {type(item).__name__}"
        )
    if type(value) is not int:
        raise InferenceError(
            f"ramp image frame_idx must be an integer, got "
            f"{type(value).__name__}: {value!r}"
        )
    return value


def validate_ramp_images(images: object, num_frames: int) -> None:
    """Require one start image and one endpoint guide, with no scheduled copies."""
    if not isinstance(images, list):
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} requires images to be a list"
        )
    endpoint = num_frames - 1
    indices = [_frame_index(item) for item in images]
    if indices.count(0) != 1:
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} requires exactly one start image"
        )
    if indices.count(endpoint) != 1:
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} requires exactly one endpoint "
            f"guide at frame {endpoint}"
        )
    unexpected = [index for index in indices if index not in (0, endpoint)]
    if unexpected:
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} rejects repeated scheduled end "
            f"guides at frames {unexpected}"
        )


def _validate_stage1_endpoint(conditionings: object, num_frames: int) -> None:
    if not isinstance(conditionings, list):
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} requires stage_1_conditionings "
            "to be a list"
        )
    endpoint = num_frames - 1
    guided_frames = [
        item.frame_idx
        for item in conditionings
        if type(getattr(item, "frame_idx", None)) is int
    ]
    if guided_frames.count(endpoint) != 1:
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} requires exactly one encoded "
            f"endpoint guide at frame {endpoint}"
        )
    unexpected = [frame for frame in guided_frames if frame != endpoint]
    if unexpected:
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} rejects encoded scheduled end "
            f"guides at frames {unexpected}"
        )


def preserve_encoded_temporal_attention_ramp(
    data: dict,
    value: object,
    num_frames: int,
    runtime_provenance: dict | None = None,
) -> tuple[float, ...] | None:
    """Validate endpoint identity and store the expanded schedule in the bundle."""
    strengths = normalize_end_image_attention_strengths(value, num_frames)
    if strengths is None:
        return None
    validate_ramp_images(data.get("images"), num_frames)
    _validate_stage1_endpoint(data.get("stage_1_conditionings"), num_frames)
    if runtime_provenance is None:
        raise InferenceError("LTX 2.5 runtime provenance is required for temporal attention")
    capability = data.get("ltx25_temporal_attention_capability")
    if not isinstance(capability, dict) or capability.get("version") != CAPABILITY_VERSION:
        raise InferenceError(f"LTX 2.5 encode capability mismatch: {capability!r}")
    data[END_IMAGE_ATTENTION_STRENGTHS] = list(strengths)
    data[RUNTIME_PROVENANCE] = dict(runtime_provenance)
    return strengths


def resolve_denoise_temporal_attention_ramp(
    data: dict,
    params: dict,
    runtime_provenance: dict | None = None,
) -> tuple[float, ...] | None:
    """Resolve bundle/request schedules and reject stale or ambiguous identity."""
    num_frames = data.get("num_frames")
    bundle = normalize_end_image_attention_strengths(
        data.get(END_IMAGE_ATTENTION_STRENGTHS), num_frames
    )
    requested = normalize_end_image_attention_strengths(
        params.get(END_IMAGE_ATTENTION_STRENGTHS), num_frames
    )
    if END_IMAGE_ATTENTION_STRENGTHS in params and requested != bundle:
        raise InferenceError(
            f"{END_IMAGE_ATTENTION_STRENGTHS} does not match the encoded bundle"
        )
    strengths = requested if END_IMAGE_ATTENTION_STRENGTHS in params else bundle
    if strengths is None:
        return None
    validate_ramp_images(data.get("images"), num_frames)
    _validate_stage1_endpoint(data.get("stage_1_conditionings"), num_frames)
    if runtime_provenance is None or data.get(RUNTIME_PROVENANCE) != runtime_provenance:
        raise InferenceError("encoded bundle LTX 2.5 runtime identity does not match the active runtime")
    return strengths


def sample_video_query_strengths(
    strengths: Sequence[float],
    positions,
    num_video_tokens: int,
):
    """Sample decoded-frame strengths at actual latent-query time positions."""
    import torch

    if positions.ndim not in (3, 4) or positions.shape[1] < 1:
        raise InferenceError(
            "video positions must have shape [B, axes, tokens] or "
            f"[B, axes, tokens, bounds], got {tuple(positions.shape)}"
        )
    if num_video_tokens < 1 or num_video_tokens > positions.shape[2]:
        raise InferenceError(
            f"num_video_tokens must be in [1, {positions.shape[2]}], got {num_video_tokens}"
        )
    frame_weights = torch.as_tensor(
        strengths, device=positions.device, dtype=torch.float32
    )
    if frame_weights.ndim != 1 or frame_weights.numel() < 1:
        raise InferenceError("expanded attention strengths must be a non-empty vector")
    times = positions[:, 0, :num_video_tokens].to(torch.float32)
    if times.ndim == 3:
        times = times.mean(dim=-1)
    first = times.amin(dim=1, keepdim=True)
    span = times.amax(dim=1, keepdim=True) - first
    if frame_weights.numel() > 1 and torch.any(span <= 0):
        raise InferenceError("video query temporal layout has no positive span")
    coordinates = torch.zeros_like(times)
    if frame_weights.numel() > 1:
        coordinates = (times - first) * (frame_weights.numel() - 1) / span
    lower = coordinates.floor().to(torch.long).clamp(max=frame_weights.numel() - 1)
    upper = (lower + 1).clamp(max=frame_weights.numel() - 1)
    fraction = coordinates - lower
    return frame_weights[lower] * (1.0 - fraction) + frame_weights[upper] * fraction


def endpoint_guide_log_bias(query_strengths, guide_token_count: int):
    """Return a compact broadcast view for only video-query→endpoint-guide scores."""
    import torch

    if query_strengths.ndim != 2:
        raise InferenceError(
            f"query strengths must have shape [B, N], got {tuple(query_strengths.shape)}"
        )
    if type(guide_token_count) is not int or guide_token_count < 1:
        raise InferenceError(
            f"guide_token_count must be positive, got {guide_token_count!r}"
        )
    if not torch.all(torch.isfinite(query_strengths)):
        raise InferenceError("query strengths must be finite")
    if torch.any((query_strengths < 0) | (query_strengths > 1)):
        raise InferenceError("query strengths must be in [0, 1]")
    finfo = torch.finfo(query_strengths.dtype)
    bias = torch.full_like(query_strengths, finfo.min)
    positive = query_strengths > 0
    bias[positive] = torch.log(query_strengths[positive].clamp(min=finfo.tiny))
    return bias.unsqueeze(-1).expand(-1, -1, guide_token_count)


def require_temporal_attention_bridge(pipeline: object, strengths: Sequence[float] | None) -> None:
    """Require the pinned runner's executable both-stage capability before GPU work."""
    if strengths is None:
        return
    capability_fn = getattr(pipeline, "temporal_attention_capability", None)
    capability = capability_fn() if callable(capability_fn) else None
    if not isinstance(capability, dict) or capability.get("version") != CAPABILITY_VERSION:
        raise InferenceError(f"LTX 2.5 temporal-attention execution capability is unavailable: {capability!r}")
    if capability.get("both_stages") is not True or capability.get("asymmetric_video_query_to_endpoint_key") is not True:
        raise InferenceError(f"LTX 2.5 temporal-attention execution capability is incomplete: {capability!r}")
