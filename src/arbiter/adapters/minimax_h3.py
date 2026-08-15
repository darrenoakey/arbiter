"""MiniMax H3 cloud video adapter with durable provider-task recovery."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from arbiter.adapters.base import InferenceError, ModelAdapter
from arbiter.adapters.registry import register


def _httpx() -> Any:
    """Import httpx on demand.

    The registry imports EVERY adapter module inside EVERY per-model venv, so a
    module-level third-party import here is an outage for unrelated models: the
    whisper venv has no httpx, and this import killed the whisper-large worker
    at startup ("load failed: subprocess died"), taking transcription — and any
    pipeline that verifies vocals — down with it. Only MiniMax code paths need
    httpx, and they only ever run in a venv that has it.
    """
    return importlib.import_module("httpx")


_API_BASE = "https://api.minimax.io"
_PROVIDER_MODEL = "MiniMax-H3"
_STATE_FILE = "minimax-h3-state.json"
_SCHEMA_VERSION = 1
_MAX_GENERATIONS = 3
_POLL_SECONDS = 10.0
_SUPPORTED_RATIOS = frozenset({"16:9", "9:16", "1:1", "4:3", "3:4"})
_SUPPORTED_RESOLUTIONS = frozenset({"768P", "2K"})
_PRIVATE_PROVIDER_ID = "darren.private.encrypted-file.v1"


def _fail(message: str) -> InferenceError:
    return InferenceError(f"minimax-h3: {message}")


def _validated_params(params: dict) -> dict:
    prompt = params.get("prompt")
    duration = params.get("duration")
    resolution = params.get("resolution")
    ratio = params.get("ratio")
    if not isinstance(prompt, str) or not prompt or len(prompt) > 7000:
        raise _fail("prompt must be a non-empty string of at most 7000 characters")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, int)
        or not 4 <= duration <= 15
    ):
        raise _fail("duration must be an integer between 4 and 15")
    if resolution not in _SUPPORTED_RESOLUTIONS:
        raise _fail("resolution must be exactly 768P or 2K")
    if ratio is not None and ratio not in _SUPPORTED_RATIOS:
        raise _fail("ratio is not supported")

    validated = {
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
    }
    if ratio is not None:
        validated["ratio"] = ratio
    for key in ("first_frame_path", "last_frame_path"):
        value = params.get(key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise _fail(f"{key} must be a staged file path")
            path = Path(value)
            if not path.is_file() or path.stat().st_size == 0:
                raise _fail(f"{key} is missing or empty")
            validated[key] = str(path)
    if "last_frame_path" in validated and "first_frame_path" not in validated:
        raise _fail("last_frame_path requires first_frame_path")
    return validated


def _staging_roots(output_dir: Path) -> tuple[Path, Path]:
    resolved_output = output_dir.resolve()
    output_root = next(
        (parent for parent in (resolved_output, *resolved_output.parents) if parent.name == "output"),
        None,
    )
    if output_root is None:
        raise _fail("worker output directory does not identify the staging store")
    return output_root.parent / "inbox", output_root


def _validated_staged_frame(value: str, roots: tuple[Path, Path], key: str) -> None:
    candidate = Path(value)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise _fail(f"{key} must be a canonical staged file path")
    for root in roots:
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise _fail(f"{key} must not contain symlinks")
        if candidate.is_file() and candidate.stat().st_size > 0:
            return
    raise _fail(f"{key} must be a non-empty file in the staging store")


def _validate_staged_frames(params: dict, output_dir: Path) -> None:
    if not any(params.get(key) is not None for key in ("first_frame_path", "last_frame_path")):
        return
    roots = _staging_roots(output_dir)
    for key in ("first_frame_path", "last_frame_path"):
        value = params.get(key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise _fail(f"{key} must be a staged file path")
            _validated_staged_frame(value, roots, key)


def _image_data_url(path_text: str) -> str:
    path = Path(path_text)
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".heif": "image/heif",
    }.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _provider_payload(params: dict) -> dict:
    content: list[dict] = [{"type": "text", "text": params["prompt"]}]
    for key, role in (
        ("first_frame_path", "first_frame"),
        ("last_frame_path", "last_frame"),
    ):
        if key in params:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(params[key])},
                    "role": role,
                }
            )
    payload = {
        "model": _PROVIDER_MODEL,
        "content": content,
        "duration": params["duration"],
        "resolution": params["resolution"],
    }
    if "ratio" in params and len(content) == 1:
        payload["ratio"] = params["ratio"]
    return payload


def _request_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _fail("provider state is corrupt; refusing a paid submission") from error
    if not isinstance(state, dict) or state.get("schema_version") != _SCHEMA_VERSION:
        raise _fail("provider state schema is unsupported; refusing a paid submission")
    if state.get("phase") not in {"submitting", "submitted", "terminal"}:
        raise _fail("provider state phase is invalid; refusing a paid submission")
    generation = state.get("generation_count")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or not 1 <= generation <= _MAX_GENERATIONS
    ):
        raise _fail("provider state generation count is invalid")
    request_hash = state.get("request_hash")
    if (
        not isinstance(request_hash, str)
        or len(request_hash) != 64
        or any(character not in "0123456789abcdef" for character in request_hash)
    ):
        raise _fail("provider state request hash is invalid")
    if state["phase"] == "submitted" and (
        not isinstance(state.get("task_id"), str) or not state["task_id"]
    ):
        raise _fail("submitted provider state has no task id")
    if state["phase"] == "terminal" and state.get("terminal_status") not in {
        "succeeded",
        "failed",
    }:
        raise _fail("terminal provider state is invalid")
    return state


def _submission_generation(state: dict | None, request_hash: str) -> int | None:
    if state is None:
        return 1
    if state["request_hash"] != request_hash:
        raise _fail("provider state belongs to a different request")
    if state["phase"] == "submitting":
        raise _fail(
            "prior submission outcome is ambiguous; refusing a second paid task"
        )
    if state["phase"] == "submitted":
        return None
    if state["terminal_status"] == "succeeded":
        return None
    if state["generation_count"] >= _MAX_GENERATIONS:
        raise _fail("provider terminal failures exhausted the replacement limit")
    return state["generation_count"] + 1


def _decode_api_key(raw: bytes) -> str:
    try:
        key = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _fail("credential is corrupt") from None
    if not key or any(ord(character) < 33 or ord(character) > 126 for character in key):
        raise _fail("credential is corrupt")
    return key


def _load_api_key(cancel_flag: threading.Event) -> str:
    try:
        from daz_secrets import Client, load_default_config

        config = load_default_config()
        if (
            config.fallback_provider_path is not None
            or config.fallback_provider_id is not None
        ):
            raise _fail("credential provider fallback is forbidden")
        if config.provider_id != _PRIVATE_PROVIDER_ID:
            raise _fail("credential provider identity is not authorized")
        raw = Client(config).get("minimax", "api_key", cancel=cancel_flag).value
    except InferenceError:
        raise
    except Exception:
        raise _fail("credential unavailable") from None
    return _decode_api_key(raw)


def _submit_task(api_key: str, payload: dict) -> str:
    httpx = _httpx()
    try:
        response = httpx.post(
            f"{_API_BASE}/v2/video_generation",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=120.0,
        )
    except httpx.HTTPError:
        raise _fail("provider submission transport failed") from None
    if response.status_code >= 400:
        raise _fail(f"provider submission failed with HTTP {response.status_code}")
    try:
        body = response.json()
        task_id = body.get("task_id") or (body.get("task") or {}).get("id")
    except (ValueError, AttributeError) as error:
        raise _fail("provider submission response was invalid") from error
    if not isinstance(task_id, str) or not task_id:
        raise _fail("provider submission response omitted task identity")
    return task_id


def _query_task(api_key: str, task_id: str) -> tuple[str, str | None]:
    httpx = _httpx()
    try:
        response = httpx.get(
            f"{_API_BASE}/v2/query/video_generation/{task_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
    except httpx.HTTPError:
        raise _fail("provider status transport failed") from None
    if response.status_code >= 400:
        raise _fail(f"provider status failed with HTTP {response.status_code}")
    try:
        body = response.json()
        task = body.get("task") if isinstance(body.get("task"), dict) else body
        status = str(task.get("status") or "").lower()
        content = task.get("content") if isinstance(task.get("content"), dict) else {}
        url = content.get("url") or task.get("file_url") or task.get("download_url")
    except (ValueError, AttributeError) as error:
        raise _fail("provider status response was invalid") from error
    return status, url if isinstance(url, str) and url else None


def _download_result(url: str, destination: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise _fail("provider returned an invalid result location")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    httpx = _httpx()
    try:
        with httpx.stream("GET", url, timeout=600.0, follow_redirects=True) as response:
            if response.status_code >= 400:
                raise _fail(
                    f"provider result download failed with HTTP {response.status_code}"
                )
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if temporary.stat().st_size == 0:
            raise _fail("provider result download was empty")
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    except httpx.HTTPError:
        raise _fail("provider result download transport failed") from None
    finally:
        temporary.unlink(missing_ok=True)


@register
class MiniMaxH3Adapter(ModelAdapter):
    model_id = "minimax-h3"

    def load(self, device: str = "cuda") -> None:
        return None

    def unload(self) -> None:
        return None

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        self._check_cancel(cancel_flag)
        _validate_staged_frames(params, output_dir)
        validated = _validated_params(params)
        payload = _provider_payload(validated)
        request_hash = _request_hash(payload)
        output_dir.mkdir(parents=True, exist_ok=True)
        state_path = output_dir / _STATE_FILE
        result_path = output_dir / "result.mp4"
        state = _read_state(state_path)
        generation = _submission_generation(state, request_hash)

        if (
            state is not None
            and state["phase"] == "terminal"
            and state["terminal_status"] == "succeeded"
        ):
            if not result_path.is_file() or result_path.stat().st_size == 0:
                raise _fail("terminal success state has no durable result")
            return {
                "format": "mp4",
                "file": "result.mp4",
                "provider_model": _PROVIDER_MODEL,
            }

        api_key = _load_api_key(cancel_flag)
        while True:
            self._check_cancel(cancel_flag)
            if generation is not None:
                state = {
                    "schema_version": _SCHEMA_VERSION,
                    "request_hash": request_hash,
                    "phase": "submitting",
                    "generation_count": generation,
                }
                _write_state(state_path, state)
                task_id = _submit_task(api_key, payload)
                state = {**state, "phase": "submitted", "task_id": task_id}
                _write_state(state_path, state)
                generation = None

            if state is None or state["phase"] != "submitted":
                raise _fail("provider recovery state is inconsistent")
            status, url = _query_task(api_key, state["task_id"])
            if status in {"success", "succeeded"}:
                if url is None:
                    raise _fail("provider succeeded without a result location")
                _download_result(url, result_path)
                state = {**state, "phase": "terminal", "terminal_status": "succeeded"}
                _write_state(state_path, state)
                return {
                    "format": "mp4",
                    "file": "result.mp4",
                    "provider_model": _PROVIDER_MODEL,
                }
            if status in {"fail", "failed", "cancel", "cancelled"}:
                state = {**state, "phase": "terminal", "terminal_status": "failed"}
                _write_state(state_path, state)
                generation = _submission_generation(state, request_hash)
                continue
            cancel_flag.wait(_POLL_SECONDS)

    def estimate_time(self, params: dict) -> float:
        duration = params.get("duration", 10)
        if isinstance(duration, bool) or not isinstance(duration, int):
            duration = 10
        return float(max(4, min(15, duration)) * 30000)
