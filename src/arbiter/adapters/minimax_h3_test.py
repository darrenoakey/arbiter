import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import arbiter.adapters  # noqa: F401 - imports the complete built-in registry
from arbiter.adapters.base import InferenceError
from arbiter.adapters.minimax_h3 import (
    _MAX_GENERATIONS,
    _decode_api_key,
    _provider_payload,
    _read_state,
    _request_hash,
    _submission_generation,
    _validate_staged_frames,
    _validated_params,
    _write_state,
    MiniMaxH3Adapter,
)
from arbiter.adapters.registry import list_registered


def test_validation_and_provider_content_use_staged_frames(tmp_path):
    first = tmp_path / "first.png"
    last = tmp_path / "last.jpg"
    first.write_bytes(b"png")
    last.write_bytes(b"jpeg")
    params = _validated_params(
        {
            "prompt": "A tracking shot",
            "duration": 12,
            "resolution": "2K",
            "ratio": "16:9",
            "first_frame_path": str(first),
            "last_frame_path": str(last),
        }
    )
    payload = _provider_payload(params)
    assert payload["model"] == "MiniMax-H3"
    assert [item.get("role") for item in payload["content"]] == [
        None,
        "first_frame",
        "last_frame",
    ]
    assert "ratio" not in payload
    assert all(
        item["image_url"]["url"].startswith("data:image/")
        for item in payload["content"][1:]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": ""},
        {"prompt": "x" * 7001},
        {"duration": True},
        {"duration": 3},
        {"duration": 16},
        {"resolution": "1080P"},
        {"ratio": "21:9"},
        {"last_frame_path": "/missing.jpg"},
    ],
)
def test_validation_rejects_contract_near_neighbors(change):
    params = {"prompt": "shot", "duration": 4, "resolution": "768P"} | change
    with pytest.raises(InferenceError):
        _validated_params(params)


@pytest.mark.parametrize("raw", [b"", b"key\n", b"key value", b"\xff"])
def test_corrupt_credentials_are_rejected_before_transport(raw):
    with pytest.raises(InferenceError, match="credential is corrupt"):
        _decode_api_key(raw)


def test_state_write_is_durable_and_round_trips(tmp_path):
    path = tmp_path / "state.json"
    state = {
        "schema_version": 1,
        "request_hash": "a" * 64,
        "phase": "submitted",
        "generation_count": 1,
        "task_id": "task-1",
    }
    _write_state(path, state)
    assert _read_state(path) == state
    assert json.loads(path.read_text()) == state
    assert not list(tmp_path.glob("*.tmp"))


def test_ambiguous_submission_fails_closed_without_replacement():
    state = {
        "schema_version": 1,
        "request_hash": "request",
        "phase": "submitting",
        "generation_count": 1,
    }
    with pytest.raises(InferenceError, match="ambiguous"):
        _submission_generation(state, "request")


def test_submitted_recovery_adopts_task_and_terminal_failure_is_bounded():
    submitted = {
        "schema_version": 1,
        "request_hash": "request",
        "phase": "submitted",
        "generation_count": 1,
        "task_id": "task-1",
    }
    assert _submission_generation(submitted, "request") is None
    terminal = {**submitted, "phase": "terminal", "terminal_status": "failed"}
    assert _submission_generation(terminal, "request") == 2
    terminal["generation_count"] = _MAX_GENERATIONS
    with pytest.raises(InferenceError, match="replacement limit"):
        _submission_generation(terminal, "request")


def test_request_hash_pins_all_provider_content():
    base = {"prompt": "shot", "duration": 4, "resolution": "768P", "ratio": "1:1"}
    original = _request_hash(_provider_payload(_validated_params(base)))
    changed = _request_hash(
        _provider_payload(_validated_params(base | {"duration": 5}))
    )
    assert original != changed


def test_text_only_payload_preserves_every_h3_contract_field():
    payload = _provider_payload(
        _validated_params(
            {
                "prompt": "orbit the subject",
                "duration": 15,
                "resolution": "2K",
                "ratio": "9:16",
            }
        )
    )
    assert payload == {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "orbit the subject"}],
        "duration": 15,
        "resolution": "2K",
        "ratio": "9:16",
    }


def test_adapter_and_deploy_config_are_discoverable():
    registered = list_registered()
    assert "minimax-h3" in registered
    assert "minimax-h3-local" in registered
    root = Path(__file__).parents[3]
    config = json.loads((root / "config/spark/minimax-h3.model.json").read_text())
    assert config["max_concurrent"] == 1
    assert config["max_instances"] == 1
    assert config["memory_gb"] > 0


def test_adapter_rejects_noncanonical_frames_before_payload_read(tmp_path):
    output_dir = tmp_path / "output/jobs/job-safe"
    inbox = tmp_path / "inbox"
    output_dir.mkdir(parents=True)
    (inbox / "nested").mkdir(parents=True)
    staged = inbox / "frame.png"
    staged.write_bytes(b"frame")
    _validate_staged_frames({"first_frame_path": str(staged)}, output_dir)

    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    outside.write_bytes(b"outside")
    linked = inbox / "linked.png"
    linked.symlink_to(staged)
    traversal = inbox / "nested" / ".." / "frame.png"
    try:
        for unsafe in (outside, linked, traversal):
            with pytest.raises(
                InferenceError, match="staged|staging|symlink|canonical"
            ):
                MiniMaxH3Adapter().infer(
                    {
                        "prompt": "shot",
                        "duration": 4,
                        "resolution": "768P",
                        "first_frame_path": str(unsafe),
                    },
                    output_dir,
                    threading.Event(),
                )
    finally:
        outside.unlink(missing_ok=True)


def test_terminal_provider_state_recovers_and_publishes_durable_result(tmp_path):
    output_dir = tmp_path / "output/jobs/job-recovery"
    output_dir.mkdir(parents=True)
    params = {"prompt": "shot", "duration": 4, "resolution": "768P"}
    request_hash = _request_hash(_provider_payload(_validated_params(params)))
    _write_state(
        output_dir / "minimax-h3-state.json",
        {
            "schema_version": 1,
            "request_hash": request_hash,
            "phase": "terminal",
            "generation_count": 1,
            "task_id": "provider-task-1",
            "terminal_status": "succeeded",
        },
    )
    (output_dir / "result.mp4").write_bytes(b"durable-video")
    result = MiniMaxH3Adapter().infer(params, output_dir, threading.Event())
    assert result == {
        "format": "mp4",
        "file": "result.mp4",
        "provider_model": "MiniMax-H3",
    }


def test_registry_imports_in_a_venv_without_httpx():
    """The registry imports every adapter module inside every per-model venv.

    whisper-large runs in its own venv with no httpx, so a module-level
    `import httpx` here killed that worker at startup ("load failed: subprocess
    died") and took transcription down with it. Reproduce the venv by hiding
    httpx from the import system in a child interpreter.
    """
    source_root = Path(arbiter.adapters.__file__).resolve().parents[2]
    child = textwrap.dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder

        class BlockHttpx(MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "httpx" or name.startswith("httpx."):
                    raise ModuleNotFoundError("No module named 'httpx'")
                return None

        sys.meta_path.insert(0, BlockHttpx())
        for cached in [n for n in sys.modules if n == "httpx" or n.startswith("httpx.")]:
            del sys.modules[cached]

        import arbiter.adapters
        from arbiter.adapters.registry import list_registered

        assert "minimax-h3" in list_registered(), list_registered()
        assert "minimax-h3-local" in list_registered(), list_registered()
        print("registry-ok")
        """
    )
    environment = dict(os.environ, PYTHONPATH=str(source_root))
    completed = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "registry-ok" in completed.stdout
