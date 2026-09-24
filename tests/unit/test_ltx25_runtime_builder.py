"""Immutable LTX 2.5 runtime construction and activation regressions."""

from __future__ import annotations

import json
import signal
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_ltx25_runtime as builder


def source_archive() -> Path:
    manifest = builder.load_manifest()
    archive = builder.default_source_archive(manifest)
    if not archive.is_file():
        raise AssertionError(f"checksum-pinned LTX source archive is required at {archive}")
    return archive


def test_build_is_content_addressed_strict_and_cpu_verified(tmp_path: Path) -> None:
    runtime = builder.build_runtime(source_archive(), tmp_path / "runtimes")
    repeated = builder.build_runtime(source_archive(), tmp_path / "runtimes")
    assert repeated == runtime
    assert runtime.name == builder.load_manifest()["release_id"]
    provenance = json.loads((runtime / builder.PROVENANCE_NAME).read_text())
    assert provenance == builder.provenance(builder.load_manifest())
    result = subprocess.run(
        [
            sys.executable,
            str(builder.REPOSITORY_ROOT / "runtime" / "ltx25" / "verify_runtime.py"),
            str(runtime),
            "--skip-runner-import",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "LTX25_TEMPORAL_ATTENTION_CPU_OK"


def test_source_archive_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / source_archive().name
    changed.write_bytes(source_archive().read_bytes() + b"drift")
    with pytest.raises(RuntimeError, match="source archive checksum mismatch"):
        builder.build_runtime(changed, tmp_path / "runtimes")
    assert not (tmp_path / "runtimes").exists()


def test_activation_records_and_restores_exact_previous_config(tmp_path: Path) -> None:
    runtime = builder.build_runtime(source_archive(), tmp_path / "runtimes")
    config = tmp_path / "local" / "config.toml"
    config.parent.mkdir(parents=True)
    original = '[service]\nname = "arbiter"\n\n[ltx25]\nruntime_path = "/previous/release"\n'
    config.write_text(original)
    rollback = tmp_path / "activation.rollback"
    builder.prepare_activation(config, rollback)
    builder.activate_runtime(runtime, config, rollback)
    builder.verify_activation(runtime, config)
    activated = config.read_text()
    assert f'runtime_path = "{runtime.resolve()}"' in activated
    assert 'previous_runtime_path = "/previous/release"' in activated
    assert json.loads(rollback.read_text())["state"] == "activated"
    builder.rollback_activation(config, rollback)
    assert config.read_text() == original
    assert json.loads(rollback.read_text())["state"] == "restored"
    builder.rollback_activation(config, rollback)
    assert config.read_text() == original


def test_interrupted_activation_is_restored_by_real_cli_processes(tmp_path: Path) -> None:
    runtime = builder.build_runtime(source_archive(), tmp_path / "runtimes")
    config = tmp_path / "local" / "config.toml"
    config.parent.mkdir(parents=True)
    original = b'[service]\nname = "arbiter"\n'
    config.write_bytes(original)
    rollback = tmp_path / "activation.rollback"
    script = builder.REPOSITORY_ROOT / "scripts" / "build_ltx25_runtime.py"
    prepare = [
        sys.executable,
        str(script),
        "prepare-activation",
        "--config",
        str(config),
        "--rollback-record",
        str(rollback),
    ]
    activate = [
        sys.executable,
        str(script),
        "activate",
        str(runtime),
        "--config",
        str(config),
        "--rollback-record",
        str(rollback),
    ]
    subprocess.run(prepare, check=True, capture_output=True, text=True)
    interrupted = subprocess.run(
        ["sh", "-c", f"{shlex.join(activate)}; kill -TERM $$"],
        capture_output=True,
        text=True,
    )
    assert interrupted.returncode == -signal.SIGTERM
    assert config.read_bytes() != original
    rollback_command = [
        sys.executable,
        str(script),
        "rollback",
        "--config",
        str(config),
        "--rollback-record",
        str(rollback),
    ]
    subprocess.run(rollback_command, check=True, capture_output=True, text=True)
    subprocess.run(rollback_command, check=True, capture_output=True, text=True)
    assert config.read_bytes() == original


@pytest.mark.parametrize("record", [None, b'{"version":1'])
def test_rollback_without_complete_record_fails_closed(tmp_path: Path, record: bytes | None) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[service]\nname = "unchanged"\n')
    rollback = tmp_path / "activation.rollback"
    if record is not None:
        rollback.write_bytes(record)
    before = config.read_bytes()
    with pytest.raises(RuntimeError, match="rollback record"):
        builder.rollback_activation(config, rollback)
    assert config.read_bytes() == before
