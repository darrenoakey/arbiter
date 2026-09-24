"""Build and activate the checksum-pinned LTX 2.5 runtime."""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "runtime" / "ltx25" / "manifest.json"
PROVENANCE_NAME = "arbiter-ltx25-runtime.json"
ROLLBACK_RECORD_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def canonical_repository_root() -> Path:
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = (REPOSITORY_ROOT / common_path).resolve()
    return common_path.parent


def default_source_archive(manifest: dict) -> Path:
    return canonical_repository_root() / "local" / "ltx25-runtime-sources" / manifest["source"]["archive"]


def default_output_root() -> Path:
    return canonical_repository_root() / "local" / "ltx25-runtimes"


def verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} checksum mismatch: expected {expected}, got {actual}")


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"source archive escapes destination: {member.name}")
        bundle.extractall(destination, filter="data")


def provenance(manifest: dict) -> dict:
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


def verify_runtime(runtime: Path, manifest: dict) -> None:
    for relative, expected in manifest["patched_files"].items():
        verify_hash(runtime / relative, expected, f"patched runtime file {relative}")
    record_path = runtime / PROVENANCE_NAME
    if not record_path.is_file():
        raise RuntimeError(f"runtime provenance is missing: {record_path}")
    actual = json.loads(record_path.read_text())
    if actual != provenance(manifest):
        raise RuntimeError(f"runtime provenance mismatch at {record_path}")


def build_runtime(source_archive: Path, output_root: Path) -> Path:
    manifest = load_manifest()
    verify_hash(source_archive, manifest["source"]["archive_sha256"], "LTX source archive")
    patch_path = MANIFEST_PATH.parent / manifest["patch"]["file"]
    verify_hash(patch_path, manifest["patch"]["sha256"], "LTX temporal-attention patch")

    destination = output_root / manifest["release_id"]
    if destination.exists():
        verify_runtime(destination, manifest)
        return destination

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{manifest['release_id']}.", dir=output_root))
    try:
        safe_extract(source_archive, staging)
        verify_hash(staging / "video_fast_gpu.py", manifest["source"]["runner_sha256"], "custom runner")
        verify_hash(
            staging / "tests" / "test_video_fast_gpu.py",
            manifest["source"]["runner_test_sha256"],
            "custom runner regression",
        )
        subprocess.run(
            ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch_path)],
            cwd=staging,
            check=True,
        )
        (staging / PROVENANCE_NAME).write_text(json.dumps(provenance(manifest), indent=2) + "\n")
        verify_runtime(staging, manifest)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return destination


def replace_ltx25_section(text: str, values: dict[str, str]) -> str:
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if line.strip() == "[ltx25]"), None)
    end = len(lines)
    if start is not None:
        end = next(
            (index for index in range(start + 1, len(lines)) if lines[index].lstrip().startswith("[")),
            len(lines),
        )
    block = ["[ltx25]", *[f"{key} = {json.dumps(value)}" for key, value in values.items()]]
    if start is None:
        result = lines + ([""] if lines else []) + block
    else:
        result = lines[:start] + block + lines[end:]
    return "\n".join(result).rstrip() + "\n"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_replace(path: Path, content: bytes, suffix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{suffix}")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _durable_unlink(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


@contextlib.contextmanager
def _activation_lock(rollback_path: Path):
    lock_path = rollback_path.with_name(f".{rollback_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _config_state(config_path: Path) -> tuple[bool, bytes]:
    return (config_path.exists(), config_path.read_bytes() if config_path.exists() else b"")


def _rollback_record(config_path: Path, existed: bool, content: bytes, state: str) -> dict:
    return {
        "version": ROLLBACK_RECORD_VERSION,
        "config_path": str(config_path.resolve()),
        "config_existed": existed,
        "config_sha256": hashlib.sha256(content).hexdigest(),
        "config_base64": base64.b64encode(content).decode("ascii"),
        "state": state,
    }


def _write_rollback_record(rollback_path: Path, record: dict) -> None:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    _durable_replace(rollback_path, payload, "new")


def _load_rollback_record(config_path: Path, rollback_path: Path) -> tuple[dict, bytes]:
    if not rollback_path.is_file():
        raise RuntimeError(f"activation rollback record is missing: {rollback_path}")
    try:
        record = json.loads(rollback_path.read_text())
        content = base64.b64decode(record["config_base64"], validate=True)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"activation rollback record is invalid: {rollback_path}: {error}") from error
    expected_keys = {
        "version", "config_path", "config_existed", "config_sha256", "config_base64", "state"
    }
    if set(record) != expected_keys:
        raise RuntimeError(f"activation rollback record has unexpected fields: {rollback_path}")
    if record["version"] != ROLLBACK_RECORD_VERSION:
        raise RuntimeError(f"activation rollback record version is unsupported: {record['version']!r}")
    if record["config_path"] != str(config_path.resolve()):
        raise RuntimeError(f"activation rollback record targets a different config: {record['config_path']!r}")
    if type(record["config_existed"]) is not bool:
        raise RuntimeError("activation rollback record config_existed must be boolean")
    if record["state"] not in {"prepared", "activated", "restored"}:
        raise RuntimeError(f"activation rollback record state is invalid: {record['state']!r}")
    if hashlib.sha256(content).hexdigest() != record["config_sha256"]:
        raise RuntimeError(f"activation rollback record checksum mismatch: {rollback_path}")
    return record, content


def prepare_activation(config_path: Path, rollback_path: Path) -> None:
    with _activation_lock(rollback_path):
        if rollback_path.exists():
            record, original = _load_rollback_record(config_path, rollback_path)
            current_exists, current = _config_state(config_path)
            if record["state"] != "restored" or current_exists != record["config_existed"] or current != original:
                raise RuntimeError(f"unresolved activation rollback record already exists: {rollback_path}")
        existed, original = _config_state(config_path)
        _write_rollback_record(
            rollback_path, _rollback_record(config_path, existed, original, "prepared")
        )


def activate_runtime(runtime: Path, config_path: Path, rollback_path: Path) -> None:
    manifest = load_manifest()
    verify_runtime(runtime, manifest)
    with _activation_lock(rollback_path):
        record, original = _load_rollback_record(config_path, rollback_path)
        current_exists, current = _config_state(config_path)
        if record["state"] != "prepared":
            raise RuntimeError(f"activation rollback record is not prepared: {record['state']!r}")
        if current_exists != record["config_existed"] or current != original:
            raise RuntimeError("activation config changed after rollback preparation")
        current_settings = tomllib.loads(original.decode()) if original else {}
        previous = str(current_settings.get("ltx25", {}).get("runtime_path", ""))
        updated = replace_ltx25_section(
            original.decode(),
            {
                "runtime_path": str(runtime.resolve()),
                "previous_runtime_path": previous,
                "release_id": manifest["release_id"],
                "capability_version": manifest["capability_version"],
                "patch_sha256": manifest["patch"]["sha256"],
            },
        )
        _durable_replace(config_path, updated.encode(), "ltx25-new")
        record["state"] = "activated"
        _write_rollback_record(rollback_path, record)


def verify_activation(runtime: Path, config_path: Path) -> None:
    manifest = load_manifest()
    verify_runtime(runtime, manifest)
    if not config_path.is_file():
        raise RuntimeError(f"activation config is missing: {config_path}")
    settings = tomllib.loads(config_path.read_text()).get("ltx25", {})
    expected = provenance(manifest)
    if Path(settings.get("runtime_path", "")).resolve() != runtime.resolve():
        raise RuntimeError(f"configured LTX runtime does not match {runtime}")
    for key in ("release_id", "capability_version", "patch_sha256"):
        if settings.get(key) != expected[key]:
            raise RuntimeError(f"configured LTX {key} does not match runtime provenance")


def rollback_activation(config_path: Path, rollback_path: Path) -> None:
    with _activation_lock(rollback_path):
        record, original = _load_rollback_record(config_path, rollback_path)
        current_exists, current = _config_state(config_path)
        if record["state"] == "restored":
            if current_exists != record["config_existed"] or current != original:
                raise RuntimeError("restored activation config no longer matches rollback record")
            return
        if record["config_existed"]:
            _durable_replace(config_path, original, "ltx25-rollback")
        elif config_path.exists():
            _durable_unlink(config_path)
        record["state"] = "restored"
        _write_rollback_record(rollback_path, record)


def finalize_activation(config_path: Path, rollback_path: Path) -> None:
    with _activation_lock(rollback_path):
        record, _original = _load_rollback_record(config_path, rollback_path)
        if record["state"] != "activated":
            raise RuntimeError(f"cannot finalize activation in state {record['state']!r}")
        _durable_unlink(rollback_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-archive", type=Path)
    build.add_argument("--output-root", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("runtime", type=Path)
    verify_activation_parser = subparsers.add_parser("verify-activation")
    verify_activation_parser.add_argument("runtime", type=Path)
    verify_activation_parser.add_argument("--config", type=Path, required=True)
    activate = subparsers.add_parser("activate")
    activate.add_argument("runtime", type=Path)
    activate.add_argument("--config", type=Path, required=True)
    activate.add_argument("--rollback-record", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-activation")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--rollback-record", type=Path, required=True)
    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--config", type=Path, required=True)
    rollback.add_argument("--rollback-record", type=Path, required=True)
    finalize = subparsers.add_parser("finalize-activation")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--rollback-record", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    if args.command == "build":
        source = args.source_archive or default_source_archive(manifest)
        output = args.output_root or default_output_root()
        print(build_runtime(source, output))
    elif args.command == "verify":
        verify_runtime(args.runtime, manifest)
        print(args.runtime.resolve())
    elif args.command == "verify-activation":
        verify_activation(args.runtime, args.config)
        print(args.runtime.resolve())
    elif args.command == "activate":
        activate_runtime(args.runtime, args.config, args.rollback_record)
        print(args.runtime.resolve())
    elif args.command == "prepare-activation":
        prepare_activation(args.config, args.rollback_record)
        print(args.rollback_record.resolve())
    elif args.command == "rollback":
        rollback_activation(args.config, args.rollback_record)
        print(config_path if (config_path := args.config).exists() else "removed")
    else:
        finalize_activation(args.config, args.rollback_record)
        print(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
