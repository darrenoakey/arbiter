"""Checksum-verified deterministic packaging for the LTX 2.5 source archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
MANIFEST_PATH = REPOSITORY_ROOT / "runtime" / "ltx25" / "manifest.json"
OVERLAY_PATHS = ("video_fast_gpu.py", "tests/test_video_fast_gpu.py")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_id_for(manifest: dict, archive_sha256: str) -> str:
    source = manifest["source"]
    return (
        f"{source['commit'][:7]}-{source['runner_sha256'][:8]}-temporal-"
        f"{manifest['patch']['sha256'][:12]}-src{archive_sha256[:12]}"
    )


def archive_name_for(manifest: dict, archive_sha256: str) -> str:
    source = manifest["source"]
    return (
        f"ltx25-{source['commit'][:7]}-custom-{source['runner_sha256'][:8]}"
        f"-src{archive_sha256[:12]}.tar.gz"
    )


def _reject_name(name: str) -> str:
    parts = [part for part in name.replace("\\", "/").split("/") if part not in ("", ".")]
    if not parts or parts[0] == ".." or any(part == ".." for part in parts):
        raise RuntimeError(f"archive member escapes destination: {name}")
    return "/".join(parts)


def _reject_link(linkname: str) -> str:
    if linkname.startswith("/") or "\\" in linkname:
        raise RuntimeError(f"symlink target is not relative: {linkname}")
    if any(part == ".." for part in linkname.split("/")):
        raise RuntimeError(f"symlink target escapes its directory: {linkname}")
    return linkname


def _records_from_git_archive(archive: Path, overlays: dict[str, bytes]) -> list[tuple]:
    files: dict[str, tuple[bytes, bool]] = {}
    links: dict[str, str] = {}
    directories: set[str] = set()
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            name = _reject_name(member.name)
            if member.isdir():
                directories.add(name)
            elif member.issym():
                links[name] = _reject_link(member.linkname)
            elif member.isfile():
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"git archive member has no payload: {name}")
                files[name] = (extracted.read(), bool(member.mode & 0o111))
            else:
                raise RuntimeError(f"unsupported git archive member {name} type {member.type!r}")
    unknown = sorted(set(overlays) - set(files))
    if unknown:
        raise RuntimeError(f"overlay path is not in the git archive: {', '.join(unknown)}")
    for relative, payload in overlays.items():
        files[relative] = (payload, files[relative][1])
    for name in (*files, *links):
        parent = name
        while "/" in parent:
            parent = parent.rsplit("/", 1)[0]
            directories.add(parent)
    records: list[tuple] = [("dir", name, b"", "") for name in directories]
    records.extend(("file", name, data, "x" if executable else "") for name, (data, executable) in files.items())
    records.extend(("symlink", name, b"", linkname) for name, linkname in links.items())
    records.sort(key=lambda record: record[1].encode())
    names = [record[1] for record in records]
    if len(names) != len(set(names)):
        raise RuntimeError("git archive contains duplicate member names")
    return records


def render_archive(records: list[tuple]) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for kind, name, data, extra in records:
            info = tarfile.TarInfo(name)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = extra
                info.mode = 0o777
            else:
                info.type = tarfile.REGTYPE
                info.mode = 0o755 if extra == "x" else 0o644
                info.size = len(data)
            archive.addfile(info, io.BytesIO(data) if kind == "file" else None)
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0, compresslevel=9) as handle:
        handle.write(payload.getvalue())
    return compressed.getvalue()


def package_git_archive(archive: Path, overlays: dict[str, bytes]) -> bytes:
    return render_archive(_records_from_git_archive(archive, overlays))


def load_overlays(pairs: list[str], expectations: dict[str, str]) -> dict[str, bytes]:
    overlays: dict[str, bytes] = {}
    for pair in pairs:
        relative, separator, source = pair.partition("=")
        if not separator or not relative or not source:
            raise RuntimeError(f"overlay must be RELATIVE=PATH, got {pair!r}")
        payload = Path(source).read_bytes()
        expected = expectations.get(relative)
        if expected is None:
            raise RuntimeError(f"overlay {relative} has no expected sha256")
        actual = sha256_bytes(payload)
        if actual != expected:
            raise RuntimeError(f"overlay {relative} checksum mismatch: expected {expected}, got {actual}")
        overlays[relative] = payload
    missing = sorted(set(expectations) - set(overlays))
    if missing:
        raise RuntimeError(f"expected overlay was not provided: {', '.join(missing)}")
    return overlays


def attest_content(archive: Path, manifest: dict, patch_path: Path) -> None:
    from scripts import build_ltx25_runtime as builder

    builder.verify_hash(patch_path, manifest["patch"]["sha256"], "LTX temporal-attention patch")
    with tempfile.TemporaryDirectory(prefix="ltx25-attest-") as temporary:
        destination = Path(temporary)
        builder.safe_extract(archive, destination)
        builder.verify_hash(
            destination / "video_fast_gpu.py",
            manifest["source"]["runner_sha256"],
            "custom runner",
        )
        builder.verify_hash(
            destination / "tests" / "test_video_fast_gpu.py",
            manifest["source"]["runner_test_sha256"],
            "custom runner regression",
        )
        patched = subprocess.run(
            ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch_path)],
            cwd=destination,
            check=False,
            capture_output=True,
            text=True,
        )
        if patched.returncode != 0:
            detail = (patched.stderr or patched.stdout).strip()
            raise RuntimeError(f"temporal-attention patch failed: {detail}")
        for relative, expected in manifest["patched_files"].items():
            builder.verify_hash(destination / relative, expected, f"patched runtime file {relative}")


def install_archive(archive: Path, name: str, sources_dir: Path) -> Path:
    sources_dir.mkdir(parents=True, exist_ok=True)
    target = sources_dir / name
    temporary = sources_dir / f".{name}.partial"
    data = archive.read_bytes()
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    if sha256_file(target) != sha256_bytes(data):
        raise RuntimeError(f"installed archive checksum changed at {target}")
    return target


def pin_archive(archive: Path, manifest_path: Path, sources_dir: Path) -> Path:
    manifest = json.loads(manifest_path.read_text())
    preserved = {
        "commit": manifest["source"]["commit"],
        "tree": manifest["source"]["tree"],
        "runner_sha256": manifest["source"]["runner_sha256"],
        "runner_test_sha256": manifest["source"]["runner_test_sha256"],
        "patch_sha256": manifest["patch"]["sha256"],
        "patched_files": dict(manifest["patched_files"]),
        "capability_version": manifest["capability_version"],
    }
    attest_content(archive, manifest, manifest_path.parent / manifest["patch"]["file"])
    digest = sha256_file(archive)
    manifest["source"]["archive"] = archive_name_for(manifest, digest)
    manifest["source"]["archive_sha256"] = digest
    manifest["release_id"] = release_id_for(manifest, digest)
    if manifest["source"]["commit"] != preserved["commit"]:
        raise RuntimeError("pin changed source commit")
    if manifest["source"]["tree"] != preserved["tree"]:
        raise RuntimeError("pin changed source tree")
    if manifest["source"]["runner_sha256"] != preserved["runner_sha256"]:
        raise RuntimeError("pin changed runner hash")
    if manifest["source"]["runner_test_sha256"] != preserved["runner_test_sha256"]:
        raise RuntimeError("pin changed runner test hash")
    if manifest["patch"]["sha256"] != preserved["patch_sha256"]:
        raise RuntimeError("pin changed patch hash")
    if manifest["patched_files"] != preserved["patched_files"]:
        raise RuntimeError("pin changed patched file hashes")
    if manifest["capability_version"] != preserved["capability_version"]:
        raise RuntimeError("pin changed capability version")
    text = json.dumps(manifest, indent=2) + "\n"
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial")
    temporary.write_text(text)
    os.replace(temporary, manifest_path)
    return install_archive(archive, manifest["source"]["archive"], sources_dir)


def canonical_sources_dir() -> Path:
    from scripts import build_ltx25_runtime as builder

    return builder.canonical_repository_root() / "local" / "ltx25-runtime-sources"


def _parse_expects(values: list[str]) -> dict[str, str]:
    expectations: dict[str, str] = {}
    for value in values:
        relative, separator, digest = value.partition("=")
        if not separator or not relative or len(digest) != 64:
            raise RuntimeError(f"expect must be RELATIVE=SHA256, got {value!r}")
        expectations[relative] = digest
    return expectations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack")
    pack.add_argument("--git-archive", type=Path, required=True)
    pack.add_argument("--overlay", action="append", required=True)
    pack.add_argument("--expect", action="append", required=True)
    pack.add_argument("--output", type=Path, required=True)
    attest = subparsers.add_parser("attest")
    attest.add_argument("--archive", type=Path, required=True)
    attest.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    pin = subparsers.add_parser("pin")
    pin.add_argument("--archive", type=Path, required=True)
    pin.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    pin.add_argument("--sources-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "pack":
        overlays = load_overlays(args.overlay, _parse_expects(args.expect))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(package_git_archive(args.git_archive, overlays))
        print(sha256_file(args.output))
        return 0
    if args.command == "attest":
        manifest = json.loads(args.manifest.read_text())
        attest_content(args.archive, manifest, args.manifest.parent / manifest["patch"]["file"])
        digest = sha256_file(args.archive)
        if digest != manifest["source"]["archive_sha256"]:
            raise RuntimeError(
                f"source archive checksum mismatch: expected {manifest['source']['archive_sha256']}, got {digest}"
            )
        if manifest["release_id"] != release_id_for(manifest, digest):
            raise RuntimeError(f"release_id does not match archive {digest}")
        print(digest)
        return 0
    installed = pin_archive(args.archive, args.manifest, args.sources_dir or canonical_sources_dir())
    print(f"{installed} {sha256_file(installed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
