"""Deterministic LTX 2.5 source archive packaging."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts import build_ltx25_runtime as builder
from scripts import package_ltx25_source as packager


def _git_archive(path: Path) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        directory = tarfile.TarInfo("pkg")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o775
        directory.mtime = 1789466997
        directory.uid = 501
        directory.gid = 20
        archive.addfile(directory)
        info = tarfile.TarInfo("pkg/runner.py")
        info.mode = 0o664
        info.mtime = 1789466997
        info.uid = 501
        data = b"print('before')\n"
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("pkg/ALIAS.md")
        link.type = tarfile.SYMTYPE
        link.linkname = "runner.py"
        link.mode = 0o777
        link.mtime = 1789466997
        archive.addfile(link)
        nested = "pkg/" + ("nested/" * 16) + "deep.txt"
        deep = tarfile.TarInfo(nested)
        deep.mode = 0o775
        deep.mtime = 42
        body = b"deep\n"
        deep.size = len(body)
        archive.addfile(deep, io.BytesIO(body))
    path.write_bytes(payload.getvalue())


def test_pack_is_deterministic_and_normalizes_metadata(tmp_path: Path) -> None:
    source = tmp_path / "head.tar"
    _git_archive(source)
    overlay = b"print('after')\n"
    first = packager.package_git_archive(source, {"pkg/runner.py": overlay})
    second = packager.package_git_archive(source, {"pkg/runner.py": overlay})
    assert first == second
    assert first[:4] == b"\x1f\x8b\x08\x00"
    assert first[4:8] == b"\x00\x00\x00\x00"
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        members = archive.getmembers()
        assert [member.mtime for member in members] == [0] * len(members)
        assert [member.uid for member in members] == [0] * len(members)
        assert [member.gid for member in members] == [0] * len(members)
        runner = archive.getmember("pkg/runner.py")
        assert runner.mode == 0o644
        assert archive.extractfile(runner).read() == overlay
        assert archive.getmember("pkg").mode == 0o755
        assert archive.getmember("pkg/ALIAS.md").linkname == "runner.py"
        deep = next(member for member in members if member.name.endswith("deep.txt"))
        assert deep.mode == 0o755
        assert len(deep.name) > 100


def test_overlay_checksum_and_missing_path_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "head.tar"
    _git_archive(source)
    overlay = tmp_path / "runner.py"
    overlay.write_bytes(b"print('after')\n")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        packager.load_overlays([f"pkg/runner.py={overlay}"], {"pkg/runner.py": "0" * 64})
    digest = packager.sha256_bytes(overlay.read_bytes())
    loaded = packager.load_overlays([f"pkg/runner.py={overlay}"], {"pkg/runner.py": digest})
    assert loaded["pkg/runner.py"] == overlay.read_bytes()
    with pytest.raises(RuntimeError, match="not in the git archive"):
        packager.package_git_archive(source, {"missing.py": b"nope"})


def test_pin_keeps_source_runner_patch_and_patched_hashes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(packager.MANIFEST_PATH.read_text())
    original_runner = manifest["source"]["runner_sha256"]
    original_patch = manifest["patch"]["sha256"]
    original_patched = dict(manifest["patched_files"])
    manifest["release_id"] = "46172b9-8b350b3f-temporal-085d91d22bd6"
    manifest["source"]["archive"] = "ltx25-46172b9-custom-8b350b3f.tar.gz"
    manifest["source"]["archive_sha256"] = "ee5de2ef67fe98d962d62d5ae59bcd76e9a86e020cd86cdc5803962a9acad27b"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (tmp_path / "temporal-attention.patch").write_bytes(
        (packager.MANIFEST_PATH.parent / manifest["patch"]["file"]).read_bytes()
    )
    archive = builder.default_source_archive(builder.load_manifest())
    if not archive.is_file():
        raise AssertionError(f"checksum-pinned LTX source archive is required at {archive}")
    installed = packager.pin_archive(archive, manifest_path, tmp_path / "sources")
    pinned = json.loads(manifest_path.read_text())
    assert pinned["source"]["runner_sha256"] == original_runner
    assert pinned["source"]["runner_test_sha256"] == manifest["source"]["runner_test_sha256"]
    assert pinned["source"]["commit"] == manifest["source"]["commit"]
    assert pinned["source"]["tree"] == manifest["source"]["tree"]
    assert pinned["patch"]["sha256"] == original_patch
    assert pinned["patched_files"] == original_patched
    assert pinned["capability_version"] == manifest["capability_version"]
    assert pinned["release_id"] != "46172b9-8b350b3f-temporal-085d91d22bd6"
    assert pinned["release_id"].startswith("46172b9-8b350b3f-temporal-085d91d22bd6-src")
    assert pinned["release_id"].endswith(f"-src{pinned['source']['archive_sha256'][:12]}")
    assert installed.name == pinned["source"]["archive"]
    assert packager.sha256_file(installed) == pinned["source"]["archive_sha256"]
    assert installed.read_bytes() == archive.read_bytes()


def test_pinned_archive_attests_without_changing_checks() -> None:
    manifest = builder.load_manifest()
    archive = builder.default_source_archive(manifest)
    if not archive.is_file():
        raise AssertionError(f"checksum-pinned LTX source archive is required at {archive}")
    assert packager.sha256_file(archive) == manifest["source"]["archive_sha256"]
    assert manifest["release_id"] == packager.release_id_for(manifest, manifest["source"]["archive_sha256"])
    assert manifest["source"]["archive_sha256"] != "ee5de2ef67fe98d962d62d5ae59bcd76e9a86e020cd86cdc5803962a9acad27b"
    packager.attest_content(archive, manifest, packager.MANIFEST_PATH.parent / manifest["patch"]["file"])
