import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/verified_post_stop.py"
SPEC = importlib.util.spec_from_file_location("verified_post_stop", SCRIPT)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)
SHA = "a" * 64
REVISION = "b" * 40
METADATA = ("artifact: go1.24\n\tpath\tarbiter/cmd/arbiter\n"
            "\tbuild\tGOOS=linux\n\tbuild\tGOARCH=arm64\n\tbuild\tvcs=git\n"
            f"\tbuild\tvcs.revision={REVISION}\n\tbuild\tvcs.modified=false\n")


def test_streaming_digest_real_files(tmp_path):
    artifact = tmp_path / "artifact"
    content = bytes(range(256)) * 32769
    artifact.write_bytes(content)
    assert wrapper.digest(artifact) == hashlib.sha256(content).hexdigest()
    artifact.write_bytes(b"")
    assert wrapper.digest(artifact) == hashlib.sha256(b"").hexdigest()
    with pytest.raises(OSError):
        wrapper.digest(tmp_path / "missing")


@pytest.mark.parametrize("old,new", [
    ("GOOS=linux", "GOOS=darwin"), ("GOARCH=arm64", "GOARCH=amd64"),
    ("vcs.modified=false", "vcs.modified=true"), (REVISION, "c" * 40),
    ("vcs=git", "vcs=hg"), ("arbiter/cmd/arbiter", "arbiter/cmd/other"),
    ("\tbuild\tGOOS=linux\n", ""),
])
def test_metadata_rejects_wrong_missing_dirty_identity(old, new):
    with pytest.raises(ValueError):
        wrapper.revision_from_metadata(METADATA.replace(old, new), REVISION)


def test_metadata_accepts_only_exact_candidate():
    assert wrapper.revision_from_metadata(METADATA, REVISION) == REVISION
    with pytest.raises(ValueError, match="duplicate"):
        wrapper.revision_from_metadata(METADATA + "\tbuild\tGOOS=linux\n", REVISION)


@pytest.mark.parametrize("sha,revision", [("A" * 64, REVISION), (SHA + ";id", REVISION),
                                          (SHA, "HEAD"), (SHA, REVISION + "\n"), ("", REVISION)])
def test_approval_argv_rejects_shell_or_ambiguous_identity(sha, revision):
    with pytest.raises(ValueError):
        wrapper.approval_command(sha, revision)


def test_exact_prepared_commands():
    assert wrapper.FINALIZER == (*wrapper.SSH, wrapper.WORK + "run-final-log-layout")
    assert wrapper.approval_command(SHA, REVISION) == [
        *wrapper.SSH, "/usr/bin/sudo", "-n", "/usr/bin/timeout", "--signal=TERM",
        "--kill-after=2s", "5s", "/usr/bin/python3",
        "/home/darren/storage-repair-20261104/finalize-log-layout.py",
        "--approve-candidate", SHA, REVISION]
    assert wrapper.SSH == ("/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
                           "darren@10.0.0.254")


def test_exact_approval_response_parser():
    response = {"status": "approved", "sha256": SHA, "source_revision": REVISION,
                "approval": "explicit greenline post-stop candidate"}
    wrapper.validate_approval(json.dumps(response), SHA, REVISION)
    for key in response:
        changed = dict(response, **{key: "wrong"})
        with pytest.raises(ValueError):
            wrapper.validate_approval(json.dumps(changed), SHA, REVISION)
        missing = dict(response)
        del missing[key]
        with pytest.raises(ValueError):
            wrapper.validate_approval(json.dumps(missing), SHA, REVISION)
    for text in ["", "not JSON", "[]", "null", json.dumps(dict(response, extra=True))]:
        with pytest.raises(ValueError):
            wrapper.validate_approval(text, SHA, REVISION)


def test_real_readonly_ssh_hash_and_fail_closed_mismatch():
    actual = wrapper.remote_digest()
    assert wrapper.require_hex(actual, 64) == actual
    wrapper.prove_installed(actual)
    different = ("0" if actual[0] != "0" else "1") + actual[1:]
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        wrapper.prove_installed(different)


def test_real_go_metadata_and_stale_dirty_artifacts(tmp_path):
    def command(*args):
        return wrapper.output(list(args), timeout=30)

    command("git", "init", "-q", str(tmp_path))
    (tmp_path / "go.mod").write_text("module arbiter\n\ngo 1.24\n")
    source = tmp_path / "cmd/arbiter/main.go"
    source.parent.mkdir(parents=True)
    source.write_text('package main\nfunc main() {}\n')
    command("git", "-C", str(tmp_path), "add", ".")
    command("git", "-C", str(tmp_path), "-c", "user.name=Wrapper Test", "-c",
            "user.email=wrapper-test@localhost", "commit", "-qm", "real metadata fixture")
    revision = command("git", "-C", str(tmp_path), "rev-parse", "HEAD").strip()
    artifact = tmp_path / "arbiter-linux-arm64"
    command("go", "-C", str(tmp_path), "build", "-o", str(artifact), "./cmd/arbiter")
    metadata = command("go", "version", "-m", str(artifact))
    assert f"vcs.revision={revision}" in metadata
    assert "vcs.modified=false" in metadata
    if "GOOS=linux" in metadata and "GOARCH=arm64" in metadata:
        assert wrapper.local_candidate(tmp_path) == (wrapper.digest(artifact), revision)
    else:
        with pytest.raises(ValueError, match="Linux ARM64"):
            wrapper.local_candidate(tmp_path)
    os.utime(artifact, ns=(1, 1))
    with pytest.raises(ValueError, match="predates"):
        wrapper.local_candidate(tmp_path)
    source.write_text(source.read_text() + "// changed\n")
    with pytest.raises(ValueError, match="dirty"):
        wrapper.local_candidate(tmp_path)
    artifact.unlink()
    artifact.symlink_to(source)
    with pytest.raises(ValueError, match="regular artifact"):
        wrapper.local_candidate(tmp_path)


def test_explicit_cli_and_real_subprocess_errors(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    with pytest.raises(subprocess.CalledProcessError):
        wrapper.output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"])
    with pytest.raises(subprocess.TimeoutExpired):
        wrapper.output([sys.executable, "-c", "while True: pass"], timeout=0.1)
    with pytest.raises(ValueError, match="regular artifact"):
        wrapper.local_candidate(tmp_path)
