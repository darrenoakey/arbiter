#!/usr/bin/env python3
# Deliberately Spark-specific. Configure this script's absolute argv in the
# existing post_stop_command only for an authorized maintenance deployment.
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSH = ("/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
       "darren@10.0.0.254")
REMOTE_BINARY = "/home/darren/src/arbiter/arbiter-go"
WORK = "/home/darren/storage-repair-20261104/"
FINALIZER = (*SSH, WORK + "run-final-log-layout")


def output(command, timeout=5):
    # Inherit the generic hook's process group: its 120s kill includes SSH.
    # Never start a detached session that could outlive the outer hook.
    return subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, check=True, timeout=timeout, shell=False).stdout


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_hex(value, length):
    if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"invalid {length}-digit hexadecimal identity: {value!r}")
    return value


def revision_from_metadata(metadata, revision):
    require_hex(revision, 40)
    fields = {}
    for line in metadata.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[0] == "build" and "=" in parts[1]:
            key, value = parts[1].split("=", 1)
            if key in fields:
                raise ValueError(f"duplicate Go build field: {key}")
            fields[key] = value
    required = {"GOOS": "linux", "GOARCH": "arm64", "vcs": "git",
                "vcs.revision": revision, "vcs.modified": "false"}
    if any(fields.get(key) != value for key, value in required.items()):
        raise ValueError(f"artifact is not a clean Linux ARM64 build of {revision}: {fields}")
    if "\tpath\tarbiter/cmd/arbiter" not in metadata.splitlines():
        raise ValueError("artifact is not arbiter/cmd/arbiter")
    return revision


def local_candidate(root=ROOT):
    artifact = root / "arbiter-linux-arm64"
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError(f"candidate must be a regular artifact: {artifact}")
    revision = output(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    if output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError(f"tracked candidate sources are dirty: {root}")
    tracked = output(["git", "-C", str(root), "ls-files", "-z"]).split("\0")
    if any((root / name).stat().st_mtime_ns > artifact.stat().st_mtime_ns
           for name in tracked if name):
        raise ValueError(f"candidate predates checked-out sources; rebuild {artifact}")
    metadata = output(["go", "version", "-m", str(artifact)])
    revision_from_metadata(metadata, revision)
    return digest(artifact), revision


def remote_digest():
    result = output([*SSH, "/usr/bin/sha256sum", "--", REMOTE_BINARY], timeout=10)
    match = re.fullmatch(r"([0-9a-f]{64})  " + re.escape(REMOTE_BINARY) + r"\n", result)
    if not match:
        raise ValueError(f"unexpected remote SHA256 response: {result!r}")
    return match[1]


def prove_installed(expected):
    require_hex(expected, 64)
    actual = remote_digest()
    if actual != expected:
        raise ValueError(f"installed candidate SHA256 mismatch: expected {expected}, got {actual}")


def approval_command(sha, revision):
    require_hex(sha, 64)
    require_hex(revision, 40)
    return [*SSH, "/usr/bin/sudo", "-n", "/usr/bin/timeout", "--signal=TERM",
            "--kill-after=2s", "5s", "/usr/bin/python3", WORK + "finalize-log-layout.py",
            "--approve-candidate", sha, revision]


def validate_approval(text, sha, revision):
    response = json.loads(text)
    expected = {"status": "approved", "sha256": sha, "source_revision": revision,
                "approval": "explicit greenline post-stop candidate"}
    if response != expected:
        raise ValueError(f"candidate approval did not attest exact identity: {response!r}")


def execute():
    sha, revision = local_candidate()
    prove_installed(sha)
    # The prepared remote launcher independently bounds finalization + recovery.
    # Stream its diagnostics; never hide a recovery failure or substitute a binary.
    subprocess.run(FINALIZER, stdin=subprocess.DEVNULL, check=True, timeout=110, shell=False)
    prove_installed(sha)
    validate_approval(output(approval_command(sha, revision), timeout=10), sha, revision)
    print(f"verified installed candidate approved: {sha} ({revision})", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify installed candidate, finalize storage, approve exact SHA")
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args(argv)
    try:
        execute()
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"verified post-stop failed closed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
