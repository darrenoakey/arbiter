#!/usr/bin/env python3
"""Standalone adapter test harness for Arbiter.

Tests the complete adapter lifecycle: load -> infer -> verify output -> unload.
Runs outside the full arbiter server. Optionally reserves memory first.

Usage:
    python -m arbiter.adapter_test sadtalker
    python -m arbiter.adapter_test sonic
    python -m arbiter.adapter_test sadtalker --reserve 5
    python -m arbiter.adapter_test --list
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Project root (two levels up from src/arbiter/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_ASSETS = PROJECT_ROOT / "test-assets"

ARBITER_URL = os.environ.get("ARBITER_URL", "http://localhost:8400")


def _ensure_test_assets():
    """Create test assets if they don't exist."""
    TEST_ASSETS.mkdir(exist_ok=True)

    portrait = TEST_ASSETS / "portrait.png"
    audio = TEST_ASSETS / "test_3s.wav"

    if not portrait.exists():
        # Try to copy from SadTalker examples
        src = Path("/home/darren/src/talking-head/local-sadtalker/examples/source_image/art_0.png")
        if src.exists():
            shutil.copy2(str(src), str(portrait))
        else:
            print(f"ERROR: No test portrait at {portrait} or {src}")
            sys.exit(1)

    if not audio.exists():
        # Generate a 3-second sine wave
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "sine=frequency=440:duration=3",
                "-ar", "16000", "-ac", "1",
                str(audio),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"ERROR: Failed to create test audio: {result.stderr.decode()[-200:]}")
            sys.exit(1)

    return portrait, audio


def _reserve_memory(gb: float) -> str | None:
    """Reserve VRAM via arbiter API. Returns reservation_id or None."""
    try:
        import requests
        resp = requests.post(
            f"{ARBITER_URL}/v1/reserve",
            json={"memory_gb": gb, "label": "adapter-test"},
            timeout=30,
        )
        if resp.status_code == 201:
            rid = resp.json()["reservation_id"]
            print(f"  Reserved {gb}GB (id: {rid})")
            return rid
        print(f"  WARNING: Reservation failed ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"  WARNING: Could not reserve memory: {e}")
    return None


def _release_reservation(rid: str):
    """Release a reservation via arbiter API."""
    try:
        import requests
        requests.delete(f"{ARBITER_URL}/v1/reserve/{rid}", timeout=10)
        print(f"  Released reservation {rid}")
    except Exception:
        pass


def _probe_video(path: str) -> dict:
    """Use ffprobe to get video info."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration,codec_name",
            "-of", "json",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return {}
    import json
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    return streams[0] if streams else {}


def test_adapter(model_id: str, reserve_gb: float = 0, verbose: bool = False):
    """Test a single adapter's full lifecycle."""
    print(f"\n{'='*60}")
    print(f"Testing adapter: {model_id}")
    print(f"{'='*60}")

    portrait, audio = _ensure_test_assets()
    results = {"load": "SKIP", "infer": "SKIP", "verify": "SKIP", "unload": "SKIP"}
    reservation_id = None

    try:
        # Import adapter
        from arbiter.adapters import registry
        try:
            adapter_cls = registry.get_adapter_class(model_id)
        except KeyError as e:
            print(f"  FAIL: {e}")
            return False

        adapter = adapter_cls()

        # Reserve memory if requested
        if reserve_gb > 0:
            reservation_id = _reserve_memory(reserve_gb)

        # Test load
        print(f"\n  [LOAD] Loading {model_id}...")
        t0 = time.monotonic()
        try:
            adapter.load("cuda")
            load_time = time.monotonic() - t0
            results["load"] = "PASS"
            print(f"  [LOAD] PASS ({load_time:.1f}s)")
        except Exception as e:
            results["load"] = "FAIL"
            print(f"  [LOAD] FAIL: {e}")
            return False

        # Test infer
        print(f"\n  [INFER] Running inference...")
        output_dir = Path(tempfile.mkdtemp(prefix=f"adapter-test-{model_id}-"))
        cancel_flag = threading.Event()
        params = {
            "image_file": str(portrait),
            "audio_file": str(audio),
        }
        # Add model-specific defaults
        if model_id == "sadtalker":
            params["size"] = 256
            params["facerender"] = "pirender"

        t0 = time.monotonic()
        try:
            result = adapter.infer(params, output_dir, cancel_flag)
            infer_time = time.monotonic() - t0
            results["infer"] = "PASS"
            print(f"  [INFER] PASS ({infer_time:.1f}s)")
            if verbose:
                print(f"    Result: {result}")
        except Exception as e:
            results["infer"] = "FAIL"
            print(f"  [INFER] FAIL: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return False

        # Verify output
        print(f"\n  [VERIFY] Checking output...")
        try:
            assert "format" in result, "Missing 'format' in result"
            assert "file" in result, "Missing 'file' in result"

            out_file = output_dir / result["file"]
            assert out_file.exists(), f"Output file missing: {out_file}"
            assert out_file.stat().st_size > 1000, f"Output too small: {out_file.stat().st_size} bytes"

            # ffprobe validation
            probe = _probe_video(str(out_file))
            assert probe, "ffprobe returned no streams"
            width = int(probe.get("width", 0))
            height = int(probe.get("height", 0))
            duration = float(probe.get("duration", 0))
            codec = probe.get("codec_name", "unknown")

            assert width > 0, "Video has zero width"
            assert height > 0, "Video has zero height"
            assert duration > 0.5, f"Video too short: {duration}s"

            results["verify"] = "PASS"
            print(f"  [VERIFY] PASS ({width}x{height}, {duration:.1f}s, {codec})")
        except AssertionError as e:
            results["verify"] = "FAIL"
            print(f"  [VERIFY] FAIL: {e}")
            return False
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

        # Test unload
        print(f"\n  [UNLOAD] Unloading {model_id}...")
        t0 = time.monotonic()
        try:
            adapter.unload()
            unload_time = time.monotonic() - t0
            results["unload"] = "PASS"
            print(f"  [UNLOAD] PASS ({unload_time:.1f}s)")
        except Exception as e:
            results["unload"] = "FAIL"
            print(f"  [UNLOAD] FAIL: {e}")
            return False

    finally:
        if reservation_id:
            _release_reservation(reservation_id)

    # Summary
    print(f"\n  {'─'*40}")
    all_pass = all(v == "PASS" for v in results.values())
    for step, status in results.items():
        marker = "✓" if status == "PASS" else "✗" if status == "FAIL" else "○"
        print(f"  {marker} {step}: {status}")
    print(f"\n  {'PASS' if all_pass else 'FAIL'}: {model_id}")
    return all_pass


def list_adapters():
    """List all registered adapters."""
    from arbiter.adapters import registry
    print("Registered adapters:")
    for model_id in registry.list_registered():
        print(f"  - {model_id}")


def main():
    parser = argparse.ArgumentParser(description="Arbiter adapter test harness")
    parser.add_argument("model_id", nargs="?", help="Model ID to test")
    parser.add_argument("--list", action="store_true", help="List available adapters")
    parser.add_argument("--all", action="store_true", help="Test all adapters")
    parser.add_argument("--reserve", type=float, default=0, help="Reserve VRAM (GB) before testing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.list:
        list_adapters()
        return

    if args.all:
        from arbiter.adapters import registry
        models = registry.list_registered()
        passed = 0
        failed = 0
        for model_id in models:
            if test_adapter(model_id, args.reserve, args.verbose):
                passed += 1
            else:
                failed += 1
        print(f"\n{'='*60}")
        print(f"Results: {passed} passed, {failed} failed")
        sys.exit(1 if failed > 0 else 0)

    if not args.model_id:
        parser.print_help()
        sys.exit(1)

    success = test_adapter(args.model_id, args.reserve, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
