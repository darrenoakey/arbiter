#!/usr/bin/env python3
"""Quick test for the lora-train adapter — runs 10 iterations on 20 samples."""
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

# Reserve memory first if arbiter is running
ARBITER_URL = os.environ.get("ARBITER_URL", "http://localhost:8400")
reservation_id = None

try:
    import requests
    resp = requests.post(
        f"{ARBITER_URL}/v1/reserve",
        json={"memory_gb": 20, "label": "lora-train-test"},
        timeout=30,
    )
    if resp.status_code == 201:
        reservation_id = resp.json()["reservation_id"]
        print(f"Reserved 20GB (id: {reservation_id})")
    else:
        print(f"Reservation failed ({resp.status_code}): {resp.text}")
        print("Continuing anyway — arbiter may evict models as needed")
except Exception as e:
    print(f"Could not reserve memory: {e}")
    print("Continuing anyway — arbiter may not be running")

try:
    from arbiter.adapters.lora_train import LoraTrainAdapter

    adapter = LoraTrainAdapter()
    output_dir = Path(tempfile.mkdtemp(prefix="lora-train-test-"))

    print("\n[LOAD] Loading training libraries...")
    t0 = time.monotonic()
    adapter.load("cuda")
    print(f"[LOAD] OK ({time.monotonic() - t0:.1f}s)")

    print("\n[TRAIN] Running 10 iterations on 20 samples...")
    t0 = time.monotonic()
    result = adapter.infer(
        params={
            "data_dir": "/home/darren/training/test-leo/data",
            "model_name": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            "run_name": "test-tiny",
            "lora_rank": 16,
            "batch_size": 2,
            "grad_accum_steps": 1,
            "max_iters": 10,
            "max_seq_length": 512,
            "save_steps": 10,
            "eval_steps": 10,
        },
        output_dir=output_dir,
        cancel_flag=threading.Event(),
    )
    train_time = time.monotonic() - t0
    print(f"[TRAIN] OK ({train_time:.1f}s)")
    print(f"  Result: {json.dumps(result, indent=2)}")

    # Verify adapter files exist
    adapter_dir = output_dir / "adapter"
    print(f"\n[VERIFY] Checking adapter output at {adapter_dir}")
    assert adapter_dir.exists(), "Adapter directory not found"
    files = list(adapter_dir.rglob("*"))
    print(f"  Files: {[f.name for f in files]}")
    safetensors = [f for f in files if f.suffix == ".safetensors"]
    assert len(safetensors) > 0, "No safetensors files found"
    print(f"  Safetensors: {[f.name for f in safetensors]}")

    summary_file = output_dir / "summary.json"
    assert summary_file.exists(), "summary.json not found"
    summary = json.loads(summary_file.read_text())
    print(f"  Train loss: {summary.get('train_loss')}")
    print(f"  Runtime: {summary.get('train_runtime')}s")

    print(f"\n[UNLOAD] Cleaning up...")
    adapter.unload()

    print(f"\n{'='*40}")
    print(f"ALL PASS — training adapter works!")
    print(f"Adapter saved to: /home/darren/training/test-tiny/adapter/")

    # Cleanup test output (keep the training dir for inspection)
    shutil.rmtree(output_dir, ignore_errors=True)

finally:
    if reservation_id:
        try:
            requests.delete(f"{ARBITER_URL}/v1/reserve/{reservation_id}", timeout=10)
            print(f"Released reservation {reservation_id}")
        except Exception:
            pass
