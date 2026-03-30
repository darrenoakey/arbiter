#!/usr/bin/env python3
"""Test lora-train via arbiter API — submits a real job and polls for completion."""
import json
import sys
import time

import requests

ARBITER_URL = "http://localhost:8400"

# Submit training job
print("Submitting lora-train job (10 iters, 20 samples)...")
resp = requests.post(
    f"{ARBITER_URL}/v1/jobs",
    json={
        "type": "lora-train",
        "params": {
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
    },
    timeout=30,
)

if resp.status_code not in (200, 201, 202):
    print(f"FAIL: Submit returned {resp.status_code}: {resp.text}")
    sys.exit(1)

job = resp.json()
job_id = job["job_id"]
print(f"Job submitted: {job_id}")

# Poll for completion
start = time.monotonic()
while True:
    time.sleep(5)
    elapsed = time.monotonic() - start
    resp = requests.get(f"{ARBITER_URL}/v1/jobs/{job_id}", timeout=10)
    status = resp.json()["status"]
    print(f"  [{elapsed:.0f}s] Status: {status}")

    if status == "completed":
        result = resp.json().get("result", {})
        print(f"\nSUCCESS!")
        print(f"  Train loss: {result.get('train_loss')}")
        print(f"  Runtime: {result.get('train_runtime_seconds')}s")
        print(f"  Adapter: {result.get('adapter_path')}")
        break
    elif status == "failed":
        error = resp.json().get("error", "unknown")
        print(f"\nFAILED: {error}")
        sys.exit(1)
    elif elapsed > 900:
        print("\nTIMEOUT after 10 minutes")
        sys.exit(1)

print("\nALL PASS")
