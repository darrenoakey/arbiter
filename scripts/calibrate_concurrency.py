#!/usr/bin/env python3
"""Calibrate moondream concurrency: test 1-4 simultaneous inferences.

Loads moondream once in a single process and runs concurrent inferences
using threads (same as the concurrent-mode subprocess worker would).

Uses real images from Lorem Picsum.
"""
import base64
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

# How many images to pre-download
N_IMAGES = 8
# Concurrency levels to test
CONCURRENCY_LEVELS = [1, 2, 3, 4]
# Inferences per concurrency level
INFERENCES_PER_LEVEL = 8


def download_images(n):
    """Download n real images from Lorem Picsum."""
    print(f"Downloading {n} test images from picsum.photos...", flush=True)
    images = []
    for i in range(n):
        url = f"https://picsum.photos/512/512?random={i}"
        req = Request(url, headers={"User-Agent": "arbiter-calibrate"})
        with urlopen(req, timeout=15) as resp:
            data = resp.read()
            images.append(data)
        print(f"  [{i+1}/{n}] {len(data)} bytes", flush=True)
    return images


def run_calibration():
    print("=" * 60)
    print("Moondream Concurrency Calibration")
    print("=" * 60)

    # Download test images
    images = download_images(N_IMAGES)

    # Load moondream
    print("\nLoading moondream3...", flush=True)
    import torch
    from transformers import AutoModelForCausalLM
    from PIL import Image

    model = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream3-preview",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    print("Model loaded.", flush=True)

    # Warm up (first inference triggers FlexAttention JIT)
    print("Warming up (JIT compile)...", flush=True)
    warmup_img = Image.open(io.BytesIO(images[0])).convert("RGB")
    model.caption(warmup_img, length="short")
    print("Warm-up done.\n", flush=True)

    # Track VRAM
    vram_base = torch.cuda.memory_allocated() / (1024**3)
    print(f"Base VRAM: {vram_base:.2f} GB\n")

    # Run calibration at each concurrency level
    results = {}

    for concurrency in CONCURRENCY_LEVELS:
        print(f"--- Testing concurrency={concurrency} ---", flush=True)

        # Check VRAM before
        vram_before = torch.cuda.memory_allocated() / (1024**3)

        errors = 0
        timings = []
        lock = threading.Lock()

        def do_inference(idx):
            img_data = images[idx % len(images)]
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            start = time.perf_counter()
            try:
                result = model.caption(img, length="short")
                elapsed = time.perf_counter() - start
                with lock:
                    timings.append(elapsed)
                return elapsed, result.get("caption", "")[:50]
            except Exception as e:
                elapsed = time.perf_counter() - start
                with lock:
                    timings.append(elapsed)
                return elapsed, f"ERROR: {e}"

        batch_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for i in range(INFERENCES_PER_LEVEL):
                futures.append(pool.submit(do_inference, i))

            for i, future in enumerate(as_completed(futures)):
                elapsed, caption = future.result()
                status = "OK" if not caption.startswith("ERROR") else caption
                print(f"  [{i+1}/{INFERENCES_PER_LEVEL}] {elapsed:.2f}s - {status}", flush=True)
                if caption.startswith("ERROR"):
                    errors += 1

        batch_elapsed = time.perf_counter() - batch_start
        vram_after = torch.cuda.memory_allocated() / (1024**3)
        vram_peak = torch.cuda.max_memory_allocated() / (1024**3)

        avg_latency = sum(timings) / len(timings) if timings else 0
        throughput = INFERENCES_PER_LEVEL / batch_elapsed if batch_elapsed > 0 else 0

        results[concurrency] = {
            "concurrency": concurrency,
            "total_inferences": INFERENCES_PER_LEVEL,
            "errors": errors,
            "batch_time_s": round(batch_elapsed, 2),
            "avg_latency_s": round(avg_latency, 2),
            "min_latency_s": round(min(timings), 2) if timings else 0,
            "max_latency_s": round(max(timings), 2) if timings else 0,
            "throughput_per_s": round(throughput, 3),
            "vram_gb": round(vram_after, 2),
            "vram_peak_gb": round(vram_peak, 2),
        }

        print(f"  Batch time: {batch_elapsed:.2f}s")
        print(f"  Avg latency: {avg_latency:.2f}s")
        print(f"  Throughput: {throughput:.3f} inferences/s")
        print(f"  VRAM: {vram_after:.2f} GB (peak: {vram_peak:.2f} GB)")
        print(f"  Errors: {errors}")

        # Reset peak VRAM tracking for next level
        torch.cuda.reset_peak_memory_stats()
        print(flush=True)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Conc':>5} {'Batch(s)':>10} {'Avg(s)':>8} {'Tput(/s)':>10} {'VRAM(GB)':>10} {'Errors':>7}")
    for c in CONCURRENCY_LEVELS:
        r = results[c]
        print(f"{c:>5} {r['batch_time_s']:>10.2f} {r['avg_latency_s']:>8.2f} {r['throughput_per_s']:>10.3f} {r['vram_peak_gb']:>10.2f} {r['errors']:>7}")

    # Recommendation
    best = max(results.values(), key=lambda r: r["throughput_per_s"] if r["errors"] == 0 else 0)
    print(f"\nRecommendation: max_concurrent = {best['concurrency']}")
    print(f"  Throughput: {best['throughput_per_s']:.3f}/s (vs {results[1]['throughput_per_s']:.3f}/s at concurrency=1)")
    if best["concurrency"] > 1:
        speedup = best["throughput_per_s"] / results[1]["throughput_per_s"]
        print(f"  Speedup: {speedup:.2f}x")

    # Save results
    out_dir = Path("local/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "moondream-concurrency.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_calibration()
