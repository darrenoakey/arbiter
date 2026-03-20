"""Calibration orchestrator — measures model performance characteristics."""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .profiler import measure_vram, get_gpu_info


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def run_calibration(model_name: str, concurrency_levels: list[int], samples: int = 10):
    """Run full calibration for a single model."""
    import torch
    from arbiter.adapters.registry import get_adapter_class

    adapter_cls = get_adapter_class(model_name)
    adapter = adapter_cls()

    gpu_info = get_gpu_info()
    results = {
        "model_name": model_name,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": gpu_info,
        "measurements": {},
    }

    # Phase 1: Load/unload timing
    print("Phase 1: Measuring load/unload times...")
    load_times = []
    unload_times = []

    for i in range(3):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()

        t0 = time.perf_counter()
        adapter.load("cuda")
        torch.cuda.synchronize()
        load_time = (time.perf_counter() - t0) * 1000
        load_times.append(load_time)

        mem_loaded = torch.cuda.memory_allocated()
        memory_gb = (mem_loaded - baseline) / (1024**3)

        t0 = time.perf_counter()
        adapter.unload()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        unload_time = (time.perf_counter() - t0) * 1000
        unload_times.append(unload_time)

        print(f"  Iteration {i+1}: load={load_time:.0f}ms, unload={unload_time:.0f}ms, vram={memory_gb:.2f}GB")

    results["measurements"]["load_time_ms"] = statistics.median(load_times)
    results["measurements"]["unload_time_ms"] = statistics.median(unload_times)
    results["measurements"]["memory_after_load_gb"] = round(memory_gb, 3)

    # Phase 2: Concurrency profiling
    print("\nPhase 2: Concurrency profiling...")
    adapter.load("cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Create a temporary output dir for inference results
    cal_output = _PROJECT_ROOT / "output" / "calibration_tmp"
    cal_output.mkdir(parents=True, exist_ok=True)

    import threading
    cancel = threading.Event()

    profiles = []
    for level in concurrency_levels:
        print(f"\n  Concurrency level: {level}")
        torch.cuda.reset_peak_memory_stats()
        idle_mem = torch.cuda.memory_allocated()

        latencies = []
        errors = 0

        for batch in range(max(1, samples // level)):
            batch_latencies = []
            with ThreadPoolExecutor(max_workers=level) as pool:
                futures = []
                for j in range(level):
                    job_dir = cal_output / f"cal_{level}_{batch}_{j}"
                    job_dir.mkdir(parents=True, exist_ok=True)
                    # Use estimate_time to get a test params dict
                    test_params = _get_test_params(model_name)
                    futures.append(pool.submit(_timed_infer, adapter, test_params, job_dir, cancel))

                for f in as_completed(futures):
                    try:
                        elapsed = f.result()
                        batch_latencies.append(elapsed)
                    except Exception as e:
                        errors += 1
                        print(f"    Error: {e}")

            latencies.extend(batch_latencies)

        if not latencies:
            print(f"    All samples failed at concurrency {level}, stopping.")
            break

        peak_mem = torch.cuda.max_memory_allocated()
        peak_gb = peak_mem / (1024**3)
        mem_per_req = (peak_mem - idle_mem) / (1024**3) / level if level > 0 else 0

        profile = {
            "concurrency": level,
            "avg_inference_time_ms": round(statistics.mean(latencies), 1),
            "p50_ms": round(statistics.median(latencies), 1),
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else latencies[-1], 1),
            "memory_peak_gb": round(peak_gb, 3),
            "memory_per_request_gb": round(mem_per_req, 3),
            "throughput_rps": round(level / (statistics.mean(latencies) / 1000), 2) if latencies else 0,
            "errors": errors,
            "samples": len(latencies),
        }
        profiles.append(profile)
        print(f"    avg={profile['avg_inference_time_ms']:.0f}ms  peak_vram={profile['memory_peak_gb']:.2f}GB  throughput={profile['throughput_rps']:.1f}rps")

        # Stop if latency is degrading badly (>3x single-request)
        if len(profiles) > 1 and profiles[-1]["avg_inference_time_ms"] > profiles[0]["avg_inference_time_ms"] * 3:
            print(f"    Latency degraded >3x, stopping higher concurrency.")
            break

    adapter.unload()

    results["measurements"]["concurrency_profiles"] = profiles

    # Determine recommended max_concurrent
    if profiles:
        single_lat = profiles[0]["avg_inference_time_ms"]
        recommended = 1
        for p in profiles:
            if p["avg_inference_time_ms"] <= single_lat * 2 and p["errors"] == 0:
                recommended = p["concurrency"]
        results["measurements"]["recommended_max_concurrent"] = recommended
    else:
        results["measurements"]["recommended_max_concurrent"] = 1

    # Generate config entry
    avg_by_level = {}
    for p in profiles:
        avg_by_level[str(p["concurrency"])] = p["avg_inference_time_ms"]

    results["config_entry"] = {
        "memory_gb": round(results["measurements"]["memory_after_load_gb"], 2),
        "max_concurrent": results["measurements"]["recommended_max_concurrent"],
        "avg_inference_ms": profiles[0]["avg_inference_time_ms"] if profiles else 5000,
        "load_ms": round(results["measurements"]["load_time_ms"], 0),
    }

    # Save results
    cal_dir = _PROJECT_ROOT / "local" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    out_file = cal_dir / f"{model_name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f" Calibration complete: {model_name}")
    print(f" Load: {results['measurements']['load_time_ms']:.0f}ms")
    print(f" VRAM: {results['measurements']['memory_after_load_gb']:.2f}GB")
    print(f" Recommended max_concurrent: {results['measurements']['recommended_max_concurrent']}")
    print(f" Results: {out_file}")
    print(f"{'='*60}")

    # Clean up temp dir
    import shutil
    shutil.rmtree(cal_output, ignore_errors=True)

    return results


def _timed_infer(adapter, params, output_dir, cancel_flag):
    """Run inference and return elapsed ms."""
    t0 = time.perf_counter()
    adapter.infer(params, output_dir, cancel_flag)
    return (time.perf_counter() - t0) * 1000


def _get_test_params(model_name: str) -> dict:
    """Get realistic test parameters for a model."""
    from .inputs import get_test_image_b64, get_test_audio_b64

    img = get_test_image_b64()
    audio = get_test_audio_b64()

    defaults = {
        "flux-schnell": {"prompt": "A red fox in a forest", "width": 512, "height": 512, "steps": 4, "seed": 42},
        "birefnet": {"image": img},
        "moondream": {"image": img, "task": "caption", "length": "short"},
        "whisper-large": {"audio": audio, "language": "en"},
        "tts-custom": {"text": "Hello world, this is a calibration test.", "speaker": "Aiden", "language": "English"},
        "tts-clone": {"text": "Hello world, this is a calibration test.", "ref_audio": audio, "language": "English"},
        "tts-design": {"text": "Hello world, this is a calibration test.", "voice_description": "A clear neutral voice."},
        "sonic": {"image": img, "audio": audio},
        "ltx2": {"images": [img], "audio": audio, "resolution": "small"},
        "aesthetic-scorer": {"image": img},
    }
    return defaults.get(model_name, {})
