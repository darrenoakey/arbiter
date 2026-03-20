"""Report formatting for calibration results."""
from __future__ import annotations

import json
from pathlib import Path


def print_report(results: dict):
    """Print a human-readable calibration report."""
    name = results["model_name"]
    m = results["measurements"]

    print(f"\n{'='*60}")
    print(f" Calibration Report: {name}")
    print(f"{'='*60}")
    print(f" Hardware: {results['hardware'].get('gpu', 'unknown')}")
    print(f" Date: {results['calibrated_at']}")
    print()
    print(f" Load time:   {m['load_time_ms']:.0f}ms")
    print(f" Unload time: {m['unload_time_ms']:.0f}ms")
    print(f" VRAM (idle): {m['memory_after_load_gb']:.2f} GB")
    print()

    profiles = m.get("concurrency_profiles", [])
    if profiles:
        print(f" {'Conc':<6} {'Avg ms':<10} {'P50 ms':<10} {'P95 ms':<10} {'Peak GB':<10} {'RPS':<8}")
        print(f" {'-'*54}")
        for p in profiles:
            print(
                f" {p['concurrency']:<6} {p['avg_inference_time_ms']:<10.0f} "
                f"{p['p50_ms']:<10.0f} {p['p95_ms']:<10.0f} "
                f"{p['memory_peak_gb']:<10.2f} {p['throughput_rps']:<8.1f}"
            )

    rec = m.get("recommended_max_concurrent", 1)
    print(f"\n RECOMMENDED max_concurrent: {rec}")

    if "config_entry" in results:
        print(f"\n Config entry:")
        print(f"   {json.dumps(results['config_entry'], indent=2)}")

    print(f"{'='*60}")
