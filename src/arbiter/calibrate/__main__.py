"""CLI entry point for calibration: python -m arbiter.calibrate"""
from __future__ import annotations

import argparse
import sys

from .runner import run_calibration


def main():
    parser = argparse.ArgumentParser(description="Arbiter Model Calibration")
    parser.add_argument("model", nargs="?", help="Model ID to calibrate")
    parser.add_argument("--all", action="store_true", help="Calibrate all registered models")
    parser.add_argument("--concurrency", default="1,2,4,8", help="Comma-separated concurrency levels")
    parser.add_argument("--samples", type=int, default=10, help="Samples per concurrency level")
    args = parser.parse_args()

    if not args.all and not args.model:
        parser.print_help()
        sys.exit(1)

    levels = [int(x) for x in args.concurrency.split(",")]

    if args.all:
        from arbiter.adapters.registry import list_registered
        models = list_registered()
        if not models:
            print("No models registered. Import adapters first.")
            sys.exit(1)
    else:
        models = [args.model]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f" Calibrating: {model_name}")
        print(f"{'='*60}\n")
        try:
            run_calibration(model_name, levels, args.samples)
        except Exception as e:
            print(f"ERROR calibrating {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
