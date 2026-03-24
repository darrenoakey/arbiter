# Arbiter

Unified GPU model serving platform. Manages multiple heterogeneous ML models on a single GPU with smart memory management, shortest-job-first scheduling, and persistent job queuing.

## What it does

Arbiter runs as a single server that:
- **Loads and unloads models on demand** based on incoming requests and available VRAM
- **Schedules inference** using shortest-job-first (SJF) — shorter jobs and already-loaded models get priority
- **Manages GPU memory** within a configurable budget (default 100GB of 128GB), with LRU eviction of idle models
- **Persists the job queue** to SQLite — survives server crashes transparently
- **Pipelines model loading** — loads the next model while the current one is running inference
- **Logs everything** in structured JSONL for debugging and capacity planning

## Supported Models

| Model | Type | VRAM | Job Types |
|-------|------|------|-----------|
| FLUX.1-schnell | Image generation | ~12GB | `image-generate`, `image-edit` |
| BiRefNet HR | Background removal | ~2GB | `background-remove` |
| Moondream2 | Vision-language | ~4GB | `caption`, `query`, `detect` |
| Whisper Large-v3 | Transcription | ~3GB | `transcribe` |
| Qwen3-TTS (3 variants) | Text-to-speech | ~4GB each | `tts-custom`, `tts-clone`, `tts-design` |
| Sonic (group) | Talking head video | ~15GB | `talking-head` |
| LTX-2 (group) | Video generation | ~55GB | `video-generate` |

## Quick Start

```bash
# Setup
cd arbiter
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Copy default config
cp local/config.default.json local/config.json

# Run tests
./run test

# Start server
./run server

# Check status
./run ps
./run health
```

## API

All interactions are async — submit a job, get a job ID, poll for results.

```bash
# Submit a job
curl -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"type": "image-generate", "params": {"prompt": "a red fox", "steps": 4}}'

# Poll for result
curl http://localhost:8400/v1/jobs/{job_id}

# Cancel a job
curl -X DELETE http://localhost:8400/v1/jobs/{job_id}

# Upload a reference file (upload once, reuse in many jobs)
curl -X POST http://localhost:8400/v1/refs -F file=@voice.wav
# => {"ref_id": "a1b2c3d4e5f6.wav"}

# Use reference file in a job (no re-upload needed)
curl -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"type": "tts-clone", "params": {"text": "Hello", "ref_audio_file": "ref:a1b2c3d4e5f6.wav"}}'

# List / delete reference files
curl http://localhost:8400/v1/refs
curl -X DELETE http://localhost:8400/v1/refs/a1b2c3d4e5f6.wav

# System status (includes GPU utilization %)
curl http://localhost:8400/v1/ps

# Health check
curl http://localhost:8400/v1/health
```

## CLI

```bash
./run server              # Start server
./run ps                  # Show loaded models + VRAM
./run jobs                # Show job queue
./run submit <type> '{}'  # Submit a job
./run cancel <id>         # Cancel a job
./run test                # Run fast tests
./run calibrate <model>   # Calibrate a model
./run calibrate --all     # Calibrate all models
```

## Configuration

Edit `local/config.json` (gitignored). See `local/config.default.json` for the full schema.

Key settings:
- `vram_budget_gb`: VRAM limit (default 100 of 128GB — leaves headroom)
- `models.<id>.memory_gb`: VRAM when loaded
- `models.<id>.max_concurrent`: Max simultaneous inferences
- `models.<id>.keep_alive_seconds`: How long to keep model loaded after last use
- `models.<id>.avg_inference_ms`: Average inference time (for SJF scheduling)
- `models.<id>.load_ms`: Model load time (for SJF scheduling)

## Calibration

Before production use, calibrate each model to get accurate VRAM and timing values:

```bash
./run calibrate whisper-large
./run calibrate --all
```

Results are saved to `local/calibration/<model>.json` with a `config_entry` field ready to paste into `local/config.json`.

## Architecture

```
Client -> FastAPI Server -> Scheduler (SJF) -> Memory Manager -> Worker Pool -> Model Adapters
                               |                    |
                          SQLite Queue         VRAM Tracking
                          (persistent)        (LRU eviction)
```

## Directory Structure

- `src/arbiter/` — Source code
- `src/arbiter/adapters/` — One adapter per model
- `src/arbiter/calibrate/` — Calibration tool
- `tests/` — Test suite
- `local/` — Config and calibration results (gitignored)
- `output/` — Logs, job results, reference files, SQLite DB (gitignored)
