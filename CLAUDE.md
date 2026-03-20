# Arbiter — Development Guide

## Architecture

Arbiter is a unified GPU model server. It manages 11+ ML models on a single NVIDIA Grace Blackwell (128GB VRAM, 100GB budget). Key components:

- **`src/arbiter/server.py`** — FastAPI server (port 8400). Entry point.
- **`src/arbiter/scheduler.py`** — SJF scheduler. Picks jobs, loads models, dispatches to workers.
- **`src/arbiter/memory.py`** — GPU memory manager. VRAM budget, LRU eviction, keep-alive.
- **`src/arbiter/store.py`** — SQLite job persistence. Crash recovery.
- **`src/arbiter/worker.py`** — Thread pool for inference dispatch.
- **`src/arbiter/config.py`** — Pydantic config from `local/config.json`.
- **`src/arbiter/schemas.py`** — All API request/response schemas.
- **`src/arbiter/log.py`** — JSONL structured event logger.
- **`src/arbiter/cli.py`** — CLI that talks to the server over HTTP.
- **`src/arbiter/adapters/`** — One file per model. Each wraps load/unload/infer.
- **`src/arbiter/calibrate/`** — Measures VRAM, latency, concurrency per model.

## How to Run

```bash
./run server          # Start API server
./run test            # Fast tests (<30s, unit + integration)
./run test-all        # All tests except calibration
./run calibrate <model>  # Calibrate one model
./run ps              # Show loaded models
```

## How to Add a New Model

**IMPORTANT: Calibration is REQUIRED before any new model goes to production.**

### Step 1: Create the adapter

Create `src/arbiter/adapters/<model_name>.py`:

```python
from .base import ModelAdapter
from .registry import register

@register
class MyModelAdapter(ModelAdapter):
    model_id = "my-model"

    def __init__(self):
        self._model = None

    def load(self, device="cuda"):
        # Load model onto GPU
        self._model = ...

    def unload(self):
        del self._model
        self._model = None
        self._cleanup_gpu()

    def infer(self, params, output_dir, cancel_flag):
        self._check_cancel(cancel_flag)
        # Run inference, write result to output_dir
        # Return metadata dict: {"format": "png", "width": 1024, ...}
        ...

    def estimate_time(self, params):
        return 5000  # ms
```

### Step 2: Add to adapters/__init__.py

Add an import line so the adapter auto-registers:
```python
from . import my_model  # noqa: F401
```

### Step 3: Add job type to schemas.py

1. Add to `JobType` enum
2. Add to `JOB_TYPE_TO_MODEL` mapping
3. Create a params Pydantic model
4. Add to `JOB_TYPE_PARAMS` mapping

### Step 4: Add to config

Add the model entry to `local/config.default.json` with estimated values:
```json
"my-model": {
    "memory_gb": 4,
    "max_concurrent": 1,
    "keep_alive_seconds": 300,
    "avg_inference_ms": 5000,
    "load_ms": 10000,
    "auto_download": "org/model-name"
}
```

### Step 5: Run calibration (REQUIRED)

```bash
./run calibrate my-model
```

This produces `local/calibration/my-model.json` with measured values. Update `local/config.json` with the `config_entry` from the results.

### Step 6: Run tests

```bash
./run test
```

## Testing

- **Unit tests** (`tests/unit/`): No GPU needed. Test scheduler, memory manager, store, config.
- **Integration tests** (`tests/integration/`): FastAPI TestClient with mock adapters.
- **Calibration tests** (`tests/calibration/`): Manual only, per model.

All tests: `./run test` (excludes calibration and slow tests).

## Config

`local/config.json` (gitignored). Falls back to `local/config.default.json`.

Key per-model fields:
- `memory_gb` — VRAM when loaded (from calibration)
- `max_concurrent` — Max parallel inferences (from calibration)
- `avg_inference_ms` — Average inference time in ms (from calibration)
- `load_ms` — Model load time in ms (from calibration)
- `keep_alive_seconds` — Keep loaded after last use (default 300)
- `auto_download` — HuggingFace repo ID for auto-download

## Logs

JSONL at `output/logs/arbiter-YYYY-MM-DD.jsonl`. Events: job lifecycle, model load/evict, memory snapshots.

## Key Design Decisions

- **Single process, ThreadPoolExecutor**: PyTorch releases GIL during CUDA ops. Threads share GPU memory efficiently.
- **SJF scheduling**: `priority = avg_inference_ms + (load_ms if not loaded else 0)`. Shortest jobs run first. Already-loaded models get natural priority.
- **SQLite queue**: Persistent, crash-recoverable. On restart, incomplete jobs are re-queued.
- **Async job API**: Submit → poll → get result. Server crashes are transparent to clients.
- **LRU eviction**: Models idle past keep_alive_seconds get evicted. When memory is needed, oldest idle model goes first.
- **Group adapters**: Sonic (8 sub-models) and LTX-2 (phased pipeline) load/unload atomically.
