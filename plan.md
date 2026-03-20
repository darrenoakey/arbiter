# Arbiter — Unified GPU Model Serving Platform

## Context

Five separate projects under `/home/darren/src` (images, moondream-station, talking-head, tts, ltx2-spark) each load their own ML models onto a single Grace Blackwell GPU (128GB VRAM). They run independent servers/processes that compete for memory and kill each other. There is no coordination of model loading, concurrency, or scheduling.

**Goal**: A single server that manages all models — loading/unloading them based on demand, scheduling inference with shortest-job-first, persisting the queue through crashes, and maximizing GPU utilization within a 100GB VRAM budget.

**Product name**: Arbiter (arbitrates GPU resources)
**Location**: `/home/darren/src/arbiter/`

---

## Directory Structure

```
arbiter/
├── src/
│   └── arbiter/
│       ├── __init__.py
│       ├── server.py              # FastAPI app, lifespan, routes
│       ├── config.py              # Pydantic config loader
│       ├── memory.py              # GPU Memory Manager
│       ├── scheduler.py           # Scheduler loop + SJF scoring
│       ├── store.py               # SQLite job persistence
│       ├── worker.py              # Job dispatch to thread pool
│       ├── log.py                 # JSONL structured logger
│       ├── cli.py                 # CLI entry point
│       ├── schemas.py             # All Pydantic request/response models
│       ├── adapters/
│       │   ├── __init__.py        # Auto-imports all, populates registry
│       │   ├── base.py            # ModelAdapter ABC, GroupAdapter ABC
│       │   ├── registry.py        # model_id -> adapter class mapping
│       │   ├── flux.py            # flux-schnell (text2img, img2img)
│       │   ├── birefnet.py        # birefnet (background removal)
│       │   ├── moondream.py       # moondream (vision-language)
│       │   ├── whisper_large.py   # whisper-large (transcription)
│       │   ├── tts_custom.py      # tts-custom
│       │   ├── tts_clone.py       # tts-clone
│       │   ├── tts_design.py      # tts-design
│       │   ├── sonic.py           # sonic (group: SVD+UNet+Audio2Token+Audio2Bucket+WhisperTiny+YOLOFace+RIFE)
│       │   └── ltx2.py            # ltx2 (group: phased transformer+VAE+LoRA+Upsampler+Gemma3)
│       └── calibrate/
│           ├── __init__.py
│           ├── __main__.py        # CLI entry: python -m arbiter.calibrate
│           ├── runner.py          # Calibration orchestrator
│           ├── profiler.py        # VRAM measurement utilities
│           ├── inputs.py          # Realistic test inputs per model
│           └── report.py          # Human-readable + JSON output
├── tests/
│   ├── conftest.py                # Shared fixtures (mock config, mock VRAM tracker)
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_scheduler.py      # SJF algorithm, priority, re-scoring
│   │   ├── test_memory.py         # Fit checks, eviction, keep-alive
│   │   ├── test_store.py          # SQLite persistence, crash recovery
│   │   └── test_job_lifecycle.py  # State transitions
│   ├── integration/
│   │   ├── test_api.py            # FastAPI TestClient (mock adapters)
│   │   ├── test_cli.py            # CLI subprocess tests
│   │   └── test_small_model.py    # Real GPU: loads whisper-tiny, runs inference
│   └── calibration/               # Manual only, per model
│       └── test_calibrate.py
├── assets/                        # Small test files for calibration
│   ├── test_audio_3s.wav
│   ├── test_face_512.png
│   └── test_scene_512.jpg
├── local/                         # .gitignored
│   ├── config.json                # Runtime config (from calibration)
│   ├── config.default.json        # Template (committed)
│   ├── models/                    # Symlinks to model weights
│   └── calibration/               # Per-model calibration results
├── output/                        # .gitignored
│   ├── arbiter.db                 # SQLite job queue
│   ├── jobs/                      # Result files by job_id
│   └── logs/                      # JSONL logs
├── run                            # Entry script (bash)
├── README.md
├── CLAUDE.md
├── pyproject.toml
└── .gitignore                     # local/, output/, .venv/, __pycache__/
```

---

## Models (11 adapters, 2 groups)

| Adapter ID | Model | Est. VRAM | Max Concurrent | Framework | Source |
|-----------|-------|----------|----------------|-----------|--------|
| `flux-schnell` | FLUX.1-schnell | ~12GB | 1 | Diffusers | images/server.py |
| `birefnet` | BiRefNet HR | ~2GB | 2 | Transformers | images/server.py |
| `moondream` | Moondream2 | ~4GB | 1 | Transformers | moondream-station/server.py |
| `whisper-large` | Whisper Large-v3 | ~3GB | 1 | whisper | ltx2-spark/transcribe.py |
| `tts-custom` | Qwen3-TTS CustomVoice | ~4GB | 1 | qwen-tts | tts/worker.py |
| `tts-clone` | Qwen3-TTS Base | ~4GB | 1 | qwen-tts | tts/worker.py |
| `tts-design` | Qwen3-TTS VoiceDesign | ~4GB | 1 | qwen-tts | tts/worker.py |
| **`sonic`** | SVD+UNet+Audio2Token+Audio2Bucket+WhisperTiny+YOLOFace+RIFE | ~15GB | 1 | PyTorch group | talking-head/Sonic/ |
| **`ltx2`** | Transformer+VAE+LoRA+Upsampler+Gemma3 (phased) | ~55GB peak | 1 | PyTorch group | ltx2-spark/ |

---

## Component Design

### 1. Configuration (`config.py`)

**File**: `local/config.json` (gitignored). Template: `local/config.default.json` (committed).

```json
{
  "vram_budget_gb": 100,
  "host": "0.0.0.0",
  "port": 8400,
  "default_keep_alive_seconds": 300,
  "models": {
    "flux-schnell": {
      "memory_gb": 12, "max_concurrent": 1, "keep_alive_seconds": 300,
      "avg_inference_ms": 2000, "load_ms": 15000,
      "auto_download": "black-forest-labs/FLUX.1-schnell"
    },
    "birefnet": {
      "memory_gb": 2, "max_concurrent": 2, "keep_alive_seconds": 300,
      "avg_inference_ms": 200, "load_ms": 5000,
      "auto_download": "ZhengPeng7/BiRefNet_HR"
    },
    "moondream": {
      "memory_gb": 4, "max_concurrent": 1, "keep_alive_seconds": 300,
      "avg_inference_ms": 2000, "load_ms": 8000,
      "auto_download": "vikhyatk/moondream2"
    },
    "whisper-large": {
      "memory_gb": 3, "max_concurrent": 1, "keep_alive_seconds": 120,
      "avg_inference_ms": 10000, "load_ms": 6000,
      "auto_download": null,
      "model_path": null
    },
    "tts-custom": {
      "memory_gb": 4, "max_concurrent": 1, "keep_alive_seconds": 300,
      "avg_inference_ms": 5000, "load_ms": 10000,
      "auto_download": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    },
    "tts-clone": {
      "memory_gb": 4, "max_concurrent": 1, "keep_alive_seconds": 300,
      "avg_inference_ms": 5000, "load_ms": 10000,
      "auto_download": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    },
    "tts-design": {
      "memory_gb": 4, "max_concurrent": 1, "keep_alive_seconds": 300,
      "avg_inference_ms": 5000, "load_ms": 10000,
      "auto_download": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    },
    "sonic": {
      "memory_gb": 15, "max_concurrent": 1, "keep_alive_seconds": 600,
      "avg_inference_ms": 45000, "load_ms": 20000,
      "group": true
    },
    "ltx2": {
      "memory_gb": 55, "max_concurrent": 1, "keep_alive_seconds": 600,
      "avg_inference_ms": 120000, "load_ms": 30000,
      "group": true
    }
  }
}
```

Validated by Pydantic. Env var overrides: `ARBITER_VRAM_BUDGET_GB`, `ARBITER_PORT`.

### 2. GPU Memory Manager (`memory.py`)

**Model states**: `UNLOADED -> LOADING -> LOADED (idle) <-> ACTIVE -> EVICTING -> UNLOADED`

**Core data structure**:
```python
@dataclass
class ModelSlot:
    model_id: str
    state: ModelState
    adapter: ModelAdapter
    memory_gb: float           # from config (static)
    active_count: int          # in-flight inferences
    last_active: float         # monotonic time when active_count hit 0
    keep_alive_s: float
    load_event: asyncio.Event  # signaled when loading completes
```

**Key operations**:
- `ensure_loaded(model_id)` -- if loaded, increment active_count. If not, check budget, evict if needed, load in thread pool.
- `release(model_id)` -- decrement active_count, start keep-alive timer.
- `_evict_for(needed_gb)` -- LRU eviction: select idle models sorted by last_active ascending, evict enough to free needed_gb.
- **Keep-alive loop** -- background task every 10s, evicts models idle past their keep_alive_seconds.
- **Pipeline overlap** -- scheduler can call `ensure_loaded(model_B)` while model_A is running, if VRAM budget permits both.

**Groups**: Treated as a single ModelSlot with total_memory_gb. GroupAdapter.load() loads all sub-models atomically.

### 3. Scheduler (`scheduler.py`)

**Persistent queue**: SQLite (`output/arbiter.db`) with WAL journal mode.

```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    state TEXT NOT NULL,      -- queued|scheduled|running|completed|failed|cancelled
    priority REAL NOT NULL,   -- SJF score (lower = run sooner)
    payload TEXT NOT NULL,    -- JSON request params
    result TEXT,              -- JSON result metadata
    error TEXT,
    created_at REAL, started_at REAL, finished_at REAL
);
```

**SJF scoring**:
```python
priority = avg_inference_ms + (load_ms if model not loaded else 0)
```
Re-scored when model load state changes. Tiebreaker: `created_at` (FIFO).

**Scheduler loop** (single asyncio task):
1. Pick lowest-priority queued job where model has concurrency capacity
2. Mark as `scheduled`
3. `await memory_manager.ensure_loaded(job.model_id)` (may evict + load)
4. Mark as `running`, dispatch to thread pool
5. **Speculatively pre-load** next job's model if VRAM budget allows (pipeline overlap)

**Crash recovery** on startup: `UPDATE jobs SET state='queued' WHERE state IN ('scheduled','running')`

### 4. Async Job API (`server.py`)

All endpoints. No streaming. Data transfer only (base64 for binary).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /v1/jobs` | Submit | Returns `{job_id, status: "queued", estimated_seconds}` (202) |
| `GET /v1/jobs/{id}` | Poll | Returns status; when completed, includes base64 result |
| `DELETE /v1/jobs/{id}` | Cancel | Cancels queued/running job |
| `GET /v1/jobs` | List | Filter by `?state=queued,running` etc. |
| `GET /v1/ps` | Status | Loaded models, VRAM used, active/queued counts |
| `GET /v1/health` | Health | `{status: "ok", uptime_seconds}` |

**Job types** (maps to model_id):

| `type` field | Model | Input | Output |
|-------------|-------|-------|--------|
| `image-generate` | flux-schnell | prompt, width, height, steps, seed | PNG base64 |
| `image-edit` | flux-schnell | prompt, image (base64), strength | PNG base64 |
| `background-remove` | birefnet | image (base64) | PNG RGBA base64 |
| `caption` | moondream | image (base64), length | JSON text |
| `query` | moondream | image (base64), question | JSON text |
| `detect` | moondream | image (base64), object | JSON bboxes |
| `transcribe` | whisper-large | audio (base64), language | JSON segments |
| `tts-custom` | tts-custom | text, speaker, language, temperature | WAV base64 |
| `tts-clone` | tts-clone | text, ref_audio (base64), temperature | WAV base64 |
| `tts-design` | tts-design | text, voice_description, temperature | WAV base64 |
| `talking-head` | sonic | image (base64), audio (base64) | MP4 base64 |
| `video-generate` | ltx2 | images (base64[]), audio (base64), resolution | MP4 base64 |

**Result storage**: `output/jobs/{job_id}/result.{png|wav|mp4|json}`. GET endpoint reads file and base64-encodes in response.

### 5. Model Adapter Interface (`adapters/base.py`)

```python
class ModelAdapter(ABC):
    model_id: str
    request_schema: type        # Pydantic model
    response_schema: type       # Pydantic model

    def load(self, device: torch.device) -> None: ...
    def unload(self) -> None: ...
    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict: ...
    def estimate_time(self, params: dict) -> float: ...

class GroupAdapter(ModelAdapter):
    """Atomic load/unload of multiple co-dependent sub-models."""
    def load(self): ...   # loads all sub-models; rolls back on failure
    def unload(self): ... # unloads all, gc.collect(), empty_cache()
```

**Cancellation**: `cancel_flag` is a `threading.Event`. Adapters check it at natural breakpoints (between diffusion steps, TTS chunks, pipeline phases).

**Adapter <-> existing code strategy**:
- **Import from source** for complex code: Sonic (adds Sonic repo to sys.path), LTX-2 (pip-installs ltx-core/ltx-pipelines packages)
- **Rewrite** for simple code: FLUX (~50 lines), BiRefNet (~30 lines), Moondream (~20 lines), Whisper (~5 lines), TTS (~30 lines per variant)

### 6. Worker Threading

**Single-process, ThreadPoolExecutor.** Rationale:
- PyTorch releases GIL during CUDA kernels -- threads get real parallelism for GPU work
- Separate processes would waste ~500MB each on CUDA context overhead
- Existing codebase already uses ThreadPoolExecutor successfully
- Concurrency bounded by `max_concurrent` per model (usually 1)

Pool size: 8 threads (enough for worst case: a few models each at max_concurrent=2).

### 7. JSONL Logging (`log.py`)

File: `output/logs/arbiter-YYYY-MM-DD.jsonl`, one per day.

Events: `job.submitted`, `job.scheduled`, `job.started`, `job.completed`, `job.failed`, `job.cancelled`, `model.load_start`, `model.load_done`, `model.evict_start`, `model.evict_done`, `memory.snapshot` (every 60s), `server.start`, `server.stop`.

Each line: `{"ts": 1711000000.123, "event": "...", "model_id": "...", "job_id": "...", ...}`

### 8. CLI (`cli.py`)

Communicates via HTTP to the running server.

```
./run server              # Start Arbiter
./run ps                  # Show loaded models + VRAM
./run jobs                # Show queue
./run submit <type> ...   # Submit a job
./run cancel <id>         # Cancel a job
./run test                # Fast tests
./run calibrate <model>   # Calibrate one model
./run calibrate --all     # Calibrate all
```

### 9. Calibration System (`calibrate/`)

Per model, measures:
- Load/unload time (3 iterations, median)
- VRAM when loaded + idle
- At concurrency levels 1,2,4,8,16: avg latency, p50/p95/p99, peak VRAM, memory per request, throughput
- Recommended max_concurrent (highest where avg < 2x single-request latency)

Stores results as JSON in `local/calibration/<model>.json` with a `config_entry` field ready to paste into config.json. Uses realistic test inputs from `assets/`.

### 10. Auto-Download

When an adapter's `load()` is called and the model isn't found locally, check `auto_download` in config. If set (HuggingFace repo ID), download via `huggingface_hub` to `local/models/`.

---

## Implementation Phases

### Phase 1: Foundation (no GPU needed)
- `config.py` -- Pydantic config loader + validation
- `store.py` -- SQLite job store (CRUD, crash recovery)
- `log.py` -- JSONL logger
- `adapters/base.py` + `registry.py` -- abstract interfaces
- `schemas.py` -- all request/response Pydantic models
- Unit tests for all above

### Phase 2: Core Engine
- `memory.py` -- GPU memory manager (test with mock adapters)
- `scheduler.py` -- scheduler loop + SJF (test with mock memory manager)
- `worker.py` -- job dispatch + result storage
- Unit tests for scheduler, memory, worker

### Phase 3: API + CLI
- `server.py` -- FastAPI app, all endpoints, lifespan, crash recovery
- `cli.py` -- CLI tool
- Integration tests (TestClient with mock adapters)

### Phase 4: Adapters (parallel, each independent)
- `birefnet.py` -- simplest, good first test
- `whisper_large.py` -- exercises audio handling
- `moondream.py` -- exercises multi-task dispatch
- `flux.py` -- exercises aspect ratio logic
- `tts_custom.py`, `tts_clone.py`, `tts_design.py` -- three variants
- `sonic.py` -- first GroupAdapter, 8 sub-models
- `ltx2.py` -- most complex, phased loading

### Phase 5: Calibration + Integration
- `calibrate/` -- calibration tool
- Generate real config.json from calibration results
- End-to-end tests with real models
- `run` script, README.md, CLAUDE.md

---

## Key Challenges & Mitigations

1. **LTX-2 at 55GB prevents co-residency with most models.** Handled naturally: scheduler evicts everything idle before loading LTX-2. After LTX-2 finishes and times out, smaller models reload.

2. **LTX-2 internal phased loading.** The adapter reports 55GB peak to memory manager but internally manages sub-model loading/unloading per phase (transformer -> VAE -> upsampler). This is conservative but safe.

3. **Sonic sys.path/chdir requirements.** Adapter adds Sonic dir to sys.path and uses absolute paths throughout. No chdir.

4. **TTS voice cloning needs temp files.** Adapter writes ref_audio bytes to a named temp file, creates voice_clone_prompt, deletes temp file immediately.

5. **Sonic ffmpeg subprocess.** Adapter uses `subprocess.run` with pipes for in-memory muxing. No temp files for video output.

---

## Verification Plan

1. **Unit tests** (`./run test`): Config loading, scheduler SJF, memory manager eviction, store persistence, job lifecycle state machine. All run without GPU. Target: <30s.

2. **Integration tests** (`./run test`): API endpoint tests with mock adapters via FastAPI TestClient. CLI subprocess tests. Target: <60s.

3. **GPU smoke test** (`./run test-all`): Load whisper-tiny (smallest model, ~0.5GB), run real inference, verify VRAM tracking, verify unload frees memory.

4. **Calibration** (`./run calibrate <model>`): Run per new model. Produces measured config values. Manual trigger only.

5. **End-to-end**: Submit jobs for every model type, verify queue -> schedule -> load -> infer -> result cycle. Verify crash recovery by killing server mid-inference and restarting.
