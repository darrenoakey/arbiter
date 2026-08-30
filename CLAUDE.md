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

## Memory Containment (critical — do not regress)

On the GB10 there is NO discrete VRAM: GPU memory IS the 119.5GB unified system pool, and **CUDA allocations are NOT charged to the process memory cgroup** (verified: a `MemoryMax=4G` scope let a CUDA process allocate 8GB unhindered). A runaway adapter therefore exhausts physical RAM while keeping a tiny RSS, so the kernel OOM killer can't find it — the host livelocks with no OOM log and needs a physical reset. systemd `MemoryMax`/slices/`systemd-oomd` CANNOT contain this; containment must happen in the application layer.

The fix (commit `8429f62`): the Go server (`cmd/arbiter/proc.go`) passes `ARBITER_MEMORY_GB=<inst.memoryGB>` to every worker; `src/arbiter/worker_main.py` `_apply_cuda_memory_cap()` calls `torch.cuda.set_per_process_memory_fraction(declared*1.15/total, ceil 0.92)` at startup. An overshoot then raises a catchable `CUDA out of memory` (job fails, caller retries) instead of wedging the host. The cap derives from each model's declared `memory_gb` — so an inaccurate declaration produces an inaccurate cap. If a model legitimately CUDA-OOMs under its cap, raise its `memory_gb` in `local/config.json` (vllm/llm workers are separate binaries that self-bound via `--gpu-memory-utilization` and are unaffected). Host backstops (not in this repo): SBSA hardware watchdog (`RuntimeWatchdogSec=10s` → auto-reboot on livelock) and `vm.swappiness=10`.

### ⛔ NEVER run CUDA work on spark outside arbiter — never ever ever ever ever ever do that again

**No direct `ssh` python/torch/CUDA training or inference jobs on spark. No exceptions.** The memory cap above is applied by `worker_main.py` inside arbiter-managed workers ONLY. A CUDA process started by hand (ssh + `python train.py`, a nohup'd fit, a "quick" GPU probe) has **no cap**, and on the GB10's unified memory it can exhaust all 119.5GB of system RAM while showing a tiny RSS — the OOM killer cannot see it, the host livelocks, and recovery needs a physical reset. Every GPU job — training included (rvc-train, voice-fit, lora-train) — MUST go through the arbiter queue (`POST /v1/jobs`), where the worker gets its `ARBITER_MEMORY_GB` cap. If a GPU capability you need has no adapter yet, **add an adapter** (see "How to Add a New Model") — do not improvise a side-channel run. This rule exists because we livelocked the host doing exactly this (voxsmith fit run directly over ssh, 2026-07). Local CPU work on other machines is unaffected; this rule is about CUDA on spark.

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
- Runtime registration/update goes through the Go control plane: `POST /v1/models` can add a new configured model live, and `PATCH /v1/models/{id}` with `reload_workers=true` replaces only that model's workers so other adapters keep serving.
- Live model registration persists through `SaveModelConfig` into `local/config.json`. Keep those writes serialized and atomic: concurrent registration can otherwise leave a truncated JSON file that crashes startup/reload. Runtime-created model configs may omit optional pointer fields such as `pressure_index`; scheduler code must default nil `PressureIndex` to `1.0` instead of dereferencing it.
- Live model registration persists through `SaveModelConfig` into `local/config.json`. Keep those writes serialized and atomic: concurrent registration can otherwise leave a truncated JSON file that crashes startup/reload. Runtime-created model configs may omit optional pointer fields such as `pressure_index`; scheduler code must default nil `PressureIndex` to `1.0` instead of dereferencing it.

## Logs

Two streams on spark, neither in the repo-local `output/` or `local_output/` dirs (those are stale):
- **JSONL event log** (job lifecycle, model.scaled/auto_wake, vram): `$ARBITER_OUTPUT_DIR/logs/arbiter-YYYY-MM-DD.jsonl` — in production `/mnt/arbiter-store/output/logs/`.
- **stdout/slog** (HTTP access lines incl. PATCH callers, scheduler decisions): managed by `auto` at `~/local/auto/output/logs/arbiter/YYYY/MM/` — find the live file via `ls -l /proc/$(pgrep -f arbiter-go)/fd/1`. An HTTP line with `remote=127.0.0.1` means the call was made on spark itself (e.g. through an ssh session).

## Reference Files

Arbiter supports "reference files" — binary files uploaded once and reused across multiple jobs without re-uploading.

### API

- `POST /v1/refs` — Upload a file (multipart `file` field or raw body with `?filename=`). Returns `{"ref_id": "abc123.wav"}`.
- `GET /v1/refs` — List all reference files.
- `GET /v1/refs/{id}` — Download a reference file.
- `DELETE /v1/refs/{id}` — Delete a reference file.

Files are stored in `output/refs/`.

### Usage in Jobs

Pass a `ref:` prefix in any `_file` parameter. This works with every adapter via `_resolve_media()`:

```json
{
  "type": "tts-clone",
  "params": {
    "text": "Hello",
    "ref_text": "Reference transcript",
    "ref_audio_file": "ref:abc123.wav"
  }
}
```

The `ref:` prefix is resolved in `_resolve_media()` in `src/arbiter/adapters/base.py`. No adapter-specific changes are needed.

## Key Design Decisions

- **Single process, ThreadPoolExecutor**: PyTorch releases GIL during CUDA ops. Threads share GPU memory efficiently.
- **No Arbiter still-image generation.** This unconditional owner policy is
  enforced at API routing, persistent config loading/mutation, registration,
  reload, auto-wake, worker startup, and retained adapter load/infer boundaries.
  Flux/Flux2/Schnell/Kontext/LoRA and Z-Image aliases are denied even when
  disguised as another job type. User-facing image creation/editing belongs to
  the Mac mini Codex image service. BiRefNet `background-remove` and every LTX2
  video stage remain supported.
- **SJF scheduling**: `priority = avg_inference_ms + (load_ms if not loaded else 0)`. Shortest jobs run first. Already-loaded models get natural priority.
- **SQLite queue**: Persistent, crash-recoverable. On restart, incomplete jobs are re-queued.
- **Dedup followers**: Duplicate requests are persisted as jobs with `state=following` and `error=following:<original_job_id>`. Startup recovery must requeue `scheduled`/`running` jobs, then reconcile followers so none remain attached to terminal or missing originals.
- **Async job API**: Submit → poll → get result. Server crashes are transparent to clients.
- **LRU eviction**: Models idle past keep_alive_seconds get evicted. When memory is needed, oldest idle model goes first.
- **Group adapters**: Sonic (8 sub-models) and LTX-2 (phased pipeline) load/unload atomically.

## Calibration Results (Grace Blackwell GB10, 128GB VRAM)

| Model | VRAM (GB) | Load Time | Inference Time | Max Concurrent |
|-------|-----------|-----------|----------------|----------------|
| birefnet | 0.83 | 5.4s | 1.0s | 2 |
| moondream (v3) | 17.28 | 142s | 103s | 1 |
| whisper-large | 5.88 | 11.3s | 1.8s | 1 |
| tts-custom | 3.89 | 43s | 4s | 1 |
| tts-clone | 3.91 | 44s | ~4s | 1 |
| tts-design | ~3.9 | ~43s | ~5s | 1 |
| sonic (group) | 4.84 | 11s | ~45s | 1 |
| ltx2 (group) | ~55 | ~30s | ~120s | 1 |

Note: Moondream3 uses substantially more memory than moondream2.
Sonic at 5GB is much lighter than the 15GB estimate.

## Known Issues & Compatibility

### transformers version
- **Must use transformers 4.57.3** — qwen-tts pins this exact version
- moondream3 and BiRefNet work on 4.57.3
- transformers 5.x breaks qwen-tts import (`check_model_inputs` removed)

### Model-specific notes
- **Moondream3** (`moondream/moondream3-preview`): Upgraded from moondream2. Uses `dtype=` not `torch_dtype=` (deprecated). First inference triggers FlexAttention JIT compilation (~extra 30s). Consider torch.compile for production speed.
- **Whisper large-v3**: NOT thread-safe for concurrent calls. Errors at concurrency >= 2.
- **TTS output format**: `generate_custom_voice()` returns numpy arrays, not torch tensors. Adapter handles both via hasattr check.
- **TTS-clone**: Requires `ref_text` parameter alongside `ref_audio` for voice cloning.
- **Sonic**: Requires a real face in the input image — fails with "cannot access local variable 'bbox_s'" if no face detected.
- **LTX-2**: `load()` is instant (~2ms) — only creates config objects. Heavy model loading happens inside `infer()` per-phase. Memory manager should budget 55GB for the full job duration.
- **BiRefNet**: Needs `kornia` package.
- **vLLM chat worker**: `cmd/vllm-chat-worker` wraps `vllm serve`. When checking whether the child died during readiness polling, use a goroutine running `cmd.Wait()` and select on that channel; `exec.Cmd.ProcessState` stays nil for a zombie child until `Wait()` runs, which can leave Arbiter jobs stuck in `scheduled` with `active_jobs=1`.

## Key Dependencies

Core: fastapi, uvicorn, pydantic, httpx
ML: torch 2.10+cu130, transformers==4.57.3, diffusers, openai-whisper, qwen-tts
Model-specific: kornia (birefnet), omegaconf opencv-python-headless (sonic)
Packages: ltx-core, ltx-pipelines (installed from /home/darren/src/ltx2-spark/packages/)

## Model Weight Locations

Weights owned by Arbiter (in local/models/):
- `local/models/ltx2/` — LTX-2 checkpoints (moved from /home/darren/models/ltx2/)
- `local/models/sonic/` — Sonic checkpoints (moved from talking-head/Sonic/checkpoints/)
- Symlinks exist at the old locations pointing back here

HuggingFace cache (loaded by repo ID, shared ~/.cache/huggingface/):
- BiRefNet, Moondream3, Qwen3-TTS variants

Whisper cache: ~/.cache/whisper/large-v3.pt

Note: tts-design (Qwen3-TTS-12Hz-1.7B-VoiceDesign) is NOT downloaded yet.

## Running as a Daemon

Arbiter runs under `auto` (process manager):

```bash
# It's already registered:
auto ps                    # Check status
auto start arbiter         # Start
auto stop arbiter          # Stop
auto restart arbiter       # Restart
auto log arbiter           # View logs
```

## Invariants (do not break)

**One queue, one path.** Every inference (chat completion, streaming, image, TTS — everything) goes through `store.CreateJob` → scheduler picks → dispatch goroutine → release. No "fast paths," no "admin bypasses," no endpoints that proxy directly to a worker. The activeJobs counter is the only gate; nothing else may touch it. If you find yourself adding a `ReserveExternal`-style helper, stop — it always becomes a leak vector.

**Streaming chat** uses the same queue via the stream-handoff registry: API handler creates a `chat-completion-stream` job, registers a handoff under the job ID, the scheduler's dispatch goroutine hands the picked instance back through `instCh` and blocks on `doneCh` until the handler finishes proxying SSE. Slot accounting is identical to non-streaming.

**PickInstance** must never return an instance that lacks capacity. Loading instances at capacity also count — piling jobs onto a not-yet-loaded worker just stacks them up to fire all at once when load completes.

**VRAM bookkeeping.** `usedGB` must equal sum(`memoryGB` for instances where `vramHeld == true`) ± float-drift tolerance. If no instance holds VRAM, `usedGB` must be zero. `AuditVRAMConsistency(ctx)` enforces this on every load/unload — if it fires, dump every instance's state and root-cause the leaking path.

**Terminal jobs stay terminal.** `UpdateState` must not allow late scheduler/dispatch updates to move `completed`, `failed`, or `cancelled` jobs back to `queued`, `scheduled`, `running`, or `following`. Crash recovery must not requeue any active-state row that already has `finished_at`; mark it failed instead so active slots cannot be resurrected and stranded after cancellation races.

**Terminal jobs stay terminal.** `UpdateState` must not allow late scheduler/dispatch updates to move `completed`, `failed`, or `cancelled` jobs back to `queued`, `scheduled`, `running`, or `following`. Crash recovery must not requeue any active-state row that already has `finished_at`; mark it failed instead so active slots cannot be resurrected and stranded after cancellation races.

**No silently dead models.** A model with queued work must always be able to make progress eventually. Scale-to-zero (`max_instances=0`) is a legitimate *temporary* operation, but the scheduler's auto-wake guard (`autoWakeParkedModels` in scheduler.go) scales any parked model with queued jobs back to 1 after `auto_wake_seconds` (default 300s; negative disables) and persists it — because the 2026-06-09 gemma4 outage proved an operator who parks a model and crashes leaves it dead forever (the 0 is persisted across restarts while jobs queue silently). Do not add code paths that can leave a model permanently unable to serve its queue.

## Multi-Host Model Offload (live since 2026-06-23)

Arbiter can offload a model to a remote executor (Apple Silicon Mac running ollama/MLX) while staying the sole front door. spark remains the only queue/brain; clients always hit spark:8400. Config (top-level `hosts` map + per-model `placements`) lives in `local/config.json`:

```json
"hosts": { "boringstack": {"addr":"http://10.0.0.42:11435","kind":"mlx","budget_gb":96} },
"models": { "llm:qwen3.6-27b": {
  "placements": ["boringstack"],
  "remote_enabled": true,
  "adapter_params": {"remote_model_tag": "qwen3.6:27b-q8_0"},
  "max_concurrent": 2
}}
```

- **One Instance per placement host** (`main.go setupInstances`): local (spark) = subprocess pool as today; each remote host = one `RemoteHTTPBackend` instance (`backend.go`). Chat jobs POST to `{addr}/v1/chat/completions`; `embed-text` jobs POST to Ollama `{addr}/api/embed`. `PickInstanceForJob` walks `placements` in order, skipping confirmed-absent or `excluded_hosts`, returning the first with capacity. spark is local + always reachable = guaranteed final fallback.
- **Wiring requires a restart.** `setupInstances` + `HostMonitor` read `cfg.Hosts` only at startup. To add/change a `hosts` block or a model's `placements`, edit `local/config.json` on spark and **restart** arbiter (a PATCH cannot create a remote instance). The kill-switches and reachability operate live on already-created instances.
- **Remote = ZERO audited VRAM (invariant).** Remote instances never touch `usedGB`/`AuditVRAMConsistency`. Their capacity is advisory in a SEPARATE `remoteHostBudget` and surfaced in the SEPARATE `/v1/ps` `remote_hosts` panel — never mixed into `vram_actual_gb`.
- **Transparent mid-job failover.** On CONFIRMED remote absence (liveness poll `{addr}/api/version` 3×4s fails, or dial/conn-reset/EHOSTUNREACH) — NEVER on mere slowness — `tryFailover` requeues running→queued (allowed: no `finished_at`), appends the host to durable `excluded_hosts`, and the scheduler re-picks down the chain. The active-cancel hook (`backend.Cancel`) fires from the poll so failover is seconds, not the inference timeout. Idempotency: terminal-stays-terminal + in-flight ownership guarantee exactly one result even if the dead host responds late. `excluded_hosts` is append-only and NOT cleared on `host.recovered` (a recovered host won't re-absorb already-excluded jobs; spark catches them).
- **Streaming = buffer-on-Mac / replay-from-spark.** For a remote `chat-completion-stream` job, arbiter buffers the FULL completion then emits SSE locally — so a mid-gen host loss yields 0 client bytes (no broken stream), failover is invisible. Local/spark instances keep the direct SSE proxy.
- **Kill-switches (instant, one curl, work even if the host is down).** Per-model `PATCH /v1/models/{id} {"remote_enabled":false}` pins that model to spark + drains in-flight remote jobs; global `PATCH /v1/remote {"enabled":false}` does it fleet-wide. Both persist (`SaveModelConfig`/`PatchRemoteDisabled`).
- **Remote-servable models are NOT gated by spark-local VRAM/load-CB.** `getFullModels` computes `remoteServable = RemoteAllowedFor(model) && ModelHasReachableRemoteCapacity(model)` and skips BOTH the local VRAM-feasibility gate AND the load circuit-breaker for such a model — otherwise spark GPU pressure (e.g. ltx2 holding 56GB) starves/freezes/fails a model that actually runs on a remote box. A VRAM-insufficiency load failure (incl. the `waitForInProgressLoad` race-loser path via `Instance.lastLoadInsufficientMem`) requeues, never counts toward `maxLoadAttempts`. When the remote is absent or kill-switched, both gates apply normally (job falls back to spark, which must fit).
- **gemma fallback caveat:** spark gemma declares `memory_gb=90` and cannot load while ltx2 holds the GPU. With both Macs absent + spark ltx2-saturated, a gemma job legitimately QUEUES (not fails) until spark frees up. The intended fast fallback is a second Mac (darrens-mbp). gemma uses plain MLX `gemma4-26b-32k` (NO MTP — benchmarked slower at batch=1) and is a reasoning model: low `max_tokens` → empty `content` + `finish_reason:length`; pass generous `max_tokens` (≥256).
- **Remote-ONLY models are valid (no spark placement).** `llm:qwen3.6-27b` has `placements: ["boringstack"]` with NO `"spark"` — `setupInstances` creates only the remote instance, so no `worker_cmd` or weights are needed on spark. `remote_model_tag: "qwen3.6:27b-q8_0"` is the official non-MTP Ollama Q8 build selected on the 128 GB M5 Max after real warm throughput, schema/tool conformance, and repeated stability probes; `memory_gb: 30` is advisory-only for the remote host, and `pressure_index: 0` holds zero spark VRAM/bandwidth. If boringstack is absent there is no fallback: jobs queue until it returns.
- **The Mac chat fleet is intentionally mixed (2026-08-30).** Boringstack runs Ollama through the LAN proxy at `10.0.0.42:11435` (`kind: "mlx"`); Nativ and its dedicated model repository were removed from that machine. `darrens-mbp` remains a Nativ host for models whose MLX-community tags have not migrated, and macmini remains the Ollama chat/embed host at `10.0.0.46:11435`. Never place a Nativ-only MLX-community tag on boringstack: Ollama requires an installed Ollama tag. Ollama host liveness uses `/api/version` and loaded-model discovery uses `/api/ps`.
- **Embeddings must be numerically equivalent across placements.** `embed-text` uses `nomic-embed-text:latest` GGUF F16 on macmini, and `nomic-ai/nomic-embed-text-v1.5` F16 on spark (`placements: ["macmini","spark"]` — boringstack left the embed pool when its ollama was removed). The remote request enforces 8192 context, task prefixes (`search_document`, `search_query`, `classification`, or `clustering`), Ollama mean-pooling/L2 normalization, 768 dimensions, finite values, and exact response ordering. A wrong remote model tag fails as a job error instead of silently mixing vector spaces. Verified same-text remote versus spark cosine similarity: 0.99999924.
- See memory [[multimachine-arbiter-project]] / [[multimachine-impl-spec]] / [[fleet-topology]] for the full design + per-phase history.

**Note on this CLAUDE.md.** Sections above describe the older Python architecture in `src/arbiter/`. The live system is the Go server in `cmd/arbiter/` (scheduler.go, proc.go, api.go, store.go). Treat the Python file paths as historical until that section is rewritten.
