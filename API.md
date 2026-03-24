# Arbiter API Reference

Complete API reference for client developers integrating with the Arbiter GPU model server.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Endpoints Reference](#2-endpoints-reference) — Jobs, Bulk Status, Reference Files, Model Management, Reservations, System Status, Health
3. [Job Types Reference](#3-job-types-reference)
4. [Client Workflow](#4-client-workflow)
5. [Model Memory & Scheduling](#5-model-memory--scheduling)
6. [Rate Limits & Quotas](#6-rate-limits--quotas)

---

## 1. Overview

### Base URL

```
http://localhost:8400
```

### Core Concepts

- **Async job model**: All work is asynchronous. Submit a job, receive a `job_id`, then poll for the result. There is no synchronous inference endpoint.
- **Three ways to pass binary data**:
  1. **Base64 inline**: Encode binary as base64 and pass directly in the JSON params (e.g., `"image": "<base64>"`).
  2. **File path**: Pass a local filesystem path in a `_file` param (e.g., `"image_file": "/path/to/file.png"`).
  3. **Reference file**: Upload a file once via `POST /v1/refs`, then reference it in any job with a `ref:` prefix (e.g., `"image_file": "ref:a1b2c3.png"`). The file persists until explicitly deleted. Ideal for files reused across many jobs.
- **No authentication**: Arbiter is designed for internal/local use and does not require API keys or tokens.
- **Persistent queue**: Jobs are stored in SQLite and survive server restarts. If Arbiter crashes, in-flight jobs are automatically re-queued on the next startup.

### Job Lifecycle

Every job passes through these states in order:

```
queued -> scheduled -> running -> completed
                                   \-> failed
         \-> cancelled
```

| State       | Description                                          |
|-------------|------------------------------------------------------|
| `queued`    | Job accepted, waiting in the priority queue          |
| `scheduled` | Scheduler has picked this job; model is loading      |
| `running`   | Model is loaded and inference is in progress         |
| `completed` | Inference finished successfully; result is available |
| `failed`    | Inference failed; `error` field contains the reason  |
| `cancelled` | Job was cancelled via `DELETE /v1/jobs/{id}`         |

---

## 2. Endpoints Reference

### POST /v1/jobs -- Submit a Job

Submit a new inference job to the queue.

**Request**

```
POST /v1/jobs
Content-Type: application/json
```

| Field    | Type   | Required | Description                                      |
|----------|--------|----------|--------------------------------------------------|
| `type`   | string | Yes      | One of the 12 job type strings (see Section 3)   |
| `params` | object | No       | Parameters specific to the job type (default `{}`) |

**Request Body Example**

```json
{
  "type": "image-generate",
  "params": {
    "prompt": "A cat sitting on a windowsill at sunset",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "seed": 42
  }
}
```

**Response (202 Accepted)**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "model": "flux-schnell",
  "estimated_seconds": 12.0
}
```

| Field               | Type        | Description                                                                 |
|---------------------|-------------|-----------------------------------------------------------------------------|
| `job_id`            | string      | Unique job identifier (UUID)                                                |
| `status`            | string      | Always `"queued"` on submission                                             |
| `model`             | string      | The model that will process this job                                        |
| `estimated_seconds` | float/null  | Estimated total time including model load (if needed) and inference          |

**Error Responses**

| Status | Condition                         | Body Example                                              |
|--------|-----------------------------------|-----------------------------------------------------------|
| 400    | Unknown job type                  | `{"detail": "Unknown job type: foo"}`                     |
| 400    | Model not configured on server    | `{"detail": "Model not configured: flux-schnell"}`        |
| 400    | Invalid parameters for job type   | `{"detail": "Invalid params: Field required: 'prompt'"}`  |
| 422    | Malformed JSON / missing `type`   | Standard FastAPI validation error                         |

---

### GET /v1/jobs/{job_id} -- Get Job Status / Result

Retrieve the current status of a job. When the job is completed, the response includes the full result with base64-encoded output data.

**Request**

```
GET /v1/jobs/{job_id}
```

**Response -- Queued/Scheduled/Running**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "running",
  "model": "flux-schnell",
  "created_at": 1711000000.0,
  "started_at": 1711000002.5,
  "finished_at": null,
  "error": null,
  "result": null
}
```

**Response -- Completed**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "model": "flux-schnell",
  "created_at": 1711000000.0,
  "started_at": 1711000002.5,
  "finished_at": 1711000014.5,
  "error": null,
  "result": {
    "format": "png",
    "width": 1024,
    "height": 1024,
    "data": "/9j/4AAQSkZJRgABAQ..."
  }
}
```

The `result` object varies by job type (see Section 3 for each type's result schema). The `data` field contains the base64-encoded output file.

**Response -- Failed**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "failed",
  "model": "flux-schnell",
  "created_at": 1711000000.0,
  "started_at": 1711000002.5,
  "finished_at": 1711000003.1,
  "error": "CUDA out of memory",
  "result": null
}
```

**Response -- Cancelled**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "cancelled",
  "model": "flux-schnell",
  "created_at": 1711000000.0,
  "started_at": null,
  "finished_at": 1711000001.2,
  "error": null,
  "result": null
}
```

| Field         | Type        | Description                                         |
|---------------|-------------|-----------------------------------------------------|
| `job_id`      | string      | The job identifier                                  |
| `status`      | string      | One of: `queued`, `scheduled`, `running`, `completed`, `failed`, `cancelled` |
| `model`       | string      | Model ID that processed (or will process) this job  |
| `created_at`  | float       | Unix timestamp when job was submitted               |
| `started_at`  | float/null  | Unix timestamp when inference began                 |
| `finished_at` | float/null  | Unix timestamp when job completed or failed         |
| `error`       | string/null | Error message if job failed                         |
| `result`      | object/null | Result payload when completed (varies by job type)  |

**Error Responses**

| Status | Condition      | Body Example                                  |
|--------|----------------|-----------------------------------------------|
| 404    | Job not found  | `{"detail": "Job not found: nonexistent-id"}` |

---

### DELETE /v1/jobs/{job_id} -- Cancel a Job

Cancel a queued or in-progress job. Jobs that are already finished cannot be cancelled.

**Request**

```
DELETE /v1/jobs/{job_id}
```

**Response -- Successfully Cancelled (200)**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "cancelled"
}
```

**Response -- Job Already Finished (200)**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "message": "Job already finished"
}
```

**Error Responses**

| Status | Condition                     | Body Example                                  |
|--------|-------------------------------|-----------------------------------------------|
| 404    | Job not found                 | `{"detail": "Job not found: nonexistent-id"}` |
| 409    | Could not cancel (race cond.) | `{"detail": "Could not cancel job"}`          |

---

### GET /v1/jobs -- List Jobs

List jobs with optional filters.

**Request**

```
GET /v1/jobs?state=queued&model=flux-schnell&limit=50
```

| Query Param | Type   | Required | Default | Description                                    |
|-------------|--------|----------|---------|------------------------------------------------|
| `state`     | string | No       | all     | Filter by state: `queued`, `running`, `completed`, `failed`, `cancelled` |
| `model`     | string | No       | all     | Filter by model ID (e.g. `flux-schnell`)       |
| `limit`     | int    | No       | 100     | Maximum number of jobs to return               |

**Response (200)**

```json
[
  {
    "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "type": "image-generate",
    "model": "flux-schnell",
    "status": "completed",
    "created_at": 1711000000.0,
    "started_at": 1711000002.5,
    "finished_at": 1711000014.5
  },
  {
    "job_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "type": "caption",
    "model": "moondream",
    "status": "queued",
    "created_at": 1711000010.0,
    "started_at": null,
    "finished_at": null
  }
]
```

Note: The list endpoint does not include `result` or `error` fields. Use `GET /v1/jobs/{job_id}` to retrieve the full result for a specific job.

---

### GET /v1/ps -- System Status

Returns a snapshot of the current system state: VRAM usage, loaded models, and job queue counts.

**Request**

```
GET /v1/ps
```

**Response (200)**

```json
{
  "vram_budget_gb": 100.0,
  "vram_used_gb": 37.0,
  "gpu_utilization_pct": 86,
  "models": [
    {
      "id": "flux-schnell",
      "state": "loaded",
      "memory_gb": 32.0,
      "active_jobs": 1,
      "queued_jobs": 3,
      "idle_seconds": null
    },
    {
      "id": "sonic",
      "state": "loaded",
      "memory_gb": 5.0,
      "active_jobs": 0,
      "queued_jobs": 0,
      "idle_seconds": 142.3
    },
    {
      "id": "moondream",
      "state": "unloaded",
      "memory_gb": 18.0,
      "active_jobs": 0,
      "queued_jobs": 1,
      "idle_seconds": null
    }
  ],
  "queue": {
    "queued": 4,
    "running": 1,
    "completed": 57,
    "failed": 2,
    "cancelled": 0
  }
}
```

| Field            | Type  | Description                                        |
|------------------|-------|----------------------------------------------------|
| `vram_budget_gb`      | float | Total VRAM budget configured for the server             |
| `vram_used_gb`        | float | VRAM currently used by loaded models                    |
| `gpu_utilization_pct` | int   | GPU compute utilization percentage (0-100, -1 if unavailable) |
| `models`              | array | Status of each registered model                         |
| `queue`               | object| Job counts by state across all models                   |

Each model object:

| Field          | Type        | Description                                   |
|----------------|-------------|-----------------------------------------------|
| `id`           | string      | Model identifier                              |
| `state`        | string      | `loaded` or `unloaded`                        |
| `memory_gb`    | float       | VRAM consumed when loaded                     |
| `active_jobs`  | int         | Jobs currently running on this model          |
| `queued_jobs`  | int         | Jobs waiting in queue for this model          |
| `idle_seconds` | float/null  | Seconds since last inference (null if active or unloaded) |

---

### POST /v1/refs -- Upload a Reference File

Upload a file once for reuse across multiple jobs. Reference files persist until explicitly deleted.

**Request (multipart)**

```
POST /v1/refs
Content-Type: multipart/form-data
```

Attach the file as the `file` field.

**Request (raw body)**

```
POST /v1/refs?filename=voice.wav
Content-Type: application/octet-stream

<raw file bytes>
```

**Response (201 Created)**

```json
{
  "ref_id": "a1b2c3d4e5f6.wav",
  "size_bytes": 128000,
  "filename": "voice.wav"
}
```

The `ref_id` includes the file extension from the original filename. Use it in jobs with the `ref:` prefix:

```json
{
  "type": "tts-clone",
  "params": {
    "text": "Hello world",
    "ref_text": "Reference transcript",
    "ref_audio_file": "ref:a1b2c3d4e5f6.wav"
  }
}
```

This works with any `_file` parameter on any job type (e.g., `image_file`, `audio_file`, `ref_audio_file`).

**Error Responses**

| Status | Condition                          | Body Example                                  |
|--------|------------------------------------|-----------------------------------------------|
| 400    | Missing file or empty body         | `{"detail": "empty file"}`                    |
| 400    | Raw upload without `?filename=`    | `{"detail": "raw upload requires ?filename= query param"}` |

**curl Examples**

```bash
# Upload via multipart
curl -X POST http://localhost:8400/v1/refs -F file=@voice_sample.wav

# Upload via raw body
curl -X POST "http://localhost:8400/v1/refs?filename=voice.wav" \
  --data-binary @voice_sample.wav
```

---

### GET /v1/refs -- List Reference Files

List all uploaded reference files.

**Response (200)**

```json
[
  {
    "ref_id": "a1b2c3d4e5f6.wav",
    "size_bytes": 128000,
    "created_at": 1711000000
  },
  {
    "ref_id": "f0e1d2c3b4a5.png",
    "size_bytes": 2048000,
    "created_at": 1711000050
  }
]
```

---

### GET /v1/refs/{id} -- Download a Reference File

Download a reference file by its `ref_id`.

**Response (200)**

Returns the raw file bytes with the appropriate `Content-Type` header.

**Error Responses**

| Status | Condition      | Body Example                    |
|--------|----------------|---------------------------------|
| 404    | Ref not found  | `{"detail": "ref not found"}`   |

---

### DELETE /v1/refs/{id} -- Delete a Reference File

Permanently delete a reference file. Jobs already using this ref will fail if the file is needed again.

**Response (200)**

```json
{
  "ref_id": "a1b2c3d4e5f6.wav",
  "status": "deleted"
}
```

**Error Responses**

| Status | Condition      | Body Example                    |
|--------|----------------|---------------------------------|
| 404    | Ref not found  | `{"detail": "ref not found"}`   |

---

### GET /v1/health -- Health Check

Simple liveness check.

**Request**

```
GET /v1/health
```

**Response (200)**

```json
{
  "status": "ok",
  "uptime_seconds": 3621.5
}
```

| Field            | Type   | Description                          |
|------------------|--------|--------------------------------------|
| `status`         | string | Always `"ok"` if server is running   |
| `uptime_seconds` | float  | Seconds since server started         |

---

### POST /v1/jobs/status -- Bulk Poll Job Status

Poll up to 1000 jobs in a single request. Returns statuses in request order with `null` for unknown IDs. Result metadata is included but file data is not (use `GET /v1/jobs/{id}` to fetch file data for individual completed jobs).

**Request**

```
POST /v1/jobs/status
Content-Type: application/json

{"job_ids": ["abc123", "def456", "unknown789"]}
```

**Response (200)**

```json
{
  "jobs": [
    {
      "job_id": "abc123",
      "status": "completed",
      "model": "moondream",
      "type": "caption",
      "created_at": 1774045000.0,
      "started_at": 1774045001.0,
      "finished_at": 1774045003.0,
      "result": {"caption": "A dog sitting on grass", "format": "json"}
    },
    {
      "job_id": "def456",
      "status": "running",
      "model": "flux-schnell",
      "type": "image-generate",
      "created_at": 1774045100.0,
      "started_at": 1774045110.0
    },
    null
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `jobs` | array | One entry per requested ID, in request order. `null` if ID not found. |

Each non-null entry has the same fields as `GET /v1/jobs/{id}` except `result.data` is omitted (no inline file bytes).

---

### PATCH /v1/models/{model_id} -- Update Model Configuration

Change `max_instances` at runtime. Persists to `local/config.json`.

**Request**

```
PATCH /v1/models/moondream
Content-Type: application/json

{"max_instances": 8}
```

**Response (200)**

```json
{
  "model_id": "moondream",
  "max_instances": 8,
  "previous_max_instances": 4,
  "added": 4,
  "removed": 0,
  "condemned": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | The model that was updated |
| `max_instances` | int | New instance count |
| `previous_max_instances` | int | Previous instance count |
| `added` | int | New instances created (unloaded, loaded on demand) |
| `removed` | int | Instances removed immediately (were idle or unloaded) |
| `condemned` | int | Instances with running jobs that will auto-remove when done |

**Scaling behavior:**

- **Scale up**: New instances start stopped and load on demand when the scheduler needs them.
- **Scale down**: Idle instances are evicted and removed immediately. Instances with active jobs are "condemned" — they finish their work, then auto-evict. No jobs are killed.
- **`max_instances: 0`**: Disables all dispatch for the model. Queued jobs wait until you scale back up. Useful for temporarily prioritizing other models.

---

### DELETE /v1/models/{model_id}/queue -- Clear Queued Jobs

Cancel all queued (not running) jobs for a model.

**Request**

```
DELETE /v1/models/moondream/queue
```

**Response (200)**

```json
{
  "model_id": "moondream",
  "cancelled": 15
}
```

| Field | Type | Description |
|-------|------|-------------|
| `cancelled` | int | Number of queued jobs cancelled |

Running jobs are not affected. Combine with `max_instances: 0` to fully drain a model.

---

### DELETE /v1/models/{model_id}/running -- Kill Running Jobs

Cancel all queued AND running jobs for a model. Sends SIGUSR1 to all active instances to abort in-progress inference.

**Request**

```
DELETE /v1/models/moondream/running
```

**Response (200)**

```json
{
  "model_id": "moondream",
  "cancelled_queued": 5,
  "cancelled_running": 3
}
```

| Field | Type | Description |
|-------|------|-------------|
| `cancelled_queued` | int | Queued/scheduled jobs cancelled in the store |
| `cancelled_running` | int | Running jobs signalled to abort |

Running jobs receive a cancel signal and will transition to `cancelled` state shortly after. This is a hard stop — use `DELETE /v1/models/{model_id}/queue` if you only want to clear the queue without killing running work.

---

### POST /v1/reserve -- Reserve VRAM Budget

Reserve VRAM budget space to prevent the scheduler from using it for model loads. Useful for adapter testing or manual GPU work.

**Request**

```
POST /v1/reserve
Content-Type: application/json

{"memory_gb": 64, "label": "adapter testing"}
```

**Response (201)**

```json
{
  "reservation_id": "abc123",
  "memory_gb": 64.0,
  "label": "adapter testing"
}
```

### GET /v1/reserve -- List Active Reservations

Returns all active VRAM reservations.

### DELETE /v1/reserve/{id} -- Release a Reservation

Releases a previously created VRAM reservation.

---

## 3. Job Types Reference

### 3.1 image-generate

Generate an image from a text prompt using Flux Schnell.

**Model**: `flux-schnell` (32 GB VRAM)

**Parameters**

| Parameter      | Type   | Required | Default | Description                                           |
|----------------|--------|----------|---------|-------------------------------------------------------|
| `prompt`       | string | Yes      | --      | Text description of the desired image                 |
| `width`        | int    | No       | 1024    | Output width in pixels                                |
| `height`       | int    | No       | 1024    | Output height in pixels                               |
| `aspect_ratio` | string | No       | null    | Aspect ratio override (e.g. `"16:9"`, `"3:2"`)       |
| `steps`        | int    | No       | 4       | Number of diffusion steps (more = higher quality)     |
| `seed`         | int    | No       | 42      | Random seed for reproducibility                       |
| `transparent`  | bool   | No       | false   | Generate with transparent background                  |

**Result Object**

```json
{
  "format": "png",
  "width": 1024,
  "height": 1024,
  "data": "<base64-encoded PNG>"
}
```

**Timing Estimates**

| Metric         | Value     |
|----------------|-----------|
| VRAM           | 32 GB     |
| Model load     | ~248 s    |
| Inference      | ~12 s     |
| Cold start     | ~260 s    |

**curl Example**

```bash
# Submit job
curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "type": "image-generate",
    "params": {
      "prompt": "A golden retriever playing in autumn leaves, oil painting style",
      "width": 1024,
      "height": 1024,
      "steps": 4,
      "seed": 123
    }
  }'

# Response:
# {"job_id":"abc123","status":"queued","model":"flux-schnell","estimated_seconds":12.0}

# Poll for result
curl -s http://localhost:8400/v1/jobs/abc123

# When completed, decode the image:
curl -s http://localhost:8400/v1/jobs/abc123 | jq -r '.result.data' | base64 -d > output.png
```

---

### 3.2 image-edit

Transform an existing image guided by a text prompt (image-to-image).

**Model**: `flux-schnell` (32 GB VRAM)

**Parameters**

| Parameter     | Type   | Required | Default | Description                                           |
|---------------|--------|----------|---------|-------------------------------------------------------|
| `prompt`      | string | Yes      | --      | Text description of the desired transformation        |
| `image`       | string | Yes      | --      | Base64-encoded input image                            |
| `width`       | int    | No       | 1024    | Output width in pixels                                |
| `height`      | int    | No       | 1024    | Output height in pixels                               |
| `steps`       | int    | No       | 4       | Number of diffusion steps                             |
| `seed`        | int    | No       | 42      | Random seed for reproducibility                       |
| `strength`    | float  | No       | 0.75    | How much to transform (0.0 = no change, 1.0 = full)  |
| `transparent` | bool   | No       | false   | Generate with transparent background                  |

**Result Object**

```json
{
  "format": "png",
  "width": 1024,
  "height": 1024,
  "data": "<base64-encoded PNG>"
}
```

**Timing Estimates**

| Metric         | Value     |
|----------------|-----------|
| VRAM           | 32 GB     |
| Model load     | ~248 s    |
| Inference      | ~12 s     |
| Cold start     | ~260 s    |

**curl Example**

```bash
# Encode your input image
IMAGE_B64=$(base64 -w0 input.png)

# Submit job
curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"image-edit\",
    \"params\": {
      \"prompt\": \"Make it look like a watercolor painting\",
      \"image\": \"$IMAGE_B64\",
      \"strength\": 0.6,
      \"steps\": 4,
      \"seed\": 99
    }
  }"

# Poll and decode
curl -s http://localhost:8400/v1/jobs/{job_id} | jq -r '.result.data' | base64 -d > edited.png
```

---

### 3.3 background-remove

Remove the background from an image, producing a transparent PNG.

**Model**: `birefnet` (1 GB VRAM)

**Parameters**

| Parameter | Type   | Required | Default | Description                    |
|-----------|--------|----------|---------|--------------------------------|
| `image`   | string | Yes      | --      | Base64-encoded input image     |

**Result Object**

```json
{
  "format": "png",
  "data": "<base64-encoded PNG with alpha channel>"
}
```

**Timing Estimates**

| Metric         | Value    |
|----------------|----------|
| VRAM           | 1 GB     |
| Model load     | ~5.4 s   |
| Inference      | ~1 s     |
| Cold start     | ~6.4 s   |

**curl Example**

```bash
IMAGE_B64=$(base64 -w0 photo.jpg)

curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"background-remove\",
    \"params\": {
      \"image\": \"$IMAGE_B64\"
    }
  }"

# Poll and decode
curl -s http://localhost:8400/v1/jobs/{job_id} | jq -r '.result.data' | base64 -d > no-bg.png
```

---

### 3.4 caption

Generate a text caption describing an image.

**Model**: `moondream` (18 GB VRAM)

**Parameters**

| Parameter | Type   | Required | Default    | Description                                      |
|-----------|--------|----------|------------|--------------------------------------------------|
| `image`   | string | Yes      | --         | Base64-encoded input image                       |
| `length`  | string | No       | `"normal"` | Caption length: `"short"`, `"normal"`, or `"long"` |

**Result Object**

```json
{
  "text": "A golden retriever sitting on a wooden porch, looking at the camera with its tongue out. The background shows a garden with blooming flowers."
}
```

**Timing Estimates**

| Metric         | Value     |
|----------------|-----------|
| VRAM           | 18 GB     |
| Model load     | ~142 s    |
| Inference      | ~103 s    |
| Cold start     | ~245 s    |

**curl Example**

```bash
IMAGE_B64=$(base64 -w0 photo.jpg)

curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"caption\",
    \"params\": {
      \"image\": \"$IMAGE_B64\",
      \"length\": \"long\"
    }
  }"

# Poll for result
curl -s http://localhost:8400/v1/jobs/{job_id} | jq -r '.result.text'
```

---

### 3.5 query

Ask a question about an image and get a text answer (visual question answering).

**Model**: `moondream` (18 GB VRAM)

**Parameters**

| Parameter  | Type   | Required | Default | Description                    |
|------------|--------|----------|---------|--------------------------------|
| `image`    | string | Yes      | --      | Base64-encoded input image     |
| `question` | string | Yes      | --      | Question about the image       |

**Result Object**

```json
{
  "text": "There are three people in the image."
}
```

**Timing Estimates**

| Metric         | Value     |
|----------------|-----------|
| VRAM           | 18 GB     |
| Model load     | ~142 s    |
| Inference      | ~103 s    |
| Cold start     | ~245 s    |

**curl Example**

```bash
IMAGE_B64=$(base64 -w0 photo.jpg)

curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"query\",
    \"params\": {
      \"image\": \"$IMAGE_B64\",
      \"question\": \"How many people are in this image?\"
    }
  }"

# Poll for result
curl -s http://localhost:8400/v1/jobs/{job_id} | jq -r '.result.text'
```

---

### 3.6 detect

Detect and locate objects in an image by label.

**Model**: `moondream` (18 GB VRAM)

**Parameters**

| Parameter | Type   | Required | Default | Description                           |
|-----------|--------|----------|---------|---------------------------------------|
| `image`   | string | Yes      | --      | Base64-encoded input image            |
| `object`  | string | Yes      | --      | Object label to detect (e.g. `"car"`) |

**Result Object**

```json
{
  "objects": [
    {"label": "car", "x_min": 0.12, "y_min": 0.34, "x_max": 0.56, "y_max": 0.78},
    {"label": "car", "x_min": 0.60, "y_min": 0.40, "x_max": 0.85, "y_max": 0.72}
  ]
}
```

Bounding box coordinates are normalized to [0, 1] relative to image dimensions.

**Timing Estimates**

| Metric         | Value     |
|----------------|-----------|
| VRAM           | 18 GB     |
| Model load     | ~142 s    |
| Inference      | ~103 s    |
| Cold start     | ~245 s    |

**curl Example**

```bash
IMAGE_B64=$(base64 -w0 street.jpg)

curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"detect\",
    \"params\": {
      \"image\": \"$IMAGE_B64\",
      \"object\": \"car\"
    }
  }"

# Poll for result
curl -s http://localhost:8400/v1/jobs/{job_id} | jq '.result.objects'
```

---

### 3.7 transcribe

Transcribe speech audio to text.

**Model**: `whisper-large` (6 GB VRAM)

**Parameters**

| Parameter  | Type   | Required | Default | Description                                                |
|------------|--------|----------|---------|------------------------------------------------------------|
| `audio`    | string | Yes      | --      | Base64-encoded audio file (WAV, MP3, FLAC, etc.)           |
| `language` | string | No       | `"en"`  | Language code (e.g. `"en"`, `"es"`, `"fr"`, `"de"`, `"ja"`) |

**Result Object**

```json
{
  "text": "Hello, this is a test of the transcription system.",
  "language": "en"
}
```

**Timing Estimates**

| Metric         | Value     |
|----------------|-----------|
| VRAM           | 6 GB      |
| Model load     | ~11.3 s   |
| Inference      | ~1.8 s    |
| Cold start     | ~13.1 s   |

**curl Example**

```bash
AUDIO_B64=$(base64 -w0 recording.wav)

curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"transcribe\",
    \"params\": {
      \"audio\": \"$AUDIO_B64\",
      \"language\": \"en\"
    }
  }"

# Poll for result
curl -s http://localhost:8400/v1/jobs/{job_id} | jq -r '.result.text'
```

---

### 3.8 tts-custom

Generate speech audio from text using built-in voices.

**Model**: `tts-custom` (4 GB VRAM)

**Parameters**

| Parameter     | Type   | Required | Default     | Description                              |
|---------------|--------|----------|-------------|------------------------------------------|
| `text`        | string | Yes      | --          | Text to speak                            |
| `speaker`     | string | No       | `"Aiden"`   | Built-in voice name                      |
| `language`    | string | No       | `"English"` | Output language                          |
| `temperature` | float  | No       | 0.9         | Sampling temperature (lower = more stable) |

**Result Object**

```json
{
  "format": "wav",
  "duration_seconds": 4.2,
  "data": "<base64-encoded WAV>"
}
```

**Timing Estimates**

| Metric         | Value     |
|----------------|-----------|
| VRAM           | 4 GB      |
| Model load     | ~43 s     |
| Inference      | ~4 s      |
| Cold start     | ~47 s     |

**curl Example**

```bash
curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "type": "tts-custom",
    "params": {
      "text": "Hello! Welcome to the Arbiter text-to-speech system.",
      "speaker": "Aiden",
      "language": "English",
      "temperature": 0.8
    }
  }'

# Poll and decode
curl -s http://localhost:8400/v1/jobs/{job_id} | jq -r '.result.data' | base64 -d > speech.wav
```

---

### 3.9 tts-clone

Generate speech using a cloned voice from a reference audio sample.

**Model**: `tts-clone` (4 GB VRAM)

**Parameters**

| Parameter     | Type   | Required | Default     | Description                                                  |
|---------------|--------|----------|-------------|--------------------------------------------------------------|
| `text`        | string | Yes      | --          | Text to speak                                                |
| `ref_audio`   | string | Yes*     | --          | Base64-encoded reference audio of the voice to clone         |
| `ref_audio_file` | string | Yes* | --          | Alternative: file path or `ref:` reference (e.g., `"ref:abc123.wav"`) |
| `ref_text`    | string | No       | null        | Transcript of the reference audio (improves cloning quality) |
| `language`    | string | No       | `"English"` | Output language                                              |
| `temperature` | float  | No       | 0.9         | Sampling temperature                                         |

**Result Object**

```json
{
  "format": "wav",
  "duration_seconds": 5.1,
  "data": "<base64-encoded WAV>"
}
```

**Timing Estimates**

| Metric         | Value     |
|----------------|-----------|
| VRAM           | 4 GB      |
| Model load     | ~43 s     |
| Inference      | ~4 s      |
| Cold start     | ~47 s     |

*Provide either `ref_audio` (base64) or `ref_audio_file` (path or ref), not both.

**curl Examples**

```bash
# Option 1: Base64 inline
REF_AUDIO_B64=$(base64 -w0 reference_voice.wav)

curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"tts-clone\",
    \"params\": {
      \"text\": \"This sentence will be spoken in the cloned voice.\",
      \"ref_audio\": \"$REF_AUDIO_B64\",
      \"ref_text\": \"This is the transcript of the reference audio clip.\",
      \"language\": \"English\",
      \"temperature\": 0.9
    }
  }"

# Option 2: Reference file (upload once, reuse forever)
curl -s -X POST http://localhost:8400/v1/refs -F file=@reference_voice.wav
# => {"ref_id": "a1b2c3d4e5f6.wav", ...}

curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "type": "tts-clone",
    "params": {
      "text": "This sentence will be spoken in the cloned voice.",
      "ref_audio_file": "ref:a1b2c3d4e5f6.wav",
      "ref_text": "This is the transcript of the reference audio clip.",
      "language": "English",
      "temperature": 0.9
    }
  }'

# Poll and decode
curl -s http://localhost:8400/v1/jobs/{job_id} | jq -r '.result.data' | base64 -d > cloned_speech.wav
```

---

### 3.10 tts-design

Generate speech using a voice synthesized from a text description.

**Model**: `tts-design` (4 GB VRAM)

**Parameters**

| Parameter           | Type   | Required | Default                      | Description                          |
|---------------------|--------|----------|------------------------------|--------------------------------------|
| `text`              | string | Yes      | --                           | Text to speak                        |
| `voice_description` | string | No       | `"A clear neutral voice."`   | Natural language description of the desired voice characteristics |
| `language`          | string | No       | `"English"`                  | Output language                      |
| `temperature`       | float  | No       | 0.9                          | Sampling temperature                 |

**Result Object**

```json
{
  "format": "wav",
  "duration_seconds": 3.8,
  "data": "<base64-encoded WAV>"
}
```

**Timing Estimates**

| Metric         | Value     |
|----------------|-----------|
| VRAM           | 4 GB      |
| Model load     | ~43 s     |
| Inference      | ~5 s      |
| Cold start     | ~48 s     |

**curl Example**

```bash
curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "type": "tts-design",
    "params": {
      "text": "Good morning! Here is your daily news briefing.",
      "voice_description": "A warm, deep male voice with a British accent. Calm and authoritative.",
      "language": "English",
      "temperature": 0.85
    }
  }'

# Poll and decode
curl -s http://localhost:8400/v1/jobs/{job_id} | jq -r '.result.data' | base64 -d > designed_voice.wav
```

---

### 3.11 talking-head

Generate a lip-synced talking head video from a portrait image and audio clip.

**Model**: `sonic` (5 GB VRAM)

**Parameters**

| Parameter       | Type   | Required | Default | Description                                               |
|-----------------|--------|----------|---------|-----------------------------------------------------------|
| `image`         | string | Yes      | --      | Base64-encoded portrait image (face clearly visible)      |
| `audio`         | string | Yes      | --      | Base64-encoded audio clip to lip-sync                     |
| `dynamic_scale` | float  | No       | 1.0     | Controls amount of head/face movement (higher = more)     |
| `seed`          | int    | No       | null    | Random seed for reproducibility                           |

**Result Object**

```json
{
  "format": "mp4",
  "duration_seconds": 8.5,
  "data": "<base64-encoded MP4 video>"
}
```

**Timing Estimates**

| Metric         | Value     |
|----------------|-----------|
| VRAM           | 5 GB      |
| Model load     | ~11 s     |
| Inference      | ~45 s     |
| Cold start     | ~56 s     |

**curl Example**

```bash
IMAGE_B64=$(base64 -w0 portrait.jpg)
AUDIO_B64=$(base64 -w0 speech.wav)

curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"talking-head\",
    \"params\": {
      \"image\": \"$IMAGE_B64\",
      \"audio\": \"$AUDIO_B64\",
      \"dynamic_scale\": 1.0
    }
  }"

# Poll and decode
curl -s http://localhost:8400/v1/jobs/{job_id} | jq -r '.result.data' | base64 -d > talking_head.mp4
```

---

### 3.12 video-generate

Generate a video from reference images and audio (text/audio-to-video).

**Model**: `ltx2` (55 GB VRAM)

**Parameters**

| Parameter    | Type     | Required | Default   | Description                                                |
|--------------|----------|----------|-----------|------------------------------------------------------------|
| `images`     | string[] | Yes      | --        | List of base64-encoded reference images                    |
| `audio`      | string   | Yes      | --        | Base64-encoded audio track                                 |
| `transcript` | string   | No       | null      | Text transcript to guide video content                     |
| `resolution` | string   | No       | `"large"` | Output resolution: `"small"`, `"medium"`, or `"large"`     |
| `fps`        | int      | No       | 24        | Frames per second                                          |
| `seed`       | int      | No       | 42        | Random seed for reproducibility                            |

**Result Object**

```json
{
  "format": "mp4",
  "duration_seconds": 12.0,
  "fps": 24,
  "data": "<base64-encoded MP4 video>"
}
```

**Timing Estimates**

| Metric         | Value      |
|----------------|------------|
| VRAM           | 55 GB      |
| Model load     | ~30 s      |
| Inference      | ~120 s     |
| Cold start     | ~150 s     |

**curl Example**

```bash
IMG1_B64=$(base64 -w0 frame1.jpg)
IMG2_B64=$(base64 -w0 frame2.jpg)
AUDIO_B64=$(base64 -w0 soundtrack.wav)

curl -s -X POST http://localhost:8400/v1/jobs \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"video-generate\",
    \"params\": {
      \"images\": [\"$IMG1_B64\", \"$IMG2_B64\"],
      \"audio\": \"$AUDIO_B64\",
      \"transcript\": \"A serene forest scene transitions to a mountain lake at sunrise.\",
      \"resolution\": \"large\",
      \"fps\": 24,
      \"seed\": 42
    }
  }"

# Poll and decode
curl -s http://localhost:8400/v1/jobs/{job_id} | jq -r '.result.data' | base64 -d > generated_video.mp4
```

---

## 4. Client Workflow

### Typical Polling Pattern

All Arbiter interactions follow the same three-step pattern: submit, poll, retrieve.

#### Python Example

```python
import base64
import time
import requests

ARBITER = "http://localhost:8400"

def run_job(job_type: str, params: dict, poll_interval: float = 1.0) -> dict:
    """Submit a job and poll until it completes. Returns the result dict."""

    # Step 1: Submit
    resp = requests.post(f"{ARBITER}/v1/jobs", json={
        "type": job_type,
        "params": params,
    })
    resp.raise_for_status()
    job = resp.json()
    job_id = job["job_id"]
    print(f"Submitted {job_type} job: {job_id} (est. {job['estimated_seconds']}s)")

    # Step 2: Poll
    while True:
        resp = requests.get(f"{ARBITER}/v1/jobs/{job_id}")
        resp.raise_for_status()
        status = resp.json()

        if status["status"] == "completed":
            print(f"Job completed in {status['finished_at'] - status['created_at']:.1f}s")
            return status["result"]

        if status["status"] == "failed":
            raise RuntimeError(f"Job failed: {status['error']}")

        if status["status"] == "cancelled":
            raise RuntimeError("Job was cancelled")

        time.sleep(poll_interval)


# --- Usage Examples ---

# Generate an image
result = run_job("image-generate", {
    "prompt": "A cat in a spacesuit floating through a nebula",
    "width": 1024,
    "height": 1024,
})
with open("output.png", "wb") as f:
    f.write(base64.b64decode(result["data"]))


# Transcribe audio
with open("recording.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

result = run_job("transcribe", {"audio": audio_b64, "language": "en"})
print(result["text"])


# Caption an image
with open("photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

result = run_job("caption", {"image": image_b64, "length": "long"})
print(result["text"])
```

#### curl Polling Script

```bash
#!/usr/bin/env bash
# Usage: ./poll_job.sh <job_id>

JOB_ID="$1"
BASE="http://localhost:8400"

while true; do
  RESPONSE=$(curl -s "$BASE/v1/jobs/$JOB_ID")
  STATUS=$(echo "$RESPONSE" | jq -r '.status')

  case "$STATUS" in
    completed)
      echo "Job completed!"
      echo "$RESPONSE" | jq '.result'
      exit 0
      ;;
    failed)
      echo "Job failed: $(echo "$RESPONSE" | jq -r '.error')"
      exit 1
      ;;
    cancelled)
      echo "Job was cancelled."
      exit 1
      ;;
    *)
      echo "Status: $STATUS ..."
      sleep 1
      ;;
  esac
done
```

### Using Reference Files

For files reused across many jobs (e.g., a voice sample for TTS cloning), upload once and reference by ID:

```python
# Upload once
resp = requests.post(f"{ARBITER}/v1/refs", files={"file": open("voice.wav", "rb")})
ref_id = resp.json()["ref_id"]  # e.g., "a1b2c3d4e5f6.wav"

# Reuse in any number of jobs — no re-upload, no base64 encoding
for text in texts:
    run_job("tts-clone", {
        "text": text,
        "ref_audio_file": f"ref:{ref_id}",
        "ref_text": "Reference transcript",
    })

# Clean up when done
requests.delete(f"{ARBITER}/v1/refs/{ref_id}")
```

Every job type that accepts binary data supports `_file` params. For example:
- `image_file` for image-edit, background-remove, caption, query, detect, talking-head
- `audio_file` for transcribe, talking-head
- `ref_audio_file` for tts-clone

All `_file` params accept either an absolute filesystem path or a `ref:` reference.

### Decoding Base64 Results

All binary results (images, audio, video) are returned as base64 strings in the `result.data` field.

**Python**

```python
import base64

raw_bytes = base64.b64decode(result["data"])
with open(f"output.{result['format']}", "wb") as f:
    f.write(raw_bytes)
```

**Bash**

```bash
curl -s http://localhost:8400/v1/jobs/{job_id} \
  | jq -r '.result.data' \
  | base64 -d > output.png
```

**JavaScript/Node.js**

```javascript
const result = await fetch(`http://localhost:8400/v1/jobs/${jobId}`).then(r => r.json());
const buffer = Buffer.from(result.result.data, 'base64');
require('fs').writeFileSync(`output.${result.result.format}`, buffer);
```

### Error Handling Best Practices

1. **Always check HTTP status codes.** A 202 on submit means accepted, not completed. A 4xx means your request was rejected before queueing.

2. **Handle all terminal states.** When polling, check for `completed`, `failed`, and `cancelled`. Do not assume a job will always succeed.

3. **Use estimated_seconds for adaptive polling.** The submit response includes an estimate. Use it to set your initial delay before the first poll:

   ```python
   estimated = submit_response["estimated_seconds"] or 5.0
   time.sleep(estimated * 0.8)  # Wait for 80% of the estimate, then start polling
   ```

4. **Set a timeout.** Do not poll forever. Set a reasonable maximum wait time based on the job type:

   ```python
   MAX_WAIT = estimated * 3  # Give it 3x the estimate
   start = time.time()
   while time.time() - start < MAX_WAIT:
       # ... poll ...
   else:
       # Cancel the job and handle the timeout
       requests.delete(f"{ARBITER}/v1/jobs/{job_id}")
   ```

5. **Cancel jobs you no longer need.** If the user navigates away or the result is no longer needed, cancel the job to free up the queue:

   ```python
   requests.delete(f"{ARBITER}/v1/jobs/{job_id}")
   ```

---

## 5. Model Memory & Scheduling

### Model Reference Table

All values are from calibration on NVIDIA Grace Blackwell (128 GB VRAM, 100 GB budget).

| Model           | VRAM (GB) | Load Time | Inference Time | Max Concurrent | Keep-Alive | Used By                              |
|-----------------|-----------|-----------|----------------|----------------|------------|--------------------------------------|
| `flux-schnell`  | 32        | 248 s     | 12 s           | 1              | 300 s      | image-generate, image-edit           |
| `birefnet`      | 1         | 5.4 s     | 1 s            | 2              | 300 s      | background-remove                    |
| `moondream`     | 18        | 142 s     | 103 s          | 1              | 300 s      | caption, query, detect               |
| `whisper-large` | 6         | 11.3 s    | 1.8 s          | 1              | 120 s      | transcribe                           |
| `tts-custom`    | 4         | 43 s      | 4 s            | 1              | 300 s      | tts-custom                           |
| `tts-clone`     | 4         | 43 s      | 4 s            | 1              | 300 s      | tts-clone                            |
| `tts-design`    | 4         | 43 s      | 5 s            | 1              | 300 s      | tts-design                           |
| `sonic`         | 5         | 11 s      | 45 s           | 1              | 600 s      | talking-head                         |
| `ltx2`          | 55        | 30 s      | 120 s          | 1              | 600 s      | video-generate                       |

### SJF Scheduling

Arbiter uses Shortest Job First (SJF) scheduling to minimize average wait time. When a job is submitted, it receives a priority score:

```
priority = avg_inference_ms + (load_ms if model is not loaded else 0)
```

**Lower scores run first.** This means:

- A `background-remove` job (birefnet loaded, priority = 1,000) will run before an `image-generate` job (flux-schnell not loaded, priority = 260,000).
- Jobs whose model is already in VRAM get a significant priority boost because the `load_ms` penalty is zero.
- Priorities are re-scored whenever a model is loaded or unloaded, so the queue adapts dynamically.

### Memory Management

The memory manager enforces a VRAM budget (default 100 GB) using LRU eviction:

1. **Loading**: When a job is scheduled, its model is loaded into VRAM if not already present. If there is not enough free VRAM, idle models are evicted oldest-first until space is available.

2. **Keep-alive**: After a model finishes its last job, it stays loaded for `keep_alive_seconds` (configurable per model). This avoids reloading for bursty workloads.

3. **Eviction**: When the keep-alive timer expires and no jobs are queued for that model, it is unloaded from VRAM. Models with active jobs are never evicted.

4. **Budget enforcement**: The total VRAM of all loaded models must not exceed `vram_budget_gb`. If a model cannot fit even after evicting all idle models, the job waits until enough VRAM is freed.

### Tips for Client Developers

- **Batch similar jobs together.** If you need to generate 10 images, submit them all at once. They will all use `flux-schnell`, which only needs to be loaded once. The keep-alive timer resets after each inference.

- **Expect cold-start delays.** The first job for a model after it has been evicted will take significantly longer (load time + inference time). Check the table above for load times. Subsequent jobs for the same model will be much faster.

- **Small jobs get priority.** If you need a quick `background-remove` (1 second) and a slow `video-generate` (2+ minutes), the background removal will almost always run first regardless of submission order.

- **Use `GET /v1/ps` to understand system state.** Before submitting a large batch, check which models are loaded and how much VRAM is free. This helps you estimate wait times.

- **Largest models dominate VRAM.** `ltx2` (55 GB) and `flux-schnell` (32 GB) cannot be loaded simultaneously on a 100 GB budget if `moondream` (18 GB) is also loaded. Plan your workflow accordingly.

---

## 6. Rate Limits & Quotas

**There are no rate limits.** Arbiter is designed for internal use on a dedicated GPU server.

- Submit as many jobs as you want. The queue is unbounded.
- Jobs are persisted in SQLite and survive server restarts. If Arbiter crashes or is restarted, all queued jobs will resume automatically. Jobs that were in the `running` or `scheduled` state at crash time are re-queued.
- There are no per-client quotas or authentication. Every request is treated equally.
- The only practical limit is GPU throughput: jobs are processed one at a time per model (except `birefnet` which allows 2 concurrent). Submitting more jobs just makes the queue longer.
