![](banner.jpg)

# Arbiter

Arbiter is your personal GPU workstation manager for vision analysis, background removal, audio, speech, talking heads, and video. Still-image generation is actively disabled; use the Mac mini Codex image service for image creation or editing.

You submit a job (transcribe a recording, remove a background, clone a voice), and Arbiter handles the rest: loading the right model, running your request, and giving you back the result when it's ready.

---

## What can it do?

Arbiter supports these AI tasks out of the box:

| What you want | Job type |
|---|---|
| Remove the background from a photo | `background-remove` |
| Describe what's in an image | `caption` |
| Ask a question about an image | `query` |
| Find and locate objects in an image | `detect` |
| Transcribe speech from an audio file | `transcribe` |
| Turn text into speech (built-in voices) | `tts-custom` |
| Turn text into speech in someone's voice | `tts-clone` |
| Turn text into speech in a voice you describe | `tts-design` |
| Make a portrait photo talk in sync with audio | `talking-head` |
| Generate a video from images and audio | `video-generate` |

---

## Getting started

### 1. Install

```bash
cd arbiter
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp local/config.default.json local/config.json
```

The defaults work fine to begin with. You can tweak settings later.

### 3. Start the server

```bash
./run server
```

Arbiter starts up at `http://localhost:8400` and runs quietly in the background.

### 4. Check it's working

```bash
./run health
```

You should see something like `{"status": "ok", ...}`. You're ready to go!

---

## How to use it

Every request follows the same simple pattern:

1. **Submit** a job and get back a job ID
2. **Wait** a moment (or check right away)
3. **Pick up** your result when it's ready

Here's what that looks like in Python:

```python
import base64, time, requests

ARBITER = "http://localhost:8400"

def run_job(job_type, params):
    # Submit the job
    resp = requests.post(f"{ARBITER}/v1/jobs", json={"type": job_type, "params": params})
    job_id = resp.json()["job_id"]

    # Wait for the result
    while True:
        status = requests.get(f"{ARBITER}/v1/jobs/{job_id}").json()
        if status["status"] == "completed":
            return status["result"]
        if status["status"] in ("failed", "cancelled"):
            raise Exception(status.get("error", "job ended"))
        time.sleep(1)
```

---

## A guide to every feature

### Removing a background

```python
with open("photo.jpg", "rb") as f:
    image = base64.b64encode(f.read()).decode()

result = run_job("background-remove", {"image": image})

with open("photo-nobg.png", "wb") as f:
    f.write(base64.b64decode(result["data"]))
```

The result is a PNG with a transparent background — ready to drop into any design.

### Describing an image

```python
with open("photo.jpg", "rb") as f:
    image = base64.b64encode(f.read()).decode()

result = run_job("caption", {
    "image": image,
    "length": "long",  # "short", "normal", or "long"
})
print(result["text"])
```

### Asking a question about an image

```python
result = run_job("query", {
    "image": image,
    "question": "How many people are in this photo?",
})
print(result["text"])  # "There are three people in the image."
```

### Finding objects in an image

```python
result = run_job("detect", {
    "image": image,
    "object": "car",
})
for obj in result["objects"]:
    print(obj)  # {"label": "car", "x_min": 0.12, "y_min": 0.34, "x_max": 0.56, "y_max": 0.78}
```

Bounding box coordinates are given as fractions of the image size (0.0 to 1.0), so they work regardless of image dimensions.

### Transcribing audio

```python
with open("recording.wav", "rb") as f:
    audio = base64.b64encode(f.read()).decode()

result = run_job("transcribe", {
    "audio": audio,
    "language": "en",  # optional — supports many languages
})
print(result["text"])
```

Supports WAV, MP3, FLAC, and other common audio formats.

### Text-to-speech with built-in voices

```python
result = run_job("tts-custom", {
    "text": "Hello! Welcome to Arbiter.",
    "speaker": "Aiden",  # built-in voice name
    "language": "English",
})

with open("speech.wav", "wb") as f:
    f.write(base64.b64decode(result["data"]))
```

### Cloning a voice

Give Arbiter a short recording of any voice, and it will generate new speech in that voice.

```python
with open("my_voice_sample.wav", "rb") as f:
    ref_audio = base64.b64encode(f.read()).decode()

result = run_job("tts-clone", {
    "text": "Hello! This is spoken in my voice.",
    "ref_audio": ref_audio,
    "ref_text": "This is what was said in the reference recording.",
})
```

The `ref_text` field (transcript of your sample) is optional but improves quality.

### Designing a voice from a description

Don't have a voice sample? Just describe what you want.

```python
result = run_job("tts-design", {
    "text": "Good morning, and welcome to the show.",
    "voice_description": "A warm, deep male voice with a British accent. Calm and authoritative.",
})
```

### Making a portrait talk

Combine a photo and a speech clip to create a lip-synced talking head video.

```python
with open("portrait.jpg", "rb") as f:
    image = base64.b64encode(f.read()).decode()
with open("speech.wav", "rb") as f:
    audio = base64.b64encode(f.read()).decode()

result = run_job("talking-head", {
    "image": image,
    "audio": audio,
    "dynamic_scale": 1.0,  # higher = more head movement
})

with open("talking.mp4", "wb") as f:
    f.write(base64.b64decode(result["data"]))
```

The photo needs a clearly visible face.

### Generating video from images and audio

```python
with open("frame1.jpg", "rb") as f:
    img1 = base64.b64encode(f.read()).decode()
with open("soundtrack.wav", "rb") as f:
    audio = base64.b64encode(f.read()).decode()

result = run_job("video-generate", {
    "images": [img1],
    "audio": audio,
    "transcript": "A serene forest scene at sunrise.",
    "resolution": "large",  # "small", "medium", or "large"
})
```

---

## Uploading files once, using them many times

If you're using the same audio sample or image across many jobs — like a voice reference for batch text-to-speech — upload it once and reuse it as many times as you like. No need to send it again and again.

```python
# Upload once
resp = requests.post(f"{ARBITER}/v1/refs", files={"file": open("voice.wav", "rb")})
ref_id = resp.json()["ref_id"]  # e.g. "a1b2c3d4e5f6.wav"

# Use in as many jobs as you like
for text in ["Hello!", "How are you?", "Goodbye!"]:
    run_job("tts-clone", {
        "text": text,
        "ref_audio_file": f"ref:{ref_id}",
        "ref_text": "This is what was said in the reference clip.",
    })

# Clean up when done
requests.delete(f"{ARBITER}/v1/refs/{ref_id}")
```

Reference files work with any job that accepts images, audio, or other binary inputs.

---

## Checking what's happening

The command line gives you a quick view of everything:

```bash
./run ps      # See which models are loaded and how much memory they're using
./run jobs    # See the job queue
./run health  # Quick check that the server is alive
```

Or hit the API directly:

```bash
curl http://localhost:8400/v1/ps
```

```json
{
  "vram_budget_gb": 100.0,
  "vram_used_gb": 37.0,
  "models": [
    {"id": "birefnet", "state": "loaded", "active_jobs": 1, "queued_jobs": 3},
    {"id": "whisper-large", "state": "unloaded", "queued_jobs": 0}
  ],
  "queue": {"queued": 4, "running": 1, "completed": 57, "failed": 2}
}
```

---

## Polling many jobs at once

If you've submitted a batch of jobs, you can check them all in one go instead of making a separate request for each one:

```python
job_ids = ["abc123", "def456", "ghi789"]
resp = requests.post(f"{ARBITER}/v1/jobs/status", json={"job_ids": job_ids})
for status in resp.json()["jobs"]:
    if status and status["status"] == "completed":
        # Fetch the full result (with file data) for this one
        full = requests.get(f"{ARBITER}/v1/jobs/{status['job_id']}").json()
```

Up to 1,000 job IDs per request, returned in the same order you sent them.

---

## Submitting from the command line

You can also submit jobs directly without writing any code:

```bash
./run submit background-remove '{"image_file": "/mnt/arbiter-store/inbox/photo.png"}'
./run cancel <job-id>
```

---

## Tips and tricks

**Model process configuration is closed by default.** Runtime model APIs do not
accept arbitrary commands or environment variables. `worker_cmd` is restricted
to repository-owned, model-compatible workers, while `adapter_params` accepts
only the typed keys documented in [API.md](API.md). Built-in vision, audio,
talking-head, composite, and LTX2 adapters use job parameters for tuning.

**Submit batches together.** Group audio transcription, captions, and other work that uses the same model to reduce cold loads.

**Shorter jobs jump ahead in line.** Arbiter uses "shortest job first" scheduling. A quick background removal will almost always run before a long video generation — even if the video was submitted first. This keeps average wait times low for everyone.

**Use `estimated_seconds` to time your first check.** When you submit a job, the response includes an estimate of how long it'll take. Wait about 80% of that time before you start polling — there's no point checking every second when a job is expected to take two minutes.

```python
job = requests.post(f"{ARBITER}/v1/jobs", json={...}).json()
time.sleep(job["estimated_seconds"] * 0.8)
# Now start polling
```

**Cancel jobs you no longer need.** If the result is no longer needed, cancel the job so it doesn't hold up the queue for others:

```python
requests.delete(f"{ARBITER}/v1/jobs/{job_id}")
```

**The first job after a long pause takes longer.** Models are unloaded from memory when they haven't been used in a while. The first job will include startup time — anywhere from a few seconds to a few minutes depending on the model. Jobs after that will be much faster.

**Check what's loaded before a big batch.** Before submitting many jobs, use `/v1/ps` to see which models are already warmed up. If your model is listed as `loaded`, your first job will start immediately. If not, factor in a startup delay.

**Your queue survives restarts.** If you need to restart the server, your queued jobs will still be there when it comes back up. Nothing is lost.

**Set a timeout.** Don't poll forever. If a job takes much longer than expected, cancel it and try again:

```python
max_wait = (job["estimated_seconds"] or 60) * 3
start = time.time()
while time.time() - start < max_wait:
    status = requests.get(f"{ARBITER}/v1/jobs/{job_id}").json()
    if status["status"] == "completed":
        break
    time.sleep(1)
else:
    requests.delete(f"{ARBITER}/v1/jobs/{job_id}")
    raise TimeoutError("Job took too long")
```

---

## Troubleshooting

**My first request is slow.** That's normal — the model is starting up. Use `./run ps` to see what's loading. Once it's ready, requests will be much faster.

**A job failed.** Check the error message:

```python
status = requests.get(f"{ARBITER}/v1/jobs/{job_id}").json()
print(status["error"])
```

**The server won't start.** Make sure your virtual environment is active (`source .venv/bin/activate`) and you're in the right directory. Try `./run health` — if you get a connection error, the server isn't running yet.

**I want to see what's happening in detail.** Logs are saved daily to `output/logs/arbiter-YYYY-MM-DD.jsonl`. Each line is a structured record of what the server is doing — jobs submitted, models loading, inferences completing — which you can search or tail as needed.

---

## License

[CC BY-NC 4.0](https://darren-static.waft.dev/license) — free to use and modify, but no commercial use without permission.
