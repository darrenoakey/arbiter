# Voice pipeline adapters — demucs, rvc-train, rvc-convert

Backend for host-voice conversion in the multi-character music-video system.
A Suno cover carries the right beat/structure but a generic voice; RVC v2
converts a vocal stem to a trained host identity. Flow: **demucs** isolates the
cover's vocals → **rvc-convert** imposes the host timbre → remix over the
demucs accompaniment.

## Job types

```
{"type": "demucs", "params": {
    "audio_b64" | "audio_file": "<music clip>",
    "return_b64": false                # inline vocals_b64 + accompaniment_b64 in result
}}
# result: {vocals: "vocals.wav", accompaniment: "accompaniment.wav", samplerate: 44100, ...}
# (htdemucs two-stem; files always written to the job output dir)

{"type": "rvc-train", "params": {
    "name": "leo-laporte",             # stable model id -> /home/darren/rvc-models/<name>/
    "dataset_b64" | "dataset_file": "<zip of wavs | dir of wavs | single wav>",
    "epochs": 300, "sample_rate": 40000, "batch_size": 4, "f0_method": "rmvpe"
}}
# result: {model_id, model_path, index_path, sample_rate, epochs, train_wavs, ...}

{"type": "rvc-convert", "params": {
    "model": "leo-laporte",            # trained model id, or an absolute .pth path
    "audio_b64" | "audio_file": "<vocal stem>",
    "transpose": 0, "index_rate": 0.5, "f0_method": "rmvpe",
    "return_b64": false
}}
# result: {format: "wav", file: "result.wav", audio: "result.wav", audio_b64?: "..."}
```

`*_file` params must live inside `ARBITER_INBOX_PATH` (stage via
`arbiter_client.stage_file`); otherwise use the `*_b64` inputs. Large stems
returned as base64 travel through the worker stdout pipe — its ceiling was
raised to 96MB (`proc.go`); `return_b64` caps per stem and files are always
written, so oversized outputs are fetched as files instead.

## Backend: Applio (RVC v2), fairseq-free

RVC training/inference uses **[IAHispano/Applio](https://github.com/IAHispano/Applio)**
via its headless `core.py` CLI. Applio loads ContentVec through HuggingFace
`transformers` and rmvpe as a plain torch checkpoint — **no fairseq**, so the
aarch64 / Python-3.12 fairseq build wall is avoided. demucs runs its own
low-level API in a subprocess (see gotchas).

Trained models persist at `/home/darren/rvc-models/<model_id>/` (`model.pth` +
`model.index` + `info.json`) — outside the repo, so they survive deploys.

## Dedicated venvs (spark-only, NOT synced by deploy)

`venvs/` and the Applio checkout are machine state; `deploy-to-spark.sh` only
syncs `src/arbiter/` + binaries. Recreate on a fresh spark:

```bash
cd /home/darren/src/arbiter

# --- demucs venv ---
python3.12 -m venv venvs/demucs
venvs/demucs/bin/pip install --upgrade pip wheel setuptools
venvs/demucs/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
venvs/demucs/bin/pip install numpy soundfile
venvs/demucs/bin/pip install --no-deps demucs==4.1.0          # NOTE: --no-deps, see sphn gotcha
venvs/demucs/bin/pip install dora-search einops julius lameenc openunmix pyyaml tqdm safetensors huggingface-hub
echo /home/darren/src/arbiter/src > venvs/demucs/lib/python3.12/site-packages/arbiter.pth

# --- rvc venv (Applio deps, but KEEP the cu130 torch, not Applio's pinned cu128) ---
python3.12 -m venv venvs/rvc
venvs/rvc/bin/pip install --upgrade pip wheel setuptools
venvs/rvc/bin/pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130
git clone --depth 1 https://github.com/IAHispano/Applio.git /home/darren/src/Applio
grep -viE '^(torch|torchvision|torchaudio)==' /home/darren/src/Applio/requirements.txt > /tmp/req.notorch.txt
venvs/rvc/bin/pip install -r /tmp/req.notorch.txt            # faiss-cpu==1.13.2 has an aarch64 wheel
echo /home/darren/src/arbiter/src > venvs/rvc/lib/python3.12/site-packages/arbiter.pth

# --- Applio assets (embedder + rmvpe + RVC-v2 pretrained G/D) ---
cd /home/darren/src/Applio
/home/darren/src/arbiter/venvs/rvc/bin/python core.py prerequisites --pretraineds_hifigan True --models True --exe True
cp assets/config_template.json assets/config.json           # REQUIRED, see extract_model gotcha
```

`local/config.json` entries (each pins its worker to the right venv):

```json
"demucs":      {"memory_gb": 8,  "max_concurrent": 1, "max_instances": 1, "keep_alive_seconds": 120, "max_runtime_seconds": 1800,  "avg_inference_ms": 15000,   "load_ms": 3000, "pressure_index": 0.5, "worker_cmd": ["/home/darren/src/arbiter/venvs/demucs/bin/python", "-m", "arbiter.worker_main", "demucs"]},
"rvc-train":   {"memory_gb": 20, "max_concurrent": 1, "max_instances": 1, "keep_alive_seconds": 60,  "max_runtime_seconds": 86400, "avg_inference_ms": 1800000, "load_ms": 3000, "pressure_index": 1,   "worker_cmd": ["/home/darren/src/arbiter/venvs/rvc/bin/python", "-m", "arbiter.worker_main", "rvc-train"]},
"rvc-convert": {"memory_gb": 8,  "max_concurrent": 1, "max_instances": 1, "keep_alive_seconds": 120, "max_runtime_seconds": 1800,  "avg_inference_ms": 30000,   "load_ms": 3000, "pressure_index": 0.5, "worker_cmd": ["/home/darren/src/arbiter/venvs/rvc/bin/python", "-m", "arbiter.worker_main", "rvc-convert"]}
```

## Gotchas (all bit during bring-up)

- **demucs 4.1.0 pulls `sphn`, a Rust dep with no aarch64 wheel; its
  `audiopus_sys` build fails.** Install demucs `--no-deps` + its other deps; the
  low-level modules (`demucs.apply`/`pretrained`/`audio`) don't need sphn. The
  `python -m demucs` CLI does (`demucs.api` imports sphn at module load), so the
  adapter runs a small subprocess using the low-level API instead.
- **demucs must run in a subprocess, not in the worker thread.** In-process
  `apply_model` inside the arbiter worker died silently; a fresh subprocess is
  robust (and matches the pilot's proven invocation).
- **`torchaudio.save` needs torchcodec (absent/awkward on aarch64) in torch
  2.11+.** Write wavs with `soundfile` instead.
- **Applio's `extract_model` reads `assets/config.json`; a headless clone lacks
  it**, so the deployable-weight extraction fails *silently* (caught + printed,
  non-fatal) and training yields only `G_/D_` checkpoints, no usable model. Copy
  `config_template.json` → `config.json` (the adapter also self-heals this in
  `load()`).
- **`save_every_epoch` must be ≤ `total_epoch`** or no weight is saved at the
  final epoch (this build doesn't force a completion save). The adapter clamps
  it.
- **Do NOT let Applio install torch** (it pins `2.7.1+cu128`, wrong for the GB10
  CUDA 13 / Blackwell sm_121). Install the cu130 torch first, then Applio's
  requirements minus the torch lines.

## Training defaults for a 2-10 min single-speaker dataset

40 kHz, ~200-300 epochs, batch 4, f0 rmvpe, ContentVec embedder, HiFi-GAN
vocoder, faiss index built (`index_rate 0` disables it at inference).
