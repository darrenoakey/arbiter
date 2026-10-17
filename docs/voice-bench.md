# voice-bench — MiniCPM-o-4.5 vs Moshi vs Kyutai cascade head-to-head

Arbiter adapter suite for the praxis task 171 benchmark (project 87 "AI ESP32
box"). Measures EOU→first-audio, interrupt latency, and talk-over quality for
three local full-duplex voice stacks on the GB10, with in-process wall-clock
timing around the model APIs. Results feed the task-162 evaluation document
(`~/src/ai-esp32-box/low-latency-voice-models-evaluation-2026-10-17.md`).

## Job types / models

| job type | model id | venv (spark, real dir) | weights |
|---|---|---|---|
| `voice-bench-minicpmo` | `voice-bench-minicpmo` | `venvs/voice-bench-minicpmo` | `openbmb/MiniCPM-o-4_5` (audio-only, sdpa, bf16) |
| `voice-bench-moshi` | `voice-bench-moshi` | `venvs/voice-bench` | `kyutai/moshiko-pytorch-bf16` |
| `voice-bench-kyutai` | `voice-bench-kyutai` | `venvs/voice-bench` | `kyutai/stt-1b-en_fr` + `Qwen/Qwen3-1.7B` + `kyutai/pocket-tts-without-voice-cloning` |

Params (one job = one system × scenario × run):
`scenario` (`eou`|`interrupt`|`talkover`), `run` int, `u1_file..u4_file`
(staged 24 kHz wavs).

## Spark venv recreation

`venvs/` is spark-only state (not in git). From scratch:

```bash
# shared venv for moshi + kyutai cascade
python3 -m venv ~/src/arbiter/venvs/voice-bench
cp ~/src/arbiter/venvs/qwentts/lib/python3.12/site-packages/arbiter.pth \
   ~/src/arbiter/venvs/voice-bench/lib/python3.12/site-packages/
VB=~/src/arbiter/venvs/voice-bench
$VB/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
$VB/bin/pip install "transformers>=4.53" accelerate sentencepiece soundfile hf_transfer
# moshi via --no-deps (avoids the sphn rust build; moshi.models never imports sphn)
$VB/bin/pip install --no-deps moshi
$VB/bin/pip install "numpy<2.3,>=1.26" "safetensors<0.8.0,>=0.4.0" "huggingface-hub<1.0.0,>=0.24" "einops<0.9,>=0.7" "sentencepiece<0.3,>=0.2.0" "tqdm>=4.48"
$VB/bin/pip install pocket-tts

# MiniCPM-o venv (pinned transformers per OpenBMB docs)
python3 -m venv ~/src/arbiter/venvs/voice-bench-minicpmo
cp ~/src/arbiter/venvs/qwentts/lib/python3.12/site-packages/arbiter.pth \
   ~/src/arbiter/venvs/voice-bench-minicpmo/lib/python3.12/site-packages/
VM=~/src/arbiter/venvs/voice-bench-minicpmo
$VM/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
$VM/bin/pip install "transformers==4.51.0" accelerate "minicpmo-utils[all]" hf_transfer
```

## Model weights

HF cache dirs on spark (root disk — `/mnt/t9` cache is full, do not use):

```bash
export HF_HOME=/home/darren/hf-voicebench HF_HUB_ENABLE_HF_TRANSFER=1
VB=~/src/arbiter/venvs/voice-bench
$VB/bin/hf download kyutai/moshiko-pytorch-bf16     # ~16 GB
$VB/bin/hf download kyutai/stt-1b-en_fr             # ~3 GB
$VB/bin/hf download kyutai/pocket-tts-without-voice-cloning   # kyutai/pocket-tts itself is LICENSE-GATED
$VB/bin/hf download Qwen/Qwen3-1.7B
$VB/bin/hf download openbmb/MiniCPM-o-4_5           # ~19 GB (any venv)
```

## Registering the models (after code deploy)

```bash
for m in voice-bench-moshi voice-bench-kyutai voice-bench-minicpmo; do :; done
curl -X POST http://10.0.0.254:8400/v1/models -H 'Content-Type: application/json' -d '{
  "model_id": "voice-bench-moshi", "memory_gb": 26, "max_instances": 1,
  "avg_inference_ms": 60000, "load_ms": 180000, "keep_alive_seconds": 1800,
  "worker_cmd": ["/home/darren/src/arbiter/venvs/voice-bench/bin/python", "-m", "arbiter.worker_main", "voice-bench-moshi"]}'
# kyutai: memory_gb 16, same worker_cmd with voice-bench-kyutai
# minicpmo: memory_gb 32, worker_cmd .../venvs/voice-bench-minicpmo/bin/python ... voice-bench-minicpmo
```

## Benchmark assets + run

Four synthetic utterances (macOS `say`, 24 kHz mono) live in
`~/src/ai-esp32-box/voice-bench-assets/`; stage via `arbiter_client.stage_file()`
and submit `voice-bench-*` jobs with `force: true`.
