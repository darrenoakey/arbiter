# tts-kokoro adapter

Kokoro-82M TTS — a tiny, Apache-licensed model that is **hundreds of times
faster** than the Qwen3-TTS family. Used by book-reader to read whole books
quickly. It has **no voice cloning**: a voice is one of ~54 named "voice packs"
(256-dim style tensors), optionally blended with weights, plus a speed factor.

## Job type

```
{"type": "tts-kokoro", "params": {
    "text": "Hello world",
    "voice": "af_heart",                 # name, or blend "af_heart*0.6+am_michael*0.4"
    "speed": 1.0,                          # speech-rate multiplier
    "lang_code": ""                        # "" → derive from voice prefix (a=Am EN, b=Br EN, ...)
}}
```

Output: 24000 Hz mono `result.wav` (same rate as Qwen3-TTS, so downstream
ffmpeg concat is unchanged).

## Dedicated venv (spark-only, NOT synced by deploy)

The adapter runs in its own venv via `worker_cmd` in `local/config.json`
(`venvs/kokoro/bin/python`), isolating kokoro's deps from `venvs/qwentts`.
`venvs/` is machine state, not code — `deploy-to-spark.sh` only syncs
`src/arbiter/` + binaries, so this survives deploys but must be recreated by
hand on a fresh spark:

```bash
cd /home/darren/src/arbiter
sudo apt-get install -y espeak-ng                 # misaki g2p fallback
python3.12 -m venv venvs/kokoro
venvs/kokoro/bin/pip install --upgrade pip wheel
venvs/kokoro/bin/pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
venvs/kokoro/bin/pip install kokoro soundfile "misaki[en]"
# make the `arbiter` package importable inside this venv (same trick qwentts uses):
cp venvs/qwentts/lib/python3.12/site-packages/arbiter.pth \
   venvs/kokoro/lib/python3.12/site-packages/arbiter.pth
```

`local/config.json` entry:

```json
"tts-kokoro": {
  "auto_download": "hexgrad/Kokoro-82M",
  "memory_gb": 2, "max_concurrent": 1, "max_instances": 1,
  "keep_alive_seconds": 600, "load_ms": 8000, "avg_inference_ms": 600,
  "max_runtime_seconds": 300, "pressure_index": 0.1,
  "worker_cmd": ["/home/darren/src/arbiter/venvs/kokoro/bin/python", "-m", "arbiter.worker_main", "tts-kokoro"]
}
```

## Future: true voice cloning (kvoicewalk)

Kokoro voices are fixed. To turn an arbitrary target voice (e.g. a Qwen3-TTS
"voice-design" sample) into a real kokoro `.pt` style tensor, use
[RobViren/kvoicewalk](https://github.com/RobViren/kvoicewalk): a random-walk +
Resemblyzer/Whisper scoring optimiser that evolves a style tensor to match a
20-30s mono 24 kHz reference. Cost: ~30-90 min GPU per voice (one-time).

The intended hybrid model: pre-evolve a few hundred custom voices once, drop
the `.pt` files into the kokoro voice bank, and let the book-reader voice-mapper
choose from ~500 voices instead of 54 — without paying clone cost per book.
