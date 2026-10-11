# YuE2 music generation through Arbiter (`music-generate-yue2`)

Job type `music-generate-yue2` (model `yue2`) serves **m-a-p/YuE2-3B** — the
September 2026 YuE2 release (HKUST / M·A·P / Tokenwave.AI et al., arXiv
2609.33757). YuE2 is an AR–NAR Mixture-of-Transformers full-song generator:
the backbone writes an editable melody-and-chord **score** plus semantic
tokens autoregressively, flow matching produces acoustic latents, and a VAE
decodes 48 kHz stereo audio. Creation, covers, and score-edited generation
differ only in where the score comes from.

Its sibling on spark is `music-generate` (model `music-generate`), which
serves **ACE-Step v1.5 XL SFT** (`ACE-Step/acestep-v15-xl-sft-diffusers`,
StepFun × ACE Studio, Apache-2.0). Quick orientation:

| | YuE2 (`music-generate-yue2`) | ACE-Step v1.5 (`music-generate`) |
|---|---|---|
| Architecture | AR–NAR MoT planner + score, flow-matching acoustic decoder | LM planner + Diffusion Transformer (Sana DCAE tokenizer) |
| Output | 48 kHz stereo | 48 kHz stereo (measured on spark) |
| Planning | Editable melody/chord score (`cot=full`), melody only, or off; external ABC score accepted | No symbolic stage; direct tags+BPM/key/time-sig control |
| WildSongBench (Sept 2026, SongBench Avg) | 6.73 (best-of-8: 6.96 — ahead of Suno v5/v5.5/v6 and Mureka 9) | 6.01 |
| Weight license | CC BY-NC 4.0 + creator permission (free for personal/creator use incl. monetized outputs; companies need a license) | Apache 2.0 (unambiguous commercial use) |
| Speed claim (upstream) | ~71 s for a 3.6-min song on an RTX 4090 (full CoT) | <2 s/song on A100, <10 s on RTX 3090 |
| Declared VRAM (spark) | 24 GB (worker self-caps at 22 GiB CUDA) | 28 GB |

Rule of thumb: YuE2 scores higher on structure and gives you the editable
score artifacts; ACE-Step is dramatically faster and carries no licensing
questions. Measure both on spark before picking one for a workload.

## Input

```json
{"type": "music-generate-yue2", "params": {"style": "indie pop rock, female vocals, driving bass, 120 bpm", "lyrics": "[Verse]\n...lines...\n[Chorus]\n...lines...", "cot": "full", "seed": 42, "format": "mp3"}}
```

| Param | Default | Notes |
|---|---|---|
| `style` | — | **Required.** Free-text genre/mood/instrumentation description. |
| `lyrics` | — | **Required.** Section tags (`[Verse]`, `[Chorus]`, ...) structure the song. |
| `cot` | `full` | `full` = editable melody+chord plan (best structure); `melody` = melody plan, free accompaniment (covers); `off` = direct generation. |
| `abc` | — | External ABC score; requires `cot` `full` or `melody`. |
| `seed` | — | Integer for reproducible plans. |
| `cfg_scale` | 1.0 (1.01 for `cot=off`) | Semantic CFG. |
| `format` | `mp3` | `mp3` (320 kbps CBR) / `wav` / `flac` / `ogg`. |

Result: `{"format", "sample_rate", "duration", "channels", "style", "cot",
"truncated", "e2e_seconds", "file": "result.mp3"}` — resolve through the
shared output tree (`arbiter_client.local_result()`); YuE2's own artifacts
(score, semantic tokens, latents) stay inside the worker.

## Worker environment

The worker venv is `venvs/yue2` on spark, created from scratch with torch
`2.10.0+cu130` (the +cu130 build is mandatory — plain `torch` is CPU-only)
plus the `yue2-infer` package. `yue2-infer` is **not on PyPI**: production
runs 0.1.5 from a wheel kept at `~/yue2-pkg/` on spark; upstream's current
release is 0.1.6. Full recreation steps live in `requirements/yue2.txt` —
follow them verbatim; the venv MUST be a real directory (worker policy
rejects symlinked venv paths) and must carry the copied `arbiter.pth`.

Weights `m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae` resolve from the shared HF cache
(`/mnt/t9/hf-cache`, symlinked into `~/.cache/huggingface/hub`). The model
config sets no `auto_download` — hub fetches belong inside the audited
worker. The pipeline enforces a 22 GiB per-process CUDA cap
(`memory_budget_gib`), which is what keeps a runaway load from crashing the
GB10's unified-memory host (see the spark skill).

## Long operations

Cold load is minutes (the live config budgets 300 s); generation runs
~1.5 s per lyric character on top of a ~60 s pipeline floor (GB10
measurements — see `estimate_time`). Submit jobs asynchronously and sleep;
never hold a turn polling, and never load the model from a spark shell —
it runs only inside the arbiter worker.

## Spark measurements

<!-- Numbers here must come from real arbiter-job results (a forced job
with fixed seed), never from upstream marketing claims. Verified GB10
head-to-head, 2026-10-12, identical style/lyrics/seed 42001, both cold
workers: ACE-Step 1.5 (job 3deef6a16078) 81 s job wall for 120 s of audio;
YuE2-3B (job c3c3dd50de2c) 470 s job wall (e2e_seconds 467.8) for 119.4 s
of audio. Earlier YuE2 data point: 141.5 s e2e on 2026-09-17
(conv-33c7179a18042d39). -->
