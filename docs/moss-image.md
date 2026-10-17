# MOSS-Image-5B (MOSS-Image-SFT) comparison adapter

The owner asked on 2026-10-17 to test the newly released OpenMOSS
**MOSS-Image-SFT** (~5B single-stream DiT + Qwen3.5 hybrid text encoder,
Apache 2.0) head to head against the other image models (Qwen-Image-2.1
family, FLUX 3 Image Klein, Iris-3B, Ming-Image). This adapter exists only
for that comparison. It is **not** wired into `generate_image`,
`daz-agent-sdk` or the Mac mini Codex IGS route, and it is never a fallback
when IGS is unavailable.

| | |
|---|---|
| Job type | `moss-image` |
| Model id | `moss-image-sft` |
| HF repo / revision | `OpenMOSS-Team/MOSS-Image-SFT` @ `fc5de0d` |
| Weights | `/home/darren/nvme-models/moss-image/MOSS-Image-SFT` (spark NVMe) |
| Worker venv | `venvs/mossimage` (torch 2.14.0+cu130, diffusers 0.38.0, transformers 5.20.1) |
| Declared `memory_gb` | 36 (retuned from the measured 28.8 GB peak reserved on 2026-10-17; worker cap is `memory_gb × 1.15`) |

Policy: the model id is reachable only through the identically named job
type family (`moss-image` → `moss-image-sft`), pinned in Python
`image_policy.py` and Go `model_policy.go` (`moss_image_policy_test.go`,
`tests/unit/test_moss_image.py`). `moss-image` (the generic alias),
`moss-image-base` (the Base checkpoint), quantized community builds, LoRA
variants, and renamed copies (`OpenMOSS-Team/MOSS-Image-SFT` as a model id)
stay denied.

## Request

```json
{"type": "moss-image",
 "params": {"prompt": "a red fox in fresh snow, golden hour", "seed": 7}}
```

Card-exact defaults: 50 steps, CFG `guidance_scale` 3.0, negative-branch
skip-layer guidance `uncond_skip_layers=[22]` (`[]` explicitly disables;
the user guide warns detail collapses without it), negative prompt empty,
1024×1024. Sides must be multiples of 16 within 256-2048 with ≤ 4M total
pixels; steps 1-100; guidance 0-10. T2I only: `image*`/`references`,
`cfg_scale`/`true_cfg_scale`, `slg_*` and `num_images_per_prompt` all fail
closed, so a comparison row can never be silently run under a different
condition.

## Result

`result.png` plus a JSON result containing:

- `prompt`, `negative_prompt`, `seed`, `width`, `height`, `steps`,
  `guidance_scale`, `uncond_skip_layers`
- `precision` — transformer/text encoder bfloat16, VAE + initial noise
  float32
- `provenance` — HF repo + revision, weights dir, verified file count,
  torch/CUDA/device
- `timings` — `generate_seconds` (GPU-synchronized), `png_write_seconds`,
  `transformer_forwards` (CFG runs cond + uncond per step), `render_index`,
  load/verify seconds
- `memory` — resident allocated/reserved after load, peak allocated/reserved
  during the render (bytes and GB), declared GB

## Fail-closed load

`load()` refuses to run when:

- `ARBITER_MEMORY_GB` is missing (the worker is not under the Arbiter CUDA
  cap);
- the weights dir is not a real directory at the pinned path;
- any required path (`model_index.json`, `pipeline_moss_image.py`,
  `scheduler/`, `transformer/`, `text_encoder/`, `tokenizer/`, `vae/`) is
  missing;
- any file the pipeline reads is not from snapshot revision `fc5de0d`, or its
  content hash differs from the Hugging Face snapshot metadata — this
  includes the remote-code `pipeline_moss_image.py`, so the model-shipped
  code is exactly what the owner reviewed when asking for this adapter;
- the loaded transformer config is not dim 3072 / 42 layers, or the
  pipeline/scheduler classes are not the MOSS remote-code classes.

## Runtime notes

- The pipeline is the model's own `pipeline_moss_image.py` via diffusers
  `custom_pipeline` + `trust_remote_code`, loaded from the local snapshot
  (no hub access at runtime).
- The GB10 mmap→cuda workaround (`base._pipe_to_cuda_cloned`) is applied
  after `from_pretrained`.
- Cancellation is checked before every transformer forward via a pre-hook.
- The text encoder is a Qwen3.5 hybrid (linear + full attention) — first
  load on GB10 compiles/cache-warms triton paths; budget `load_ms` ~240 s.
- Recreation: `requirements/moss-image.txt` (venv + snapshot_download).

## Benchmark protocol

Matched-prompt comparison with the other image models uses the house set
(1024², seeds 42001-42003 — botanist, Sydney poster, LUMEN). Submit every
render as an Arbiter job with `force: true` and label outputs
`moss-image-sft`; see the skill's still-image policy section for the
comparison-sheet convention.

## Measured on GB10 (2026-10-17, first run, 1024², house prompts)

| prompt (seed) | moss-image-sft (50 steps, CFG 3.0, skip 22) | flux3-klein-bf16 (16 steps) | qwen-image-2.1-turbo (8 steps) |
|---|---|---|---|
| botanist (42001) | 242.7 s | 19.7 s | 28.8 s |
| Sydney poster (42002) | 329.0 s | 18.1 s | 26.6 s |
| LUMEN perfume (42003) | 341.8 s | 33.3 s | 20.7 s |

- Cold load 51.9 s total: 36.1 s hash verification of every snapshot file
  (including the remote-code pipeline module) + ~15.8 s pipeline load and
  GB10 mmap-clone→CUDA move. Declared `memory_gb` retuned 40 → 36 from the
  measured 28.8 GB peak reserved (worker cap 36 × 1.15 = 41.4 GB).
- Text rendering: `SYDNEY`, `HARBOUR NIGHTS` and `LUMEN` all spelled
  correctly, composition on-brief and in the same family as the
  flux3-klein renders; moss favours a more designed/posterised layout.
- Throughput context: moss runs CFG with two transformer forwards per step
  (cond + uncond-with-skip) at 50 steps — ~7.3× the DiT forwards of the
  distilled 16-step flux3-klein — so the 10-18× wall-time gap is
  architecture-expected, not a regression.

Jobs: moss `b2e86defcbe0` (botanist), `e79d87047fd4` (Sydney),
`e524777631f0` (LUMEN); flux3-klein-bf16 `7dcb359169dc`, `d94cedb5421a`,
`fb642fed9556`; qwen-image-turbo `31a4b1af3547`, `c46b3d8d4740`,
`b78fdc82880f`. Outputs under `/mnt/arbiter-store/output/jobs/<id>/`.
