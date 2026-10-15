# FLUX 3 Image Klein benchmark adapters

The owner asked for this benchmark on 2026-10-16: FLUX 3 Image Klein (distilled, 5.6B) at
BF16, FP8 and NVFP4, compared head to head and against Qwen-Image-2.1. These
adapters exist only for that benchmark. They are **not** wired into
`generate_image`, `daz-agent-sdk` or the Mac mini Codex IGS route, and they are
never a fallback when IGS is unavailable.

| Job type = model id | DiT | Declared `memory_gb` |
|---|---|---|
| `flux3-klein-bf16`  | repo root `model.safetensors` (bf16) | 36 |
| `flux3-klein-fp8`   | `variants/fp8r` (234 rowwise FP8 linears) | 32 |
| `flux3-klein-nvfp4` | `variants/nvfp4` (198 NVFP4 linears, FlashInfer) | 32 |

The upstream B200 reference measured about 31, 26 and 25 GB. Each declaration
leaves headroom for the first render. The worker CUDA cap is
`memory_gb × 1.15`, set by `worker_main._apply_cuda_memory_cap`. After the
first real renders, set each `memory_gb` from the measured
`memory.peak_reserved_gb`.

Policy: each id is reachable only through its identically named job type
(Python `image_policy.py` and Go `model_policy.go`). `flux3`, `flux3-klein`,
base, int8, LoRA and renamed copies stay denied. `FLUX3_WEIGHTS_ROOT` is an
adapter param allowlisted for these three ids only. It must be an absolute,
canonical, real directory.

## Request

```json
{"type": "flux3-klein-fp8",
 "params": {"prompt": "a red fox in fresh snow, golden hour", "seed": 7,
            "width": 1024, "height": 1024, "steps": 16}}
```

The defaults are 16 steps, 1024×1024 and seed 42. Sides must be multiples of 16 between 64 and 4096. Steps run from 1 to 64. Unknown keys are rejected:
`guidance`, `negative_prompt`, `pup`, images and boxes all fail. A benchmark
row therefore cannot silently run under a different condition. Text-to-image
uses the distilled flow map with no CFG and no prompt upsampler. The prompt is
used exactly as given.

## Result

`result.png` plus a JSON result containing:

- `prompt`, `seed`, `width`, `height`, `steps`, `guidance: null`,
  `prompt_upsampling: false`, `sampler_alpha`, `variant`, `model_id`
- `precision` — DiT dtype, quantization, quantized-linear counts actually
  present in the loaded DiT
- `provenance` — HF repo + revision, DiT path + sha256, source repo + commit,
  package location, torch/CUDA/device, FlashInfer version (nvfp4)
- `timings` — `generate_seconds` (GPU-synchronized), `png_write_seconds`,
  `dit_forwards`, `render_index` (1 = first render after load), load/verify
  seconds
- `memory` — resident allocated/reserved after load, peak allocated/reserved
  during the render (bytes and GB), declared GB

## Fail-closed load

`load()` refuses to run in any of these cases:

- `ARBITER_MEMORY_GB` is missing, meaning the worker is not under the Arbiter
  CUDA cap.
- The `flux3` distribution was not installed from git commit `45e57c9…`.
- Any file the pipeline reads (DiT manifest/config/weights, `ae.safetensors`,
  every `text_encoder/` file) is not from snapshot revision `8454017…`, or its
  content hash differs from the Hugging Face snapshot metadata.
- The DiT `model.safetensors` or `config.json` differs from the manifest
  sha256.
- The manifest or config quantization does not match the requested variant.
- The loaded DiT does not contain exactly the expected quantized-linear class
  and count.

The pipeline is `flux3.pipeline.Pipeline` with an explicit DiT dict,
text-encoder directory and `ae.safetensors` path. `flux3.generate.Models` is
not used, because it imports vLLM for the upsampler and loads the safety
classifier. No `WEIGHTS_DIR` is read.

Cancellation is checked before every DiT forward pass, which is one per step.

## Runtime

See `requirements/flux3.txt` for the exact venv recipe and pins
(`venvs/flux3`, torch 2.14.0+cu130, flashinfer 0.6.8.post1). Weights live at
`/home/darren/nvme-models/flux3/flux-3-image-klein`. Never run these models
over ssh: every render goes through `POST /v1/jobs`.
