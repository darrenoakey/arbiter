# P-H3-Image — offline adapter preparation (not deployed or benchmarked)

Owner request 2026-11-03: compare official `PrunaAI/P-H3-Image` with the prior
image tests.  This tree contains local-only adapter/policy/schema/test work.
**No weight or runtime download, installation, upstream execution, Arbiter
submission, production deployment, or greenline release has occurred.**

## Immutable source

- HF: `PrunaAI/P-H3-Image` @ `7f98acef503cf2bcfba821a78cb1cf1a9928b2df`.
- Official pipeline `PH3ImageModularPipeline`, declaring diffusers
  `0.42.0.dev0`; 33.1B transformer, 4B text encoder, prompt upsampler, VAE.
- License: MiniMax H3 Community License. Australia is not excluded; the EU,
  UK, Republic of Korea and USA are excluded absent separate authorization.
- Complete official per-directory expanded trees are in the task's
  `source-evidence/official-runtime-tree.json`; 15 safetensors files total
  99,797,308,280 bytes. All runtime metadata/license files total
  99,845,388,933 bytes. Without the prompt upsampler, weights are
  88,438,696,120 bytes (82.35 GiB), before metadata.
- Official smaller **Comfy** alternative, not interchangeable with this
  diffusers adapter: `PrunaAI/P-H3-Image-Comfy` @
  `44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543`. W4A8 transformer/text encoder,
  int8 VAE and bf16 prompt upsampler = **39,390,906,877 bytes** (36.69 GiB).
  Omitting the upsampler = 20,084,601,325 bytes (18.71 GiB). Vendor-reported
  8 MP peak is ~21 GB, not a local measurement.

## Runtime approval gate

The official quickstart asks for `git+https://github.com/huggingface/diffusers`,
`transformers`, `accelerate`, `torch`, and `pillow`. The checkpoint declares
0.42.0.dev0; installed Spark venvs contain 0.40.0.dev0 / 0.41.0.dev0 and are
not presumed compatible. **An exact official GitHub commit supporting the
checkpoint has not yet been selected, reviewed, or approved.** Do not install
a floating master branch or trust model-supplied code.

The adapter has no `trust_remote_code`, pins all official Git-blob/LFS content
IDs, and refuses symlinks, changed weights, changed pipeline metadata or a
changed 8-step sampling grid. Nevertheless its load/render API calls are
**unverified** until the exact upstream diffusers implementation is reviewed.
Prompt upsampling is currently rejected; the default is prompt-as-written.
Only bf16 diffusers weights are supported, not Comfy quantized files.

## Proposed local configuration

- Job type `p-h3-image`; model ID `p-h3-image-bf16` only.
- Real venv `venvs/ph3image`; pinned official source and hash manifest needed.
- `max_instances=1`, `max_concurrent=1`, Spark placement, allocator segments,
  proposed declared memory 84 GB (subject to actual 8 MP peak/offload proof).
- Proposed snapshot path `/mnt/t9/models/p-h3-image/P-H3-Image`. **It does not
  fit now.** Root has ~3.0 GB and `/mnt/t9` ~10.1 GB available. At least
  99.85 GB metadata/weight payload plus temporary-download and OS headroom is
  required. `/mnt/arbiter-store` is a directory on the same full root disk,
  not a mounted spare share. No deletion, cleanup or new mount was performed.
- A new built-in adapter needs the repository's normal release process. Live
  `POST /v1/models` cannot add an unknown image adapter without a code deploy.
  Release path: local worktree tests → explicit approval → `greenline submit`
  → full check → drain to zero active jobs → deploy/restart/health. **Do not
  submit merely to finish this preparation task.**

## Comparison arms

1. Like-for-like: house botanist/Sydney/LUMEN, 1024×1024, seeds
   42001/42002/42003, exact recovered prompts (pending), prompt upsampling
   off, checkpoint sigmas, 8 steps/no CFG. Warm renders with queue/load split.
2. Native: 2912×2912, same seeds, prompt-as-written, separately labelled
   native-8MP. Optional prompt-upsampling arm requires a reviewed
   implementation and must record expanded prompt plus separate timing.
3. Quantization changes precision and is never silently described as bf16 or
   like-for-like. If Comfy W4A8 is chosen, prepare a separate reviewed Comfy
   runtime adapter instead of mixing its files into this diffusers adapter.

Results contain prompt/seed, revision, precision, width/height, sigmas, cold
load and verification, per-render generation time, PNG write time, render
index, actual CUDA allocated/reserved peaks, and runtime versions. Queue time
is read from Arbiter job timestamps, not mixed into pure generation time.
