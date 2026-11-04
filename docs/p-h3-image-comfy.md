# P-H3-Image Comfy W4A8 — offline adapter preparation

## Status

Owner-requested comparison only; not an IGS/generate_image/SDK route. **No
weights, runtime installation, GPU execution, job, deployment or Greenline
submission has occurred.** GB10 compatibility, actual peak memory and timing
remain unmeasured. The old diffusers scaffold was removed because no upstream
PH3Image pipeline was found.

## Immutable supported source

- Runtime: `https://github.com/comfyanonymous/ComfyUI.git` commit
  `576b642e77b4c42f84afe583c8e8f0eef7637618` (`feat: Support PH3Image`
  #17506). The adapter verifies the commit and PH3-relevant source blob IDs.
- Checkpoint: `PrunaAI/P-H3-Image-Comfy` revision
  `44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543`; W4A8 transformer/text encoder
  and int8-convrot VAE, exactly 20,084,601,325 bytes. All three LFS hashes
  are pinned in `adapters/p_h3_image_comfy.py`.
- License: MiniMax H3 Community License. Australian local use is not in the
  excluded EU/UK/Republic of Korea/USA territories. No new agreement accepted.

## Storage finding

Preferred existing authorized storage is Mac mini
`/Volumes/ssd_4/arbiter/models/p-h3-image` with **1.63 TiB free**, exposed
through the existing intended mapping `/mnt/arbiter-store/models/p-h3-image`.
No relocation or Spark cleanup is needed for this payload.

At inspection, Spark's existing CIFS mount was absent. TCP/445 connected, but
SMB3 protocol negotiation timed out and the kernel logged `cifs_mount failed
w/return code = -112`. The Mac mini has the writable `arbiter-data` share
configured but no running `smbd` process. **No mount, security, SMB service,
restart or reconfiguration was attempted.** This is the remaining storage
blocker, not capacity. Dedicated cleanup owner conv-af0761673c160cb4 has the
payload and protected-path facts; this task performs no cleanup or downloads.

## Adapter scope

- Exact job/model: `p-h3-image-comfy` / `p-h3-image-comfy-w4a8`; all other
  precision/upsampler/LoRA/video-component substitutions fail closed.
- In-process `execution.PromptExecutor` following official `main.py`; no
  HTTP listener and no GPU execution outside Arbiter. Before importing Comfy
  modules, the pinned checkout is inserted into `sys.path`; built-in nodes
  initialize with both custom nodes and API nodes disabled.
- Explicit `folder_paths.add_model_folder_path` entries restrict
  `diffusion_models`, `text_encoders` and `vae` to the pinned snapshot.
  `folder_paths.set_output_directory` is called per Arbiter job, so PNGs are
  collected from that job's directory, not a shared runtime/output tree.
- Seven logical stages distilled into a 12-node prompt graph from Pruna
  workflow `9d43b9c5-ca8f-4a60-837d-d342773e8b8a`: loaders, 1.0/1.0 model
  shift, PH3 conditioning, fixed noise/sigma schedule, Euler sampler, VAE
  decode and PNG save. Prompt upsampler absent.
- Model directory real-path and three full SHA-256 checks before load.
- Result records revision/precision/settings, render index, generation/write
  and load timings, actual CUDA allocated/reserved peaks.
- Proposed single worker with 28 GB declaration (vendor says ~21 GB at 8 MP;
  GB10 must be measured). Memory cap and cancellation/unload checks require
  installed-runtime smoke tests before release.

## Unverified installed-runtime details

The pinned runtime path is `/home/darren/src/arbiter/runtime/ComfyUI`, a real
checkout whose HEAD and PH3 source blobs are verified before loading. The
adapter configures model search roots, output paths and built-in nodes as
above. Actual `PromptExecutor` initialization/imports, GB10 W4A8 kernel
support and real model output still require an installed-runtime smoke test;
offline local doubles cannot prove those. Do not deploy without that check.

## Tests

Offline only: **67 Python tests passed** (adapter plus existing image-policy
and job-type suites), and focused Go policy tests passed. Tests use local
files and a lightweight local execution double for our runner boundary; they
do not import ComfyUI/torch or claim a render. Alias gaps found by the Go
negative test were closed in both policy implementations.

## Next benchmark plan after approval/storage/runtime

Use recovered primary Flux/Qwen prompts verbatim at 1024², seeds
42001/42002/42003, upsampling off. Separate native 2912² arm. Keep queue,
cold load/verification and generation timings distinct; retain logs/original
PNGs/contact sheets and assess hands, detail, adherence and exact typography.
All downloads/installs/releases must run detached and be slept on.
