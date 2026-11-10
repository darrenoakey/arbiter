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

## Local storage and approval state (2026-11-04 19:12 UTC)

Prepared model location is `/mnt/t9/models/p-h3-image`, not the unmounted
`/mnt/arbiter-store` path. The approved physical T9 filesystem UUID is
`ec088732-0cb2-4fc4-935e-f5d6f9a3b73b`. Before a future download, prove the
exact mount/UUID, fail if absent, and use a local directory there; never fall
back to Spark root or automatically mount anything.

Fresh available bytes: root 803,936,972,800; T9 64,384,659,456. Storage owner
conv-6f92a2a821b950fa requires root use <200 GB, models/logs on T9, and a
**50,000,000,000-byte T9 free reserve**. Thus usable T9 space is only
14,384,659,456 bytes: **5,699,941,869 more bytes are required for weights
alone**, before the unmeasured temporary-download/venv/runtime overhead.
Recovery is the storage owner's task, not this adapter's. Do not clean or
compress anything as part of P-H3.

Installation/download approval is pending with the parent. No explicit
installation or adapter-release denial has occurred, but the original scope
held both. Direct Spark-to-Mini SSH was explicitly denied and will not be
retried. The shared drain/restart hold is binding until the music MP4 is
verified. A production release is a separate decision from installing files.
The local path removes CIFS as a model-loading dependency, not the storage
reserve, install approval, or render hold.

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

## Nov 10 bounded audit (supersedes stale holds/counts above)

MiniMaxAI/MiniMax-H3-Image@492fd8c4eca666f1085f57e30738c1e0d86b3439
is a distinct Nov10 still-image release, not P-H3's old video base. No new
model install, licence acceptance, endpoint or job is authorized by this
adapter. Source/licence/resolution audit and delivered historical sheets with
NOT RUN cells are in ~/src/p-h3-image-comparison-20261103/CANDIDATE-AUDIT-20261110.md.
Music owner record14000 cleared the video hold after final delivery; do not
repeat the stale music hold. Separate production-release authority is still
unresolved. T9 currently has333.014GB free, above the50GB reserve.

Offline corrections: PromptExecutor gets explicit CLASSIC/cache_args rather
than its invalid None default; repeated load no longer depends on stale
local module bindings; worker setup duration is measured; cached sampler
replay is rejected rather than reported as warm inference. Whole-pipeline
and copy duration are named as such. Pure-inference and isolated cold-load
fields remain None, not invented. Cancellation during inference and genuinely
uncached same-seed warm-repeat support remain pre-release requirements.
Fresh verification:68 Python tests and2 focused Go policy tests passed.
Go cold compilation was detached after a120s foreground timeout; its bounded
verdict completed successfully. No GPU, runtime import, model output or
production-release test occurred.
