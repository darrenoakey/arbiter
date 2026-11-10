# MiniMax-H3-Image — local still-image comparison

Exact job/model: `minimax-h3-image`, isolated from the existing local H3 video
adapters and from every MiniMax cloud route. Source revision
`492fd8c4eca666f1085f57e30738c1e0d86b3439`, pinned Diffusers commit
`f598e6e6a99528858b962c1399c45b59a6c25484`. The model and every job remain
local; no prompt enhancement, cloud endpoint, editing or CFG is offered.

## GB10 staged loading

The three BF16 weight groups total about 126 GiB, exceeding the usable unified
memory. Ordinary CPU offload does not fix that on GB10 because CPU and CUDA
tensors use the same physical memory. This adapter therefore uses the pinned
modular runtime's selective `load_components` / `unload_components` lifecycle:

1. Verify all snapshot content and the Diffusers commit, load the Qwen text
   encoder, tokenizer and processor, and encode the literal prompt.
2. Move the small prompt embeddings out of the encoder, free the complete
   encoder and all its activations, and verify CUDA allocation is zero.
3. Load transformer, VAE and scheduler, transfer embeddings back to CUDA,
   denoise for the fixed six steps and decode the PNG.
4. On a later prompt, free transformer/VAE before re-loading the encoder.

The large stages never coexist. Text encoding, encoder freeing, model loading,
six-step denoising, VAE decoding and PNG writing are measured independently.
The declared 75 GB worker budget limits the largest single CUDA stage; the
weights are still BF16 and no quantization or model substitution occurs.

The pinned modular pipeline lacks `enable_model_cpu_offload` and
`enable_sequential_cpu_offload`; don't add those APIs as a purported fix.

## Benchmark lanes

`controlled-1024` requests and renders exactly 1024×1024 with
`min_generation_area=1024`, the runtime's documented minimum-canvas off switch.
`recommended-4096` requests/renders 4096×4096. Requested, rendered and
 delivered sizes are checked and recorded separately. Native/recommended and
controlled timings are never compared as equal work.

The per-job `license_confirmed: true` and `license_confirmation_ref` are
required by the community licence's explicit eligibility/authorization gate.
Third-party hosted-service flags are refused.

## Reproducible runtime

The real-directory `venvs/minimaxh3image` is a venv overlay sharing the
existing qwenimage package directory for torch 2.14.0+cu130,
transformers 5.17.0 and accelerate 1.15.0. Only the pinned Diffusers commit is
installed in the overlay. The actual worker venv parent must never be a
symlink. Existing qwenimage packages are not modified.

Downloads and installs live under `/home/darren/h3-image-install-20261111/`,
with the snapshot at
`/home/darren/nvme-models/minimax-h3-image/MiniMax-H3-Image` (physical T9).
Installation, downloads, real GPU jobs and releases must be detached with
persistent state/logs and slept on. Offline tests do not establish a model
render; only the Arbiter job result and inspected PNG do.
