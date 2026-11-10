# MiniMax-H3-Image (P-H3 comparison adapter)

`src/arbiter/adapters/minimax_h3_image.py` serves the **new** (2026-11-10)
`MiniMaxAI/MiniMax-H3-Image` still-image checkpoint — not the older MiniMax-H3
video model (`minimax-h3-local` / `minimax-fast-h3`). It exists for the P-H3
image comparison only. This document records what was verified, what the
adapter enforces, and what remains unresolved. Nothing here is a model run:
no weights were downloaded, no GPU work was run, and no licence agreement was
accepted while writing it.

## Verified runtime API (source audit, 2026-11-10)

Evidence: `p-h3-image-comparison-20261103/source-evidence/new-h3-image-20261110/`
(model repo at pin `492fd8c4eca666f1085f57e30738c1e0d86b3439`; Diffusers tree at
commit `f598e6e6a99528858b962c1399c45b59a6c25484`).

- The official in-process runtime is Diffusers' modular pipeline:
  `MiniMaxH3ImageModularPipeline` / `MiniMaxH3ImageBlocks`, loading
  `transformer_diffusers/` (`MiniMaxH3ImageTransformer2DModel`) and
  `vae_diffusers/` (`AutoencoderKLMiniMaxH3Image`) per
  `modular_model_index.json`. Usage per the class docstring:
  `ModularPipeline.from_pretrained(...)`, `load_components(dtype=torch.bfloat16)`,
  then call with `prompt`, `height`, `width`, `min_generation_area`,
  `generator`, `output_type="pil"`; the blocks declare one `images` output.
- Distilled twice over: guidance is baked in — there is **no**
  `negative_prompt` and **no** `guidance_scale` input (guidance 1 only), and
  sampling uses the fixed six-step sigma grid in `modular_model_index.json`
  (`[1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724,
  0.23200887441635132, 0.0]`), so there is no `num_inference_steps` either.
- No prompt enhancement, API, or cloud route is used; text-to-image only.

### The `min_generation_area` trap (confirmed in the pinned source)

`MiniMaxH3ImagePrepareCanvasStep` defaults `min_generation_area` to
4194304 (2048×2048): a smaller request is rendered on a larger same-aspect
canvas and resized back to the requested size after decoding
(`MiniMaxH3ImageDecodeStep`, via the image processor's LANCZOS resize). So a
default 1024×1024 run is rendered at 2048×2048 — not native. Passing `1024`
is the runtime's documented off switch. Canvas sides round to multiples of 32.

## Adapter protocol

Only two fixed lanes, so a benchmark row can never be mislabelled:

| lane | requested | `min_generation_area` | rendered | delivered |
|---|---|---|---|---|
| `controlled-1024` | 1024×1024 | 1024 (trap off) | 1024×1024 | 1024×1024 |
| `recommended-4096` | 4096×4096 | 4194304 (default; scales nothing at 16 MP) | 4096×4096 | 4096×4096 |

Every result records requested / rendered / delivered sizes separately and
the render fails if the delivered image is not exactly the requested size.
Seed uses a CPU `torch.Generator`. Caller-supplied `width`/`height` that
disagree with the lane, `steps`, `sigmas`, `min_generation_area`,
`negative_prompt`, any CFG knob other than omitted/`guidance_scale=1`,
image/edit inputs, and prompt-enhancement/API/cloud/hosted-service params
all fail instead of being silently ignored.

Fail-closed load: requires the Arbiter worker CUDA cap
(`ARBITER_MEMORY_GB`); verifies every file under `LICENSE`,
`model_index.json`, `modular_model_index.json`, `scheduler/`, `text_encoder/`,
`tokenizer/`, `processor/`, `transformer_diffusers/`, `vae_diffusers/`
against the pinned snapshot metadata; requires the installed `diffusers`
distribution to be the pinned git commit above; then checks the loaded
pipeline/transformer/VAE/scheduler class names and the exact sigma grid.
An FP4/NVFP4 declaration in the transformer config fails the load — quantized
weights are never reused or relabelled as the bf16 benchmark snapshot, and
the old H3 video snapshot (no modular index) cannot load here.

## Licence gate (unresolved requirement)

The MiniMax-H3-Image licence requires separate prior **written authorization**
from MiniMax (via https://platform.minimax.io/license) if the annualized
revenue of the user and/or affiliates exceeds USD 1M, **or** the model is
provided as a Hosted Service to third parties. Whether this deployment is
covered is **unverified**. Therefore every job must carry
`license_confirmed: true` plus a `license_confirmation_ref` (8–200 chars)
identifying the local eligibility determination or written authorization
relied on; the reference is recorded in the result. Absent that, inference
refuses. Third-party hosted-service flags are rejected outright.

## Unresolved requirements / limitations

1. **Licence eligibility** (above) must be determined/authorized before any
   real run; the approved P-H3 install does not cover this model's licence.
2. **Weights are not present**: the bf16 snapshot at the pin is ~135 GB
   across 54 files. It must be downloaded deliberately (never duplicated from,
   or substituted with, the old H3/NVFP4 snapshot) to
   `/home/darren/nvme-models/minimax-h3-image/MiniMax-H3-Image`.
3. **Runtime install**: the worker venv needs `diffusers` installed from the
   pinned git commit `f598e6e6…` (a PyPI install fails the load gate). See
   the optional `requirements/minimax-h3-image.txt` note — installing it is
   **not** authorized by the P-H3 approval and has not been done.
4. **No authoritative memory-capable GB10 route is proven** for this snapshot.
   The adapter loads bf16 in-process under the Arbiter worker CUDA cap and
   fails if it does not fit; it does not invent offload/quantization routes.
5. **Parent wiring still required** (shared files this task did not touch):
   add `minimax_h3_image` to `adapters/__init__.py`, model/job-type/schema
   and image-policy entries, and config/memory declarations.
6. Existing full old-H3 encoder hashes matched 14/14 in the audit, but no
   current full old-H3 snapshot was verified; that has no bearing here —
   this adapter shares no weights with it.

## Tests

`tests/unit/test_minimax_h3_image.py` — unit/static tests only (real on-disk
snapshot hashes, parser, licence gate, canvas math). They are **not** model
runs and prove no GPU behaviour.
