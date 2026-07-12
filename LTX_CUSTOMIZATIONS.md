# LTX-2.3 Video Pipeline — Complete Customization Record

**Why this file exists:** the LTX video path on spark is heavily modified from
stock LTX-2. Every customization below was deliberate and hard-won. Read this
in full before changing *anything* in the LTX path — a naive "upgrade to the
latest LTX" will silently destroy all of it.

**Two repos are involved:**

| Repo | Location | Holds |
|------|----------|-------|
| `arbiter` (source of truth) | Mac `/Volumes/T9/darrenoakey/src/arbiter` → deploy → spark `~/src/arbiter` | The 4 LTX **adapters** (`src/arbiter/adapters/ltx2*.py`) |
| `ltx2-spark` | spark `~/src/ltx2-spark` (own git repo, 3 commits) | `video_fast_gpu.py` (the custom `FastPipeline`), `constants.py`, the vendored `packages/ltx-core` + `packages/ltx-pipelines` |

The adapters call into `FastPipeline` via `sys.path.insert(0, "/home/darren/src/ltx2-spark")`.

---

## A. The "hollow out" — ltx2-spark is not the upstream LTX repo

`ltx2-spark` git history:
- `ed913a2` / `62d249c` — pre-arbiter snapshots preserving the original model code.
- `1e29ea5` — **"Hollow out: replace model code with Arbiter API calls."** The CLI
  (`pipeline.py`, `story.py`, `assemble.py`, `images.py`, `transcribe.py`) no longer
  runs models — it submits jobs to Arbiter. The *only* code that touches GPU is
  `video_fast_gpu.py:FastPipeline`, driven exclusively by the arbiter adapters.

Implication: there is no "pull latest LTX-2.3 and reinstall" path. The model
runner is our own `FastPipeline`, not Lightricks' inference scripts.

## B. Weights are LTX-2.3, wired through a custom model dir

`constants.py`:
- `CHECKPOINT      = ~/models/ltx2/ltx-2.3-22b-distilled.safetensors`
- `UPSCALER        = ~/models/ltx2/ltx-2.3-spatial-upscaler-x2-1.0.safetensors`
- `DISTILLED_LORA  = ~/models/ltx2/ltx-2.3-22b-distilled-lora-384.safetensors`
- `GEMMA_DIR       = ~/models/ltx2` (holds `text_encoder/`, `tokenizer/`)

`~/models/ltx2/` is a hand-built layout: the three `*.safetensors` are **symlinks**
into the HF cache snapshot `models--Lightricks--LTX-2.3/snapshots/cd784f3198e9ec3efec60b66a0fd78aafe413a86/`,
alongside real local dirs `audio_vae/ connectors/ text_encoder/ tokenizer/
transformer/ vae/ vocoder/` and `model_index.json`. The adapter is *named*
`ltx2` for historical reasons but runs **LTX-2.3 (22B distilled)** weights.

Custom `DEFAULT_NEGATIVE_PROMPT` — a long hand-tuned negative prompt including
audio-quality negatives ("silent or muted audio, distorted voice, robotic
voice, echo, off-sync audio, mismatched lip sync"). Not the stock LTX negative.

Custom `RESOLUTION_PRESETS` (must stay /64 divisible for the two-stage upsample):
`small (512,768)`, `small-portrait (768,512)`, `large (704,1280)`,
`large-portrait (1280,704)`.

## C. Phased model loading (the 7-phase FastPipeline)

Stock LTX loads/unloads its 7 sub-models *per chunk* (≈350 loads for a
50-chunk video). `FastPipeline.generate_all_chunks` batches **all chunks
through each stage with the model loaded once** (7 loads total):

1. text encoder → encode every unique prompt (cached by prompt string) → unload
2. audio encoder → encode every chunk's audio slice → unload
3. load video encoder + spatial upsampler (kept resident across phases 3–5)
4. stage-1 transformer → denoise **all** chunks → unload
5. upsample **all** chunks + build stage-2 image conditionings → unload video encoder/upsampler
6. stage-2 transformer (= stage-1 ledger **+ distilled LoRA**) → denoise all → unload
7. video decoder → decode all chunks → `chunk_NNN.npy` (+ `_last.jpg`) → unload

Intermediate latents are pushed to **CPU** (`.cpu()`) between phases to keep
GPU resident set small. `stage_2_ledger = stage_1_ledger.with_additional_loras(
DISTILLED_LORA, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)` — the distilled LoRA is only
applied for stage-2, with the ComfyUI renaming map.

## D. Audio-cue conditioning — "taking cues from the audio"

This is the biggest semantic customization. The video is generated **driven by
the audio track**, not just a text prompt:

- Per chunk, the matching audio slice is cut by time (`decode_audio_from_file(
  audio_path, device, start, dur)`), VAE-encoded (`vae_encode_audio`), and
  trimmed to `AudioLatentShape.from_duration(batch=1, duration=frames/fps,
  channels=8, mel_bins=16)`.
- Denoising uses a **dual MultiModalGuider**: a video guider
  (`cfg 3.0, stg 1.0, rescale 0.7, modality 3.0, skip_step 0, stg_blocks [28]`)
  **and a separate audio guider** (`cfg 3.5, modality 3.5`) with its own
  negative context.
- `initial_audio_latent` is threaded into **both** stage-1 and stage-2
  `denoise_video_only` calls, so audio conditions the whole diffusion, not just
  a post-hoc mux.
- The audio latent is persisted through the file handoff
  (`encoded.pt` → `stage1_output.pt` → stage-2) so the split adapters stay
  audio-conditioned.
- Final mp4 is muxed with the **time-sliced original audio** (`-ss start -t
  dur ... -shortest`) so the rendered chunk lines up with its source audio.

If a future LTX swap changes the guider/audio-VAE API, **audio conditioning is
the first thing that silently breaks** (video still renders, just ignoring
audio). Always A/B test that motion follows the beat after any LTX change.

## E. Split into three separate adapters (encode → denoise1 → denoise2)

Besides the monolithic `ltx2` adapter, the pipeline is split into three
independently-scheduled Arbiter models so the scheduler isn't forced to budget
the full ~90 GB for the whole job:

| Adapter (`model_id`) | Loads | ≈VRAM | Output file |
|---|---|---|---|
| `ltx2-encode` (`ltx2_encode.py`) | text enc + gemma embed proc + audio enc | ~12 GB | `encoded.pt` |
| `ltx2-denoise1` (`ltx2_denoise1.py`) | video enc + spatial upsampler + stage-1 transformer | ~42 GB | `stage1_output.pt` |
| `ltx2-denoise2` (`ltx2_denoise2.py`) | stage-2 transformer + video decoder (**pre-loaded in `load()`, kept resident ≈40 GB**) | ~40 GB | `result.mp4` |

- All three are `GroupAdapter`s. Model caching across jobs via
  `_enc_loaded` / `_dn1_loaded` / `_s2_loaded` flags +
  `_ensure_encode_models` / `_ensure_denoise1_models` /
  `unload_*` methods in `FastPipeline`.
- **Backwards-compatible file schema:** `ltx2_denoise2` accepts
  `denoise1_file` *or* `stage1_file`; `FastPipeline.run_stage1` (legacy
  monolithic stage-1) writes the exact same `torch.save` dict as the split
  `save_denoise1_output`, so old `stage1_output.pt` files still feed denoise2.
- `ltx2.py` keeps a **single-segment fast path** (`_infer_single_chunk` →
  `generate_single_chunk`) used when there's exactly one segment and
  `_transformer_s1` is already resident.

## F. GPU-lock pipelining inside each split adapter

Each split adapter implements the 3-phase pattern from `OPERATIONS.md`:

- `__init__`: `self._gpu_lock = threading.Lock()`.
- `infer()`: **Phase 1** CPU/IO prep (no lock) → **Phase 2** `with
  self._gpu_lock:` GPU forward passes → **Phase 3** CPU/IO tail (no lock).
- Configured `max_concurrent = 2` so Arbiter dispatches two `infer()` calls to
  the same worker; one holds the GPU lock and runs the model while the other
  does its audio decode / `torch.load` / ffmpeg encode in parallel. Shared
  model weights make the lock mandatory (concurrent forward passes on shared
  CUDA modules crash).

`FastPipeline` method triplets back this: `load_encode_input` /
`run_encode_gpu` / `save_encode_output`, `load_denoise1_input` /
`run_denoise1_gpu` / `save_denoise1_output`, `load_stage2_input` /
`run_stage2_gpu` / `save_stage2_output` (+ `run_*` convenience wrappers).

## G. Memory-mapped / `map_location` file reads — deliberate, asymmetric

The file handoff between adapters is the overlappable CPU/IO work. The
`torch.load` `map_location` is chosen **per stage on purpose**:

- `load_denoise1_input`: `torch.load(encoded_path, map_location="cpu",
  weights_only=False)` — keep tensors **off GPU** so the read fully overlaps
  another concurrent call's GPU phase; tensors are moved to device later inside
  the locked GPU phase.
- `load_stage2_input`: `torch.load(stage1_path, map_location=self.device,
  weights_only=False)` — deliberately loads **straight to GPU** because
  `stage2_conds` is an **opaque container** with no generic `.to(device)`
  (the documented "opaque container in torch.load" gotcha in
  `OPERATIONS.md`). The dominant cost (NFS read + unpickle) still overlaps; the
  inline CPU→GPU copy is cheap. **Do not "normalise" these two to the same
  `map_location` — it will either crash stage-2 (tensors split cpu/cuda) or
  kill the denoise1 pipelining win.**

`weights_only=False` is required (the payload contains non-tensor container
objects, not a plain state dict).

## H. Other adapter-side orchestration to preserve

- **Model-only spatial lattice:** the dev denoiser patches the 8x VAE latent in
  2x2 blocks, so model dimensions must be divisible by 16. Beezle supplies
  1920x1088 for a 1080p render; `ltx2_denoise2` center-crops the decoded frames
  to the requested 1920x1080 before NVENC. Never expose 1088p or resize it back
  to 1080p. 3840x2160 already satisfies the lattice and is unchanged.
- **Odd frame count**: LTX-2 requires an odd `num_frames`; adapters do
  `if num_frames % 2 == 0: num_frames += 1`.
- **Per-chunk seed**: `seed + chunk_index` — deterministic but varied per chunk.
- **Chunk overlap**: `_assemble_mp4` drops the **first frame** of every chunk
  after the first (it overlaps the previous chunk's last frame).
- **Encoders**: multi-chunk assemble uses `libx264 -crf 18`; per-chunk
  `denoise2` uses `h264_nvenc` (GPU encode, overlaps next chunk's GPU denoise).
- **Dual input paths**: every media input accepts base64 (`audio_b64`,
  `start_image_b64`, `end_image_b64`) *and* staged spark file paths
  (`audio_file`, `start_image_file`, `end_image_file`, `encoded_file`,
  `denoise1_file`/`stage1_file`).
- **Cancellation via progress_fn**: `progress_fn` raises `CancelledException`
  when `cancel_flag` is set, so cancellation is honoured mid-denoise.
- **estimate_time** per stage: encode 15 s, denoise1 180 s, denoise2 120 s,
  monolithic `ltx2` ≈ 800 ms/output-frame.
- **Debug hook**: `/tmp/ltx_debug_mem` (touch file) → `_ensure_encode_models`
  writes per-submodel load timings + CUDA peak to `/tmp/ltx_debug_mem.log`.

## I. Environment / packaging

- LTX adapters run in the arbiter **main `.venv`** (not a dedicated venv).
  `ltx_core` / `ltx_pipelines` are `pip install -e` from
  `~/src/ltx2-spark/packages/`. `ltx-core` pyproject pins `torch~=2.7`,
  `transformers>=4.52` — loose enough for the arbiter stack, but **a
  transformers v5 / torch 2.12 bump must be smoke-tested against LTX**, the
  Gemma text encoder especially.
- Arbiter config registers `ltx2`, `ltx2-encode`, `ltx2-denoise1`,
  `ltx2-denoise2` as separate models with their own memory_gb / max_concurrent.

---

## Upgrade path within LTX-2.3 (1.0 → 1.1)

`Lightricks/LTX-2.3` ships **1.1** variants (≈1 month newer) that are weight-only
swaps — *no code change*, same 22B arch, **improved audio**:

| constant | current (1.0) | 1.1 upgrade |
|---|---|---|
| `CHECKPOINT` | `ltx-2.3-22b-distilled.safetensors` | `ltx-2.3-22b-distilled-1.1.safetensors` |
| `DISTILLED_LORA` | `ltx-2.3-22b-distilled-lora-384.safetensors` | `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` |
| `UPSCALER` | `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (hotfix) |

`distilled` and its `distilled-lora` are a **matched pair — upgrade both or
neither**. Verify the 1.1 distilled still uses `STAGE_2_DISTILLED_SIGMA_VALUES`
from `ltx_pipelines.utils.constants` before trusting output; re-run an
audio-driven A/B (does motion still follow the beat?) after the swap.
