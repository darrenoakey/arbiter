# LTX 2.5 acceleration A/B benchmark (opt-in, capped)

Code: `src/arbiter/adapters/ltx25_acceleration.py`, with a single opt-in hook in
`ltx25_denoise1.py`. Tests: `tests/unit/test_ltx25_acceleration.py`. The tests
run on CPU and cover parsing, caps, weight identity, attestation, timing
bookkeeping and constants checked against the pinned source archive. They do
**not** show GPU correctness. Only real Arbiter jobs, checked with
ffprobe/visual/audio review, can show that.

## Opting in

The opt-in is the job param `acceleration_benchmark` on `ltx25-denoise1`. When it
is absent the production path runs, and that path is unchanged. No config,
manifest, default, cap or security setting changes.

| Param | Rule |
|---|---|
| `acceleration_benchmark` | exactly `"stock"`, `"distilled"` or `"distilled-conv"` |
| `encoded_file`, `audio_file`, `start_time`, `fps` | same meaning as production |
| `num_inference_steps` | stock: absent or `30`. distilled*: must be absent |
| `a2v_guidance_scale` | stock: absent or `3.0` (the historical `5` is rejected). distilled*: must be absent |
| `stage1_guiding_keyframes`, `generated_keyframes`, `generated_keyframe_positions`, `stage2_drop_end_image`, `end_image_strength` | only allowed at their no-op default |
| `end_image_attention_strengths` | honoured on both stages, same as production |

Caps are checked against the **encoded bundle**: `num_frames <= 49` with
`num_frames % 8 == 1`, `height <= 512`, `width <= 768`, and both sides
divisible by 64.

## Profiles

| | stock | distilled | distilled-conv |
|---|---|---|---|
| transformer | dev bf16 (`792a2bad…`) | official distilled bf16, separate file (`31eb3cad…`, HF rev `6c7e5e5`) | same as distilled |
| stage 1 | `LTX2Scheduler` 30 steps, GuidedDenoiser CFG 3 / STG 1 @28 / rescale 0.7 / modality 3, Euler | `DISTILLED_SIGMAS` (8 steps), SimpleDenoiser (CFG 1, STG 0, modality 1), `EulerAncestralDiffusionStep(1,1)` + `euler_ancestral_denoising_loop(noise_seed=seed+10000)` | same |
| stage 2 | dev + distilled LoRA-450 @1.0, runner sigmas `[.909,.725,.422,0]` | same distilled transformer, no LoRA, `STAGE_2_DISTILLED_SIGMAS` `[.909375,.725,.421875,0]`, deterministic Euler | same |
| decoder | DiffVAE `CHUNKED_EAGER` (`847e14ca…`) | DiffVAE `CHUNKED_EAGER` | conv VAE (`685b06ee…`) |

- Seed: the runner convention `seed + chunk_index` for the noiser. The
  ancestral loop adds `+10000`, as installed `distilled.py` does.
- dev + LoRA is **not** the official student and is never used for the
  distilled profiles.
- The conv VAE's `encoder.*` and `per_channel_statistics.*` tensors are
  byte-identical to the DiffVAE file (86/86 tensors, checked on spark). Latents
  are therefore compatible; only the decoder changes.
- No `blackwell_dsl`, no compile and no new dependencies.

All profiles share the music-video adaptation:

- The original audio latent is frozen in both stages and the original
  soundtrack is muxed by NVENC. This is not stock joint T2AV.
- The bundle's stage-1 conditionings (first, last, repeated-end, guides) are
  used unchanged.
- Stage 2 is rebuilt from the bundle images by the runner.
- After decode, the exact first and last pixels are copied in. This is a pixel
  copy and does not show generative adherence.

## Safety properties

- **Pinned runner.** The stock path replays the runner's `run_denoise_gpu`
  phase by phase, so build time and warm denoise time can be measured
  separately. If `runtime/ltx25/manifest.json` pins any runner other than
  `1c8ea922…`, the benchmark refuses to run. If the runner changes, re-audit
  the replay first.
- **Installed constants checked.** At run time the installed sigmas, ancestral
  constants and runner paths are compared with the pinned values. The
  distilled checkpoint must also select the ancestral sampler.
- **Fixed weight paths.** No job param can choose a weight path.
- **Content attestation.** Every selected weight is checked for size and
  SHA-256 before load. A match is cached per process, keyed by
  (dev, inode, size, mtime). Hashing a 39 GiB file on the SMB store takes
  about 6 minutes on first use. The hash emits a real bytes-hashed stderr line
  every 1 GiB or 30 s, and the GPU idle watchdog reads that output as
  liveness.
- **One transformer at a time.** Each `DiffusionStage` builds its
  transformer, runs, and frees it. Stage objects, the temporary conv decoder
  and the decode iterator are released in a `finally` block, whether the run
  succeeds or fails, before the GPU lock is released.
- **No global patching.** Nothing is monkeypatched and the base pipeline is
  never mutated. The resident image encoder, upsampler and DiffVAE decoder
  are only read.

## Output

The job writes `result.mp4` and `benchmark.json`. The result dict carries the
same metadata under `acceleration_benchmark`:

- the contract and the sigmas, seeds and conditioning types actually used;
- weight attestations;
- runtime provenance, the runner SHA, and the transformer `model_version`;
- `timings_s`: `cpu_prep`, `attestation`, `context_to_device`,
  `stage1_build/denoise/release`, `upscale`, `stage2_conditioning`,
  `stage2_build/denoise/release`, `decode_build`, `decode`, `postprocess`,
  `gpu_total`, `encode_mux`, `total`;
- per-segment and overall CUDA peak allocated/reserved GiB.

Every boundary calls `torch.cuda.synchronize`, and peaks are reset per
segment. The worker's model load happens outside `infer`.
`worker_load.timing` reports the measured `load()` durations: runtime import,
and `FastPipeline` init plus `load_denoise_models`. Transformers are not part of
worker load; each job builds its own, as shown in `stage*_build`.
`benchmark_infers_since_load` counts benchmark jobs only. Normal
`ltx25-denoise1` jobs on the same worker are not counted, so `1` means the
first benchmark job since load, not proof of a cold process. Compare
`worker_load.timing.loaded_at_unix` with the scheduler job history to see what
else ran on that worker.
