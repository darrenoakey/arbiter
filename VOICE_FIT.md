# voice-fit adapter — Kokoro voice-pack training (voxsmith)

Fits custom [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) voice packs
(`.pt` style tensors, shape `(511,1,256)` float32) from reference audio of a
target speaker, using the **voxsmith** package: direct gradient descent on the
256-dim style vector through Kokoro's differentiable decoder, scored by a
differentiable ECAPA-TDNN speaker encoder (SpeechBrain
`spkrec-ecapa-voxceleb`), with a multi-target round-robin objective so the
fitted voice generalises across varied phonetic content instead of overfitting
one utterance. This replaces the KVoiceWalk-style random walk (~100x slower;
still in voxsmith as `--method walk`).

> ⛔ **Never run voxsmith (or any CUDA training) on spark outside arbiter.**
> On the GB10's unified memory an uncapped CUDA process can livelock the host
> (see CLAUDE.md §Memory Containment). All GPU fits go through this adapter —
> `voxsmith arbiter-fit` on the Mac is the front door. No direct ssh python
> jobs. Never ever ever ever ever ever do that again.

## Job type

```
{"type": "voice-fit", "params": {
    "name": "leo-laporte",               # required -> /home/darren/voice-models/<name>.pt
    "targets_dir": "/path/on/spark",     # dir of *.wav + sibling *.txt transcripts
    #   OR "targets_file": "<staged zip of the same layout>"  (stage via arbiter_client)
    "seed_voice": "auto",                # stock voice, or "auto" (embedding search + blend)
    "exclude": "am_adam",                # optional: drop voices from the auto seed pool
    "init_pack": "leo-laporte",          # optional: continue from voice-models/<name>.pt
                                         #   or an absolute .pt path (e.g. a staged file)
    "steps": 300, "lr": 0.05, "w_self": 0.5, "w_reg": 1.0, "eval_every": 10
}}
# result: {model_id, pack_path, seed, baseline_sim, final_sim, delta, per_target,
#          self_sim, steps, seconds, history_tail}
# The fitted pack is ALSO written as the job's result.pt, so the API inlines it
# as base64 (result.data) — remote clients need no filesystem access to spark.
```

- `*_file` params must be inside `ARBITER_INBOX_PATH` (`/mnt/arbiter-store/inbox`) —
  stage with `arbiter_client.stage_file` (the voxsmith CLI does this for you,
  with an scp fallback when the Mac-side shared mount is unavailable).
- Cancellation (SIGUSR1) is honored between optimizer steps via voxsmith's
  `cancel_check` hook; the best checkpoint so far is always saved incrementally.
- Metrics are ECAPA-TDNN cosine similarities: `baseline_sim`/`final_sim` are
  means across ALL targets; `self_sim` is cross-text consistency of the fitted
  pack (stock voices score ~0.77-0.88 on arbitrary sentence pairs).
- **voxsmith ≥ v3 (2026-07-22): naturalness terms are ON by default in
  `fit_gradient` — the adapter params are unchanged and need no redeploy.**
  `w_f0=0.3` (log-F0 mean/std match on Kokoro's F0_pred over voiced frames),
  `w_spec=0.3` (windowed mean-log-mel cosine vs reference), `w_dur_rate=0.2`
  (mean frames/phoneme pulled to the reference speaking rate), `w_dur=1.0`
  (duration-runaway guard). Best checkpoint = harmonic mean of (mean target
  sim, self sim, spectral sim). `--seed-voice auto` ranks by
  `sim − 0.3·|log2(f0 ratio)|` using cached pitch profiles at
  `/home/darren/voice-models/voices/stock/pitch_profiles.json` (refresh with
  `voxsmith pitch-profile --stock`). The decoder is gradient-checkpointed and
  the allocator is emptied per step.
- **2026-07-23 OOM root causes + fixes (voxsmith ≥ v3.1):** per-step CUDA
  telemetry now logs the TRUE step peak (reset happens after logging) plus
  live/reserved; an OOM handler dumps allocator state + `memory_summary`.
  Measured: a 500-step fit on a 20 s target peaks **~8-9 GiB/step**, live
  between steps is flat at 0.64 GiB (no cross-step leak), and memory scales
  ~12 MiB per alignment frame of realized audio. Two failure modes, both
  fixed: (1) **duration runaway** — leo-ens-s5's predicted durations inflated
  17 → 27 s in 3 steps (~13.3 GiB live at 1084 frames) → new `w_dur_total`
  total-frames budget (soft cap 800 frames = 20 s) + hard
  `MAX_ALIGN_FRAMES=880` (22 s ≈ ~10 GiB worst case); the old 1600-frame
  (40 s) cap was too loose to bite before the 13.8 GiB allocator cap did.
  (2) **two jobs in one worker process** — the scheduler placed leo-ens-s2 on
  the instance already running leo-ens-s1 (`active_jobs=1` at dispatch);
  2 × ~6.5 GiB live exceeded the SHARED 13.8 GiB
  `set_per_process_memory_fraction` cap and both OOM'd simultaneously with
  identical allocator stats. `max_concurrent=1` re-asserted via
  `PATCH /v1/models/voice-fit` (verified persisted). Separate worker
  processes are safe to run concurrently (each gets its own 13.8 GiB cap;
  device has 119.55 GiB).

Fitted packs persist at `/home/darren/voice-models/<name>.pt` (+ `<name>.json`
metrics sidecar), deploy-independent, same convention as `rvc-models`. Stock
seed voices cache under `/home/darren/voice-models/voices/stock/`.

## mode=finetune (full single-speaker fine-tune)

The same adapter also serves `params.mode="finetune"` (default `"fit"` is
unchanged): `voxsmith.finetune.finetune_full` trains decoder + predictor
(69.5M params) + the (1,256) style vector on the job's targets, freezing
BERT/text-encoder. Loss = multi-res STFT log-mag L1 + ECAPA cosine +
self-sim + duration-rate + log-RMS gain anchor (`w_gain=1.0`; without it the
decoder drifts ~12 dB quiet — ECAPA and log-mel don't pin absolute gain).
Straight-through duration alignment means the duration HEAD gets no gradient
(cannot collapse the alignment); the shared predictor trunk trains through
the en → F0Ntrain → decoder path.

- Requires `init_pack` (e.g. the speaker's ensemble pack) and `memory_gb`
  ~20 (peaks ~13 GiB at 20 s targets vs ~9 GiB for style fits; PATCH the
  model entry and restore 10.47 after).
- Artifacts: `<name>.pt` pack, `<name>-model.pth` full Kokoro state_dict,
  `<name>.json` sidecar (includes `recommended_speed`), plus `result.pt`.
- The trained style vector is NOT portable to stock weights (0.59 vs 0.82
  baseline) — ship model + pack together.
- Paris experiment (2026-07-23, jobs 75ea4f11fd9a / 2c258f5e85a0): mean
  ECAPA vs 5 held-out-protocol segments 0.817 (ensemble) → **0.942**
  (fine-tune). Full write-up in voxsmith `out/finetune_notes.md`.
- A permanent `voice-finetune` sibling job type (own model entry, own
  memory_gb) needs `JobTypeToModel` + policy-map entries and a Go
  rebuild/redeploy; the mode param on `voice-fit` is the interim path.

## Client (Mac): `voxsmith arbiter-fit`

```bash
cd /Users/darrenoakey/src/voxsmith
.venv/bin/voxsmith arbiter-fit \
    --targets-dir targets/leo \
    --name leo-laporte \
    --seed-voice auto --steps 400 --eval-every 10
# stages a targets zip, submits the job, polls, downloads voices/leo-laporte.pt
```

To refine an existing fit: `--init-pack leo-laporte` (uses the spark copy) or
`--init-pack ./local.pt` (staged).

## Dedicated venv (spark-only, NOT synced by deploy)

The worker runs in `venvs/voxsmith` (torch cu130 + kokoro + speechbrain +
the voxsmith package), pinned by `worker_cmd` in `local/config.json`.
`venvs/` is machine state; recreate on a fresh spark per
`requirements/voice-fit.txt` (which has the exact commands). Key pins:

- **torch 2.11.0+cu130 / torchaudio 2.11.0+cu130** from
  `https://download.pytorch.org/whl/cu130` — torchaudio 2.12.x has no aarch64
  cu130 wheel, and PyPI's aarch64 torch is CPU-only (useless on the GB10).
- `pip install -e /home/darren/src/voxsmith` for the voxsmith package itself
  (EDITABLE install: rsyncing the source tree is sufficient — no reinstall
  needed — but a RUNNING worker keeps its already-imported modules, so the
  worker must restart to pick up new code; with a continuous queue the
  keep-alive chains jobs in one process indefinitely).
- `arbiter.pth` in site-packages pointing at `/home/darren/src/arbiter/src`.
- The venv's `bin/python*` must be REAL BINARIES, not symlinks (arbiter's
  trusted-worker check resolves symlinks; a symlinked python collapses to the
  system interpreter and loses the venv). If recreated with plain
  `python3.12 -m venv`, copy `/usr/bin/python3.12` over `bin/python*`.
- espeak-ng (system package) for misaki G2P — already installed on spark.

## Config (local/config.json on spark)

Registered via `POST /v1/models` (persists atomically), then memory calibrated:

```json
"voice-fit": {
  "memory_gb": 12, "max_concurrent": 1, "max_instances": 3,
  "keep_alive_seconds": 60, "max_runtime_seconds": 14400,
  "avg_inference_ms": 600000, "load_ms": 4000, "pressure_index": 0.5,
  "worker_cmd": ["/home/darren/src/arbiter/venvs/voxsmith/bin/python",
                 "-m", "arbiter.worker_main", "voice-fit"]
}
```

`max_concurrent` MUST stay 1: two fits in one worker process share one CUDA
allocator and one 13.8 GiB fraction cap (2 × ~8 GiB live → both OOM;
happened 2026-07-23 to leo-ens-s1+s2). `max_instances` 3 is safe — separate
processes each get their own 13.8 GiB cap and the GB10 has 119.55 GiB.

## Calibration (measured 2026-07-22, GB10)

`local/calibration/voice-fit.json` (3-step probe on the Leo targets):

- load: 4.0s, VRAM after load: **0.39 GB** (Kokoro-82M + ECAPA are tiny)
- peak during fitting: **11.35 GB** — the autograd graph through the ISTFTNet
  decoder for ~20 s of 24 kHz audio dominates. `memory_gb` is therefore **12**
  (cap = 12 × 1.15 = 13.8 GB > 11.35 GB peak). Declaring only the 0.39 GB load
  footprint would CUDA-OOM every fit under the cap.
- recommended max_concurrent: 1 (training job; SJF est. 600s default).

Re-measured 2026-07-23 with true per-step peak telemetry (500-step fits on
20 s targets): peak **8-9 GiB/step**, flat 0.64 GiB live between steps, slope
~12 MiB per alignment frame. `MAX_ALIGN_FRAMES=880` (22 s) bounds the worst
step at ~10 GiB. The old "~1 GB live" note conflated post-`empty_cache`
live memory with the true in-step peak.

## Gotchas

- **ffmpeg `silenceremove → loudnorm` in one process deadlocks** on spark's
  ffmpeg 6.1.1/aarch64 (main thread parks in `hrtimer_nanosleep`). voxsmith's
  `preprocess_ref` therefore does loudness normalization in numpy, not ffmpeg.
- Python stdout in workers is block-buffered when redirected — run manual
  probes with `PYTHONUNBUFFERED=1` or the logs appear frozen.
- `pkill -f "voxsmith fit"` over ssh kills the remote shell itself (its own
  cmdline matches); use the `[v]oxsmith` bracket trick.
- Stock packs are 510 or 511 rows depending on the voice; voxsmith pads to
  `(511,1,256)` on save. Fitted packs replicate one vector across all rows
  (length-invariant style → maximal cross-text stability).
