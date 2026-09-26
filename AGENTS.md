<!-- >>> greenline >>> -->
## Greenline gate — how merges work here

This repo is gated by **greenline**. Read `docs/greenline.md` and `docs/DOCTRINE.md`
before writing code or tests.

**Invariants (never violate):**
- `master` == what prod runs == green, always.
- The canonical checkout is pristine — never edit it by hand.
- All work happens in worktrees branched from last-green.
- Every merge goes through the serialized gate: full `check` + real `deploy`.

**Your workflow:**
1. `greenline worktree <name>` — get a worktree at `/Users/darrenoakey/src/.greenline-worktrees/arbiter/<name>` on branch `gl/<name>`.
2. Do your work there. Co-design tests + code per docs/DOCTRINE.md (parallel-safe, namespaced, no global-state assertions, OS-assigned ports; never mock other services — make real calls fast with a content-addressed record/replay cache).
3. Commit in your worktree. Then `greenline submit` (from that worktree).
4. The gate squash-merges, runs `./run check`, fast-forwards `master`, runs `./run deploy`, and publishes. It rolls back prod automatically if deploy fails.
5. On success: `greenline done` to remove your worktree + branch.

**Long gate waits:** `greenline-wait.sh` must always be run detached (`async`)
with a `sleep` — never hold a turn polling inline. Submit from the canonical
repo path (`--repo /Users/darrenoakey/src/arbiter <branch>`) after committing
in the worktree.

**Never** commit or push on `master` — hooks hard-lock it (reference-transaction
cannot be bypassed with `--no-verify`; pre-commit/pre-push refuse too). Never edit the
canonical checkout. If the gate reports a conflict, rebase your worktree on
`master` and resubmit. If commits somehow reached `master` outside the
gate (legacy workflow, hotfix), run `greenline adopt` to gate them in place —
greenline never discards commits on `master`.

Diagnose with `greenline status` and `greenline doctor` (`--fix` to reconcile).
<!-- <<< greenline <<< -->

## Production job-store maintenance

- The production `jobs` table is tens of gigabytes. Background retention work
  must select completed candidates through `idx_jobs_completed_stats`, delete
  at most 100 rows per scheduled pass, and then yield. Both an unindexed scan
  and a large delete batch interact with the Store's writer-preferring
  `RWMutex` and can starve ordinary `GET /v1/jobs/{id}` reads even while
  `/health` and the memory-backed `/ps` endpoint remain fast.
- Do not build a new jobs-table index during `NewStore` or daemon startup. Plan
  large index builds as explicit maintenance after active work drains.
- All-history dashboard queries (`CountByStateGrouped` and
  `CompletedJobStatsGrouped`) must not hold the operational Store `RWMutex`.
  SQLite WAL and the read pool provide isolation; holding the Go read lock lets
  one queued writer starve every new primary-key job lookup for the full scan.
- A stopped Arbiter can retain port 8400 briefly while a thread finishes
  uninterruptible database/filesystem I/O. The Spark deploy must wait for the
  old listener to disappear after `auto stop`; auto's ten-second reclaim window
  is shorter than this observed kernel cleanup and an immediate start can fail.
- MiniMax H3 means the local GPU adapter. The cloud client must not register
  and must not have `config/spark/minimax-h3.model.json`. Do not restore
  `model_id="minimax-h3"`; a cloud restore over a GPU module made workers
  started as `minimax-h3-local` die with `Unknown model`. Keep
  `minimax_h3_local.py` and `minimax_fast_h3.py` as separate registrations,
  keep `config/spark/minimax-h3-local.model.json` and
  `minimax-fast-h3.model.json`, and merge local config without dropping a
  live `worker_cmd`. Deploy must drop any stale `minimax-h3` config key.
  Registry tests must assert the cloud id is absent and local plus FastH3
  stay registered. FastH3 is 4-step only (`video-generate-fast-h3` /
  `minimax-fast-h3`); it reuses the H3 NVFP4 text encoder and the
  `minimax-h3` venv. Preview weights are T2VA-only and require the native
  5-second, 1344x768 operating point. The adapter rejects first- and
  last-keyframe parameters. Its four integer clock points
  `[999, 749, 500, 250]` must be converted to separate shifted sigma schedules:
  video shift 12 and audio shift 3, each with one terminal zero. Diffusers
  accepts explicit sigmas verbatim and does not apply the shift. Audio-in is
  unsupported; callers mux their own soundtrack.

## Live model registration

- `POST /v1/models` is the no-restart path. Built-in adapters register as-is.
  A remote-only `llm:*` model MUST send `placements` (e.g. `["boringstack"]`).
  Omitting it defaults the model to local spark and worker policy returns 400
  (`no trusted built-in adapter`). PATCH accepts the same field and heals
  missing remotes. New local adapter *code* still needs a deploy; after that,
  reload that one model. Do not restart Arbiter just to add a remote LLM.

## Capabilities

- `GET /v1/capabilities` is the version-negotiation surface. It reports the
  served API major version, live `JobTypeToModel` keys, and live LLM alias
  targets. Job types and aliases version independently of `/v1`; additive-minor
  only. An alias rename requires an overlap window. Register a new job type in
  `JobTypeToModel` — do not hard-code the list in the handler.

## Completed GPU result files

- `GET /v1/jobs/{id}` synthesizes `result.result_path` from adapter `result.file`
  (basename only) when present, otherwise `result.{format}` as `result.<format>`.
  LTX 2.5 encode writes `encoded.pt`, not `result.pt`. A completed job with no
  inlined `data` is usually the poller looking at the wrong filename.

## GPU idle watchdog (false-kill contract)

- The watchdog kills a worker only when GPU util stays low AND the worker has a
  pending request it never answered. A worker that already returned (e.g.
  inference done, `result.mp4` written) must never be killed because later
  bookkeeping blocked — the scheduler releases the dispatch slot before
  share/DB completion work, and the watchdog skips workers with no pending
  request and ignores samples stalled behind a hung `nvidia-smi` call
  (2026-09-22 false kill of finished job 633b27a52464; fix on master).
- When adding post-inference completion work, keep it off the dispatch slot's
  critical path — an 18-minute blocked bundle once let the idle window elapse
  in a single tick.

## LTX 2.5 conditioning changes

- Do not `import ltx_pipelines.utils.helpers` from an adapter unit test or from
  the default denoise path. Importing `ltx_pipelines.utils` executes
  `utils/__init__.py`, which imports blocks and then torchaudio. The laptop
  gate has torch but not torchaudio. Construct `VideoGeneratedKeyframeSlots`
  from `ltx_core` directly, and lock interior positions to the
  `torch.linspace(...)[1:-1]` line in `helpers.py`.
- `generated_keyframes` defaults to 0. A positive count is interior slots only;
  it does not replace endpoint conditioning. Do not deploy this repo while an
  `ltx25-denoise1` job is running — `./run deploy` is drain-gated and the
  greenline release dies at 600s if the drain cannot finish.

## Greenline submit from agents

- Never hold a turn on `greenline submit`. Start it detached and sleep. The
  10-minute turn wall kills the release mid-check, leaves a dead lock holder,
  and does not deploy. Check output is fully buffered until the process exits,
  so a killed log can look like a hang after pytest even when Go tests are fine.


