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
- MiniMax H3 cloud and local are different adapters. Restoring the cloud
  client (`minimax_h3.py`, `model_id="minimax-h3"`) over the local GPU
  module made workers started as `minimax-h3-local` die with `Unknown model`
  and trip the load circuit-breaker. Keep `minimax_h3.py` and
  `minimax_h3_local.py` as separate registrations, keep both
  `config/spark/minimax-h3*.model.json` files, and merge the local config
  without dropping a live `worker_cmd`. Registry tests must assert both ids
  stay registered after a cloud restore.

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


