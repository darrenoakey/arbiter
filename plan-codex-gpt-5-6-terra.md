# Plan: Keep Qwen Remote While Spark Runs LTX

## Objective

Ensure `llm:qwen3.6-35b` dispatches to `boringstack`, then `darrens-mbp`, using Spark only when both remotes are unsuitable and Spark has safe capacity—without killing, cancelling, or restarting active `ltx2-dev-denoise1` work.

## Current evidence

At 2026-07-26 UTC:

- Live config is correct: `placements=["boringstack","darrens-mbp","spark"]`, `remote_enabled=true`, `max_instances=1`, `max_concurrent=1`.
- Both remote functional probes completed with `OK`: boringstack in 0.58s, `darrens-mbp` in 1.52s. Health-only checks are insufficient.
- `ltx2-dev-denoise1` remained active at 40.04GB with 102 queued.
- Qwen’s backlog subsequently drained remotely, but recent jobs waited up to ~78 minutes.
- Incident job `c9800edbf098` failed over from boringstack at 23:10:21, durably acquired `excluded_hosts=[boringstack]`, and could not fit on Spark (`need 59.1GB, only 34.0GB free`). Although boringstack recovered at 23:10:24, the exclusion remained. It progressed only when `darrens-mbp` recovered at 00:16:42.
- Source confirms per-model head-of-line selection: `scheduler.go:1631-1688`, `store.go:524-539`.
- `proc.go:1408-1418`, `1936-2008`, and `2021-2075` incorrectly include remote instances in Spark reclaim/eviction calculations. Logs consequently show remote Qwen being “VRAM”-evicted despite holding zero Spark VRAM.
- Pressure accounting is local memory-bandwidth accounting (`config.go:36`), yet remote Qwen currently reserves pressure in `scheduler.go:1709-1714`.

## Non-goals and invariants

- Never cancel, clear, reload, or kill Qwen/LTX jobs or workers.
- Never use `remote_enabled:false`, `max_instances:0`, `DEPLOY_FORCE`, or reduce Qwen’s declared memory to force Spark placement.
- Never restart Arbiter while `active_jobs>0`.
- Do not modify Spark’s deploy-target source or manually edit SQLite/config state.
- Preserve `ltx2-dev-denoise1` job identity and terminal success.
- No placement/config change is currently required.

## Phase 0 — Read-only diagnosis

Run and archive:

```bash
curl -fsS http://10.0.0.254:8400/v1/models/llm%3Aqwen3.6-35b | jq .
curl -fsS http://10.0.0.254:8400/v1/ps |
  jq '{active_jobs,queue,remote_hosts,models:[.models[]|select(.id=="llm:qwen3.6-35b" or .id=="ltx2-dev-denoise1")]}'
curl -fsS 'http://10.0.0.254:8400/v1/jobs?state=queued&model=llm%3Aqwen3.6-35b&limit=100' | jq .
```

Functional remote probes:

```bash
curl -fsS --max-time 90 http://10.0.0.42:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3.6-35B-A3B-4bit","messages":[{"role":"user","content":"Reply only OK"}],"max_tokens":1,"stream":false}'

curl -fsS --max-time 90 http://10.0.0.44:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3.6-35B-A3B-4bit","messages":[{"role":"user","content":"Reply only OK"}],"max_tokens":1,"stream":false}'
```

Inspect the authoritative scheduler log:

```bash
ssh -o BatchMode=yes darren@10.0.0.254 \
  'log=$HOME/local/auto/output/logs/arbiter/2026/07/arbiter_260725_043633.log;
   grep -E "qwen3.6-35b|host liveness|scheduler.requeue|queue-priority eviction|vram watchdog|circuit-breaker|swap_held" "$log" | tail -n 2000'
```

| Observation | Cause/action |
|---|---|
| Wrong placement order or `remote_enabled=false` | Configuration drift. Do not patch blindly; establish the persisted `local/config.json` owner/reconciliation path first. |
| Host “reachable” but one-token completion fails | Repair that host’s Nativ service; do not alter Arbiter or Spark. |
| All remote instances at `max_concurrent` | Normal bounded queue; confirm dispatch immediately after a slot frees. Increase concurrency only after host load testing. |
| Oldest job excludes recovered preferred host | Durable-exclusion recovery bug. |
| Oldest job unplaceable while newer jobs could use remote | Per-model head-of-line blocking bug. |
| Remote instance appears in Spark reclaim/eviction logs | Local/remote VRAM accounting bug. |
| Qwen held solely by pressure while a remote is available | Remote job incorrectly consuming local pressure budget. |
| Inference/load breaker active | Classify the underlying failure; do not disable remote routing. |

## Phase 1 — Immediate no-restart remediation

If Qwen is already draining remotely, make no runtime mutation.

If `darrens-mbp` is absent, restore the existing Nativ application on this Mac and require the functional probe above to pass. The host monitor will automatically mark it recovered and wake scheduling. If boringstack is healthy, new jobs should prefer it.

If both remotes pass inference but a pre-fix oldest job remains stranded by `excluded_hosts`, there is no safe public API to clear that exclusion. Do not edit SQLite or cancel the job. Proceed to the code fix and safe drain deployment.

## Phase 2 — Permanent implementation

Work only in a dedicated worktree. The canonical checkout is currently dirty and has neither GitHub CI nor `greenline.toml`; preserve those owner changes and establish Greenline before merging/deploying.

Change only inspected scheduler components:

1. `cmd/arbiter/store.go`
   - Add a transactional operation that removes a recovered host from queued jobs’ durable `excluded_hosts`.
   - Add ordered/paged queued-job retrieval for a model so scheduling can inspect beyond an unplaceable oldest job without unbounded memory use.
   - Preserve FCFS among jobs that are currently placeable.
2. `cmd/arbiter/host_liveness.go`
   - On confirmed `RECOVERED`, clear that host from queued exclusions, log the affected count, then wake the scheduler.
   - Keep absence-triggered active cancellation/failover unchanged.
3. `cmd/arbiter/scheduler.go`
   - Replace single-job per-model selection with “oldest currently placeable candidate.”
   - If an excluded/absent/local-VRAM-blocked head job cannot run, allow a later fresh Qwen job to use a healthy remote.
   - Record zero `currentPressure` for a selected remote instance.
   - Bypass the local pressure gate only when the selected candidate will actually run remotely.
   - Emit structured `scheduler.job_placement_blocked` diagnostics containing job ID, exclusions, placement order, per-host blocker, queue depth, and local free/reclaimable GB.
4. `cmd/arbiter/proc.go`
   - Exclude `inst.isRemote()` from `reclaimableIdleGBLocked`, `EvictForGB`, `EvictForGBWithQueueInfo`, and `EvictIdleNoQueueModels`.
   - Keep intentional remote keepalive unloading separate from Spark VRAM relief.
   - Preserve ordered placement in `PickInstanceForJobWithReason` (`proc.go:1241-1295`).

No `local/config.json` change is required. `PATCH /v1/models` does not support `placements` (`api.go:76-99`), so do not prescribe a runtime placement patch that cannot reconcile persistence.

## Phase 3 — Tests

Add real-store/real-manager regression coverage to existing suites:

- Reproduce the captured incident: oldest job excludes boringstack, MBP absent, Spark pinned by active LTX; verify a later Qwen job dispatches to boringstack rather than waiting.
- Recover boringstack and verify the oldest job’s exclusion clears and it becomes dispatchable.
- Verify ordering: boringstack `preferred`; MBP `spill`; Spark `fallback`.
- Verify Spark fallback is rejected when 59.1GB cannot fit beside active LTX.
- Verify remote instances contribute zero reclaimable Spark GB and are never candidates for Spark VRAM or queue-priority eviction.
- Verify remote dispatch contributes zero pressure while local dispatch retains configured pressure.
- Verify queue ordering remains FCFS among equally placeable jobs.
- Retain existing transparent-failover, terminal-state, circuit-breaker, and placement tests.

Run the complete gate, with required live forwards established:

```bash
go test ./cmd/arbiter/ -count=1
.venv/bin/python -m pytest tests/ -m 'not calibration'
```

All tests and warnings must pass.

## Phase 4 — Safe deployment and persistence

The Go scheduler cannot be hot-swapped. Deployment therefore requires a restart, but only after a proven job-safe drain.

Before draining, capture the active denoise job ID and state:

```bash
curl -fsS 'http://10.0.0.254:8400/v1/jobs?state=running&model=ltx2-dev-denoise1&limit=10' | jq .
```

Enter drain mode:

```bash
curl -fsS -X POST http://10.0.0.254:8400/v1/drain \
  -H 'Content-Type: application/json' -d '{}'
```

Poll `/v1/ps` until `draining=true` and `active_jobs=0`. Drain stops new work but lets the active denoise job finish. If the wait must be abandoned, resume safely:

```bash
curl -fsS -X POST http://10.0.0.254:8400/v1/drain \
  -H 'Content-Type: application/json' -d '{"resume":true}'
```

Deploy only through Greenline’s serialized full check/deploy gate. Never rely on `deploy-to-spark.sh`’s timeout path, because it currently proceeds after timeout. Recheck `active_jobs==0` immediately before the stop/restart step. Queued denoise jobs remain persisted and resume afterward.

Rollback is the prior Greenline-deployed commit, using the same drain requirement. No config rollback is necessary.

## Phase 5 — Live acceptance

Submit two unique, one-token async Qwen canaries close together. Poll every 0.5s. Acceptance:

- First starts within five seconds when boringstack has capacity.
- Event log records `host=boringstack`, `placement_reason=preferred`.
- When boringstack’s single Arbiter slot is occupied, the second records `host=darrens-mbp`, `placement_reason=spill`.
- Neither produces a Spark placement while remote capacity exists.
- `ltx2-dev-denoise1` remains active or completes successfully; no failed/cancelled denoise job appears.
- Qwen queued count returns to zero.
- No `queue-priority eviction` or `vram watchdog: evicting` names a remote instance.

Verify placement from the authoritative event log:

```bash
ssh -o BatchMode=yes darren@10.0.0.254 \
  'tail -n 5000 /mnt/arbiter-store/output/logs/arbiter-$(date +%F).jsonl |
   jq -c "select(.event==\"model.placed\" and .model_id==\"llm:qwen3.6-35b\")"'
```

Monitor `/v1/ps`, Qwen oldest wait, `scheduler.job_placement_blocked`, host recovery, circuit-breaker, and remote-eviction events for at least one denoise cycle. Spark fallback is proven by automated tests; do not induce a production remote outage merely to demonstrate it.
