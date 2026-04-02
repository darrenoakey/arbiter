# Dedup And Follower State Machine

This document defines Arbiter's chosen behavior for deduplicated jobs.

Goal:
- one canonical execution per dedup key
- zero stranded `following` jobs
- restart-safe recovery
- deterministic behavior for success, failure, cancellation, and operator actions

## Terms

- **Original**: the current canonical job for a dedup hash.
- **Follower**: a job in `following` state whose `error` field stores `following:<original_job_id>`.
- **Canonical execution**: the one job that is actually allowed to execute for a dedup key at a time.

## Invariants

1. At most one job per dedup key should be runnable at a time.
2. Followers must never become permanently stuck behind a terminal or missing original.
3. If the canonical job fails or is cancelled, exactly one follower is promoted to become the next canonical job.
4. If Arbiter restarts, jobs in `scheduled` or `running` state are requeued.
5. A follower never executes directly unless it has been explicitly promoted to canonical.

## Submission Rules

When a new job arrives for a dedup hash:

- If the canonical job is `completed`:
  - create a new completed job immediately
  - symlink its output dir to the completed canonical job
- If the canonical job is `queued`, `scheduled`, or `running`:
  - create a follower in `following`
- If the canonical job is `failed` or `cancelled`:
  - treat as cache miss
  - create a fresh queued canonical job

## Terminal-State Resolution

### Original completes

Chosen behavior:
- all followers become `completed`
- all followers inherit the original result
- follower output dirs symlink to the original output dir
- stale `following:<id>` markers are cleared

### Original fails

Chosen behavior:
- the failed original stays `failed`
- the oldest follower is promoted to `queued`
- every remaining follower is rebound to `following:<promoted_job_id>`
- the dedup cache is repointed to the promoted follower

Rationale:
- the original requester gets the real failure
- the waiting group still gets exactly one retry path
- we never fan out into N duplicate executions after one failure

### Original is cancelled

Chosen behavior:
- same as original failure
- oldest follower is promoted
- remaining followers are rebound to the promoted follower

Rationale:
- cancelling one canonical attempt should not silently discard every waiting duplicate

## Restart / Crash Recovery

On startup:

1. jobs in `scheduled` or `running` are reset to `queued`
2. dedup cache is rebuilt from queued canonical jobs
3. follower reconciliation runs:
   - if a follower points at a `completed` original, it completes immediately
   - if a follower points at a `failed` or `cancelled` original, one follower is promoted
   - if a follower points at a missing original row, one follower is promoted
   - if a follower points at a `queued`, `scheduled`, or `running` original, it stays `following`

Chosen behavior for restart:
- we do not resume in-process execution
- worst case is replay from the start of the canonical job
- we do not leave anything in `running` after restart

## Operator Actions

### Cancel one follower

Chosen behavior:
- cancel only that follower
- do not affect the canonical job
- do not affect sibling followers

### Cancel one canonical queued/scheduled job

Chosen behavior:
- cancel the canonical job
- promote exactly one follower
- rebind remaining followers to the promoted job

### Cancel one canonical running job

Chosen behavior:
- signal cancellation to the worker
- when the running job reaches terminal `cancelled`, normal follower promotion rules apply

### Clear model queue

Chosen behavior:
- cancel model jobs already in `queued`
- cancel model jobs in `following`

Rationale:
- queue clear is an operator intent to drop pending work, including dedup waiters

### Hard-kill or remove a model

Chosen behavior:
- cancel queued jobs for that model
- cancel follower jobs for that model
- fail active scheduled/running jobs for that model

Rationale:
- operator-forced teardown should not requeue hidden work behind followers

## Edge Cases

### Original row missing but followers remain

Chosen behavior:
- treat as terminal loss of canonical state
- promote the oldest follower
- rebind remaining followers to it

### Multiple consecutive failures

Chosen behavior:
- each failed canonical attempt promotes the next oldest follower
- the waiting group advances one canonical attempt at a time
- duplicate fan-out is never allowed

### Completed original with missing output symlink target

Chosen behavior:
- followers are still marked `completed`
- Arbiter attempts to create the symlink
- completion metadata is still copied even if the symlink already exists or must be recreated

## Non-Goals

- mid-inference resume from partial model state
- preserving worker subprocess state across restart
- per-request retry budgets distinct from dedup group behavior

## Practical Outcome

For 80 identical requests:

- one canonical job executes
- 79 jobs wait as followers
- if that canonical job completes, all 79 complete instantly
- if that canonical job fails, the next oldest follower becomes the only canonical retry
- if Arbiter restarts, the canonical job is requeued and the 79 followers remain attached to it
