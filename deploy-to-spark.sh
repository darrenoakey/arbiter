#!/bin/bash
# Deploy arbiter from this Mac to spark.
# This is the ONLY way to push code to spark — do not edit files in place on spark.
#
# Steps:
#   1. Smoke-test the Python adapter package on the target venv (spark)
#   2. Cross-compile binaries for Linux ARM64
#   3. Stop arbiter on spark
#   4. Sync Python adapters + binaries
#   5. Start arbiter
#   6. Verify health
#
# Validation lives in the greenline gate (./run check), which runs the FULL
# Go + Python suites on this exact tree immediately before the deploy; this
# script must not re-run them (measured 2026-09-20: a duplicate full Go suite
# here cost ~45s of every release for zero added coverage — the canonical
# checkout it runs from is the tree the gate just validated). The cross-
# compile below still fails on any build breakage before prod is touched.

set -euo pipefail

SPARK=${SPARK:-darren@10.0.0.254}
REMOTE=/home/darren/src/arbiter

cd "$(dirname "$0")"

ARBITER_URL="http://10.0.0.254:8400"
DEPLOY_DRAIN_TIMEOUT="${DEPLOY_DRAIN_TIMEOUT:-120}"
DEPLOY_DRAIN_LEASE="${DEPLOY_DRAIN_LEASE:-300}"
DRAIN_REQUESTED=0
drain_deadline=0

# Idempotent-redeploy fast path. Greenline runs this deploy twice per release
# (candidate deploy, then a publish deploy of the merged tree — for a
# fast-forward release byte-identical to the candidate). Redeploying the exact
# same tree reproduces every artifact the full path produces, so the repeat
# deploy collapses to a drift-heal + health check instead of a second
# drain/build/bounce (measured 2026-09-18: the repeat deploy's 120s drain
# deadline alone was most of a 256.6s release vs the 180s target).
#
# The key covers everything the full path ships or builds from: HEAD, the full
# git worktree state, and a content hash of the directories/files rsync'd or
# scp'd to spark. It is recorded on spark ONLY after a full deploy passes its
# health check; a missing or mismatched key always falls through to the full
# path, so this can never skip a deploy that would have changed production.
deploy_key() {
    {
        echo "deploy-key-v1"
        git rev-parse HEAD 2>/dev/null || echo "no-git"
        git status --porcelain=v1 2>/dev/null || true
        git diff HEAD 2>/dev/null || true
        find src/arbiter config/spark scripts/spark-host/arbiter-firewall-guard -type f 2>/dev/null \
            | LC_ALL=C sort | while IFS= read -r f; do md5 -q "$f"; done
    } | md5 -q
}

drain_request() { # renews the lease; idempotent
    curl -s --max-time 5 -X POST "$ARBITER_URL/v1/drain" \
        -H 'Content-Type: application/json' \
        -d "{\"lease_seconds\":${DEPLOY_DRAIN_LEASE}}" >/dev/null 2>&1
}

resume_if_aborted() {
    rc=$?
    if [ "$DRAIN_REQUESTED" = "1" ]; then
        echo "==> Deploy ended before the bounce (rc=$rc) — resuming arbiter dispatch"
        curl -s --max-time 5 -X POST "$ARBITER_URL/v1/drain" \
            -H 'Content-Type: application/json' -d '{"resume":true}' >/dev/null 2>&1 || true
    fi
}

# Graceful drain: ask the running arbiter to stop starting NEW jobs and let
# in-flight work finish before we bounce it, so a redeploy never kills a
# running job (e.g. a 10-min ltx2 denoise). Tolerant of an older binary that
# lacks /v1/drain. Bounded wait; override with DEPLOY_FORCE=1 to skip, or
# DEPLOY_DRAIN_TIMEOUT to change the ceiling.
#
# The default ceiling is deliberately SHORT (120s). This deploy runs inside a
# greenline release with a hard, non-negotiable 600-second end-to-end deadline;
# an ltx25 denoise chunk runs 20-45 minutes, so waiting it out guarantees the
# release is SIGTERM-killed mid-deploy (seen 2026-09-14: killed release left
# arbiter draining and wedged the whole fleet). Short jobs finish inside the
# window, and anything still in flight is REQUEUED by the shutdown path
# (scheduler.shouldRequeueForShutdown) rather than lost — it simply reruns
# after the bounce. Raise it only for a manual deploy outside the gate.
#
# The drain is requested UP FRONT — before the local build/test phase — so the
# in-flight window overlaps it instead of serializing after it (measured
# 2026-09-15: a full drain-window wait cost 120s of a 203.8s release). The
# deadline is counted from this request, so the time spent building/testing
# consumes the same 120s window: short jobs get no less protection, and a bit
# more wall-clock to finish. If a build/test step fails, the EXIT trap below
# resumes dispatch immediately; worst case the lease lapses after
# DEPLOY_DRAIN_LEASE seconds and arbiter resumes dispatch on its own.
#
# The drain is LEASED and renewed on every poll. A deploy killed here (gate
# deadline, ^C, crash) therefore cannot leave the fleet wedged: the lease
# lapses within DEPLOY_DRAIN_LEASE seconds and arbiter resumes dispatch on its
# own. The trap below is the fast path for signals we can actually catch; the
# server-side lease covers SIGKILL, which we cannot.
DEPLOY_KEY="$(deploy_key)"
DEPLOY_KEY_PATH="$REMOTE/local/.deploy-key"

if [ "${DEPLOY_FORCE:-0}" != "1" ]; then
    remote_key="$(ssh -o ConnectTimeout=5 "$SPARK" "cat '$DEPLOY_KEY_PATH' 2>/dev/null" || true)"
    if [ -n "$remote_key" ] && [ "$remote_key" = "$DEPLOY_KEY" ]; then
        echo "==> Spark already runs this exact tree ($DEPLOY_KEY) — skipping drain, build, sync, and bounce"
        # The one state heal the full path performs that a no-op deploy must
        # keep: a symlinked .venv python silently loses venv activation (see
        # the full path's comment below).
        echo "==> Ensuring .venv python is a real binary (not a symlink)..."
        ssh "$SPARK" "test -L '$REMOTE/.venv/bin/python' && cp --remove-destination /usr/bin/python3.12 '$REMOTE/.venv/bin/python' '$REMOTE/.venv/bin/python3' '$REMOTE/.venv/bin/python3.12' && echo '    converted .venv python symlinks to real binary copies' || echo '    .venv python already a real binary'"
        echo "==> Waiting for health check..."
        fast_healthy=0
        for _ in $(seq 1 20); do
            if curl -s --max-time 5 http://10.0.0.254:8400/v1/health 2>/dev/null | grep -q '"status":"ok"'; then
                fast_healthy=1
                break
            fi
            sleep 1
        done
        if [ "$fast_healthy" = "1" ]; then
            echo "    healthy"
            curl -s --max-time 5 http://10.0.0.254:8400/v1/health
            echo ""
            exit 0
        fi
        echo "    health check failed — falling through to a full redeploy"
    fi
fi

if [ "${DEPLOY_FORCE:-0}" = "1" ]; then
    echo "==> DEPLOY_FORCE=1 — skipping graceful drain (may kill in-flight jobs)"
elif drain_request; then
    DRAIN_REQUESTED=1
    trap resume_if_aborted EXIT INT TERM HUP
    drain_deadline=$(( $(date +%s) + DEPLOY_DRAIN_TIMEOUT ))
    echo "==> Drain requested (no new jobs) — in-flight work winds down while we build/test locally"
else
    echo "==> No /v1/drain on running arbiter (older binary) — proceeding without drain"
fi

echo "==> Smoke-testing Python adapter package on spark..."
# This is the exact import sequence that worker_main.py does on startup.
# If this fails, the deploy is aborted BEFORE we touch the running arbiter —
# protecting any in-flight queued work from circuit-breaker cancellation.
ssh "$SPARK" "mkdir -p /tmp/arbiter-smoke-test/arbiter/adapters"
rsync -az --delete src/arbiter/adapters/ "$SPARK:/tmp/arbiter-smoke-test/arbiter/adapters/"
if ! ssh "$SPARK" "cd /tmp/arbiter-smoke-test && PYTHONPATH=/tmp/arbiter-smoke-test:/home/darren/src/arbiter/src /home/darren/src/arbiter/.venv/bin/python -c 'from arbiter.adapters import registry; print(\"adapters loaded OK\")'" 2>&1; then
    echo "    FAILED — adapter package has import errors. Deploy aborted."
    echo "    Fix the Python imports locally and re-run deploy."
    exit 1
fi
echo "    python smoke test passed"

# Install while the current service is still healthy. A package-index failure
# must abort before drain/stop rather than create avoidable production downtime.
echo "==> Installing MiniMax H3 adapter dependency..."
ssh "$SPARK" "$REMOTE/.venv/bin/python -m pip install -q 'daz-secrets>=0.1.0a1'"

echo "==> Cross-compiling binaries..."
GOOS=linux GOARCH=arm64 go build -o arbiter-linux-arm64 ./cmd/arbiter/
GOOS=linux GOARCH=arm64 go build -o llm-worker-linux-arm64 ./cmd/llm-worker/
GOOS=linux GOARCH=arm64 go build -o vllm-chat-worker-linux-arm64 ./cmd/vllm-chat-worker/
echo "    $(md5 -q arbiter-linux-arm64 2>/dev/null || md5sum arbiter-linux-arm64 | awk '{print $1}') arbiter"
echo "    $(md5 -q llm-worker-linux-arm64 2>/dev/null || md5sum llm-worker-linux-arm64 | awk '{print $1}') llm-worker"
echo "    $(md5 -q vllm-chat-worker-linux-arm64 2>/dev/null || md5sum vllm-chat-worker-linux-arm64 | awk '{print $1}') vllm-chat-worker"

# Rejoin the drain requested before the build/test phase. Patience is
# PER-JOB, measured from each job's own start (scripts/drain_wait.py): a job
# is protected until it has run one full drain window, so a long job that
# already outlived the window costs no further wait — it is requeued by the
# shutdown path either way (measured 2026-09-20: two multi-minute training
# jobs burned ~80s of a 209.6s release waiting out a wall-clock deadline that
# was never going to let them finish). Short jobs keep the same or better
# protection they always had; the overall deadline below still bounds the
# total wait.
if [ "$DRAIN_REQUESTED" = "1" ]; then
    drain_request # renew the lease across the build/test gap
    echo "==> Draining arbiter (deadline ${DEPLOY_DRAIN_TIMEOUT}s from drain request; per-job patience ${DEPLOY_DRAIN_TIMEOUT}s from each job's start)..."
    while :; do
        verdict="$(curl -s --max-time 5 "$ARBITER_URL/v1/ps" 2>/dev/null \
            | python3 scripts/drain_wait.py "${DEPLOY_DRAIN_TIMEOUT}" 2>/dev/null || printf '0\t0')"
        active="${verdict%%$'\t'*}"
        waitable="${verdict##*$'\t'}"
        if [ "${active:-0}" = "0" ]; then
            echo "    drained — 0 in-flight jobs"
            break
        fi
        if [ "$waitable" != "1" ]; then
            echo "    long-job cut — every in-flight job already ran past the ${DEPLOY_DRAIN_TIMEOUT}s patience window; proceeding (shutdown requeues them)"
            break
        fi
        if [ "$(date +%s)" -ge "$drain_deadline" ]; then
            echo "    WARNING: still ${active} in-flight after ${DEPLOY_DRAIN_TIMEOUT}s — proceeding anyway"
            break
        fi
        echo "    ${active} job(s) still in flight; waiting..."
        sleep 10
        drain_request # renew the lease while we keep waiting
    done
fi

echo "==> Stopping arbiter on spark..."
# Past this point the old process is going away, so an abort no longer needs a
# resume call: the restarted arbiter starts undrained, and a drain request that
# never got renewed expires server-side anyway.
DRAIN_REQUESTED=0
trap - EXIT INT TERM HUP
ssh "$SPARK" "/home/darren/local/auto/run stop arbiter" 2>&1 | tail -1 || true

# A terminated arbiter can remain in uninterruptible SQLite/filesystem I/O for
# longer than auto's ten-second port-reclaim window. Starting immediately then
# fails even though the old listener has already received SIGKILL. Wait for the
# kernel to finish releasing that exact listening socket before replacing the
# binary. Do not kill unrelated listeners. If auto loses the stop race and
# respawns /home/darren/src/arbiter/arbiter-go, stop that service again — a
# healthy respawn will otherwise hold port 8400 until this wait fails.
echo "==> Waiting for the stopped arbiter to release port 8400..."
if ! ssh "$SPARK" 'deadline=$(( $(date +%s) + 300 ))
restopped=""
while lsof -nP -iTCP:8400 -sTCP:LISTEN >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "    FAILED — terminated arbiter still owns port 8400 after 300s"
        lsof -nP -iTCP:8400 -sTCP:LISTEN 2>&1 || true
        exit 1
    fi
    # auto can lose the stop race and respawn arbiter while this wait runs.
    # A healthy respawn never exits, so waiting the full 300s fails the
    # release (2026-09-22 gl/stage1-guiding-keyframes). Stop that service
    # again. A dying uninterruptible process is left alone.
    pid=$(lsof -t -nP -iTCP:8400 -sTCP:LISTEN 2>/dev/null | head -1 || true)
    if [ -n "$pid" ] && [ "$pid" != "$restopped" ]; then
        exe=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
        state=$(sed -n "s/.*) //p" "/proc/$pid/stat" 2>/dev/null | awk "{print \$1}")
        if [ "$exe" = "/home/darren/src/arbiter/arbiter-go" ] && [ "$state" != "D" ]; then
            echo "    respawned arbiter pid $pid state ${state:-unknown} still listening — stopping it again"
            /home/darren/local/auto/run stop arbiter >/dev/null 2>&1 || true
            kill -TERM "$pid" >/dev/null 2>&1 || true
            restopped=$pid
        fi
    fi
    sleep 1
done'; then
    exit 1
fi
echo "    port 8400 released"

echo "==> Ensuring .venv python is a real binary (not a symlink)..."
# resolveTrustedPythonExecutable collapses the interpreter via EvalSymlinks to
# block symlink-swap TOCTOU. If .venv/bin/python is a symlink to
# /usr/bin/python3.12, that collapse returns the SYSTEM python, which loses
# venv activation (pyvenv.cfg lookup) and every site-packages dependency
# (torch, diffusers, ltx_core, …). The sanctioned per-adapter venvs avoid
# this because `venv` copied a real binary into them; the main .venv was
# created with symlinks. Replace the symlink chain with binary copies (the
# same state a `python -m venv --copies` produces) so EvalSymlinks returns
# .venv/bin/python itself and venv activation survives. Spark-only state.
ssh "$SPARK" "test -L '$REMOTE/.venv/bin/python' && cp --remove-destination /usr/bin/python3.12 '$REMOTE/.venv/bin/python' '$REMOTE/.venv/bin/python3' '$REMOTE/.venv/bin/python3.12' && echo '    converted .venv python symlinks to real binary copies' || echo '    .venv python already a real binary'"

echo "==> Syncing Python adapters..."
rsync -az --delete src/arbiter/ "$SPARK:$REMOTE/src/arbiter/"

echo "==> Installing model configs..."
ssh "$SPARK" "mkdir -p '$REMOTE/config/spark'"
rsync -az config/spark/ "$SPARK:$REMOTE/config/spark/"
ssh "$SPARK" "$REMOTE/.venv/bin/python -c 'import glob, json, os, pathlib; root=pathlib.Path(\"$REMOTE\"); path=root/\"local/config.json\"; data=json.loads(path.read_text()); models=data.setdefault(\"models\", {}); [models.update({pathlib.Path(f).name.replace(\".model.json\", \"\"): json.loads(pathlib.Path(f).read_text())}) for f in glob.glob(str(root/\"config/spark/*.model.json\")) if \"minimax-h3-local\" not in f]; existing=models.get(\"minimax-h3-local\") or {}; local=json.loads((root/\"config/spark/minimax-h3-local.model.json\").read_text()); worker=existing.get(\"worker_cmd\"); local.update({\"worker_cmd\": worker} if worker else {}); models[\"minimax-h3-local\"]=local; temporary=path.with_name(\".config.spark-models.tmp\"); handle=temporary.open(\"w\"); json.dump(data, handle, indent=2); handle.write(\"\\n\"); handle.flush(); os.fsync(handle.fileno()); handle.close(); os.replace(temporary, path); descriptor=os.open(path.parent, os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)'"

echo "==> Uploading binaries..."
# Upload to a temporary name, then rename into place. A plain scp onto the live
# path fails with ETXTBSY — reported by scp as `dest open "...": Failure` —
# whenever any process still holds the old binary as its executable text. That
# includes a stopped-but-unreaped process whose threads are wedged in
# uninterruptible I/O (seen 2026-08-04: arbiter-go left as a zombie with live
# threads stuck in CIFS path lookup after the //10.0.0.46/arbiter-data mount
# wedged). rename(2) never opens the destination, so it swaps the directory
# entry regardless and the old inode stays alive for whatever still references
# it. Without this, the deploy stops arbiter, fails to upload, and cannot even
# roll back — leaving production down until the host is power-cycled.
scp -q arbiter-linux-arm64 "$SPARK:$REMOTE/arbiter-go.new"
scp -q llm-worker-linux-arm64 "$SPARK:$REMOTE/llm-worker.new"
scp -q vllm-chat-worker-linux-arm64 "$SPARK:$REMOTE/vllm-chat-worker.new"
ssh "$SPARK" "set -e
chmod +x $REMOTE/arbiter-go.new $REMOTE/llm-worker.new $REMOTE/vllm-chat-worker.new
mv -f $REMOTE/arbiter-go.new $REMOTE/arbiter-go
mv -f $REMOTE/llm-worker.new $REMOTE/llm-worker
mv -f $REMOTE/vllm-chat-worker.new $REMOTE/vllm-chat-worker"

echo "==> Installing arbiter firewall guard..."
# This guard is a host-level backstop, not an arbiter process, but it protects
# the exact LAN path that spark-view and clients use to reach arbiter. Refresh it
# on every deploy so a hand-fixed version on spark is never lost by a later
# binary-only deploy. The guard runs every minute via cron and removes any
# REJECT/DROP rule on port 8400 (e.g. left behind by a modulith-g0 measurement
# window or an experiment). See scripts/spark-host/README.md for details.
scp -q scripts/spark-host/arbiter-firewall-guard "$SPARK:/home/darren/bin/arbiter-firewall-guard.new"
ssh "$SPARK" "set -e
chmod +x /home/darren/bin/arbiter-firewall-guard.new
mv -f /home/darren/bin/arbiter-firewall-guard.new /home/darren/bin/arbiter-firewall-guard
# Ensure exactly one crontab entry exists for this path; preserve all other lines.
( crontab -l 2>/dev/null | grep -v '/home/darren/bin/arbiter-firewall-guard' || true ) > /tmp/cron.base
echo '* * * * * /home/darren/bin/arbiter-firewall-guard' >> /tmp/cron.base
crontab /tmp/cron.base
rm -f /tmp/cron.base
"

echo "==> Starting arbiter on spark..."
# Race with auto's restart-on-crash: when the pre-stop drain actually emptied
# the fleet, arbiter is idle and exits on SIGTERM within milliseconds. auto's
# stop can lose its own state machine race to its restart policy and bring
# arbiter right back (seen 2026-09-15: "Failed to SIGKILL ... No such
# process" during stop, then "Process arbiter is already running" on start,
# which aborted the release and forced a rollback). The resurrected process
# also predates the binary upload above, so it must never be accepted. If
# start reports the service already running, bounce it once more: a fresh,
# complete stop of the live resurrected process wins the state machine, and
# the retry starts the new binary.
start_out=""
if ! start_out=$(ssh "$SPARK" "/home/darren/local/auto/run start arbiter" 2>&1); then
    if echo "$start_out" | grep -q "already running"; then
        echo "==> auto restarted arbiter during the stop — bouncing it once more"
        ssh "$SPARK" "/home/darren/local/auto/run stop arbiter" 2>&1 | tail -1 || true
        if ! ssh "$SPARK" 'deadline=$(( $(date +%s) + 120 ))
        while lsof -nP -iTCP:8400 -sTCP:LISTEN >/dev/null 2>&1; do
            if [ "$(date +%s)" -ge "$deadline" ]; then
                echo "    FAILED — resurrected arbiter still owns port 8400 after 120s"
                exit 1
            fi
            sleep 1
        done'; then
            exit 1
        fi
        start_out=$(ssh "$SPARK" "/home/darren/local/auto/run start arbiter" 2>&1) || {
            echo "$start_out" | tail -1
            exit 1
        }
    else
        echo "$start_out" | tail -1
        exit 1
    fi
fi
echo "$start_out" | tail -1

echo "==> Waiting for health check..."
for i in $(seq 1 20); do
    if curl -s --max-time 5 http://10.0.0.254:8400/v1/health 2>/dev/null | grep -q '"status":"ok"'; then
        echo "    healthy"
        # Record the deployed content key so an identical repeat deploy
        # (greenline's publish deploy) takes the fast path above. Never
        # recorded on failure — a failed deploy must not arm the skip.
        ssh "$SPARK" "printf '%s' '$DEPLOY_KEY' > '$DEPLOY_KEY_PATH'" \
            || echo "    WARNING: could not record deploy key — next deploy takes the full path"
        curl -s --max-time 5 http://10.0.0.254:8400/v1/health
        echo ""
        exit 0
    fi
    sleep 1
done
# "not responding" is only the symptom. Arbiter exits outright when it cannot
# load its config — e.g. a security-policy rejection drops a model, which makes
# an llm_alias target unresolvable — and reporting a bare timeout hides that
# cause behind a rollback. Print whether the process is alive plus the tail of
# its own log so the real reason is in the deploy log.
echo "    FAILED — arbiter not responding after 20s"
echo "    process check:"
ssh "$SPARK" "pgrep -af '$REMOTE/arbiter-go' || echo '      arbiter-go is NOT running — it exited after start'" 2>&1 | sed 's/^/      /'
echo "    last arbiter log lines:"
ssh "$SPARK" "L=\$(ls -t /home/darren/local/auto/output/logs/arbiter/*/*/*.log 2>/dev/null | head -1); [ -n \"\$L\" ] && tail -15 \"\$L\"" 2>&1 | sed 's/^/      /'
exit 1
