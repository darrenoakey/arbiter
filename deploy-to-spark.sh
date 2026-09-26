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

abort_recovery_termination() {
    case "$1" in
        replacement-listener)
            echo "ERROR: unexpected replacement process pid $2 owns port 8400 after the drained Arbiter was stopped; deploy aborted without stopping or signaling it" >&2
            ;;
        already-running)
            echo "ERROR: Arbiter is already running after the drained instance was stopped; deploy aborted without stopping or signaling the replacement" >&2
            ;;
        *)
            echo "ERROR: unknown deployment recovery condition: $1" >&2
            ;;
    esac
    return 1
}

# Exercise the exact fail-closed recovery decision without contacting Spark.
# This is used by the focused regression tests below; ordinary deploys take no
# arguments and never enter it.
if [ "${1:-}" = "--test-recovery-abort" ]; then
    abort_recovery_termination "${2:-}" "${3:-unknown}"
    exit $?
fi

SPARK=${SPARK:-darren@10.0.0.254}
REMOTE=/home/darren/src/arbiter

cd "$(dirname "$0")"

LTX25_RUNTIME="$(python3 scripts/build_ltx25_runtime.py build | tail -1)"
LTX25_RELEASE="$(basename "$LTX25_RUNTIME")"

ARBITER_URL="http://10.0.0.254:8400"
DEPLOY_DRAIN_TIMEOUT="${DEPLOY_DRAIN_TIMEOUT:-120}"
DEPLOY_DRAIN_LEASE="${DEPLOY_DRAIN_LEASE:-300}"
DRAIN_OWNED=0
DRAIN_PROVEN=0
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
        find src/arbiter config/spark runtime/ltx25 scripts/build_ltx25_runtime.py \
            scripts/spark-host/arbiter-firewall-guard -type f 2>/dev/null \
            | LC_ALL=C sort | while IFS= read -r f; do md5 -q "$f"; done
    } | md5 -q
}

drain_request() { # renews the lease; idempotent
    response="$(curl -fsS --max-time 5 -X POST "$ARBITER_URL/v1/drain" \
        -H 'Content-Type: application/json' \
        -d "{\"lease_seconds\":${DEPLOY_DRAIN_LEASE}}")" || return 1
    printf '%s' "$response" | python3 scripts/drain_wait.py drain "$DEPLOY_DRAIN_LEASE"
}

ps_state() {
    response="$(curl -fsS --max-time 5 "$ARBITER_URL/v1/ps")" || return 1
    printf '%s' "$response" | python3 scripts/drain_wait.py ps
}

wait_state() {
    deadline_reached=$1
    response="$(curl -fsS --max-time 5 "$ARBITER_URL/v1/ps")" || return 1
    printf '%s' "$response" | python3 scripts/drain_wait.py wait "$deadline_reached"
}

resume_if_aborted() {
    rc=$?
    trap - EXIT INT TERM HUP
    if [ "$DRAIN_OWNED" = "1" ]; then
        echo "==> Deploy ended before the bounce (rc=$rc) — resuming arbiter dispatch"
        response="$(curl -fsS --max-time 5 -X POST "$ARBITER_URL/v1/drain" \
            -H 'Content-Type: application/json' -d '{"resume":true}')" || {
            echo "ERROR: failed to resume the deploy-owned Arbiter drain" >&2
            exit 1
        }
        if ! printf '%s' "$response" | python3 scripts/drain_wait.py resume; then
            echo "ERROR: Arbiter did not confirm resuming the deploy-owned drain" >&2
            exit 1
        fi
    fi
    exit "$rc"
}

# Graceful drain: ask the running arbiter to stop starting NEW jobs and let
# in-flight work finish before we bounce it, so a redeploy never kills a
# running job (e.g. a 10-min ltx2 denoise). A missing or malformed drain/status
# API aborts the deploy before any stop. DEPLOY_DRAIN_TIMEOUT changes the
# ceiling but never permits a stop while active_jobs is nonzero.
#
# The default ceiling is deliberately SHORT (120s). This deploy runs inside a
# greenline release with a hard, non-negotiable 600-second end-to-end deadline;
# an ltx25 denoise chunk runs 20-45 minutes, so the release must abort rather
# than kill it when the deadline expires. The owned leased drain is explicitly
# resumed on that abort; the lease is the crash/SIGKILL recovery boundary.
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
    if [ -n "$remote_key" ] && [ "$remote_key" = "$DEPLOY_KEY" ] && \
        ssh "$SPARK" "python3 '$REMOTE/scripts/build_ltx25_runtime.py' verify-activation \
            '$REMOTE/local/ltx25-runtimes/$LTX25_RELEASE' --config '$REMOTE/local/config.toml' >/dev/null 2>&1"; then
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

if ! initial_state="$(ps_state)"; then
    echo "ERROR: cannot validate Arbiter /v1/ps before requesting drain; deploy aborted" >&2
    exit 1
fi
initial_draining="${initial_state%%$'\t'*}"
if [ "$initial_draining" != "0" ]; then
    echo "ERROR: Arbiter is already draining; ownership is external, so deploy aborted without resuming it" >&2
    exit 1
fi

# The server exposes one global leased drain, not caller tokens. A validated
# undrained snapshot is the ownership boundary: only this deploy's subsequent
# request may be resumed by its abort trap. Never resume a drain observed as
# already active, because it belongs to another caller.
DRAIN_OWNED=1
trap resume_if_aborted EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
if ! drain_request; then
    echo "ERROR: Arbiter did not confirm the leased drain; deploy aborted" >&2
    exit 1
fi
drain_deadline=$(( $(date +%s) + DEPLOY_DRAIN_TIMEOUT ))
echo "==> Drain requested (no new jobs) — in-flight work winds down while we build/test locally"

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

echo "==> Installing immutable LTX 2.5 runtime candidate $LTX25_RELEASE..."
ssh "$SPARK" "mkdir -p '$REMOTE/local/ltx25-runtimes' '$REMOTE/runtime/ltx25' '$REMOTE/scripts'"
rsync -az runtime/ltx25/ "$SPARK:$REMOTE/runtime/ltx25/"
rsync -az scripts/build_ltx25_runtime.py "$SPARK:$REMOTE/scripts/build_ltx25_runtime.py"
if ! ssh "$SPARK" "python3 '$REMOTE/scripts/build_ltx25_runtime.py' verify '$REMOTE/local/ltx25-runtimes/$LTX25_RELEASE' >/dev/null 2>&1"; then
    remote_candidate="$REMOTE/local/ltx25-runtimes/.$LTX25_RELEASE.deploy"
    ssh "$SPARK" "test ! -e '$remote_candidate' && mkdir '$remote_candidate'"
    rsync -az --delete "$LTX25_RUNTIME/" "$SPARK:$remote_candidate/"
    ssh "$SPARK" "set -e
python3 '$REMOTE/scripts/build_ltx25_runtime.py' verify '$remote_candidate'
test ! -e '$REMOTE/local/ltx25-runtimes/$LTX25_RELEASE'
mv '$remote_candidate' '$REMOTE/local/ltx25-runtimes/$LTX25_RELEASE'"
fi

echo "==> Cross-compiling binaries..."
GOOS=linux GOARCH=arm64 go build -o arbiter-linux-arm64 ./cmd/arbiter/
GOOS=linux GOARCH=arm64 go build -o llm-worker-linux-arm64 ./cmd/llm-worker/
GOOS=linux GOARCH=arm64 go build -o vllm-chat-worker-linux-arm64 ./cmd/vllm-chat-worker/
echo "    $(md5 -q arbiter-linux-arm64 2>/dev/null || md5sum arbiter-linux-arm64 | awk '{print $1}') arbiter"
echo "    $(md5 -q llm-worker-linux-arm64 2>/dev/null || md5sum llm-worker-linux-arm64 | awk '{print $1}') llm-worker"
echo "    $(md5 -q vllm-chat-worker-linux-arm64 2>/dev/null || md5sum vllm-chat-worker-linux-arm64 | awk '{print $1}') vllm-chat-worker"

# Rejoin the drain requested before the build/test phase. Existing queued work
# is deliberately allowed to remain: drain pauses dispatch, and the queue is
# persistent across the bounce. Only validated root active_jobs==0 proves that
# no worker execution will be interrupted.
if ! drain_request; then
    echo "ERROR: failed to renew the deploy-owned drain; deploy aborted" >&2
    exit 1
fi
echo "==> Draining arbiter (deadline ${DEPLOY_DRAIN_TIMEOUT}s from drain request)..."
while :; do
    deadline_reached=0
    if [ "$(date +%s)" -ge "$drain_deadline" ]; then
        deadline_reached=1
    fi
    if ! state="$(wait_state "$deadline_reached")"; then
        echo "ERROR: cannot validate Arbiter /v1/ps while draining; deploy aborted" >&2
        exit 1
    fi
    decision="${state%%$'\t'*}"
    active="${state##*$'\t'}"
    if [ "$decision" = "drained" ]; then
        DRAIN_PROVEN=1
        echo "    drained — validated active_jobs=0 (queued jobs remain persisted)"
        break
    fi
    if [ "$decision" = "abort" ]; then
        echo "ERROR: still ${active} active job(s) after ${DEPLOY_DRAIN_TIMEOUT}s; deploy aborted without stopping Arbiter" >&2
        exit 1
    fi
    echo "    ${active} job(s) still active; waiting..."
    sleep 10
    if ! drain_request; then
        echo "ERROR: failed to renew the deploy-owned drain; deploy aborted" >&2
        exit 1
    fi
done

if [ "$DRAIN_PROVEN" != "1" ]; then
    echo "ERROR: refusing to stop Arbiter without a validated zero-active drain proof" >&2
    exit 1
fi
echo "==> Stopping arbiter on spark..."
# Past this point the old process is going away, so an abort no longer needs a
# resume call: the restarted arbiter starts undrained, and a drain request that
# never got renewed expires server-side anyway.
DRAIN_OWNED=0
trap - EXIT INT TERM HUP
ssh "$SPARK" "/home/darren/local/auto/run stop arbiter" 2>&1 | tail -1 || true

# A terminated arbiter can remain in uninterruptible SQLite/filesystem I/O for
# longer than auto's ten-second port-reclaim window. Starting immediately then
# fails even though the old listener has already received SIGKILL. Wait for the
# kernel to finish releasing that exact listening socket before replacing the
# binary. Do not kill unrelated listeners. If auto loses the stop race and
# creates a replacement listener, the earlier drain proof does not authorize
# terminating that new process: abort and leave it untouched.
echo "==> Waiting for the stopped arbiter to release port 8400..."
port_wait_rc=0
port_wait_out="$(ssh "$SPARK" 'deadline=$(( $(date +%s) + 300 ))
while lsof -nP -iTCP:8400 -sTCP:LISTEN >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "    FAILED — terminated arbiter still owns port 8400 after 300s"
        lsof -nP -iTCP:8400 -sTCP:LISTEN 2>&1 || true
        exit 1
    fi
    # auto can lose the stop race and respawn arbiter while this wait runs.
    # A healthy replacement never exits, but the zero-active proof belonged
    # to the stopped instance and cannot authorize terminating the new PID.
    # Report it immediately to the caller; a dying uninterruptible process is
    # still left alone to release its socket naturally.
    pid=$(lsof -t -nP -iTCP:8400 -sTCP:LISTEN 2>/dev/null | head -1 || true)
    if [ -n "$pid" ]; then
        exe=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
        state=$(sed -n "s/.*) //p" "/proc/$pid/stat" 2>/dev/null | awk "{print \$1}")
        if [ "$exe" = "/home/darren/src/arbiter/arbiter-go" ] && [ "$state" != "D" ]; then
            echo "$pid"
            exit 75
        fi
    fi
    sleep 1
done')" || port_wait_rc=$?
if [ "$port_wait_rc" -eq 75 ]; then
    replacement_pid="$(printf '%s\n' "$port_wait_out" | tail -1)"
    abort_recovery_termination replacement-listener "${replacement_pid:-unknown}" || exit 1
fi
if [ "$port_wait_rc" -ne 0 ]; then
    printf '%s\n' "$port_wait_out" >&2
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
ssh "$SPARK" "$REMOTE/.venv/bin/python -c 'import glob, json, os, pathlib; root=pathlib.Path(\"$REMOTE\"); path=root/\"local/config.json\"; data=json.loads(path.read_text()); models=data.setdefault(\"models\", {}); [models.update({pathlib.Path(f).name.replace(\".model.json\", \"\"): json.loads(pathlib.Path(f).read_text())}) for f in glob.glob(str(root/\"config/spark/*.model.json\")) if pathlib.Path(f).name not in {\"minimax-h3-local.model.json\", \"minimax-h3.model.json\"}]; existing=models.get(\"minimax-h3-local\") or {}; local=json.loads((root/\"config/spark/minimax-h3-local.model.json\").read_text()); worker=existing.get(\"worker_cmd\"); local.update({\"worker_cmd\": worker} if worker else {}); models[\"minimax-h3-local\"]=local; models.pop(\"minimax-h3\", None); temporary=path.with_name(\".config.spark-models.tmp\"); handle=temporary.open(\"w\"); json.dump(data, handle, indent=2); handle.write(\"\\n\"); handle.flush(); os.fsync(handle.fileno()); handle.close(); os.replace(temporary, path); descriptor=os.open(path.parent, os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)'"

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
LTX25_CONFIG="$REMOTE/local/config.toml"
LTX25_ROLLBACK="$REMOTE/local/.ltx25-activation-rollback"
rollback_ltx25_activation() {
    rc=$?
    trap - EXIT INT TERM HUP
    echo "==> Restoring previous immutable LTX 2.5 runtime activation (rc=$rc)"
    if ! ssh "$SPARK" "python3 '$REMOTE/scripts/build_ltx25_runtime.py' rollback --config '$LTX25_CONFIG' --rollback-record '$LTX25_ROLLBACK'"; then
        echo "ERROR: failed to restore previous LTX 2.5 runtime activation" >&2
        exit 1
    fi
    exit "$rc"
}
trap rollback_ltx25_activation EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
ssh "$SPARK" "python3 '$REMOTE/scripts/build_ltx25_runtime.py' prepare-activation \
    --config '$LTX25_CONFIG' --rollback-record '$LTX25_ROLLBACK'"
ssh "$SPARK" "python3 '$REMOTE/scripts/build_ltx25_runtime.py' activate \
    '$REMOTE/local/ltx25-runtimes/$LTX25_RELEASE' \
    --config '$LTX25_CONFIG' --rollback-record '$LTX25_ROLLBACK'"
# Race with auto's restart-on-crash: when the pre-stop drain actually emptied
# the fleet, arbiter is idle and exits on SIGTERM within milliseconds. auto's
# stop can lose its own state machine race to its restart policy and bring
# arbiter right back (seen 2026-09-15: "Failed to SIGKILL ... No such
# process" during stop, then "Process arbiter is already running" on start).
# The resurrected process also predates the binary upload above, so it must
# never be accepted. The earlier drain proof belonged to the stopped process,
# though, so this deployment must abort rather than stop the replacement. The
# activation rollback trap remains armed and restores the prior runtime config.
start_out=""
if ! start_out=$(ssh "$SPARK" "/home/darren/local/auto/run start arbiter" 2>&1); then
    if echo "$start_out" | grep -q "already running"; then
        abort_recovery_termination already-running || exit 1
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
        ssh "$SPARK" "python3 '$REMOTE/scripts/build_ltx25_runtime.py' finalize-activation \
            --config '$LTX25_CONFIG' --rollback-record '$LTX25_ROLLBACK'"
        trap - EXIT INT TERM HUP
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
