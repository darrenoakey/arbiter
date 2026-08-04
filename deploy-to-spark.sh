#!/bin/bash
# Deploy arbiter from this Mac to spark.
# This is the ONLY way to push code to spark — do not edit files in place on spark.
#
# Steps:
#   1. Run tests locally
#   2. Cross-compile binaries for Linux ARM64
#   3. Stop arbiter on spark
#   4. Sync Python adapters + binaries
#   5. Start arbiter
#   6. Verify health

set -euo pipefail

SPARK=${SPARK:-darren@10.0.0.254}
REMOTE=/home/darren/src/arbiter

cd "$(dirname "$0")"

echo "==> Running Go tests..."
# ARBITER_GO_TEST_SKIP: optional regex of environment-dependent tests to skip
# (some remote-host tests require live local ollama / remote Macs on the LAN).
# Default (unset) runs the FULL suite — do not set it to dodge real failures.
if [ -n "${ARBITER_GO_TEST_SKIP:-}" ]; then
    echo "    skipping env-dependent tests matching: $ARBITER_GO_TEST_SKIP"
    go test ./cmd/arbiter/ -count=1 -skip "$ARBITER_GO_TEST_SKIP" >/dev/null
else
    go test ./cmd/arbiter/ -count=1 >/dev/null
fi
echo "    go tests passed"

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

echo "==> Cross-compiling binaries..."
GOOS=linux GOARCH=arm64 go build -o arbiter-linux-arm64 ./cmd/arbiter/
GOOS=linux GOARCH=arm64 go build -o llm-worker-linux-arm64 ./cmd/llm-worker/
GOOS=linux GOARCH=arm64 go build -o vllm-chat-worker-linux-arm64 ./cmd/vllm-chat-worker/
echo "    $(md5 -q arbiter-linux-arm64 2>/dev/null || md5sum arbiter-linux-arm64 | awk '{print $1}') arbiter"
echo "    $(md5 -q llm-worker-linux-arm64 2>/dev/null || md5sum llm-worker-linux-arm64 | awk '{print $1}') llm-worker"
echo "    $(md5 -q vllm-chat-worker-linux-arm64 2>/dev/null || md5sum vllm-chat-worker-linux-arm64 | awk '{print $1}') vllm-chat-worker"

# Graceful drain: ask the running arbiter to stop starting NEW jobs and let
# in-flight work finish before we bounce it, so a redeploy never kills a
# running job (e.g. a 10-min ltx2 denoise). Tolerant of an older binary that
# lacks /v1/drain. Bounded wait; override with DEPLOY_FORCE=1 to skip, or
# DEPLOY_DRAIN_TIMEOUT to change the ceiling (default 1800s).
ARBITER_URL="http://10.0.0.254:8400"
DEPLOY_DRAIN_TIMEOUT="${DEPLOY_DRAIN_TIMEOUT:-1800}"
if [ "${DEPLOY_FORCE:-0}" = "1" ]; then
    echo "==> DEPLOY_FORCE=1 — skipping graceful drain (may kill in-flight jobs)"
elif curl -s --max-time 5 -X POST "$ARBITER_URL/v1/drain" >/dev/null 2>&1; then
    echo "==> Draining arbiter (no new jobs; waiting for in-flight to finish, max ${DEPLOY_DRAIN_TIMEOUT}s)..."
    drain_deadline=$(( $(date +%s) + DEPLOY_DRAIN_TIMEOUT ))
    while :; do
        active=$(curl -s --max-time 5 "$ARBITER_URL/v1/ps" 2>/dev/null \
            | python3 -c 'import sys,json; print(json.load(sys.stdin).get("active_jobs",0))' 2>/dev/null || echo 0)
        if [ "${active:-0}" = "0" ]; then
            echo "    drained — 0 in-flight jobs"
            break
        fi
        if [ "$(date +%s)" -ge "$drain_deadline" ]; then
            echo "    WARNING: still ${active} in-flight after ${DEPLOY_DRAIN_TIMEOUT}s — proceeding anyway"
            break
        fi
        echo "    ${active} job(s) still in flight; waiting..."
        sleep 10
    done
else
    echo "==> No /v1/drain on running arbiter (older binary) — proceeding without drain"
fi

echo "==> Stopping arbiter on spark..."
ssh "$SPARK" "/home/darren/local/auto/run stop arbiter" 2>&1 | tail -1 || true

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

echo "==> Starting arbiter on spark..."
ssh "$SPARK" "/home/darren/local/auto/run start arbiter" 2>&1 | tail -1

echo "==> Waiting for health check..."
for i in $(seq 1 20); do
    if curl -s --max-time 5 http://10.0.0.254:8400/v1/health 2>/dev/null | grep -q '"status":"ok"'; then
        echo "    healthy"
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
