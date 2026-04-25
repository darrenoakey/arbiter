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

SPARK=darren@spark
REMOTE=/home/darren/src/arbiter

cd "$(dirname "$0")"

echo "==> Running Go tests..."
go test ./cmd/arbiter/ -count=1 >/dev/null
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

echo "==> Stopping arbiter on spark..."
ssh "$SPARK" "/home/darren/local/auto/run stop arbiter" 2>&1 | tail -1 || true

echo "==> Syncing Python adapters..."
rsync -az --delete src/arbiter/ "$SPARK:$REMOTE/src/arbiter/"

echo "==> Uploading binaries..."
scp -q arbiter-linux-arm64 "$SPARK:$REMOTE/arbiter-go"
scp -q llm-worker-linux-arm64 "$SPARK:$REMOTE/llm-worker"
scp -q vllm-chat-worker-linux-arm64 "$SPARK:$REMOTE/vllm-chat-worker"
ssh "$SPARK" "chmod +x $REMOTE/arbiter-go $REMOTE/llm-worker $REMOTE/vllm-chat-worker"

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
echo "    FAILED — arbiter not responding after 20s"
exit 1
