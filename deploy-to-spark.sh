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

echo "==> Running tests..."
go test ./cmd/arbiter/ -count=1 >/dev/null
echo "    tests passed"

echo "==> Cross-compiling binaries..."
GOOS=linux GOARCH=arm64 go build -o arbiter-linux-arm64 ./cmd/arbiter/
GOOS=linux GOARCH=arm64 go build -o llm-worker-linux-arm64 ./cmd/llm-worker/
echo "    $(md5 -q arbiter-linux-arm64 2>/dev/null || md5sum arbiter-linux-arm64 | awk '{print $1}') arbiter"
echo "    $(md5 -q llm-worker-linux-arm64 2>/dev/null || md5sum llm-worker-linux-arm64 | awk '{print $1}') llm-worker"

echo "==> Stopping arbiter on spark..."
ssh "$SPARK" "/home/darren/local/auto/run stop arbiter" 2>&1 | tail -1 || true

echo "==> Syncing Python adapters..."
rsync -az --delete src/arbiter/ "$SPARK:$REMOTE/src/arbiter/"

echo "==> Uploading binaries..."
scp -q arbiter-linux-arm64 "$SPARK:$REMOTE/arbiter-go"
scp -q llm-worker-linux-arm64 "$SPARK:$REMOTE/llm-worker"
ssh "$SPARK" "chmod +x $REMOTE/arbiter-go $REMOTE/llm-worker"

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
