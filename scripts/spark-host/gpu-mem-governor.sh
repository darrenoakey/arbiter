#!/bin/bash
# gpu-mem-governor — GB10 unified-memory safety net.
#
# On the GB10 (Grace-Blackwell, unified 128GB), the NVIDIA driver does NOT
# force-evict clean page cache to satisfy GPU allocations. So when a large
# model file (read off the CIFS share) sits in page cache, big model loads /
# gen activations OOM (NV_ERR_NO_MEMORY) even though that cache is reclaimable.
# Observed: GPU "free" was 65GB of 128 with 63GB stuck in page cache; dropping
# cache restored 124GB free and the load succeeded.
#
# Fix: when MemFree is low AND there is substantial reclaimable page cache,
# drop page cache (vm.drop_caches=1, page cache only — not dentries/inodes).
# The dual condition means it only acts on the actual pathology and does NOT
# thrash during normal operation (a legitimately-resident model keeps MemFree
# low but does not leave tens of GB of reclaimable Cached lying around — and
# once dropped, the condition clears until cache refills).
#
# Helps every big model on the arbiter (ltx2, wan-flf, wan-s2v, flux2, gemma).
# Managed by `auto` (restarts on crash, starts at login). Privileged drop runs
# via passwordless sudo.
# Single-instance guard: auto can race and spawn two copies at login; the
# duplicate exits instead of double-dropping caches.
exec 9>/tmp/gpu-mem-governor.lock
flock -n 9 || { echo "2026-06-10T05:44:26+00:00 gpu-mem-governor: another instance holds the lock — exiting"; exit 0; }
FLOOR_KB=$((12 * 1024 * 1024))     # act when MemFree < 12 GB
MINCACHE_KB=$((24 * 1024 * 1024))  # ...and reclaimable Cached > 24 GB
INTERVAL=6
echo "$(date -Is) gpu-mem-governor up (floor=12GB mincache=24GB interval=${INTERVAL}s)"
while true; do
  free_kb=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
  cached_kb=$(awk '/^Cached:/{print $2}' /proc/meminfo)
  if [ "${free_kb:-0}" -lt "$FLOOR_KB" ] && [ "${cached_kb:-0}" -gt "$MINCACHE_KB" ]; then
    sync
    sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'
    newfree=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
    echo "$(date -Is) governor: dropped page cache (MemFree ${free_kb}KB->${newfree}KB, Cached was ${cached_kb}KB)"
  fi
  sleep "$INTERVAL"
done
