#!/bin/bash
# blackbox — 2s-resolution flight recorder for post-mortem of spark hard deaths.
#
# The Jun 2 / Jun 4 / Jun 10 2026 deaths left NOTHING: no kernel panic, no
# pstore record, no watchdog reset, journals running at full speed until the
# instant of silence, memory tens-of-GB free at the last 10-min sar sample.
# This records the gap sar cannot see: memory, GPU power/util/temp and the
# biggest processes every 2 seconds, synced to disk every 30 seconds, so the
# final 30s before any future death are on disk for the next boot to read.
#
# Read after an incident:  tail -200 /home/darren/local/blackbox/blackbox-<date>.log
exec 9>/tmp/blackbox.lock
flock -n 9 || { echo "2026-06-10T05:48:35+00:00 blackbox: another instance holds the lock — exiting"; exit 0; }

OUT_DIR=/home/darren/local/blackbox
INTERVAL=2
SYNC_EVERY=15   # iterations between fsyncs (15 x 2s = 30s max data loss)
echo "$(date -Is) blackbox up (interval=${INTERVAL}s sync_every=$((SYNC_EVERY*INTERVAL))s)"
i=0
while true; do
  day=$(date +%Y%m%d)
  f="$OUT_DIR/blackbox-$day.log"
  # prune logs older than 14 days once per day (first iteration after midnight)
  if [ ! -e "$f" ]; then
    find "$OUT_DIR" -name "blackbox-*.log" -mtime +14 -delete 2>/dev/null
  fi
  mem=$(awk "/^MemAvailable:/{a=\$2} /^MemFree:/{fr=\$2} /^Cached:/{c=\$2} END{printf \"avail=%dM free=%dM cached=%dM\", a/1024, fr/1024, c/1024}" /proc/meminfo)
  gpu=$(timeout 5 nvidia-smi --query-gpu=power.draw,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d " " | awk -F, "{printf \"gpu_w=%s gpu_util=%s gpu_temp=%s\", \$1, \$2, \$3}")
  [ -z "$gpu" ] && gpu="gpu_w=ERR gpu_util=ERR gpu_temp=ERR"
  load=$(cut -d" " -f1-3 /proc/loadavg)
  top3=$(ps -eo rss,comm --sort=-rss --no-headers | head -3 | awk "{printf \"%s:%dM \", \$2, \$1/1024}")
  echo "$(date -Is) $mem $gpu load=$load top=[$top3]" >> "$f"
  i=$((i+1))
  if [ $((i % SYNC_EVERY)) -eq 0 ]; then sync -d "$f" 2>/dev/null || sync; fi
  sleep "$INTERVAL"
done
