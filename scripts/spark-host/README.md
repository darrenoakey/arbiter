# spark-host — host-level protection scripts

Source of truth for the machine-local scripts that keep the spark GB10 alive
(or bring it back) when GPU workloads misbehave. These are NOT deployed by
`deploy-to-spark.sh`; they are installed once per machine and managed by
`auto` on their respective hosts. If you change one here, scp it to the right
host and `auto restart <name>`.

| Script | Runs on | auto name | Purpose |
|--------|---------|-----------|---------|
| `gpu-mem-governor.sh` | spark `~/bin/` | `gpu-mem-governor` | Drops reclaimable page cache when MemFree is low — the NVIDIA driver won't evict clean cache to satisfy GPU allocations, so big model loads OOM with tens of GB "stuck" in cache. |
| `blackbox.sh` | spark `~/bin/` | `blackbox` | 2-second flight recorder (MemAvailable, GPU power/util/temp, top-RSS) synced to disk every 30s. After any sudden death, `tail` the latest `/home/darren/local/blackbox/blackbox-*.log` FIRST — it distinguishes memory spike vs thermal climb vs instant power cut. |
| `spark-watchdog.py` | mac mini `~/bin/` | `spark-watchdog` | Pings spark every 30s; after 3 minutes down it sends Wake-on-LAN magic packets (enP7s7, `30:c5:99:3e:50:64`) every 60s until spark answers. Turns "dead until someone presses the button" into "back in ~3 minutes". |

Related layers (not in this directory):

- per-worker CUDA cap — `src/arbiter/worker_main.py` `_apply_cuda_memory_cap`
- scheduler ceiling — `VRAMBudgetGB` (90) in `cmd/arbiter/config.go`
- in-arbiter emergency kill — `cmd/arbiter/emergency_guardian.go`
- host backstops on spark — earlyoom (`/etc/default/earlyoom`), swap disabled
  (`/etc/fstab`), panic sysctls (`/etc/sysctl.d/99-crash-resilience.conf`),
  systemd SBSA watchdog (`RuntimeWatchdogSec=10s`)
- WoL persistence on spark — NetworkManager `802-3-ethernet.wake-on-lan magic`
  on "Wired connection 3"
