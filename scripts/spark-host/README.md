# spark-host — host-level protection scripts

Source of truth for the machine-local scripts that keep the spark GB10 alive
(or bring it back) when GPU workloads misbehave. These are NOT deployed by
`deploy-to-spark.sh` — **except** `arbiter-firewall-guard`, which is refreshed
on every deploy because it directly protects the arbiter LAN path that
spark-view and clients rely on. For the others, install once per machine and
manage by `auto` on their respective hosts. If you change one here, scp it to the
right host and `auto restart <name>`.

NOTE: `auto` can race at login and spawn TWO copies of a service (observed
2026-06-10: duplicate governors and blackboxes, double cache drops in the
journal). Both spark scripts therefore take an exclusive flock on a /tmp lock
file and the duplicate exits immediately — keep that guard when editing.

| Script | Runs on | auto name | Purpose |
|--------|---------|-----------|---------|
| `arbiter-firewall-guard` | spark `~/bin/` | cron `* * * * *` | Removes any iptables REJECT/DROP rule on arbiter port 8400. Prevents a modulith-g0 measurement window (or a hand-applied experiment) from leaving rules that block spark-view and LAN clients. |
| `gpu-mem-governor.sh` | spark `~/bin/` | `gpu-mem-governor` | Drops reclaimable page cache when MemFree is low — the NVIDIA driver won't evict clean cache to satisfy GPU allocations, so big model loads OOM with tens of GB "stuck" in cache. |
| `blackbox.sh` | spark `~/bin/` | `blackbox` | 2-second flight recorder (MemAvailable, GPU power/util/temp/T.Limit-headroom/clock, top-RSS) synced to disk every 30s. After any sudden death, `tail` the latest `/home/darren/local/blackbox/blackbox-*.log` FIRST — it distinguishes memory spike vs thermal climb vs instant power cut. |
| `gpu-thermal-governor.sh` | spark `~/bin/` | `gpu-thermal-governor` | THE fix for the thermal hard-power-off (blackbox-proven 2026-06-10: instant cut riding T.Limit headroom ≈0 at 89°C; GB10 shutdown fires at headroom -5). Dynamic clock governor: full speed when cool; steps max clock down 200MHz when headroom ≤6 or temp ≥87, emergency-floors to 1300MHz at headroom ≤3, recovers +100MHz/6s when headroom ≥10 and temp ≤83. Replaces the brief static `gpu-clock-cap.service`. Remove: `auto stop gpu-thermal-governor && sudo nvidia-smi -rgc`. |
| `spark-watchdog.py` | mac mini `~/bin/` | `spark-watchdog` | DOWN DETECTOR: pings spark every 30s; after 3 minutes down it logs the outage and sends WoL magic packets every 60s. NOTE: the DGX Spark does NOT support WoL (NVIDIA-confirmed, tested 2026-06-10) — the packets are kept only in case firmware ever adds it. Real remote revive = smart plug + UEFI "After Power Loss Behavior: Auto Boot" (the default). |

Related layers (not in this directory):

- per-worker CUDA cap — `src/arbiter/worker_main.py` `_apply_cuda_memory_cap`
- scheduler ceiling — `VRAMBudgetGB` (90) in `cmd/arbiter/config.go`
- in-arbiter emergency kill — `cmd/arbiter/emergency_guardian.go`
- host backstops on spark — earlyoom (`/etc/default/earlyoom`), swap disabled
  (`/etc/fstab`), panic sysctls (`/etc/sysctl.d/99-crash-resilience.conf`),
  systemd SBSA watchdog (`RuntimeWatchdogSec=10s`)
- WoL persistence on spark — NetworkManager `802-3-ethernet.wake-on-lan magic`
  on "Wired connection 3"
