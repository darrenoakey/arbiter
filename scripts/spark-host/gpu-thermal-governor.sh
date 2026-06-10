#!/bin/bash
# gpu-thermal-governor — dynamic GPU clock governor for the GB10 thermal cutoff.
#
# The spark hard-powers-off under sustained GPU load (5 deaths Jun 2-10 2026,
# blackbox-proven instant cut at GPU 81-89°C with everything else healthy).
# The GB10 reports thermal limits as T.Limit HEADROOM (degrees remaining):
# slowdown at -2, HARD SHUTDOWN at -5 past the limit. At 89°C the box rides
# ~0 headroom and one excursion past -5 between samples = found-it-dead-again.
#
# Strategy: run at FULL clocks while cool; when T.Limit headroom shrinks (or
# GPU temp nears the observed death zone) step the max clock down fast; when
# headroom recovers, step back up slowly. Hysteresis dead-band prevents
# oscillation. This trades a few % render speed under sustained load for the
# machine not turning itself off.
#
#   headroom <= 3              -> EMERGENCY: floor clocks (1300)
#   headroom <= 6 OR temp >=87 -> step down 200 MHz
#   headroom >= 10 AND temp<=83 -> step up 100 MHz (at most every 3rd tick)
#
# Managed by auto. Remove entirely: auto stop gpu-thermal-governor &&
# sudo nvidia-smi -rgc

exec 200>/tmp/gpu-thermal-governor.lock
flock -n 200 || { echo "another gpu-thermal-governor holds the lock — exiting"; exit 0; }

# CEILING IS THE REAL FIX: community-proven (NVIDIA forums mega-thread 363370,
# MODS 372469, eugr/spark-vllm-docker) that GB10 boost transients toward
# 3000MHz trip EC overcurrent protection and hard-power-off the box EVEN WHEN
# COOL (units die at 60-70C GPU). Sub-ms transients are invisible to any
# polling loop, so the ceiling must prevent them: 2150-2200 is the consensus
# safe cap (~3-4% real perf cost; sustained clock under load was ~2400-2500).
# The dynamic part below handles the slower thermal-buildup path (unmonitored
# board sensor — fan curve tracks CPU, not GPU; nanoChat thread 358280).
MAXCLK=2200
MINCLK=1300
STEP_DOWN=200
STEP_UP=100
INTERVAL=2
cap=2400                      # conservative start; adapts within seconds
ticks_since_up=0

echo "$(date -Is) thermal-governor up (max=$MAXCLK min=$MINCLK target: headroom>6, temp<87)"
sudo nvidia-smi -lgc 0,$cap >/dev/null 2>&1

while true; do
  read -r temp headroom clk <<< "$(timeout 5 nvidia-smi --query-gpu=temperature.gpu,temperature.gpu.tlimit,clocks.gr --format=csv,noheader,nounits 2>/dev/null | tr -d ',')"
  if [ -z "$headroom" ]; then sleep "$INTERVAL"; continue; fi
  new=$cap
  reason=""
  if [ "$headroom" -le 3 ]; then
    new=$MINCLK; reason="EMERGENCY headroom=$headroom"
  elif [ "$headroom" -le 6 ] || [ "$temp" -ge 87 ]; then
    new=$(( cap - STEP_DOWN )); [ $new -lt $MINCLK ] && new=$MINCLK
    reason="throttle temp=$temp headroom=$headroom"
  elif [ "$headroom" -ge 10 ] && [ "$temp" -le 83 ] && [ $cap -lt $MAXCLK ]; then
    ticks_since_up=$(( ticks_since_up + 1 ))
    if [ $ticks_since_up -ge 3 ]; then
      new=$(( cap + STEP_UP )); [ $new -gt $MAXCLK ] && new=$MAXCLK
      reason="recover temp=$temp headroom=$headroom"
      ticks_since_up=0
    fi
  fi
  if [ "$new" != "$cap" ]; then
    if sudo nvidia-smi -lgc 0,$new >/dev/null 2>&1; then
      echo "$(date -Is) cap ${cap}->${new} MHz ($reason, clk_was=${clk})"
      cap=$new
    fi
  fi
  sleep "$INTERVAL"
done
