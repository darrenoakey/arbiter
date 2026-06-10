#!/usr/bin/env python3
"""spark-watchdog — auto-revive the spark GB10 when it dies.

The spark has a recurring failure mode (4 incidents Jun 2-10 2026) where it
dies instantly under sustained GPU load — no kernel panic, no watchdog reset,
no OOM, journals running at full speed until the moment of silence — and stays
dead until someone presses the power button. Until the hardware cause is
pinned down (the blackbox flight recorder on spark is collecting evidence),
this watchdog turns "dead until Darren walks over" into "back in ~3 minutes":

  ping spark every 30s -> 6 consecutive failures (3 min) -> send Wake-on-LAN
  magic packets every 60s until it answers again.

WoL is armed on spark's enP7s7 (NetworkManager wake-on-lan=magic, verified
`Wake-on: g`). Note WoL revives suspend/soft-off states; if the failure turns
out to be a PSU hard-latch it may not fire — the logs here tell us which.
"""
import socket
import subprocess
import time
from datetime import datetime

SPARK_IP = "10.0.0.254"
SPARK_MAC = "30:c5:99:3e:50:64"  # enP7s7, the active GbE NIC
PING_INTERVAL = 30
FAILURES_BEFORE_WAKE = 6  # 6 x 30s = 3 minutes down before we act
WAKE_RETRY_INTERVAL = 60


def log(msg: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def spark_alive() -> bool:
    result = subprocess.run(
        ["ping", "-c", "2", "-W", "3000", SPARK_IP],
        capture_output=True,
    )
    return result.returncode == 0


def send_magic_packet() -> None:
    mac_bytes = bytes.fromhex(SPARK_MAC.replace(":", ""))
    packet = b"\xff" * 6 + mac_bytes * 16
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for port in (9, 7):
            s.sendto(packet, ("255.255.255.255", port))
            s.sendto(packet, (SPARK_IP, port))


def main() -> None:
    log(f"spark-watchdog up (target={SPARK_IP} mac={SPARK_MAC} "
        f"threshold={FAILURES_BEFORE_WAKE * PING_INTERVAL}s)")
    consecutive_failures = 0
    waking = False
    last_wake = 0.0
    while True:
        if spark_alive():
            if waking:
                log("spark is BACK UP")
            elif consecutive_failures > 0:
                log(f"spark recovered after {consecutive_failures} failed ping(s)")
            consecutive_failures = 0
            waking = False
        else:
            consecutive_failures += 1
            if consecutive_failures == 1:
                log("spark missed a ping")
            if consecutive_failures >= FAILURES_BEFORE_WAKE:
                now = time.time()
                if now - last_wake >= WAKE_RETRY_INTERVAL:
                    log(f"spark down {consecutive_failures * PING_INTERVAL}s — "
                        "sending Wake-on-LAN magic packets")
                    try:
                        send_magic_packet()
                    except OSError as e:
                        log(f"WoL send failed: {e}")
                    last_wake = now
                    waking = True
        time.sleep(PING_INTERVAL)


if __name__ == "__main__":
    main()
