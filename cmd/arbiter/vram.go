package main

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// nvidiaSMITimeout bounds every nvidia-smi exec. A driver lock makes the
// query block with no output (observed 14 minutes on 2026-09-22 while
// ltx25-denoise1 job 633b27a52464 had already finished). Callers treat a
// timeout like any other query failure: utilization -1, empty process map.
const nvidiaSMITimeout = 8 * time.Second

func runNvidiaSMI(args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), nvidiaSMITimeout)
	defer cancel()
	return exec.CommandContext(ctx, "nvidia-smi", args...).Output()
}

// GetPerProcessVRAM returns a map of PID -> VRAM usage in bytes.
// Uses nvidia-smi to query actual GPU memory per process.
func GetPerProcessVRAM() map[int]int64 {
	result := make(map[int]int64)

	out, err := runNvidiaSMI(
		"--query-compute-apps=pid,used_memory",
		"--format=csv,noheader,nounits",
	)
	if err != nil {
		return result
	}

	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ",", 2)
		if len(parts) != 2 {
			continue
		}
		pid, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}
		mib, err := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
		if err != nil {
			continue
		}
		result[pid] = mib * 1024 * 1024 // MiB to bytes
	}
	return result
}

// GetGPUUtilization returns GPU compute utilization as a percentage (0-100).
func GetGPUUtilization() int {
	out, err := runNvidiaSMI(
		"--query-gpu=utilization.gpu",
		"--format=csv,noheader,nounits",
	)
	if err != nil {
		return -1
	}
	line := strings.TrimSpace(string(out))
	// Multi-GPU: take first line
	if idx := strings.IndexByte(line, '\n'); idx >= 0 {
		line = line[:idx]
	}
	pct, err := strconv.Atoi(strings.TrimSpace(line))
	if err != nil {
		return -1
	}
	return pct
}

// GetGPUStatusLine returns a one-line nvidia-smi snapshot for diagnostics.
func GetGPUStatusLine() string {
	out, err := runNvidiaSMI(
		"--query-gpu=utilization.gpu,power.draw,temperature.gpu",
		"--format=csv,noheader",
	)
	if err != nil {
		return ""
	}
	line := strings.TrimSpace(string(out))
	if idx := strings.IndexByte(line, '\n'); idx >= 0 {
		line = line[:idx]
	}
	return line
}
