package main

import (
	"os/exec"
	"strconv"
	"strings"
)

// GetPerProcessVRAM returns a map of PID -> VRAM usage in bytes.
// Uses nvidia-smi to query actual GPU memory per process.
func GetPerProcessVRAM() map[int]int64 {
	result := make(map[int]int64)

	out, err := exec.Command(
		"nvidia-smi",
		"--query-compute-apps=pid,used_memory",
		"--format=csv,noheader,nounits",
	).Output()
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
	out, err := exec.Command(
		"nvidia-smi",
		"--query-gpu=utilization.gpu",
		"--format=csv,noheader,nounits",
	).Output()
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
