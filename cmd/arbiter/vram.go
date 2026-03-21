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
