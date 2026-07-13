package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// descendantPIDs returns root and all transitive descendants by walking
// /proc/<pid>/task/*/children. Cycles and missing files are tolerated. The
// result includes root.
func descendantPIDs(root int) []int {
	if root <= 0 {
		return nil
	}
	seen := map[int]bool{root: true}
	queue := []int{root}
	out := []int{root}
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		for _, child := range readChildPIDs(pid) {
			if seen[child] {
				continue
			}
			seen[child] = true
			out = append(out, child)
			queue = append(queue, child)
		}
	}
	return out
}

// readChildPIDs reads /proc/<pid>/task/*/children for the given pid and
// returns the union of immediate children. Returns nil on /proc errors so
// callers degrade gracefully on non-Linux or transient failures.
func readChildPIDs(pid int) []int {
	taskDir := fmt.Sprintf("/proc/%d/task", pid)
	tasks, err := os.ReadDir(taskDir)
	if err != nil {
		return nil
	}
	var out []int
	for _, t := range tasks {
		data, err := os.ReadFile(filepath.Join(taskDir, t.Name(), "children"))
		if err != nil {
			continue
		}
		for _, f := range strings.Fields(string(data)) {
			cpid, err := strconv.Atoi(f)
			if err == nil && cpid > 0 {
				out = append(out, cpid)
			}
		}
	}
	return out
}

// treeVRAMBytes sums VRAM (bytes) across root and all descendants, using the
// pidVRAM map produced by GetPerProcessVRAM. Descendants not in the map
// contribute zero.
func treeVRAMBytes(root int, pidVRAM map[int]int64) int64 {
	if root <= 0 || len(pidVRAM) == 0 {
		return 0
	}
	var total int64
	for _, pid := range descendantPIDs(root) {
		total += pidVRAM[pid]
	}
	return total
}

// pidRSSAnonMB reads /proc/<pid>/status RssAnon for a single pid in MB.
// Returns 0 on any error so the caller treats it as "no contribution".
func pidRSSAnonMB(pid int) float64 {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "RssAnon:") {
			var kb int64
			if _, err := fmt.Sscanf(strings.TrimSpace(strings.TrimPrefix(line, "RssAnon:")), "%d", &kb); err != nil {
				return 0
			}
			return float64(kb) / 1024
		}
	}
	return 0
}

// treeRSSAnonMB sums anonymous RSS across root and all descendants in MB.
func treeRSSAnonMB(root int) float64 {
	if root <= 0 {
		return 0
	}
	var total float64
	for _, pid := range descendantPIDs(root) {
		total += pidRSSAnonMB(pid)
	}
	return total
}
