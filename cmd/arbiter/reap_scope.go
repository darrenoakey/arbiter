package main

import (
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// workerReapPatterns are the worker command lines the Kill-time sweep looks
// for. They include the Python multiprocessing children of vLLM (which rename
// themselves to "VLLM::EngineCore" via setproctitle), since those reparent to
// init the moment vllm serve dies and then aren't reachable via PPID walks.
var workerReapPatterns = []string{
	"llama.cpp/build/bin/llama-server",
	"arbiter/vllm-chat-worker",
	"arbiter/llm-worker",
	"arbiter.worker_main",
	".venv-vllm/bin/vllm",
	"VLLM::EngineCore",
	"vllm.entrypoints",
}

// startupReapPatterns are the worker command lines swept at arbiter startup.
var startupReapPatterns = []string{
	"llama.cpp/build/bin/llama-server",
	"arbiter/vllm-chat-worker",
	"arbiter/llm-worker",
	"arbiter.worker_main",
}

// reapOwnedOrphans SIGKILLs every process whose command line matches one of
// patterns and that this arbiter (selfPid) owns as an orphan, skipping
// excludePid. It returns the PIDs it signalled.
//
// Ownership boundary. A matching process is reaped only when it is not a live
// descendant of this arbiter and either
//   - it sits in this arbiter's own cgroup (our workers, and their escaped
//     descendants, stay in our cgroup even after reparenting), or
//   - the root of its process tree, the ancestor whose parent is init, is
//     itself a worker. That is an orphaned worker tree, for example one left
//     by a previous arbiter that was started from a different session.
//
// A worker owned by another live process tree in another cgroup is never
// touched: for example a separately managed Arbiter lane running in its own
// systemd unit, whose tree is rooted at the user's systemd manager, or another
// daemon's live worker. That owner, and its unit's cgroup cleanup, is
// responsible for it. Before this boundary, any matching process that was not
// our descendant was killed, which SIGKILLed a concurrently managed training
// worker during an ordinary production model unload.
func reapOwnedOrphans(patterns []string, selfPid, excludePid int) []int {
	expr := strings.Join(patterns, "|")
	re, err := regexp.Compile(expr)
	if err != nil {
		return nil
	}
	// One pgrep for all patterns instead of one per pattern: a full command-
	// line scan of every process on the host was measured at ~1s+ per pattern
	// under load, and Kill runs this sweep on every teardown.
	output, err := exec.Command("pgrep", "-f", expr).Output()
	if err != nil {
		return nil
	}
	var reaped []int
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		pid, err := strconv.Atoi(strings.TrimSpace(line))
		if err != nil || pid <= 1 || pid == selfPid || pid == excludePid {
			continue
		}
		if killIfOwnedOrphan(pid, selfPid, re) {
			reaped = append(reaped, pid)
		}
	}
	return reaped
}
