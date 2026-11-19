//go:build linux

package main

import (
	"fmt"
	"log/slog"
	"os"
	"regexp"
	"strings"

	"golang.org/x/sys/unix"
)

// killIfOwnedOrphan SIGKILLs pid when it is an orphan this arbiter owns.
//
// The process is pinned with a pidfd before it is classified. If it exits and
// its PID is reused between classification and the signal, the signal fails
// with ESRCH instead of reaching the newcomer.
func killIfOwnedOrphan(pid, selfPid int, re *regexp.Regexp) bool {
	fd, err := unix.PidfdOpen(pid, 0)
	if err != nil {
		return false // already gone
	}
	defer unix.Close(fd)
	if !workerReapable(pid, selfPid, re) {
		slog.Debug("orphan sweep: preserving worker owned by another live process tree", "pid", pid)
		return false
	}
	return unix.PidfdSendSignal(fd, unix.SIGKILL, nil, 0) == nil
}

// workerReapable applies the ownership boundary documented on reapOwnedOrphans.
// Anything that cannot be read from /proc is preserved.
func workerReapable(pid, selfPid int, re *regexp.Regexp) bool {
	if !re.MatchString(procCmdline(pid)) {
		return false
	}
	if hasAncestor(pid, selfPid) {
		return false // a live worker of ours
	}
	if self, ok := procCgroup(selfPid); ok {
		if group, ok := procCgroup(pid); ok && group == self {
			return true
		}
	}
	root := orphanTreeRoot(pid)
	return root > 1 && re.MatchString(procCmdline(root))
}

// orphanTreeRoot returns the ancestor of pid (possibly pid itself) whose parent
// is init, or 0 when the chain cannot be read.
func orphanTreeRoot(pid int) int {
	for depth := 0; depth < 256 && pid > 1; depth++ {
		ppid := parentPid(pid)
		if ppid == 1 {
			return pid
		}
		if ppid <= 0 || ppid == pid {
			return 0
		}
		pid = ppid
	}
	return 0
}

// procCmdline returns pid's command line with NUL separators as spaces, the
// form pgrep -f matches against.
func procCmdline(pid int) string {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(strings.ReplaceAll(string(data), "\x00", " "))
}

// procCgroup returns pid's cgroup membership as listed in /proc.
func procCgroup(pid int) (string, bool) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	group := strings.TrimSpace(string(data))
	return group, err == nil && group != ""
}
