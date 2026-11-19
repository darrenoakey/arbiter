//go:build !linux

package main

import (
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"testing"
)

// TestReapOwnedOrphansFailsClosedOffLinux: without /proc and pidfds the
// ownership boundary cannot be established, so a matching non-descendant is
// never signalled. A disposable sleep with a per-run name plays the worker.
func TestReapOwnedOrphansFailsClosedOffLinux(t *testing.T) {
	name := "reapharness-" + regexp.QuoteMeta(t.Name()) + "-" + strconv.Itoa(os.Getpid())
	worker := &exec.Cmd{Path: "/bin/sleep", Args: []string{name, "30"}}
	if err := worker.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = worker.Process.Kill(); _ = worker.Wait() })
	re := regexp.MustCompile(regexp.QuoteMeta(name))
	if killIfOwnedOrphan(worker.Process.Pid, 1, re) {
		t.Fatal("killed a worker off Linux")
	}
	if worker.ProcessState != nil {
		t.Fatal("worker exited")
	}
}
