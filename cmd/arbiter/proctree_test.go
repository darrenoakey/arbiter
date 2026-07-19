//go:build linux

package main

import (
	"os"
	"os/exec"
	"strconv"
	"testing"
	"time"
)

// startSelfChild runs `sleep 30` so we have a real child process whose pid
// shows up in /proc/<parent>/task/*/children. Returns the child's pid and a
// cleanup func that kills it.
func startSelfChild(t *testing.T) (int, func()) {
	t.Helper()
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start child: %v", err)
	}
	// Wait until the child appears in /proc, with a small bounded retry — on
	// Linux the children file is updated synchronously but tests are flaky on
	// CI without a tiny pause.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat("/proc/" + strconv.Itoa(cmd.Process.Pid) + "/status"); err == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	cleanup := func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	}
	return cmd.Process.Pid, cleanup
}

func TestDescendantPIDs_IncludesSelf(t *testing.T) {
	got := descendantPIDs(os.Getpid())
	if len(got) == 0 || got[0] != os.Getpid() {
		t.Fatalf("expected self pid as first element, got %v", got)
	}
}

func TestDescendantPIDs_SeesChild(t *testing.T) {
	childPID, cleanup := startSelfChild(t)
	defer cleanup()

	got := descendantPIDs(os.Getpid())
	found := false
	for _, p := range got {
		if p == childPID {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("child pid %d not found in descendants of self %d: %v", childPID, os.Getpid(), got)
	}
}

func TestTreeVRAMBytes_SumsAcrossDescendants(t *testing.T) {
	childPID, cleanup := startSelfChild(t)
	defer cleanup()

	parent := os.Getpid()
	pidVRAM := map[int]int64{
		parent:   1_000_000_000, // 1 GB on parent
		childPID: 2_000_000_000, // 2 GB on child
		99999999: 9_000_000_000, // unrelated PID, should NOT be summed
	}
	got := treeVRAMBytes(parent, pidVRAM)
	want := int64(3_000_000_000)
	if got != want {
		t.Fatalf("treeVRAMBytes got %d, want %d", got, want)
	}
}

func TestTreeRSSAnonMB_NonZeroForLiveProcess(t *testing.T) {
	rss := treeRSSAnonMB(os.Getpid())
	if rss <= 0 {
		t.Fatalf("expected positive tree RSS for live process, got %f", rss)
	}
}

func TestTreeRSSAnonMB_ZeroForInvalidPID(t *testing.T) {
	rss := treeRSSAnonMB(0)
	if rss != 0 {
		t.Fatalf("expected 0 for pid 0, got %f", rss)
	}
}
