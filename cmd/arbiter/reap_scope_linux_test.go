//go:build linux

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"
)

// TestReapOwnedOrphansScopedToOwner reproduces the production incident in which an
// ordinary model unload's orphan sweep SIGKILLed a concurrently managed training
// worker running in its own systemd unit. It uses only disposable sleep processes
// whose command lines carry a per-run nonce, and it sweeps only that nonce, so it
// can never signal a real worker.
//
// The test process plays the arbiter. Five disposable "workers" are started:
//
//	own        live child of the arbiter                          -> preserved
//	ownOrphan  reparented away from the arbiter, same cgroup      -> reaped
//	lane       child of a live owner in a separate capped systemd
//	           service (the managed training lane shape)          -> preserved
//	prior      orphan rooted at init in another cgroup (a previous
//	           arbiter's leftover)                                 -> reaped
//	foreign    child of another live daemon rooted at init in its
//	           own scope (another Arbiter on the host)            -> preserved
//
// It must run outside any child subreaper (for example from a login session or
// a systemd-run --scope), so that orphans reparent to init as they do on Spark.
func TestReapOwnedOrphansScopedToOwner(t *testing.T) {
	if _, err := exec.LookPath("systemd-run"); err != nil {
		t.Skip("requires systemd-run --user")
	}
	if err := exec.Command("systemctl", "--user", "show-environment").Run(); err != nil {
		t.Skip("requires a systemd user manager")
	}
	nonce := fmt.Sprintf("%d-%d", os.Getpid(), time.Now().UnixNano())
	worker := "reapharness-" + nonce + "-worker"
	owner := "reapharness-" + nonce + "-owner"
	units := map[string]string{
		"lane":    "reapharness-lane-" + nonce,
		"prior":   "reapharness-prior-" + nonce,
		"foreign": "reapharness-foreign-" + nonce,
	}
	t.Cleanup(func() {
		for _, unit := range units {
			_ = exec.Command("systemctl", "--user", "stop", unit+".service", unit+".scope").Run()
		}
		_ = exec.Command("pkill", "-KILL", "-f", "reapharness-"+nonce).Run()
	})

	own := &exec.Cmd{Path: "/bin/sleep", Args: []string{worker, "300"}}
	if err := own.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = own.Process.Kill(); _ = own.Wait() })
	// Names travel in the environment, so only the disposable workers' own
	// command lines contain the worker name; the owner shells never match.
	// Backgrounded processes get /dev/null stdio so CombinedOutput returns.
	env := []string{"RH_WORKER=" + worker, "RH_OWNER=" + owner, "RH_INNER=" + `(exec -a "$RH_WORKER" sleep 300) & wait`}
	mustRun(t, env, "bash", "-c", `(exec -a "$RH_WORKER" sleep 300 </dev/null >/dev/null 2>&1 &)`)
	lane := []string{"systemd-run", "--user", "--quiet", "--collect", "--unit=" + units["lane"],
		"--property=MemoryMax=8589934592", "--property=MemorySwapMax=0",
		"--property=CPUQuota=400%", "--property=TasksMax=128",
		"--property=RuntimeMaxSec=300", "--property=KillMode=control-group"}
	for _, kv := range env {
		lane = append(lane, "--setenv="+kv)
	}
	lane = append(lane, "/bin/bash", "-c", `exec -a "$RH_OWNER" /bin/bash -c "$RH_INNER"`)
	mustRun(t, env, lane[0], lane[1:]...)
	mustRun(t, env, "systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit="+units["prior"],
		"/bin/bash", "-c", `(exec -a "$RH_WORKER" sleep 300 </dev/null >/dev/null 2>&1 &)`)
	mustRun(t, env, "systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit="+units["foreign"],
		"/bin/bash", "-c", `(exec -a "$RH_OWNER" /bin/bash -c "$RH_INNER" </dev/null >/dev/null 2>&1 &)`)

	roles := waitForRoles(t, worker, own.Process.Pid, units)
	if parentPid(roles["prior"]) != 1 || parentPid(roles["ownOrphan"]) != 1 {
		t.Fatal("orphans did not reparent to init; run outside any child subreaper")
	}
	checkLaneCaps(t, roles["lane"])
	before := map[string]string{}
	for role, pid := range roles {
		before[role] = startTime(t, pid)
	}

	pattern := regexp.QuoteMeta(worker)
	reaped := reapOwnedOrphans([]string{pattern}, os.Getpid(), 0)
	slices.Sort(reaped)
	want := []int{roles["ownOrphan"], roles["prior"]}
	slices.Sort(want)
	t.Logf("roles=%v reaped=%v", roles, reaped)
	if !slices.Equal(reaped, want) {
		t.Errorf("reaped %v; want only the owned orphans %v (ownOrphan, prior)", reaped, want)
	}
	waitGone(t, roles["ownOrphan"], roles["prior"])
	for _, role := range []string{"own", "lane", "foreign"} {
		pid := roles[role]
		if !pidAlive(pid) || startTime(t, pid) != before[role] {
			t.Errorf("%s worker pid %d was not preserved with its identity", role, pid)
		}
	}

	// The lane's own unit still owns cleanup of its cgroup.
	mustRun(t, nil, "systemctl", "--user", "stop", units["lane"]+".service")
	waitGone(t, roles["lane"])
	if left := strings.TrimSpace(output(t, "pgrep", "-f", pattern)); left == "" {
		t.Fatal("expected own and foreign workers to remain until cleanup")
	}
}

func mustRun(t *testing.T, env []string, name string, args ...string) {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Env = append(os.Environ(), env...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%s %v: %v\n%s", name, args, err, out)
	}
}

func output(t *testing.T, name string, args ...string) string {
	t.Helper()
	out, _ := exec.Command(name, args...).Output()
	return string(out)
}

// waitForRoles identifies the five disposable workers by parentage and cgroup.
func waitForRoles(t *testing.T, worker string, ownPid int, units map[string]string) map[string]int {
	t.Helper()
	self, _ := procCgroup(os.Getpid())
	deadline := time.Now().Add(10 * time.Second)
	for {
		roles := map[string]int{"own": ownPid}
		for _, field := range strings.Fields(output(t, "pgrep", "-f", regexp.QuoteMeta(worker))) {
			var pid int
			if _, err := fmt.Sscan(field, &pid); err != nil || pid == ownPid {
				continue
			}
			group, _ := procCgroup(pid)
			switch {
			case strings.Contains(group, "/"+units["lane"]+".service"):
				roles["lane"] = pid
			case strings.Contains(group, "/"+units["prior"]+".scope"):
				roles["prior"] = pid
			case strings.Contains(group, "/"+units["foreign"]+".scope"):
				roles["foreign"] = pid
			case group == self:
				roles["ownOrphan"] = pid
			}
		}
		if len(roles) == 5 {
			return roles
		}
		if time.Now().After(deadline) {
			t.Fatalf("disposable workers did not all start: %v", roles)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// checkLaneCaps reads the lane unit's enforced limits back from its cgroup.
func checkLaneCaps(t *testing.T, pid int) {
	t.Helper()
	group, _ := procCgroup(pid)
	dir := filepath.Join("/sys/fs/cgroup", strings.TrimPrefix(group, "0::"))
	for file, want := range map[string]string{
		"memory.max": "8589934592", "memory.swap.max": "0",
		"cpu.max": "400000 100000", "pids.max": "128",
	} {
		got, err := os.ReadFile(filepath.Join(dir, file))
		if err != nil || strings.TrimSpace(string(got)) != want {
			t.Fatalf("lane %s=%q, want %s (%v)", file, got, want, err)
		}
	}
}

// startTime returns field 22 of /proc/PID/stat, the process start time, which
// together with the PID identifies one process across PID reuse.
func startTime(t *testing.T, pid int) string {
	t.Helper()
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return ""
	}
	s := string(data)
	fields := strings.Fields(s[strings.LastIndexByte(s, ')')+2:])
	return fields[19]
}

func waitGone(t *testing.T, pids ...int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for _, pid := range pids {
		for pidAlive(pid) {
			if time.Now().After(deadline) {
				t.Fatalf("pid %d still alive", pid)
			}
			time.Sleep(50 * time.Millisecond)
		}
	}
}
