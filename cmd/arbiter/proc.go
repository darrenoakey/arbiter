package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// Instance represents one adapter subprocess.
// For models with max_concurrent > 1, one subprocess handles multiple concurrent jobs.
// For others, one subprocess = one job at a time.
type Instance struct {
	ModelID       string
	InstanceID    string
	MaxConcurrent int

	// host is the executor this instance runs on. Defaults to LocalHost
	// ("spark") — the local CUDA backend, indistinguishable from pre-seam
	// behavior. A non-local host marks the instance remote: it holds ZERO
	// audited VRAM (skipped by Reserve/Release/AuditVRAMConsistency).
	host string
	// backend is the execution seam. For local instances it wraps these very
	// methods (LocalProcBackend → Instance), so behavior is byte-identical;
	// Phase 2 swaps in RemoteHTTPBackend for remote placements. Callers today
	// still invoke the Instance methods directly — the field exists so the
	// seam is in place and the scheduler can route through it in Phase 2.
	backend Backend

	mu         sync.Mutex
	state      string // "stopped", "starting", "loading", "loaded", "unloading", "error"
	activeJobs int32  // atomic
	// loadInFlight is a try-lock for the ensureLoaded reserve+Load critical
	// section. It is deliberately separate from `state`: Spawn() early-returns
	// when state is anything other than stopped/error, so pre-setting state to
	// "loading" as a claim would make Spawn a no-op and break loading entirely.
	// Exactly one goroutine wins the CAS, performs the load, and clears it; any
	// concurrent ensureLoaded caller (the dispatch path racing the background
	// tryPreload goroutine for the same instance) observes it and waits instead
	// of spawning a second subprocess / double-reserving VRAM.
	loadInFlight atomic.Bool
	// lastLoadInsufficientMem records that the most recent load attempt for this
	// instance failed purely because spark VRAM couldn't be reserved (not a worker
	// crash). A waiter that loses the loadInFlight race observes the resulting
	// terminal state and uses this flag to return an insufficientMemoryError —
	// which requeues the job to wait for VRAM rather than counting a load failure
	// toward the maxLoadAttempts/circuit-breaker terminal-fail path. Without it,
	// jobs racing onto a VRAM-starved model fail spuriously under transient
	// contention (e.g. gemma fallback while ltx2 holds the GPU).
	lastLoadInsufficientMem atomic.Bool
	lastActive              time.Time
	// loadedAt marks when the current residency (keep-alive session) began — set
	// when the instance transitions into "loaded", cleared to zero on unload /
	// subprocess exit. The dashboard uses it to count "actions done since this
	// model was loaded", a progress numerator that intentionally resets whenever
	// the model is reloaded.
	loadedAt time.Time
	memoryGB float64
	vramHeld bool // true while this instance's memoryGB is counted in InstanceManager.usedGB
	// manager is set by InstanceManager.Register. markSubprocessExited needs it
	// to decrement usedGB in the same breath that it clears vramHeld — clearing
	// the flag alone strands the reservation until the 15s reconciler, and the
	// challenger that triggered the eviction fails its reservation on phantom
	// memory in the meantime (the 2026-06-10 swap-thrash livelock).
	manager *InstanceManager

	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	stderr io.ReadCloser

	// For concurrent mode: pending responses keyed by req_id
	pendingMu sync.Mutex
	pending   map[string]chan json.RawMessage

	// Background reader goroutine
	readerDone chan struct{}

	pythonBin     string
	projectRoot   string
	workerCmd     []string          // validated repository-owned worker identity (overrides python)
	adapterParams map[string]string // validated worker settings; revalidated immediately before execution

	// placementReason records why the most recent job landed on this instance's
	// host (preferred|spill|fallback). Set at dispatch by the scheduler; surfaced
	// in /v1/ps so a 3am debugger can see where work is going and why. Advisory:
	// it never gates scheduling.
	reasonMu        sync.Mutex
	placementReason string
}

// setPlacementReason records the reason the latest dispatch chose this instance.
func (inst *Instance) setPlacementReason(reason string) {
	inst.reasonMu.Lock()
	inst.placementReason = reason
	inst.reasonMu.Unlock()
}

// PlacementReason returns the latest dispatch's placement reason for /v1/ps.
func (inst *Instance) PlacementReason() string {
	inst.reasonMu.Lock()
	defer inst.reasonMu.Unlock()
	return inst.placementReason
}

// WorkerResponse is the JSON response from the Python worker.
type WorkerResponse struct {
	Status    string          `json:"status"`
	ReqID     string          `json:"req_id,omitempty"`
	Result    json.RawMessage `json:"result,omitempty"`
	Error     string          `json:"error,omitempty"`
	VRAMBytes int64           `json:"vram_bytes,omitempty"`
}

func NewInstance(modelID, instanceID string, maxConcurrent int, memoryGB float64, pythonBin, projectRoot string) *Instance {
	inst := &Instance{
		ModelID:       modelID,
		InstanceID:    instanceID,
		MaxConcurrent: maxConcurrent,
		host:          LocalHost,
		state:         "stopped",
		memoryGB:      memoryGB,
		pending:       make(map[string]chan json.RawMessage),
		pythonBin:     pythonBin,
		projectRoot:   projectRoot,
	}
	// Default backend is the local subprocess path, wrapping these very
	// methods so existing behavior is unchanged.
	inst.backend = &LocalProcBackend{inst: inst}
	return inst
}

// isRemote reports whether this instance runs on a remote executor. Remote
// instances hold ZERO audited VRAM on spark, so VRAM bookkeeping skips them.
func (inst *Instance) isRemote() bool {
	return inst.host != "" && inst.host != LocalHost
}

// hostOrLocal returns the instance's host id, defaulting empty to the local
// host ("spark") for /v1/ps display.
func (inst *Instance) hostOrLocal() string {
	if inst.host == "" {
		return LocalHost
	}
	return inst.host
}

func (inst *Instance) State() string {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	return inst.state
}

// setState forces the instance lifecycle state. Used by RemoteHTTPBackend,
// which has no subprocess to drive state transitions through Spawn/Load/Unload
// the way the local path does — it must flip state directly so PickInstance /
// ensureLoaded see "loaded"/"stopped" for a remote instance. Also stamps
// lastActive on load so the keepalive timer starts from the warm.
func (inst *Instance) setState(state string) {
	inst.mu.Lock()
	inst.state = state
	switch state {
	case "loaded":
		inst.lastActive = time.Now()
		if inst.loadedAt.IsZero() {
			inst.loadedAt = inst.lastActive
		}
	case "stopped", "unloading", "error":
		inst.loadedAt = time.Time{}
	}
	inst.mu.Unlock()
}

// LoadedAt returns when the current residency began, or the zero time if the
// instance is not currently loaded.
func (inst *Instance) LoadedAt() time.Time {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	return inst.loadedAt
}

func (inst *Instance) ActiveJobs() int {
	return int(atomic.LoadInt32(&inst.activeJobs))
}

// RSSAnon returns the anonymous RSS (heap + stack) of the worker process tree
// (root + all descendants) in MB. Subprocess accounting matters: workers like
// llm-worker shell out to llama-server, and Python adapters fork CUDA helpers
// — without tree summation, large child allocations stay invisible to the
// arbiter's memory bookkeeping. Returns 0 if not running or on non-Linux.
func (inst *Instance) RSSAnon() float64 {
	inst.mu.Lock()
	cmd := inst.cmd
	inst.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		return 0
	}
	return treeRSSAnonMB(cmd.Process.Pid)
}

func (inst *Instance) HasCapacity() bool {
	return inst.ActiveJobs() < inst.MaxConcurrent
}

func (inst *Instance) LastActive() time.Time {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	return inst.lastActive
}

// Spawn starts the Python subprocess if not already running.
func (inst *Instance) Spawn() error {
	inst.mu.Lock()
	defer inst.mu.Unlock()

	if inst.state != "stopped" && inst.state != "error" {
		return nil // already running
	}
	workerConfig := ModelConfig{
		WorkerCmd:     cloneStrings(inst.workerCmd),
		AdapterParams: cloneAdapterParams(inst.adapterParams),
	}
	if err := validateModelWorkerPolicy(inst.projectRoot, inst.ModelID, workerConfig, true); err != nil {
		inst.state = "error"
		return fmt.Errorf("worker execution policy: %w", err)
	}
	executable, err := resolveWorkerExecutable(inst.projectRoot, inst.workerCmd, inst.pythonBin)
	if err != nil {
		inst.state = "error"
		return fmt.Errorf("worker execution policy: %w", err)
	}

	// Defence-in-depth: refuse to spawn if a prior OS process is still
	// alive. If we get here it means Kill failed silently and we'd
	// otherwise create a second worker holding GPU memory in parallel
	// with the first (observed: two moondream processes both at 17 GB).
	// Better to fail the load and surface the bug than to silently leak.
	if inst.cmd != nil && inst.cmd.Process != nil {
		oldPid := inst.cmd.Process.Pid
		if pidAlive(oldPid) {
			slog.Error("Spawn: refusing — previous worker still alive",
				"instance", inst.InstanceID, "old_pid", oldPid, "state", inst.state)
			return fmt.Errorf("previous worker pid %d still alive — Kill failed", oldPid)
		}
	}

	inst.state = "starting"
	slog.Info("spawning adapter subprocess", "instance", inst.InstanceID, "model", inst.ModelID)

	var cmd *exec.Cmd
	if len(inst.workerCmd) > 0 {
		cmd = exec.Command(executable, inst.workerCmd[1:]...)
	} else {
		cmd = exec.Command(executable, "-m", "arbiter.worker_main", inst.ModelID)
	}
	cmd.Dir = inst.projectRoot
	// Start from a finite trusted environment. In particular, loader, shell,
	// interpreter, and caller-controlled PATH variables from either config or
	// Arbiter's own parent process never reach the worker.
	cmd.Env = buildCleanWorkerEnvironment(inst.projectRoot, cmd.Path, inst.memoryGB, workerConfig.AdapterParams)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		inst.state = "error"
		return fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		inst.state = "error"
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		inst.state = "error"
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		inst.state = "error"
		return fmt.Errorf("start subprocess: %w", err)
	}

	// Make worker subprocesses OOM-kill candidates before arbiter-go or auto.
	// LTX-2 and other heavy workers can use tens of GB of RAM; we want the OOM
	// killer to prefer them over the Go server and the auto process manager.
	// Writing to /proc/<pid>/oom_score_adj is Linux-only; silently ignored elsewhere.
	if cmd.Process != nil {
		oomPath := fmt.Sprintf("/proc/%d/oom_score_adj", cmd.Process.Pid)
		_ = os.WriteFile(oomPath, []byte("500"), 0o644)
	}

	inst.cmd = cmd
	inst.stdin = stdin
	inst.stdout = bufio.NewScanner(stdout)
	// Grow up to 96MB per response line. Small initial buffer keeps baseline
	// memory low; the ceiling accommodates adapters that inline base64 audio
	// (e.g. demucs two-stem or rvc-convert output for a verse) in the result
	// JSON. A single response line exceeding the max makes Scan() fail with
	// bufio.ErrTooLong, which readLoop reports as "subprocess died" even though
	// the worker is healthy — so this ceiling bounds the largest inlined payload.
	inst.stdout.Buffer(make([]byte, 256*1024), 96*1024*1024)
	inst.stderr = stderr
	inst.readerDone = make(chan struct{})
	inst.state = "unloaded"

	// Read stderr in background (adapter logs)
	go func() {
		scanner := bufio.NewScanner(stderr)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
		for scanner.Scan() {
			slog.Info("adapter", "instance", inst.InstanceID, "msg", scanner.Text())
		}
	}()

	// Background reader dispatches responses by req_id
	go inst.readLoop()

	slog.Info("adapter subprocess started", "instance", inst.InstanceID, "pid", cmd.Process.Pid)
	return nil
}

// readLoop runs in a goroutine for concurrent-mode instances.
// It reads stdout and dispatches responses to the correct pending channel by req_id.
func (inst *Instance) readLoop() {
	defer close(inst.readerDone)
	for inst.stdout.Scan() {
		// Copy the line — scanner.Bytes() is reused on next Scan()
		line := make([]byte, len(inst.stdout.Bytes()))
		copy(line, inst.stdout.Bytes())

		var resp WorkerResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			slog.Warn("non-JSON from adapter", "instance", inst.InstanceID, "line", string(line))
			continue
		}
		if resp.ReqID == "" {
			// Non-request response (e.g., load/unload) — send to default channel
			resp.ReqID = "_default"
		}
		inst.pendingMu.Lock()
		ch, ok := inst.pending[resp.ReqID]
		if ok {
			delete(inst.pending, resp.ReqID)
		}
		inst.pendingMu.Unlock()
		if ok {
			ch <- line
		} else {
			slog.Warn("no pending request for response", "instance", inst.InstanceID, "req_id", resp.ReqID)
		}
	}

	// stdout closed — subprocess is gone. Clean up references and unblock all
	// pending requests.
	inst.markSubprocessExited()

	inst.pendingMu.Lock()
	for reqID, ch := range inst.pending {
		errResp, _ := json.Marshal(WorkerResponse{
			Status: "error",
			ReqID:  reqID,
			Error:  "subprocess died",
		})
		ch <- errResp
		delete(inst.pending, reqID)
	}
	inst.pendingMu.Unlock()
}

// markSubprocessExited finalises an instance whose worker subprocess has gone
// away. Called by readLoop on stdout close. It distinguishes a deliberate
// shutdown (Unload/Kill already transitioned state to "unloading"/"stopped")
// from an unexpected death: only the latter flips state to "error". The VRAM
// bookkeeping flag is cleared in BOTH cases — a dead subprocess holds zero
// VRAM regardless of who killed it.
//
// Before this was unconditional, code paths that called Kill() without pairing
// with ReleaseMemoryFor (e.g. the inference-timeout watchdog at
// scheduler.go:558) left vramHeld=true on a stopped instance forever, and
// because ReconcileFromInstances sums vramHeld instances into `expected`, the
// leak was indistinguishable from a live reservation. Repeated timeouts then
// exhausted the budget with phantom allocations (observed: 83 GB phantom from
// llm:gemma4-26b @ 80.8 GB after one max_runtime_seconds kill).
func (inst *Instance) markSubprocessExited() {
	inst.mu.Lock()
	deliberateShutdown := inst.state == "unloading" || inst.state == "stopped"
	if deliberateShutdown {
		slog.Info("readLoop exited after deliberate shutdown", "instance", inst.InstanceID, "state", inst.state)
	} else {
		slog.Warn("readLoop exited, subprocess died", "instance", inst.InstanceID, "prior_state", inst.state)
		inst.state = "error"
	}
	// Release the VRAM reservation properly, not just the flag. Clearing
	// vramHeld without decrementing usedGB leaves the budget inflated until the
	// 15s reconciler — and because ReleaseMemoryFor keys its idempotency off
	// vramHeld, the eviction path's paired ReleaseMemoryFor call then no-ops.
	// Observed 2026-06-10: denoise2 evicted for gemma, gemma's reservation
	// failed on the phantom 56GB, 30s cooldown, denoise2 reloaded — an 8-minute
	// load wasted per cycle, forever (the same gemma job aged, won, and lost
	// five times in a row without ever running).
	heldGB := 0.0
	if inst.vramHeld {
		heldGB = inst.memoryGB
		inst.vramHeld = false
	}
	inst.loadedAt = time.Time{} // residency ended
	manager := inst.manager
	if inst.stdin != nil {
		if err := inst.stdin.Close(); err != nil {
			slog.Debug("close worker stdin after exit", "instance", inst.InstanceID, "error", err)
		}
		inst.stdin = nil
	}
	inst.cmd = nil
	inst.mu.Unlock()
	if heldGB > 0 && manager != nil {
		manager.ReleaseMemory(heldGB)
	}
}

// sendAndReceive sends a command and waits for the response.
// Uses the background readLoop to dispatch responses by req_id.
func (inst *Instance) sendAndReceive(cmd map[string]any) (*WorkerResponse, error) {
	data, _ := json.Marshal(cmd)
	data = append(data, '\n')

	inst.mu.Lock()
	if inst.stdin == nil {
		inst.mu.Unlock()
		return nil, fmt.Errorf("subprocess not running")
	}
	stdin := inst.stdin
	inst.mu.Unlock()

	// Register pending channel for this request
	reqID, _ := cmd["req_id"].(string)
	if reqID == "" {
		reqID = "_default"
	}
	ch := make(chan json.RawMessage, 1)
	inst.pendingMu.Lock()
	inst.pending[reqID] = ch
	inst.pendingMu.Unlock()

	if _, err := stdin.Write(data); err != nil {
		inst.pendingMu.Lock()
		delete(inst.pending, reqID)
		inst.pendingMu.Unlock()
		return nil, fmt.Errorf("write to subprocess: %w", err)
	}

	// Wait for response, but also watch for subprocess death.
	// Race: readLoop may have already exited before we registered ch (subprocess died
	// between our pendingMu.Unlock and stdin.Write). In that case nobody will ever
	// send to ch, so we'd block forever without selecting on readerDone.
	var raw json.RawMessage
	select {
	case raw = <-ch:
		// Normal path: response arrived (may be a "subprocess died" error from readLoop).
	case <-inst.readerDone:
		// readLoop already exited. Check if it sent to ch before closing.
		inst.pendingMu.Lock()
		delete(inst.pending, reqID)
		inst.pendingMu.Unlock()
		select {
		case raw = <-ch:
			// readLoop did manage to send to us just before exiting.
		default:
			return nil, fmt.Errorf("subprocess died")
		}
	}
	var resp WorkerResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("decode worker response: %w", err)
	}
	return &resp, nil
}

// Load sends the load command and waits for completion.
func (inst *Instance) Load(device string) error {
	if err := inst.Spawn(); err != nil {
		return err
	}

	inst.mu.Lock()
	inst.state = "loading"
	// Reset the activity timestamp so the model-health watchdog measures the
	// "loading stuck" timeout from the start of THIS load attempt, not from
	// the last successful job (which may have completed hours ago, instantly
	// tripping the stuck-loading kill on any re-load of a previously-active
	// model — observed on z-image-turbo killing itself ~5s into a fresh load).
	inst.lastActive = time.Now()
	inst.mu.Unlock()

	resp, err := inst.sendAndReceive(map[string]any{"cmd": "load", "device": device})
	if err != nil {
		inst.mu.Lock()
		inst.state = "error"
		inst.mu.Unlock()
		return fmt.Errorf("load failed: %w", err)
	}
	if resp.Status != "ok" {
		inst.mu.Lock()
		inst.state = "error"
		inst.mu.Unlock()
		return fmt.Errorf("load failed: %s", resp.Error)
	}

	inst.mu.Lock()
	// Only mark loaded if state is still "loading". The subprocess may have
	// died between receiving the load response and this point; readLoop would
	// already have set state to "error" and cleared cmd. Don't overwrite that.
	if inst.state != "loading" {
		state := inst.state
		inst.mu.Unlock()
		return fmt.Errorf("load failed: subprocess died after load response (state=%s)", state)
	}
	inst.state = "loaded"
	inst.lastActive = time.Now() // start keepalive timer from load time
	inst.loadedAt = inst.lastActive
	inst.mu.Unlock()

	slog.Info("model loaded", "instance", inst.InstanceID)
	return nil
}

// InferRaw sends an inference command without managing activeJobs.
// Used when the caller (scheduler) manages the reservation itself.
func (inst *Instance) InferRaw(jobID, jobType string, params json.RawMessage, outputDir string) (*WorkerResponse, error) {
	return inst.sendAndReceive(map[string]any{
		"cmd":        "infer",
		"req_id":     jobID,
		"params":     json.RawMessage(params),
		"output_dir": outputDir,
		"job_id":     jobID,
		"job_type":   jobType,
	})
}

// GetPort queries the llm-worker for its llama-server port.
func (inst *Instance) GetPort() (int, error) {
	resp, err := inst.sendAndReceive(map[string]any{"cmd": "get_port"})
	if err != nil {
		return 0, err
	}
	if resp.Status != "ok" {
		return 0, fmt.Errorf("get_port failed: %s", resp.Error)
	}
	var result struct {
		Port int `json:"port"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return 0, fmt.Errorf("parse port: %w", err)
	}
	return result.Port, nil
}

// Cancel sends SIGUSR1 to the subprocess to set the cancel flag.
func (inst *Instance) Cancel() error {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.cmd != nil && inst.cmd.Process != nil {
		return inst.cmd.Process.Signal(syscall.SIGUSR1)
	}
	return nil
}

// Unload IS process tree obliteration. There is no "soft unload" or "hard
// unload" — the act of unloading from the arbiter's perspective is the act
// of killing the worker subprocess and every descendant, then verifying they
// are gone. The unload is not complete until the tree is empty.
//
// The "unload" stdin command sent to the worker is a courtesy hint so it
// can release CUDA cleanly if it wants to; we don't wait for it. Kill()
// handles the rest with retries.
//
// Refuses only when running this would corrupt in-flight work (active jobs)
// or race with a concurrent load.
func (inst *Instance) Unload() error {
	if inst.ActiveJobs() > 0 {
		return fmt.Errorf("refusing to unload %s: %d active jobs", inst.InstanceID, inst.ActiveJobs())
	}
	if inst.State() == "loading" {
		return fmt.Errorf("refusing to unload %s: currently loading", inst.InstanceID)
	}

	inst.mu.Lock()
	inst.state = "unloading"
	inst.loadedAt = time.Time{} // residency ended
	stdin := inst.stdin
	inst.mu.Unlock()
	slog.Info("unloading model — initiating tree kill", "instance", inst.InstanceID)

	// Fire-and-forget courtesy hint to the worker. Don't wait — Kill() has
	// its own grace window and will SIGKILL anything that doesn't exit on
	// its own.
	if stdin != nil {
		go func() {
			data, _ := json.Marshal(map[string]any{"cmd": "unload"})
			if _, err := stdin.Write(append(data, '\n')); err != nil {
				slog.Debug("send unload hint", "instance", inst.InstanceID, "error", err)
			}
		}()
	}

	// THE unload: tree kill with retries until the entire subtree is gone.
	inst.Kill()
	slog.Info("unload complete — process tree obliterated", "instance", inst.InstanceID)
	return nil
}

// Kill terminates the subprocess AND every descendant (vllm serve →
// VLLM::EngineCore → N child threads, llm-worker → llama-server, etc.) and
// retries until the entire tree is gone. The contract: when this returns,
// no process from this instance's tree is holding the GPU.
//
// Strategy:
//  1. Send graceful "shutdown" via stdin (one-shot, gives the worker a chance
//     to checkpoint cleanly).
//  2. Wait briefly for the root to exit on its own.
//  3. Walk the descendant tree (leaves-first) and SIGKILL every PID. Repeat
//     until the tree is empty or we hit the retry cap.
//  4. Sweep by pattern in case anything reparented to init before we walked
//     the tree (e.g. vLLM's EngineCore subprocess).
//
// This is the "release the GPU at all costs" path — we don't trust state
// transitions, we verify by polling /proc.
func (inst *Instance) Kill() {
	inst.mu.Lock()
	var rootPid int
	var rootCmd *exec.Cmd
	prevReaderDone := inst.readerDone
	if inst.cmd != nil && inst.cmd.Process != nil {
		rootPid = inst.cmd.Process.Pid
		rootCmd = inst.cmd
		// One graceful attempt — well-behaved workers will release CUDA cleanly.
		if inst.stdin != nil {
			data, _ := json.Marshal(map[string]any{"cmd": "shutdown"})
			if _, err := inst.stdin.Write(append(data, '\n')); err != nil {
				slog.Debug("send worker shutdown", "instance", inst.InstanceID, "error", err)
			}
			if err := inst.stdin.Close(); err != nil {
				slog.Debug("close worker stdin", "instance", inst.InstanceID, "error", err)
			}
		}
	}
	inst.state = "stopped"
	inst.loadedAt = time.Time{} // residency ended
	inst.stdin = nil
	inst.cmd = nil
	inst.mu.Unlock()

	slog.Info("Kill: invoked", "instance", inst.InstanceID, "root_pid", rootPid)
	if rootPid == 0 {
		// Nothing in inst.cmd, but a previous Kill might have left an OS process
		// alive that we've forgotten about. Pattern sweep is the safety net.
		reapWorkersExcept(0)
		return
	}

	// CRITICAL: spawn a goroutine that calls Wait() on the root cmd. Without
	// this the root process becomes a zombie when it exits — and the NVIDIA
	// driver keeps the zombie's CUDA allocations marked "in use" until the
	// parent reaps it. Symptom: dmesg fills with "NVRM: Out of memory" and
	// new vLLM loads fail because the driver thinks VRAM is full even though
	// nvidia-smi shows nothing running.
	if rootCmd != nil {
		go func() { _ = rootCmd.Wait() }()
	}

	// ALWAYS send SIGKILL to the root at least once. The earlier "grace + then
	// only SIGKILL if pidAlive" pattern silently bailed when pidAlive returned
	// false (e.g. transient /proc read failure, or the worker was in mid-fork
	// when we sampled). That left the OS process alive and the next ensureLoaded
	// happily spawned a SECOND worker (observed: two moondream processes both
	// holding 17 GB). SIGKILL on a dead PID is harmless (ESRCH); SIGKILL on a
	// live PID is what we need.
	if err := syscall.Kill(rootPid, syscall.SIGKILL); err != nil && err != syscall.ESRCH {
		slog.Warn("Kill: root SIGKILL failed", "instance", inst.InstanceID, "pid", rootPid, "error", err)
	}

	// Now verify-and-retry: walk the descendant tree, SIGKILL everything,
	// and keep going until /proc shows none of them. 30 attempts × 500ms =
	// 15s cap. We retry forever within that window because Python+CUDA can
	// take a few seconds to actually exit after SIGKILL while CUDA cleanup
	// runs in a kernel D-state thread.
	const maxAttempts = 30
	var lastSurvivors []int
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		descendants := collectDescendants(rootPid)
		if len(descendants) == 0 {
			slog.Info("Kill: tree cleared", "instance", inst.InstanceID, "root", rootPid, "attempts", attempt)
			lastSurvivors = nil
			break
		}
		lastSurvivors = descendants
		slog.Warn("Kill: SIGKILL descendant tree",
			"instance", inst.InstanceID, "root", rootPid, "attempt", attempt, "pids", descendants)
		for _, pid := range descendants {
			if err := syscall.Kill(pid, syscall.SIGKILL); err != nil && err != syscall.ESRCH {
				slog.Warn("Kill: descendant SIGKILL failed", "instance", inst.InstanceID, "pid", pid, "error", err)
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	if len(lastSurvivors) > 0 {
		slog.Error("Kill: tree NOT fully reaped after retries — possible D-state survivor",
			"instance", inst.InstanceID, "root", rootPid, "survivors", lastSurvivors)
	}

	// Wait for the previous readLoop goroutine to actually exit before
	// returning. Otherwise a fast Spawn() that follows can race: the old
	// readLoop is still reading from inst.stdout, and when Spawn reassigns
	// inst.stdout the old loop may consume responses meant for the new
	// process or call markSubprocessExited at the wrong time, nilifying
	// inst.cmd of the new process.
	if prevReaderDone != nil {
		select {
		case <-prevReaderDone:
		case <-time.After(5 * time.Second):
			slog.Warn("Kill: readLoop did not exit within 5s",
				"instance", inst.InstanceID, "root", rootPid)
		}
	}

	// Pattern sweep for anything that escaped the tree walk (e.g. EngineCore
	// already reparented to init when we started).
	reapWorkersExcept(0)
}

// pidAlive returns true if the process exists AND is not a zombie. A zombie
// (defunct) process still has a /proc entry and accepts kill(0) — but it's
// already dead, just waiting to be reaped. Treating zombies as alive sends
// the SIGKILL-retry loop into a 7.5s no-op spin. Worse, until the parent
// calls Wait() the kernel keeps the zombie's resources (including CUDA
// allocations) marked in-use, which is what triggers the NVIDIA OOM cascade.
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	if syscall.Kill(pid, 0) != nil {
		return false
	}
	// Skip zombies. /proc/PID/status line "State:  Z (zombie)".
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "State:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 && (fields[1] == "Z" || fields[1] == "X") {
				return false
			}
			break
		}
	}
	return true
}

// collectDescendants returns rootPid plus every currently-alive transitive
// descendant, in post-order (leaves first). This ordering matters: killing
// leaves first prevents reparenting from breaking the tree mid-walk.
func collectDescendants(rootPid int) []int {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		// Non-Linux fallback: just the root if alive.
		if pidAlive(rootPid) {
			return []int{rootPid}
		}
		return nil
	}
	childrenOf := make(map[int][]int)
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		ppid := parentPid(pid)
		if ppid > 0 {
			childrenOf[ppid] = append(childrenOf[ppid], pid)
		}
	}
	var out []int
	visited := make(map[int]bool)
	var walk func(p int)
	walk = func(p int) {
		if visited[p] {
			return
		}
		visited[p] = true
		for _, c := range childrenOf[p] {
			walk(c)
		}
		if pidAlive(p) {
			out = append(out, p)
		}
	}
	walk(rootPid)
	return out
}

// reapWorkersExcept scans for leftover arbiter worker processes and kills any
// that aren't the named PID (which might still be winding down). Called from
// Instance.Kill as a belt-and-braces sweep after the descendant-tree walk.
//
// Patterns include the Python multiprocessing children of vLLM (which rename
// themselves to "VLLM::EngineCore" via setproctitle), since those reparent to
// init the moment vllm serve dies and then aren't reachable via PPID walks.
func reapWorkersExcept(excludePid int) {
	patterns := []string{
		"llama.cpp/build/bin/llama-server",
		"arbiter/vllm-chat-worker",
		"arbiter/llm-worker",
		"arbiter.worker_main",
		".venv-vllm/bin/vllm",
		"VLLM::EngineCore",
		"vllm.entrypoints",
	}
	selfPid := os.Getpid()
	for _, pattern := range patterns {
		output, err := exec.Command("pgrep", "-f", pattern).Output()
		if err != nil {
			continue
		}
		for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
			if line == "" {
				continue
			}
			pid, err := strconv.Atoi(line)
			if err != nil || pid == selfPid || pid == excludePid {
				continue
			}
			// Skip processes still rooted under the running arbiter — those
			// are healthy in-flight loads, not orphans.
			if hasAncestor(pid, selfPid) {
				continue
			}
			if err := syscall.Kill(pid, syscall.SIGKILL); err == nil {
				slog.Warn("reaped orphan worker", "pid", pid, "pattern", pattern)
			}
		}
	}
}

// parentPid returns the parent PID of pid, or 0 if unavailable.
func parentPid(pid int) int {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0
	}
	// /proc/PID/stat format: pid (comm) state ppid ...
	// comm can contain spaces/parens, so find the LAST ')' and split after it.
	s := string(data)
	idx := strings.LastIndexByte(s, ')')
	if idx < 0 || idx+2 >= len(s) {
		return 0
	}
	fields := strings.Fields(s[idx+2:])
	if len(fields) < 2 {
		return 0
	}
	ppid, _ := strconv.Atoi(fields[1])
	return ppid
}

func hasAncestor(pid int, ancestorPid int) bool {
	for pid > 1 {
		ppid := parentPid(pid)
		if ppid == ancestorPid {
			return true
		}
		if ppid <= 1 || ppid == pid {
			return false
		}
		pid = ppid
	}
	return false
}

// Reservation represents a VRAM budget reservation.
// Reservations block the scheduler from using the reserved memory for model loads.
type Reservation struct {
	ID        string    `json:"id"`
	MemoryGB  float64   `json:"memory_gb"`
	Label     string    `json:"label"`
	CreatedAt time.Time `json:"created_at"`
}

// InstanceManager manages all adapter instances for all models.
type InstanceManager struct {
	mu           sync.RWMutex
	instances    map[string]*Instance // instanceID -> Instance
	byModel      map[string][]string  // modelID -> []instanceID
	condemned    map[string]bool      // instance IDs pending removal (scale-down, permanent)
	budgetGB     float64              // single VRAM budget limit
	usedGB       float64
	reservations map[string]*Reservation
	reservedGB   float64
	pythonBin    string
	projectRoot  string
	config       *Config // shared with the rest of the arbiter; mutated under m.mu
	// remoteHosts is per-host advisory memory accounting for remote executors.
	// It is intentionally SEPARATE from usedGB/budgetGB (spark's audited
	// local-CUDA ledger). Phase 1 only builds the registry from config; Phase 2
	// polls each host and routes against it.
	remoteHosts map[string]*remoteHostBudget
	// reachable reports whether a host id is currently reachable. It is set by
	// the HostMonitor (Phase 3) so PickInstanceForJob can skip a host whose
	// liveness poll has confirmed it absent — without that, a job would be
	// dispatched to a dead box and only fail over after the inference call timed
	// out. nil means "assume everything reachable" (tests / no monitor running),
	// preserving pre-Phase-3 behavior.
	reachableMu sync.RWMutex
	reachable   func(hostID string) bool
}

// SetReachabilityFunc installs the host-reachability predicate used by
// PickInstanceForJob to skip confirmed-absent remote hosts. The HostMonitor
// calls this once at startup. Safe to call before the monitor's first poll: it
// starts every host reachable.
func (m *InstanceManager) SetReachabilityFunc(f func(hostID string) bool) {
	m.reachableMu.Lock()
	m.reachable = f
	m.reachableMu.Unlock()
}

// hostReachable consults the installed reachability predicate. Local hosts and
// the absence of a predicate both report reachable=true.
func (m *InstanceManager) hostReachable(hostID string) bool {
	if m.config.HostIsLocal(hostID) {
		return true
	}
	m.reachableMu.RLock()
	f := m.reachable
	m.reachableMu.RUnlock()
	if f == nil {
		return true
	}
	return f(hostID)
}

func NewInstanceManager(cfg *Config, pythonBin, projectRoot string) *InstanceManager {
	remoteHosts := make(map[string]*remoteHostBudget)
	for id, h := range cfg.Hosts {
		if cfg.HostIsLocal(id) {
			continue // local CUDA host is the audited usedGB/budgetGB ledger
		}
		remoteHosts[id] = newRemoteHostBudget(id, h.Addr, h.BudgetGB)
	}
	return &InstanceManager{
		instances:    make(map[string]*Instance),
		byModel:      make(map[string][]string),
		condemned:    make(map[string]bool),
		budgetGB:     cfg.VRAMBudgetGB,
		reservations: make(map[string]*Reservation),
		pythonBin:    pythonBin,
		projectRoot:  projectRoot,
		config:       cfg,
		remoteHosts:  remoteHosts,
	}
}

// RemoteHostBudget returns the advisory budget tracker for a remote host, or
// nil if the id is unknown/local. Phase 2 dispatch uses this to gate remote
// placement without ever touching the audited local VRAM ledger.
func (m *InstanceManager) RemoteHostBudget(hostID string) *remoteHostBudget {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.remoteHosts[hostID]
}

func (m *InstanceManager) Register(inst *Instance) {
	m.mu.Lock()
	defer m.mu.Unlock()
	inst.manager = m
	m.instances[inst.InstanceID] = inst
	m.byModel[inst.ModelID] = append(m.byModel[inst.ModelID], inst.InstanceID)
}

// EnsureModel ensures a model key exists in byModel, even with 0 instances.
func (m *InstanceManager) EnsureModel(modelID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.byModel[modelID]; !ok {
		m.byModel[modelID] = []string{}
	}
}

// RemoveModelRuntime removes a newly registered model whose surrounding
// persistence transaction failed. It refuses to erase active work.
func (m *InstanceManager) RemoveModelRuntime(modelID string) error {
	m.mu.Lock()
	identifiers := append([]string(nil), m.byModel[modelID]...)
	instances := make([]*Instance, 0, len(identifiers))
	for _, identifier := range identifiers {
		instance := m.instances[identifier]
		if instance != nil && instance.ActiveJobs() > 0 {
			m.mu.Unlock()
			return fmt.Errorf("remove model runtime %q: instance %q has active jobs", modelID, identifier)
		}
		if instance != nil {
			instances = append(instances, instance)
		}
	}
	delete(m.byModel, modelID)
	for _, identifier := range identifiers {
		delete(m.instances, identifier)
		delete(m.condemned, identifier)
	}
	m.mu.Unlock()
	for _, instance := range instances {
		instance.Kill()
	}
	return nil
}

func (m *InstanceManager) Get(instanceID string) *Instance {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.instances[instanceID]
}

func (m *InstanceManager) GetModelInstances(modelID string) []*Instance {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var out []*Instance
	for _, id := range m.byModel[modelID] {
		out = append(out, m.instances[id])
	}
	return out
}

// AllInstances returns every registered instance across all models. Used by the
// host liveness monitor to fire Cancel() on the remote instances of a host that
// just flipped absent.
func (m *InstanceManager) AllInstances() []*Instance {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]*Instance, 0, len(m.instances))
	for _, inst := range m.instances {
		out = append(out, inst)
	}
	return out
}

// PickInstance finds the best instance for a new job.
// Returns nil if all instances are at capacity.
// ModelHasAnyCapacity returns true if at least one instance for this model
// could accept a new job right now: a loaded/loading instance with HasCapacity,
// OR an unloaded/stopped instance (which has trivial capacity, just needs a
// cold start). Used by the scheduler's getFullModels so the "full" decision
// stays consistent with what PickInstance will actually return.
func (m *InstanceManager) ModelHasAnyCapacity(modelID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	ids := m.byModel[modelID]
	for _, id := range ids {
		inst := m.instances[id]
		state := inst.State()
		switch state {
		case "loaded", "loading":
			if inst.HasCapacity() {
				return true
			}
		case "unloaded", "stopped":
			return true
		case "error":
			if inst.HasCapacity() {
				return true
			}
		}
	}
	return false
}

// ModelHasReachableRemoteCapacity reports whether this model has at least one
// REMOTE instance that is (a) on a reachable host and (b) able to accept a job
// right now (loaded/loading with capacity, or unloaded/stopped → trivial warm).
// A remote instance consumes ZERO spark VRAM, so when this is true the
// scheduler's local-VRAM feasibility gate (getFullModels) must NOT mark the
// model "full" just because spark's GPU is busy — the job will run on the remote
// box. This is what lets gemma keep flowing to boringstack while ltx2 saturates
// the local GPU, instead of queuing behind spark VRAM pressure.
func (m *InstanceManager) ModelHasReachableRemoteCapacity(modelID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, id := range m.byModel[modelID] {
		inst := m.instances[id]
		if !inst.isRemote() {
			continue
		}
		if !m.hostReachable(inst.host) {
			continue
		}
		switch inst.State() {
		case "loaded", "loading":
			if inst.HasCapacity() {
				return true
			}
		case "unloaded", "stopped":
			return true
		case "error":
			if inst.HasCapacity() {
				return true
			}
		}
	}
	return false
}

func (m *InstanceManager) PickInstance(modelID string) *Instance {
	m.mu.RLock()
	defer m.mu.RUnlock()

	ids := m.byModel[modelID]
	if len(ids) == 0 {
		return nil
	}
	pool := make([]*Instance, 0, len(ids))
	for _, id := range ids {
		pool = append(pool, m.instances[id])
	}
	return pickFromPool(pool)
}

// pickFromPool selects the best instance among a candidate pool using the
// capacity-aware preference order. Returns nil if the pool is empty.
//
// MaxConcurrent applies regardless of lifecycle state. A loading instance that
// already has MaxConcurrent jobs reserved against it must not get more piled on
// — they would all unblock simultaneously once the load finishes and overwhelm
// the worker. So every branch below checks HasCapacity().
//
// For multi-instance models (max_instances > 1) the pool of Instance objects is
// fixed at startup, so falling through to cold-start a stopped sibling can never
// exceed max_instances. That is exactly the horizontal-scale behaviour the
// config asks for: when #0 is saturated, light up #1.
func pickFromPool(pool []*Instance) *Instance {
	if len(pool) == 0 {
		return nil
	}

	// 1. Loaded with capacity (least busy first)
	var bestLoaded *Instance
	for _, inst := range pool {
		if inst.State() == "loaded" && inst.HasCapacity() {
			if bestLoaded == nil || inst.ActiveJobs() < bestLoaded.ActiveJobs() {
				bestLoaded = inst
			}
		}
	}
	if bestLoaded != nil {
		return bestLoaded
	}

	// 2. Loading with capacity (job will wait for load to complete).
	// Preferred over cold-starting a sibling because the load is already
	// in flight — adding another cold start just wastes VRAM/time.
	for _, inst := range pool {
		if inst.State() == "loading" && inst.HasCapacity() {
			return inst
		}
	}

	// 3. Unloaded or stopped (needs cold start). HasCapacity is trivially
	// true for these (activeJobs == 0), but we still gate on it so the
	// rule "no instance is ever picked while at capacity" holds uniformly.
	for _, inst := range pool {
		s := inst.State()
		if (s == "unloaded" || s == "stopped") && inst.HasCapacity() {
			return inst
		}
	}

	// 4. Errored with capacity — retry only actual errored instances. A loaded
	// instance at MaxConcurrent must not fall through and get overfilled.
	for _, inst := range pool {
		if inst.State() == "error" && inst.HasCapacity() {
			return inst
		}
	}

	return nil
}

// PickInstanceForJob is the host-aware picker used by the dispatch path. It
// walks the model's placement chain IN ORDER (most-preferred host first) and
// returns the first host's instance that is (a) not in the job's excluded set,
// and (b) pickable per pickFromPool (has capacity / can cold-start). Lower-
// preference hosts stay unpicked while a higher one is available — so we never
// duplicate a model across boxes, only spill when the preferred host has no
// capacity or is excluded. spark (LocalHost) is always local + reachable and is
// the guaranteed final link in every offloaded model's chain.
//
// remoteEnabled=false pins the model to LOCAL hosts only (the per-model
// kill-switch): every remote host in the chain is skipped, leaving spark.
//
// Returns nil only when no eligible host has capacity right now (the scheduler
// then requeues the job and retries on the next tick).
func (m *InstanceManager) PickInstanceForJob(job *Job, remoteEnabled bool) *Instance {
	inst, _ := m.PickInstanceForJobWithReason(job, remoteEnabled)
	return inst
}

// placementReason classifies WHY a job landed on the host it did, for /v1/ps and
// the model.placed event. The chain is walked most-preferred-first:
//   - preferred: the head of the chain was chosen
//   - spill:     a higher-preference host was unavailable (full/excluded/absent)
//     so we fell to a later REMOTE host
//   - fallback:  we fell all the way to the local host (spark) because no remote
//     host in the chain was usable
//   - absent:    (annotation) the head host is confirmed unreachable right now
//   - excluded:  (annotation) the head host is in the job's excluded set
//
// The reason returned by PickInstanceForJobWithReason is the *chosen* host's
// reason (preferred/spill/fallback). "absent"/"excluded" describe why earlier
// hosts were skipped and surface as per-instance annotations in /v1/ps.
const (
	reasonPreferred = "preferred"
	reasonSpill     = "spill"
	reasonFallback  = "fallback"
	reasonAbsent    = "absent"
	reasonExcluded  = "excluded"
)

// PickInstanceForJobWithReason is PickInstanceForJob plus the placement_reason
// classifying why the chosen host was picked. A confirmed-absent remote host is
// SKIPPED here (its liveness poll flipped it absent) so a job is never dispatched
// to a dead box only to time out — it spills immediately to the next placement,
// with spark the guaranteed reachable final link.
func (m *InstanceManager) PickInstanceForJobWithReason(job *Job, remoteEnabled bool) (*Instance, string) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	mc, ok := m.config.Models[job.ModelID]
	if !ok {
		// Unknown model: fall back to the flat pool (back-compat).
		return m.pickFromModelLocked(job.ModelID), reasonPreferred
	}
	placements := mc.PlacementsOrDefault()

	skippedHigher := false // a more-preferred host was passed over
	for idx, host := range placements {
		if job.HostExcluded(host) {
			skippedHigher = true
			continue
		}
		isLocal := m.config.HostIsLocal(host)
		if !isLocal && !remoteEnabled {
			skippedHigher = true
			continue // kill-switch: skip remote hosts, spill to local
		}
		if !isLocal && !m.hostReachable(host) {
			skippedHigher = true
			continue // liveness poll confirmed this host absent — skip, don't dispatch into the dark
		}
		inst := m.instanceForHostLocked(job.ModelID, host)
		if inst == nil {
			skippedHigher = true
			continue
		}
		if isLocal && m.localColdStartShouldSpillLocked(job, inst, remoteEnabled) {
			skippedHigher = true
			continue
		}
		if !m.instanceCanAcceptPlacementLocked(job.ModelID, inst) {
			skippedHigher = true
			continue
		}
		if picked := pickFromPool([]*Instance{inst}); picked != nil {
			// pickFromPool on a single-element pool returns that instance only
			// when it has capacity / can cold-start; a returned errored
			// instance (branch 4) is still a legit retry target on its host.
			reason := reasonPreferred
			if isLocal && (idx > 0 || skippedHigher) {
				reason = reasonFallback
			} else if skippedHigher {
				reason = reasonSpill
			}
			return picked, reason
		}
		skippedHigher = true
	}
	return nil, ""
}

func (m *InstanceManager) localColdStartShouldSpillLocked(job *Job, inst *Instance, remoteEnabled bool) bool {
	state := inst.State()
	if state == "loaded" || state == "loading" {
		return false
	}
	if !remoteEnabled || !m.localHostHasActiveJobsLocked(job.ModelID) {
		return false
	}
	return m.jobHasPickableRemotePlacementLocked(job, remoteEnabled)
}

func (m *InstanceManager) localHostHasActiveJobsLocked(excludeModelID string) bool {
	for _, inst := range m.instances {
		if inst.isRemote() || inst.ModelID == excludeModelID {
			continue
		}
		if inst.ActiveJobs() > 0 {
			return true
		}
	}
	return false
}

func (m *InstanceManager) jobHasPickableRemotePlacementLocked(job *Job, remoteEnabled bool) bool {
	if !remoteEnabled {
		return false
	}
	mc, ok := m.config.Models[job.ModelID]
	if !ok {
		return false
	}
	for _, host := range mc.PlacementsOrDefault() {
		if m.config.HostIsLocal(host) || job.HostExcluded(host) || !m.hostReachable(host) {
			continue
		}
		inst := m.instanceForHostLocked(job.ModelID, host)
		if inst != nil && pickFromPool([]*Instance{inst}) != nil {
			return true
		}
	}
	return false
}

func (m *InstanceManager) instanceCanAcceptPlacementLocked(modelID string, inst *Instance) bool {
	if inst.isRemote() || inst.State() == "loaded" || inst.State() == "loading" {
		return true
	}
	return inst.memoryGB <= m.freeGBLocked()+m.reclaimableIdleGBLocked(modelID)+1e-9
}

// pickFromModelLocked is the flat (host-agnostic) pick over all of a model's
// instances. Caller holds m.mu.
func (m *InstanceManager) pickFromModelLocked(modelID string) *Instance {
	ids := m.byModel[modelID]
	if len(ids) == 0 {
		return nil
	}
	pool := make([]*Instance, 0, len(ids))
	for _, id := range ids {
		pool = append(pool, m.instances[id])
	}
	return pickFromPool(pool)
}

// instanceForHostLocked returns the model's instance placed on the given host,
// or nil. Caller holds m.mu.
func (m *InstanceManager) instanceForHostLocked(modelID, host string) *Instance {
	for _, id := range m.byModel[modelID] {
		inst := m.instances[id]
		if inst.host == host || (m.config.HostIsLocal(host) && m.config.HostIsLocal(inst.host)) {
			return inst
		}
	}
	return nil
}

func (m *InstanceManager) IsLoaded(modelID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, id := range m.byModel[modelID] {
		if m.instances[id].State() == "loaded" {
			return true
		}
	}
	return false
}

// FreeGB returns free VRAM (budget minus usage minus reservations).
func (m *InstanceManager) FreeGB() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.freeGBLocked()
}

// ReclaimableIdleGB returns the total VRAM (GB) currently held by loaded
// instances that have no active jobs — i.e. memory the scheduler can free on
// demand by evicting idle instances to make room for another model. Combined
// with FreeGB this is the true upper bound on what a not-yet-loaded model
// could be given without preempting in-flight work. Instances of
// excludeModelID are skipped: the caller is deciding whether to load that
// model and must not count its own idle residency toward "can it fit".
func (m *InstanceManager) ReclaimableIdleGB(excludeModelID string) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.reclaimableIdleGBLocked(excludeModelID)
}

func (m *InstanceManager) freeGBLocked() float64 {
	return m.budgetGB - m.usedGB - m.reservedGB
}

func (m *InstanceManager) reclaimableIdleGBLocked(excludeModelID string) float64 {
	var total float64
	for _, inst := range m.instances {
		if inst.ModelID == excludeModelID {
			continue
		}
		if inst.State() != "loaded" || inst.ActiveJobs() > 0 {
			continue
		}
		total += inst.memoryGB
	}
	return total
}

// TotalActiveJobs returns the number of in-flight jobs across every instance.
// Used by drain/graceful-shutdown to decide when it is safe to bounce.
func (m *InstanceManager) TotalActiveJobs() int {
	m.mu.RLock()
	insts := make([]*Instance, 0, len(m.instances))
	for _, inst := range m.instances {
		insts = append(insts, inst)
	}
	m.mu.RUnlock()

	total := 0
	for _, inst := range insts {
		total += inst.ActiveJobs()
	}
	return total
}

// ModelActiveJobs returns the number of in-flight jobs across all instances of
// one model. Used by the conflict-group gate to enforce mutual exclusion.
func (m *InstanceManager) ModelActiveJobs(modelID string) int {
	insts := m.GetModelInstances(modelID)
	total := 0
	for _, inst := range insts {
		total += inst.ActiveJobs()
	}
	return total
}

// BudgetGB returns the total VRAM budget.
func (m *InstanceManager) BudgetGB() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.budgetGB
}

// actualVRAMFromPidMap sums actual VRAM for all known worker PIDs.
// Caller must hold m.mu.RLock.
func (m *InstanceManager) actualVRAMFromPidMap(pidVRAM map[int]int64) float64 {
	var total float64
	for _, inst := range m.instances {
		inst.mu.Lock()
		if inst.cmd != nil && inst.cmd.Process != nil {
			// Tree-sum: worker may have spawned subprocesses (llama-server,
			// CUDA forks, ffmpeg) that hold VRAM independently.
			bytes := treeVRAMBytes(inst.cmd.Process.Pid, pidVRAM)
			total += float64(bytes) / (1024 * 1024 * 1024)
		}
		inst.mu.Unlock()
	}
	return total
}

// instanceMemSnapshot captures actual memory usage for a single loaded
// instance. Used by the memory watchdog to detect drift between configured
// estimates and reality.
type instanceMemSnapshot struct {
	InstanceID   string
	ModelID      string
	PID          int
	TreeVRAMGB   float64
	TreeRSSGB    float64
	ConfiguredGB float64
}

// snapshotInstanceMemory returns one entry per loaded instance with current
// process-tree VRAM and RSS. pidVRAM is passed in so callers can amortise the
// nvidia-smi exec across multiple consumers.
func (m *InstanceManager) snapshotInstanceMemory(pidVRAM map[int]int64) []instanceMemSnapshot {
	type pending struct {
		id      string
		modelID string
		pid     int
	}
	m.mu.RLock()
	var pendings []pending
	for _, inst := range m.instances {
		if inst.State() != "loaded" && inst.State() != "active" {
			continue
		}
		inst.mu.Lock()
		if inst.cmd != nil && inst.cmd.Process != nil {
			pendings = append(pendings, pending{
				id:      inst.InstanceID,
				modelID: inst.ModelID,
				pid:     inst.cmd.Process.Pid,
			})
		}
		inst.mu.Unlock()
	}
	configuredByModel := make(map[string]float64)
	for id, mc := range m.config.Models {
		configuredByModel[id] = mc.MemoryGB
	}
	m.mu.RUnlock()

	out := make([]instanceMemSnapshot, 0, len(pendings))
	for _, p := range pendings {
		out = append(out, instanceMemSnapshot{
			InstanceID:   p.id,
			ModelID:      p.modelID,
			PID:          p.pid,
			TreeVRAMGB:   float64(treeVRAMBytes(p.pid, pidVRAM)) / (1024 * 1024 * 1024),
			TreeRSSGB:    treeRSSAnonMB(p.pid) / 1024,
			ConfiguredGB: configuredByModel[p.modelID],
		})
	}
	return out
}

// UpdateModelMemoryGB replaces the in-memory MemoryGB for a model. Used by the
// memory watchdog after it patches local/config.json so subsequent dispatch
// decisions use the new value without an arbiter restart.
func (m *InstanceManager) UpdateModelMemoryGB(modelID string, newGB float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	mc, ok := m.config.Models[modelID]
	if !ok {
		return
	}
	mc.MemoryGB = newGB
	m.config.Models[modelID] = mc
}

// ActualVRAMGB returns the total actual VRAM in GB used by all known worker
// subprocesses, as reported by nvidia-smi. Returns 0 if nvidia-smi fails.
func (m *InstanceManager) ActualVRAMGB() float64 {
	pidVRAM := GetPerProcessVRAM()
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.actualVRAMFromPidMap(pidVRAM)
}

// EffectiveUsedGB returns max(actual VRAM, bookkeeping allocated) + reserved.
// This is the "true pressure" on the GPU.
func (m *InstanceManager) EffectiveUsedGB() (effective, actual, allocated, reserved float64) {
	actual = m.ActualVRAMGB()
	m.mu.RLock()
	allocated = m.usedGB
	reserved = m.reservedGB
	m.mu.RUnlock()
	if actual > allocated {
		effective = actual + reserved
	} else {
		effective = allocated + reserved
	}
	return
}

// ReconcileUsedGB adjusts bookkeeping usedGB upward if actual VRAM exceeds it
// by more than the given tolerance. Prevents bookkeeping from going stale.
func (m *InstanceManager) ReconcileUsedGB(actualGB, tolerance float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if actualGB > m.usedGB+tolerance {
		slog.Warn("vram reconcile: adjusting bookkeeping upward",
			"old_used_gb", m.usedGB, "actual_gb", actualGB)
		m.usedGB = actualGB
	}
}

// ReserveMemory reserves VRAM if it fits under the budget. Prefer
// ReserveMemoryFor when a specific instance owns the reservation — that path
// also marks the instance so death-path cleanup is automatic.
func (m *InstanceManager) ReserveMemory(gb float64) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.usedGB+gb > m.budgetGB-m.reservedGB {
		slog.Info("VRAM reserve failed", "needed_gb", gb, "used_gb", m.usedGB,
			"reserved_gb", m.reservedGB, "budget_gb", m.budgetGB,
			"free_gb", m.budgetGB-m.usedGB-m.reservedGB)
		return false
	}
	m.usedGB += gb
	slog.Info("VRAM reserved", "gb", gb, "used_gb", m.usedGB, "free_gb", m.budgetGB-m.usedGB-m.reservedGB)
	return true
}

// ReserveMemoryFor reserves VRAM and tags the instance so the reservation can
// be matched back to it during reconciliation or death-path cleanup. Use this
// for any reservation tied to an instance lifetime.
// ReserveMemoryFor reserves gb of VRAM budget on behalf of inst. It is
// idempotent per instance and race-safe: if inst already holds a reservation
// (vramHeld), it returns true WITHOUT double-counting. Two goroutines can race
// into ensureLoaded for the same instance (the dispatch path and the
// background tryPreload goroutine both see state=="stopped" before either has
// flipped it to "loading"); without this guard both call ReserveMemory(gb),
// usedGB is inflated by one memoryGB, and because vramHeld is a single bool the
// matching ReleaseMemoryFor only frees it once — a permanent budget leak that
// AuditVRAMConsistency flags as drift and that eventually wedges the queue.
// ReleaseMemoryFor was already idempotent the same way; this restores symmetry.
func (m *InstanceManager) ReserveMemoryFor(inst *Instance, gb float64) bool {
	// Remote instances hold ZERO audited VRAM on spark — their memory lives on
	// another box and is tracked separately (remoteHostBudget). Never touch
	// usedGB or set vramHeld for them, so AuditVRAMConsistency stays
	// local-CUDA-only. Report success: from the scheduler's view the "budget"
	// for a remote instance is always available locally.
	if inst.isRemote() {
		return true
	}
	inst.mu.Lock()
	if inst.vramHeld {
		inst.mu.Unlock()
		return true
	}
	inst.mu.Unlock()

	if !m.ReserveMemory(gb) {
		return false
	}

	inst.mu.Lock()
	if inst.vramHeld {
		// Lost the race: another goroutine reserved for this instance between
		// our check and now. Give back the extra reservation we just took so
		// usedGB stays consistent, and report success — it IS reserved.
		inst.mu.Unlock()
		m.ReleaseMemory(gb)
		return true
	}
	inst.vramHeld = true
	inst.mu.Unlock()
	return true
}

func (m *InstanceManager) ReleaseMemory(gb float64) {
	m.mu.Lock()
	m.usedGB -= gb
	if m.usedGB < 0 {
		m.usedGB = 0
	}
	m.mu.Unlock()
	slog.Info("VRAM released", "gb", gb, "used_gb_now", m.usedGB)
}

// ReleaseMemoryFor releases the VRAM reservation owned by an instance. It is
// idempotent: only the first call after a matching ReserveMemoryFor frees the
// memory. Returns the GB freed (0 when no reservation was held).
func (m *InstanceManager) ReleaseMemoryFor(inst *Instance) float64 {
	// Remote instances never reserved audited VRAM (see ReserveMemoryFor), so
	// there is nothing to release locally.
	if inst.isRemote() {
		return 0
	}
	inst.mu.Lock()
	if !inst.vramHeld {
		inst.mu.Unlock()
		return 0
	}
	gb := inst.memoryGB
	inst.vramHeld = false
	inst.mu.Unlock()
	m.ReleaseMemory(gb)
	return gb
}

// ShrinkReservationFor lowers an instance's declared memoryGB to targetGB
// and credits the difference back to free VRAM. Used by the memory watchdog
// after observing sustained over-declaration: e.g. vLLM with
// gpu-memory-utilization=0.30 and a 80GB declaration only ever uses ~25GB,
// so the other 55GB shouldn't be reserved indefinitely.
//
// Caller must have evidence (multiple samples) that actual usage is bounded
// by targetGB plus a safety margin — once we shrink, future loads will plan
// for the smaller footprint and an unexpectedly large allocation will OOM.
//
// No-op if instance doesn't currently hold VRAM, or if targetGB >= current.
// Returns the GB freed.
func (m *InstanceManager) ShrinkReservationFor(inst *Instance, targetGB float64) float64 {
	inst.mu.Lock()
	if !inst.vramHeld {
		inst.mu.Unlock()
		return 0
	}
	current := inst.memoryGB
	if targetGB >= current {
		inst.mu.Unlock()
		return 0
	}
	freed := current - targetGB
	inst.memoryGB = targetGB
	inst.mu.Unlock()

	m.mu.Lock()
	m.usedGB -= freed
	if m.usedGB < 0 {
		m.usedGB = 0
	}
	used := m.usedGB
	m.mu.Unlock()
	slog.Info("VRAM reservation shrunk to actual",
		"instance", inst.InstanceID, "from_gb", current, "to_gb", targetGB,
		"freed_gb", freed, "used_gb_now", used)
	return freed
}

// ReconcileFromInstances forces InstanceManager.usedGB to equal the sum of
// memoryGB across instances currently flagged vramHeld. This is the safety
// net for accounting leaks: any historical bug or new code path that drops a
// reservation without releasing it will be reclaimed within one watchdog
// interval. Returns the GB freed (0 when bookkeeping was already truthful).
//
// Why this is the source of truth: vramHeld is set inside ReserveMemoryFor
// under the same locks that bump usedGB, and cleared inside ReleaseMemoryFor
// under the same locks that decrement usedGB. So any divergence between
// usedGB and sum(vramHeld * memoryGB) is by definition a leak.
func (m *InstanceManager) ReconcileFromInstances() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	expected := 0.0
	for _, inst := range m.instances {
		if inst.isRemote() {
			continue // remote instances never enter the audited local ledger
		}
		inst.mu.Lock()
		if inst.vramHeld {
			expected += inst.memoryGB
		}
		inst.mu.Unlock()
	}
	const tolerance = 0.5 // GB; ignore tiny float drift
	if m.usedGB > expected+tolerance {
		freed := m.usedGB - expected
		slog.Warn("vram reconcile: reclaiming orphan bookkeeping",
			"old_used_gb", m.usedGB, "expected_gb", expected, "freed_gb", freed)
		m.dumpVRAMStateLocked("reconcile-mismatch")
		m.usedGB = expected
		return freed
	}
	return 0
}

// AuditVRAMConsistency asserts that bookkeeping (usedGB) matches the sum of
// memoryGB across instances flagged vramHeld. Call this after every load,
// unload, eviction, or anything that touches the VRAM ledger. The whole point:
// it should be trivially impossible for the ledger to drift, so if this ever
// fires we want a full snapshot of every instance's state to root-cause it.
//
// ctx is a short label ("post-load", "post-unload", "shutdown") used in the
// log so we can identify which path produced the drift.
func (m *InstanceManager) AuditVRAMConsistency(ctx string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	expected := 0.0
	holders := 0
	for _, inst := range m.instances {
		if inst.isRemote() {
			continue // remote instances hold zero audited VRAM on spark
		}
		inst.mu.Lock()
		if inst.vramHeld {
			expected += inst.memoryGB
			holders++
		}
		inst.mu.Unlock()
	}
	const tolerance = 0.5
	drift := m.usedGB - expected
	if drift < 0 {
		drift = -drift
	}
	if drift > tolerance {
		slog.Error("VRAM audit FAILED — bookkeeping drift",
			"ctx", ctx, "used_gb", m.usedGB, "expected_gb", expected,
			"drift_gb", drift, "holders", holders)
		m.dumpVRAMStateLocked("audit-drift:" + ctx)
		return
	}
	// Stronger invariant: when no instance holds VRAM, usedGB MUST be zero.
	if holders == 0 && m.usedGB > tolerance {
		slog.Error("VRAM audit FAILED — usedGB non-zero with no holders",
			"ctx", ctx, "used_gb", m.usedGB)
		m.dumpVRAMStateLocked("audit-no-holders:" + ctx)
	}
}

// dumpVRAMStateLocked logs every instance's VRAM-relevant state. Caller must
// hold m.mu. Used by audit/reconcile when a mismatch is detected so we can
// see which paths leaked.
func (m *InstanceManager) dumpVRAMStateLocked(reason string) {
	slog.Error("VRAM state dump", "reason", reason,
		"used_gb", m.usedGB, "reserved_gb", m.reservedGB, "budget_gb", m.budgetGB)
	for _, inst := range m.instances {
		inst.mu.Lock()
		slog.Error("  instance",
			"id", inst.InstanceID, "model", inst.ModelID,
			"state", inst.state, "vram_held", inst.vramHeld,
			"memory_gb", inst.memoryGB, "active_jobs", atomic.LoadInt32(&inst.activeJobs))
		inst.mu.Unlock()
	}
}

// EvictForGB tries to free enough memory by unloading idle instances.
// Uses tiered eviction: higher-numbered instances (excess capacity) are evicted
// before lower-numbered ones. A model's 4th instance is less important than any
// 3rd instance, a 3rd less important than any 2nd, etc.
//
// Eviction order within each tier:
//  1. Excess instances (models with most loaded instances first) — no minimum idle time
//  2. Last instances of a model — only if idle > 1 minute (protect recently-used models)
func (m *InstanceManager) EvictForGB(needed float64) error {
	m.mu.RLock()
	now := time.Now()

	// Count loaded instances per model
	modelLoaded := make(map[string]int)
	for _, inst := range m.instances {
		if inst.State() == "loaded" {
			modelLoaded[inst.ModelID]++
		}
	}

	type candidate struct {
		inst        *Instance
		lastUsed    time.Time
		loadedCount int // how many loaded instances this model has
	}

	var candidates []candidate
	for _, inst := range m.instances {
		st := inst.State()
		active := inst.ActiveJobs()
		if st != "loaded" || active > 0 {
			if st == "loaded" && active > 0 {
				slog.Info("evict skip: active", "instance", inst.InstanceID, "active_jobs", active)
			}
			if st == "loading" {
				slog.Info("evict skip: loading", "instance", inst.InstanceID)
			}
			continue
		}
		la := inst.LastActive()
		if la.IsZero() {
			continue
		}
		candidates = append(candidates, candidate{
			inst:        inst,
			lastUsed:    la,
			loadedCount: modelLoaded[inst.ModelID],
		})
	}
	m.mu.RUnlock()

	// Partition: excess instances (loadedCount > 1) vs last instances
	var excess, last []candidate
	for _, c := range candidates {
		if c.loadedCount > 1 {
			excess = append(excess, c)
		} else if now.Sub(c.lastUsed) >= 60*time.Second {
			// Last instance: only evict if idle > 1 minute
			last = append(last, c)
		}
	}

	// Sort excess: highest loadedCount first (evict 4th before 3rd),
	// then LRU within same tier
	for i := 0; i < len(excess); i++ {
		for j := i + 1; j < len(excess); j++ {
			if excess[j].loadedCount > excess[i].loadedCount ||
				(excess[j].loadedCount == excess[i].loadedCount && excess[j].lastUsed.Before(excess[i].lastUsed)) {
				excess[i], excess[j] = excess[j], excess[i]
			}
		}
	}
	// Sort last instances by LRU
	for i := 0; i < len(last); i++ {
		for j := i + 1; j < len(last); j++ {
			if last[j].lastUsed.Before(last[i].lastUsed) {
				last[i], last[j] = last[j], last[i]
			}
		}
	}

	ordered := append(excess, last...)

	freed := 0.0
	for _, c := range ordered {
		if freed >= needed {
			break
		}
		tier := "last"
		if c.loadedCount > 1 {
			tier = fmt.Sprintf("excess(%d loaded)", c.loadedCount)
		}
		idle := now.Sub(c.lastUsed).Seconds()
		slog.Info("evicting for memory", "instance", c.inst.InstanceID,
			"memory_gb", c.inst.memoryGB, "idle_seconds", idle, "tier", tier)
		if err := c.inst.Unload(); err != nil {
			slog.Error("eviction unload failed", "instance", c.inst.InstanceID, "error", err)
			continue
		}
		freed += m.ReleaseMemoryFor(c.inst)
		// Update loadedCount for remaining candidates of the same model
		for i := range ordered {
			if ordered[i].inst.ModelID == c.inst.ModelID {
				ordered[i].loadedCount--
			}
		}
	}

	if freed < needed {
		return fmt.Errorf("need %.1fGB but can only free %.1fGB (%d idle instances)", needed, freed, len(candidates))
	}
	return nil
}

// EvictForGBWithQueueInfo frees VRAM by unloading idle instances, preferring
// models with no queued jobs before models with queued jobs.
// Within each group, eviction is LRU (longest idle first).
//
// allowQueued, when non-nil, gates the has-queue tier: an instance whose model
// still has queued work is only evicted if allowQueued(modelID) returns true.
// This is how the scheduler's swap-patience guard prevents a cheap-job model
// from destroying an expensive model's sunk load between every job. Pass nil
// to allow all (memory-pressure relief must always be able to evict).
func (m *InstanceManager) EvictForGBWithQueueInfo(needed float64, queuedJobs map[string]int, allowQueued func(modelID string) bool) (float64, error) {
	m.mu.RLock()
	type candidate struct {
		inst     *Instance
		lastUsed time.Time
		queued   int
	}
	var candidates []candidate
	for _, inst := range m.instances {
		st := inst.State()
		active := inst.ActiveJobs()
		if st != "loaded" || active > 0 {
			continue
		}
		la := inst.LastActive()
		if la.IsZero() {
			continue
		}
		candidates = append(candidates, candidate{
			inst:     inst,
			lastUsed: la,
			queued:   queuedJobs[inst.ModelID],
		})
	}
	m.mu.RUnlock()

	// Partition: no queued jobs first, then has queued jobs
	var noQueue, hasQueue []candidate
	for _, c := range candidates {
		if c.queued == 0 {
			noQueue = append(noQueue, c)
		} else {
			if allowQueued != nil && !allowQueued(c.inst.ModelID) {
				continue // swap-patience guard: this model's queue outranks the challenger for now
			}
			hasQueue = append(hasQueue, c)
		}
	}

	// Sort each group by LRU (oldest idle first)
	sortLRU := func(s []candidate) {
		for i := 0; i < len(s); i++ {
			for j := i + 1; j < len(s); j++ {
				if s[j].lastUsed.Before(s[i].lastUsed) {
					s[i], s[j] = s[j], s[i]
				}
			}
		}
	}
	sortLRU(noQueue)
	sortLRU(hasQueue)

	ordered := append(noQueue, hasQueue...)
	freed := 0.0
	for _, c := range ordered {
		if freed >= needed {
			break
		}
		idle := time.Since(c.lastUsed).Seconds()
		group := "no_queue"
		if c.queued > 0 {
			group = fmt.Sprintf("has_queue(%d)", c.queued)
		}
		slog.Info("vram watchdog: evicting for memory pressure",
			"instance", c.inst.InstanceID, "memory_gb", c.inst.memoryGB,
			"idle_seconds", idle, "group", group)
		if err := c.inst.Unload(); err != nil {
			slog.Error("vram watchdog: eviction unload failed",
				"instance", c.inst.InstanceID, "error", err)
			continue
		}
		freed += m.ReleaseMemoryFor(c.inst)
	}

	if freed < needed {
		return freed, fmt.Errorf("need %.1fGB but can only free %.1fGB (%d idle instances)",
			needed, freed, len(candidates))
	}
	return freed, nil
}

// EvictIdleNoQueueModels unloads every loaded idle instance whose model has no
// queued, scheduled, or running work, but only when some other model does have
// pending work. This keeps warm-model residency subordinate to real queue
// pressure even when there is still nominal free VRAM.
func (m *InstanceManager) EvictIdleNoQueueModels(queuedJobs map[string]int) (int, error) {
	hasPending := false
	for _, n := range queuedJobs {
		if n > 0 {
			hasPending = true
			break
		}
	}
	if !hasPending {
		return 0, nil
	}

	m.mu.RLock()
	type candidate struct {
		inst     *Instance
		lastUsed time.Time
	}
	var candidates []candidate
	for _, inst := range m.instances {
		if queuedJobs[inst.ModelID] > 0 {
			continue
		}
		if inst.State() != "loaded" || inst.ActiveJobs() > 0 {
			continue
		}
		la := inst.LastActive()
		if la.IsZero() {
			continue
		}
		candidates = append(candidates, candidate{inst: inst, lastUsed: la})
	}
	m.mu.RUnlock()

	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].lastUsed.Before(candidates[i].lastUsed) {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	evicted := 0
	for _, c := range candidates {
		slog.Info("queue-priority eviction",
			"instance", c.inst.InstanceID,
			"model", c.inst.ModelID,
			"idle_seconds", time.Since(c.lastUsed).Seconds())
		if err := c.inst.Unload(); err != nil {
			slog.Error("queue-priority eviction failed", "instance", c.inst.InstanceID, "error", err)
			continue
		}
		m.ReleaseMemoryFor(c.inst)
		evicted++
	}
	return evicted, nil
}

// CreateReservation reserves VRAM budget space, evicting models if needed.
// Smart eviction order:
//  1. Instances idle past keepalive (would be evicted anyway)
//  2. Excess instances (model with MOST loaded instances loses an idle one first)
//  3. Regular LRU among remaining idle instances
func (m *InstanceManager) CreateReservation(memoryGB float64, label string, keepAliveSecs map[string]int) (string, error) {
	if !finitePositive(memoryGB) || memoryGB > m.budgetGB {
		return "", fmt.Errorf("memory_gb must be finite, > 0, and <= %.3g", m.budgetGB)
	}
	m.mu.Lock()
	available := m.budgetGB - m.usedGB - m.reservedGB
	m.mu.Unlock()

	if memoryGB > available {
		deficit := memoryGB - available
		if err := m.EvictForReservation(deficit, keepAliveSecs); err != nil {
			return "", err
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Re-check after eviction
	available = m.budgetGB - m.usedGB - m.reservedGB
	if memoryGB > available {
		return "", fmt.Errorf("need %.1fGB but only %.1fGB available after eviction", memoryGB, available)
	}

	id := genID()
	m.reservations[id] = &Reservation{
		ID:        id,
		MemoryGB:  memoryGB,
		Label:     label,
		CreatedAt: time.Now(),
	}
	m.reservedGB += memoryGB
	slog.Info("reservation created", "id", id, "memory_gb", memoryGB, "label", label)
	return id, nil
}

// ReleaseReservation releases a previously created reservation.
func (m *InstanceManager) ReleaseReservation(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	r, ok := m.reservations[id]
	if !ok {
		return false
	}
	m.reservedGB -= r.MemoryGB
	if m.reservedGB < 0 {
		m.reservedGB = 0
	}
	delete(m.reservations, id)
	slog.Info("reservation released", "id", id, "memory_gb", r.MemoryGB)
	return true
}

// ListReservations returns all active reservations.
func (m *InstanceManager) ListReservations() []Reservation {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]Reservation, 0, len(m.reservations))
	for _, r := range m.reservations {
		out = append(out, *r)
	}
	return out
}

// EvictForReservation frees memory using smart eviction order:
//  1. Expired keepalive instances
//  2. Excess instances (model with most loaded instances first)
//  3. Regular LRU
func (m *InstanceManager) EvictForReservation(needed float64, keepAliveSecs map[string]int) error {
	m.mu.RLock()
	now := time.Now()

	// Collect all evictable instances: loaded, no active jobs
	type candidate struct {
		inst        *Instance
		lastUsed    time.Time
		expired     bool
		loadedCount int // how many loaded instances this model has
	}

	// Count loaded instances per model
	modelLoaded := make(map[string]int)
	for _, inst := range m.instances {
		if inst.State() == "loaded" {
			modelLoaded[inst.ModelID]++
		}
	}

	var candidates []candidate
	for _, inst := range m.instances {
		st := inst.State()
		active := inst.ActiveJobs()
		if st != "loaded" || active > 0 {
			if active > 0 {
				slog.Info("evict-reservation skip: active", "instance", inst.InstanceID, "active_jobs", active)
			}
			continue
		}
		la := inst.LastActive()
		if la.IsZero() {
			continue
		}
		kas := 300 // default
		if s, ok := keepAliveSecs[inst.ModelID]; ok {
			kas = s
		}
		expired := now.Sub(la) > time.Duration(kas)*time.Second
		candidates = append(candidates, candidate{
			inst:        inst,
			lastUsed:    la,
			expired:     expired,
			loadedCount: modelLoaded[inst.ModelID],
		})
	}
	m.mu.RUnlock()

	// Sort into phases: expired first, then excess (most loaded model first), then LRU
	// Phase 1: expired keepalive, sorted LRU
	var phase1, phase2, phase3 []candidate
	for _, c := range candidates {
		if c.expired {
			phase1 = append(phase1, c)
		} else if c.loadedCount > 1 {
			phase2 = append(phase2, c)
		} else {
			phase3 = append(phase3, c)
		}
	}
	// Sort phase1 by LRU
	sortByLRU := func(s []candidate) {
		for i := 0; i < len(s); i++ {
			for j := i + 1; j < len(s); j++ {
				if s[j].lastUsed.Before(s[i].lastUsed) {
					s[i], s[j] = s[j], s[i]
				}
			}
		}
	}
	sortByLRU(phase1)
	// Sort phase2 by loaded count DESC, then LRU
	for i := 0; i < len(phase2); i++ {
		for j := i + 1; j < len(phase2); j++ {
			if phase2[j].loadedCount > phase2[i].loadedCount ||
				(phase2[j].loadedCount == phase2[i].loadedCount && phase2[j].lastUsed.Before(phase2[i].lastUsed)) {
				phase2[i], phase2[j] = phase2[j], phase2[i]
			}
		}
	}
	sortByLRU(phase3)

	ordered := append(append(phase1, phase2...), phase3...)

	freed := 0.0
	for _, c := range ordered {
		if freed >= needed {
			break
		}
		idle := now.Sub(c.lastUsed).Seconds()
		phase := "lru"
		if c.expired {
			phase = "expired"
		} else if c.loadedCount > 1 {
			phase = "excess"
		}
		slog.Info("evicting for reservation", "instance", c.inst.InstanceID,
			"memory_gb", c.inst.memoryGB, "idle_seconds", idle, "phase", phase)
		if err := c.inst.Unload(); err != nil {
			slog.Error("reservation eviction failed", "instance", c.inst.InstanceID, "error", err)
			continue
		}
		freed += m.ReleaseMemoryFor(c.inst)
	}

	if freed < needed {
		return fmt.Errorf("need %.1fGB but can only free %.1fGB (%d idle instances)", needed, freed, len(candidates))
	}
	return nil
}

// Snapshot returns the current state for the /v1/ps endpoint.
// Queries nvidia-smi for actual per-process VRAM usage.
func (m *InstanceManager) Snapshot() map[string]any {
	// Get real VRAM per PID (outside lock since it shells out)
	pidVRAM := GetPerProcessVRAM()

	m.mu.RLock()
	defer m.mu.RUnlock()

	// Group by model, sorted alphabetically
	type group struct {
		instances []*Instance
	}
	groups := make(map[string]*group)
	var modelOrder []string
	for modelID := range m.byModel {
		modelOrder = append(modelOrder, modelID)
		groups[modelID] = &group{}
		for _, id := range m.byModel[modelID] {
			groups[modelID].instances = append(groups[modelID].instances, m.instances[id])
		}
	}
	// Sort alphabetically
	for i := 0; i < len(modelOrder); i++ {
		for j := i + 1; j < len(modelOrder); j++ {
			if modelOrder[j] < modelOrder[i] {
				modelOrder[i], modelOrder[j] = modelOrder[j], modelOrder[i]
			}
		}
	}

	var models []map[string]any
	for _, modelID := range modelOrder {
		g := groups[modelID]
		totalActive := 0
		loadedCount := 0
		var totalMem float64
		for _, inst := range g.instances {
			totalActive += inst.ActiveJobs()
			s := inst.State()
			if s == "loaded" || s == "loading" {
				totalMem += inst.memoryGB
			}
			if s == "loaded" {
				loadedCount++
			}
		}

		stateStr := "unloaded"
		errorCount := 0
		for _, inst := range g.instances {
			if inst.State() == "error" {
				errorCount++
			}
		}
		if totalActive > 0 {
			stateStr = "active"
		} else if loadedCount > 0 {
			stateStr = "loaded"
		} else if len(g.instances) > 0 && g.instances[0].State() == "loading" {
			stateStr = "loading"
		} else if errorCount > 0 {
			stateStr = "error"
		}

		entry := map[string]any{
			"id":          modelID,
			"state":       stateStr,
			"memory_gb":   totalMem,
			"active_jobs": totalActive,
		}

		// Earliest residency start among currently-loaded instances, as float
		// epoch seconds. The /v1/ps cache uses it to count "actions done since the
		// model was loaded". Omitted when nothing is loaded.
		var earliestLoaded time.Time
		for _, inst := range g.instances {
			if inst.State() != "loaded" {
				continue
			}
			la := inst.LoadedAt()
			if la.IsZero() {
				continue
			}
			if earliestLoaded.IsZero() || la.Before(earliestLoaded) {
				earliestLoaded = la
			}
		}
		if !earliestLoaded.IsZero() {
			entry["loaded_at"] = float64(earliestLoaded.UnixNano()) / 1e9
		}

		if totalMem == 0 && len(g.instances) > 0 {
			entry["memory_gb"] = g.instances[0].memoryGB
		}

		// Idle time
		var idleSec *float64
		for _, inst := range g.instances {
			if inst.State() == "loaded" && inst.ActiveJobs() == 0 && !inst.LastActive().IsZero() {
				s := time.Since(inst.LastActive()).Seconds()
				if idleSec == nil || s < *idleSec {
					idleSec = &s
				}
			}
		}
		entry["idle_seconds"] = idleSec

		// Per-instance breakdown (always include for transparency)
		var instList []map[string]any
		var totalVRAMActual float64
		for _, inst := range g.instances {
			iState := inst.State()
			if inst.ActiveJobs() > 0 {
				iState = "active"
			}
			ie := map[string]any{
				"instance_id": inst.InstanceID,
				"state":       iState,
				"active_jobs": inst.ActiveJobs(),
				"host":        inst.hostOrLocal(),
			}
			if pr := inst.PlacementReason(); pr != "" {
				ie["placement_reason"] = pr
			}
			if inst.isRemote() {
				ie["remote"] = true
			}
			if m.condemned[inst.InstanceID] {
				ie["condemned"] = true
			}
			inst.mu.Lock()
			if inst.cmd != nil && inst.cmd.Process != nil {
				pid := inst.cmd.Process.Pid
				ie["pid"] = pid
				bytes := treeVRAMBytes(pid, pidVRAM)
				if bytes > 0 {
					vramGB := float64(bytes) / (1024 * 1024 * 1024)
					ie["vram_gb"] = vramGB
					totalVRAMActual += vramGB
				}
			}
			inst.mu.Unlock()
			if inst.State() == "loaded" && inst.ActiveJobs() == 0 && !inst.LastActive().IsZero() {
				s := time.Since(inst.LastActive()).Seconds()
				ie["idle_seconds"] = &s
			}
			instList = append(instList, ie)
		}
		// Count condemned instances
		condemnedCount := 0
		for _, inst := range g.instances {
			if m.condemned[inst.InstanceID] {
				condemnedCount++
			}
		}

		// Any remote placement means the model spans hosts — always show the
		// per-instance breakdown so the host + placement_reason are visible.
		hasRemote := false
		for _, inst := range g.instances {
			if inst.isRemote() {
				hasRemote = true
				break
			}
		}
		if len(g.instances) > 1 || condemnedCount > 0 || hasRemote {
			entry["instances"] = instList
			entry["loaded_instances"] = loadedCount
			entry["total_instances"] = len(m.byModel[modelID])
			if condemnedCount > 0 {
				entry["condemned_instances"] = condemnedCount
			}
		} else {
			// Single instance: still show pid and vram at top level
			if len(instList) > 0 {
				if pid, ok := instList[0]["pid"]; ok {
					entry["pid"] = pid
				}
				if vram, ok := instList[0]["vram_gb"]; ok {
					entry["vram_gb"] = vram
				}
			}
		}
		if totalVRAMActual > 0 {
			entry["memory_gb"] = totalVRAMActual
		}

		models = append(models, entry)
	}

	// Sum actual VRAM from nvidia-smi across all our subprocesses
	totalActualVRAM := m.actualVRAMFromPidMap(pidVRAM)

	// Reservations
	var reservations []map[string]any
	for _, r := range m.reservations {
		reservations = append(reservations, map[string]any{
			"id":         r.ID,
			"memory_gb":  r.MemoryGB,
			"label":      r.Label,
			"created_at": r.CreatedAt.Unix(),
		})
	}

	snap := map[string]any{
		"vram_budget_gb":    m.budgetGB,
		"vram_allocated_gb": m.usedGB,
		"vram_actual_gb":    totalActualVRAM,
		"vram_reserved_gb":  m.reservedGB,
		"vram_free_gb":      m.budgetGB - m.usedGB - m.reservedGB,
		"models":            models,
		"reservations":      reservations,
	}
	return snap
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	out := make([]string, len(values))
	copy(out, values)
	return out
}

func (m *InstanceManager) newInstance(modelID, instanceID string, cfg ModelConfig) *Instance {
	inst := NewInstance(
		modelID, instanceID,
		cfg.MaxConcurrent,
		cfg.MemoryGB,
		m.pythonBin, m.projectRoot,
	)
	inst.workerCmd = cloneStrings(cfg.WorkerCmd)
	inst.adapterParams = cloneAdapterParams(cfg.AdapterParams)
	return inst
}

func (m *InstanceManager) ApplyModelConfig(modelID string, cfg ModelConfig) {
	for _, inst := range m.GetModelInstances(modelID) {
		inst.mu.Lock()
		inst.MaxConcurrent = cfg.MaxConcurrent
		inst.workerCmd = cloneStrings(cfg.WorkerCmd)
		inst.adapterParams = cloneAdapterParams(cfg.AdapterParams)
		if state := inst.state; state == "stopped" || state == "unloaded" || state == "error" {
			inst.memoryGB = cfg.MemoryGB
		}
		inst.mu.Unlock()
	}
}

func (m *InstanceManager) retireInstanceLocked(modelID, instanceID string, result map[string]any) {
	inst := m.instances[instanceID]
	if inst == nil {
		return
	}

	ids := m.byModel[modelID]
	for i, id := range ids {
		if id == instanceID {
			m.byModel[modelID] = append(ids[:i], ids[i+1:]...)
			break
		}
	}

	state := inst.State()
	active := inst.ActiveJobs()

	if state == "stopped" || state == "unloaded" || state == "error" {
		if state == "unloaded" {
			inst.Kill()
		}
		delete(m.instances, instanceID)
		result["removed"] = result["removed"].(int) + 1
		slog.Info("instance removed", "model", modelID, "instance", instanceID, "was", state)
		return
	}

	if state == "loaded" && active == 0 {
		slog.Info("evicting idle instance", "instance", instanceID, "reason", "reload_or_scale_down")
		if err := inst.Unload(); err != nil {
			slog.Error("instance unload failed", "instance", instanceID, "error", err)
		}
		m.usedGB -= inst.memoryGB
		if m.usedGB < 0 {
			m.usedGB = 0
		}
		inst.Kill()
		delete(m.instances, instanceID)
		result["removed"] = result["removed"].(int) + 1
		slog.Info("instance removed", "model", modelID, "instance", instanceID)
		return
	}

	m.condemned[instanceID] = true
	result["condemned"] = result["condemned"].(int) + 1
	slog.Info("instance condemned", "model", modelID, "instance", instanceID,
		"state", state, "active_jobs", active)
}

// localInstanceIDsLocked returns the model's LOCAL ("#N" pool) instance ids.
// Remote placement instances ("model@host") are registered per the placement
// chain and must never be counted or retired by local pool scaling — a scale
// or reload that consumed them would leave a remote-placed model with no
// dispatchable instance until the next restart. Caller holds m.mu.
func (m *InstanceManager) localInstanceIDsLocked(modelID string) []string {
	ids := make([]string, 0, len(m.byModel[modelID]))
	for _, id := range m.byModel[modelID] {
		if inst := m.instances[id]; inst != nil && !inst.isRemote() {
			ids = append(ids, id)
		}
	}
	return ids
}

// ScaleModel changes the number of LOCAL instances for a model at runtime.
// Scale up: creates new stopped instances.
// Scale down: removes idle instances immediately, condemns active ones.
// Remote placement instances are untouched.
func (m *InstanceManager) ScaleModel(modelID string, newCount int, cfg ModelConfig) map[string]any {
	m.mu.Lock()
	currentIDs := m.localInstanceIDsLocked(modelID)
	m.mu.Unlock()

	currentCount := len(currentIDs)
	result := map[string]any{"added": 0, "removed": 0, "condemned": 0}

	if newCount == currentCount {
		return result
	}

	if newCount > currentCount {
		// Scale up. A model whose placement chain has no local host gets no
		// local pool — its capacity lives on the fleet placements. Creating a
		// local worker here (e.g. the auto-wake guard un-parking a remote-only
		// LLM) would spawn a Python adapter with no implementation for the
		// model, which just crash-loops "Unknown model" against the registry.
		if !m.config.HasLocalPlacement(cfg) {
			return result
		}
		nextIdx := m.nextInstanceIndex(modelID)
		for i := 0; i < newCount-currentCount; i++ {
			idx := nextIdx + i
			instanceID := fmt.Sprintf("%s#%d", modelID, idx)
			inst := m.newInstance(modelID, instanceID, cfg)
			m.Register(inst)
			result["added"] = result["added"].(int) + 1
			slog.Info("instance added", "model", modelID, "instance", instanceID)
		}
		return result
	}

	// Scale down: remove from the end
	toRemove := currentIDs[newCount:]

	m.mu.Lock()
	defer m.mu.Unlock()

	for _, iid := range toRemove {
		m.retireInstanceLocked(modelID, iid, result)
	}

	return result
}

// ReloadModel replaces a model's LOCAL worker processes without touching other
// models. New instances are registered first so only the target model is
// cycled. Remote placement instances have no local subprocess to cycle; they
// are left in place, but any MISSING remote placement (e.g. destroyed by an
// earlier hard-kill) is re-registered so a reload heals the dispatch set.
func (m *InstanceManager) ReloadModel(modelID string, targetCount int, cfg ModelConfig) map[string]any {
	if targetCount < 0 {
		targetCount = 0
	}

	m.mu.Lock()
	currentIDs := m.localInstanceIDsLocked(modelID)
	m.mu.Unlock()

	result := map[string]any{"added": 0, "removed": 0, "condemned": 0}
	localWanted := m.config.HasLocalPlacement(cfg)
	if localWanted {
		nextIdx := m.nextInstanceIndex(modelID)
		for i := 0; i < targetCount; i++ {
			instanceID := fmt.Sprintf("%s#%d", modelID, nextIdx+i)
			inst := m.newInstance(modelID, instanceID, cfg)
			m.Register(inst)
			result["added"] = result["added"].(int) + 1
			slog.Info("replacement instance added", "model", modelID, "instance", instanceID)
		}
	}

	m.mu.Lock()
	for _, iid := range currentIDs {
		m.retireInstanceLocked(modelID, iid, result)
	}
	m.mu.Unlock()

	result["added"] = result["added"].(int) + m.registerMissingRemotePlacements(modelID, cfg)
	return result
}

// HardKillModel forcefully terminates all worker processes for one model.
// If recreate is true, fresh stopped instances are registered afterward using cfg.
func (m *InstanceManager) HardKillModel(modelID string, recreate bool, cfg *ModelConfig) map[string]any {
	m.mu.Lock()
	currentIDs := make([]string, len(m.byModel[modelID]))
	copy(currentIDs, m.byModel[modelID])
	m.byModel[modelID] = []string{}
	m.mu.Unlock()

	result := map[string]any{"killed": 0, "recreated": 0}

	for _, iid := range currentIDs {
		m.mu.Lock()
		inst := m.instances[iid]
		delete(m.instances, iid)
		delete(m.condemned, iid)
		m.mu.Unlock()
		if inst == nil {
			continue
		}

		state := inst.State()
		if state == "loaded" || state == "loading" || state == "unloading" {
			m.ReleaseMemoryFor(inst)
		}
		inst.Kill()
		result["killed"] = result["killed"].(int) + 1
		slog.Warn("instance hard killed", "model", modelID, "instance", iid, "state", state)
	}

	if recreate && cfg != nil {
		m.EnsureModel(modelID)
		recreated := 0
		if m.config.HasLocalPlacement(*cfg) && cfg.MaxInstances != nil && *cfg.MaxInstances > 0 {
			scaleResult := m.ScaleModel(modelID, *cfg.MaxInstances, *cfg)
			recreated += scaleResult["added"].(int)
		}
		// Remote placements were destroyed above too. They are only ever
		// registered from the placement chain (never by ScaleModel), so a
		// hard-kill that skipped this would leave a remote-placed model with
		// no dispatchable instance until the next arbiter restart — every
		// queued job requeues forever ("PickInstanceForJob returned nil").
		recreated += m.registerMissingRemotePlacements(modelID, *cfg)
		result["recreated"] = recreated
	}

	return result
}

// registerMissingRemotePlacements registers a fresh remote instance for every
// remote host in the model's placement chain that has no live instance,
// mirroring setupInstances. Existing instances (any state) are left alone.
// Returns the number of instances added.
func (m *InstanceManager) registerMissingRemotePlacements(modelID string, cfg ModelConfig) int {
	added := 0
	for _, host := range cfg.PlacementsOrDefault() {
		if m.config.HostIsLocal(host) {
			continue
		}
		hc, ok := m.config.Hosts[host]
		if !ok || hc.Addr == "" {
			slog.Warn("skipping remote placement with no host addr",
				"model", modelID, "host", host)
			continue
		}
		instanceID := fmt.Sprintf("%s@%s", modelID, host)
		m.mu.RLock()
		_, exists := m.instances[instanceID]
		m.mu.RUnlock()
		if exists {
			continue
		}
		inst := NewRemoteInstance(modelID, instanceID, host, hc.Addr,
			remoteModelTag(cfg, modelID), cfg.MaxConcurrent, cfg.MemoryGB)
		m.Register(inst)
		added++
		slog.Info("re-registered remote model placement",
			"model", modelID, "host", host, "addr", hc.Addr)
	}
	return added
}

// ReleaseAndCheck decrements activeJobs and checks if the instance is condemned
// (either hard-condemned from scale-down or soft-condemned for VRAM pressure).
// If condemned and now idle, evicts in a background goroutine.
func (m *InstanceManager) ReleaseAndCheck(inst *Instance) bool {
	atomic.AddInt32(&inst.activeJobs, -1)
	inst.mu.Lock()
	inst.lastActive = time.Now()
	inst.mu.Unlock()

	m.mu.Lock()
	isCondemned := m.condemned[inst.InstanceID]
	m.mu.Unlock()

	slog.Info("job finished on instance", "instance", inst.InstanceID,
		"active_jobs_remaining", inst.ActiveJobs(), "condemned", isCondemned)

	if isCondemned && inst.ActiveJobs() == 0 {
		go m.evictCondemned(inst)
		return true
	}
	return false
}

func (m *InstanceManager) evictCondemned(inst *Instance) {
	m.mu.Lock()
	if !m.condemned[inst.InstanceID] || inst.ActiveJobs() > 0 {
		m.mu.Unlock()
		return
	}
	delete(m.condemned, inst.InstanceID)
	m.mu.Unlock()

	slog.Info("evicting condemned instance", "instance", inst.InstanceID)
	if inst.State() == "loaded" {
		if err := inst.Unload(); err != nil {
			slog.Error("condemned unload failed", "instance", inst.InstanceID, "error", err)
		}
		m.ReleaseMemoryFor(inst)
	}

	// Scale-down: remove entirely
	inst.Kill()
	m.mu.Lock()
	delete(m.instances, inst.InstanceID)
	m.mu.Unlock()
	slog.Info("condemned instance removed", "instance", inst.InstanceID)
}

func (m *InstanceManager) nextInstanceIndex(modelID string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	maxIdx := -1
	// Check all known instances (active + condemned still in map)
	for iid, inst := range m.instances {
		if inst.ModelID != modelID {
			continue
		}
		idx := parseInstanceIndex(iid, modelID)
		if idx > maxIdx {
			maxIdx = idx
		}
	}
	// Also check active list
	for _, iid := range m.byModel[modelID] {
		idx := parseInstanceIndex(iid, modelID)
		if idx > maxIdx {
			maxIdx = idx
		}
	}
	return maxIdx + 1
}

func parseInstanceIndex(instanceID, modelID string) int {
	if instanceID == modelID {
		return 0 // bare model_id
	}
	prefix := modelID + "#"
	if len(instanceID) > len(prefix) && instanceID[:len(prefix)] == prefix {
		if n, err := strconv.Atoi(instanceID[len(prefix):]); err == nil {
			return n
		}
	}
	return -1
}

// KillAll shuts down all subprocesses.
func (m *InstanceManager) KillAll() {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, inst := range m.instances {
		inst.Kill()
	}
}
