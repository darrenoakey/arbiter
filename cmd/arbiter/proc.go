package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// Instance represents one adapter subprocess.
// For models with max_concurrent > 1, one subprocess handles multiple concurrent jobs.
// For others, one subprocess = one job at a time.
type Instance struct {
	ModelID      string
	InstanceID   string
	MaxConcurrent int

	mu         sync.Mutex
	state      string // "stopped", "starting", "unloaded", "loading", "loaded", "error"
	activeJobs int32  // atomic
	lastActive time.Time
	memoryGB   float64

	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	stderr io.ReadCloser

	// For concurrent mode: pending responses keyed by req_id
	pendingMu sync.Mutex
	pending   map[string]chan json.RawMessage

	// Background reader goroutine
	readerDone chan struct{}

	pythonBin  string
	projectRoot string
}

// WorkerResponse is the JSON response from the Python worker.
type WorkerResponse struct {
	Status string           `json:"status"`
	ReqID  string           `json:"req_id,omitempty"`
	Result json.RawMessage  `json:"result,omitempty"`
	Error  string           `json:"error,omitempty"`
	VRAMBytes int64         `json:"vram_bytes,omitempty"`
}

func NewInstance(modelID, instanceID string, maxConcurrent int, memoryGB float64, pythonBin, projectRoot string) *Instance {
	return &Instance{
		ModelID:       modelID,
		InstanceID:    instanceID,
		MaxConcurrent: maxConcurrent,
		state:         "stopped",
		memoryGB:      memoryGB,
		pending:       make(map[string]chan json.RawMessage),
		pythonBin:     pythonBin,
		projectRoot:   projectRoot,
	}
}

func (inst *Instance) State() string {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	return inst.state
}

func (inst *Instance) ActiveJobs() int {
	return int(atomic.LoadInt32(&inst.activeJobs))
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

	inst.state = "starting"
	slog.Info("spawning adapter subprocess", "instance", inst.InstanceID, "model", inst.ModelID)

	cmd := exec.Command(inst.pythonBin, "-m", "arbiter.worker_main", inst.ModelID)
	cmd.Dir = inst.projectRoot
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

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

	inst.cmd = cmd
	inst.stdin = stdin
	inst.stdout = bufio.NewScanner(stdout)
	inst.stdout.Buffer(make([]byte, 10*1024*1024), 10*1024*1024) // 10MB buffer for large responses
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
	slog.Warn("readLoop exited", "instance", inst.InstanceID)
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

	raw := <-ch
	var resp WorkerResponse
	json.Unmarshal(raw, &resp)
	return &resp, nil
}

// Load sends the load command and waits for completion.
func (inst *Instance) Load(device string) error {
	if err := inst.Spawn(); err != nil {
		return err
	}

	inst.mu.Lock()
	inst.state = "loading"
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
	inst.state = "loaded"
	inst.mu.Unlock()

	slog.Info("model loaded", "instance", inst.InstanceID)
	return nil
}

// Infer sends an inference command. Increments/decrements activeJobs.
func (inst *Instance) Infer(jobID, jobType string, params json.RawMessage, outputDir string) (*WorkerResponse, error) {
	atomic.AddInt32(&inst.activeJobs, 1)
	defer func() {
		atomic.AddInt32(&inst.activeJobs, -1)
		inst.mu.Lock()
		inst.lastActive = time.Now()
		inst.mu.Unlock()
	}()

	return inst.sendAndReceive(map[string]any{
		"cmd":        "infer",
		"req_id":     jobID,
		"params":     json.RawMessage(params),
		"output_dir": outputDir,
		"job_id":     jobID,
		"job_type":   jobType,
	})
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

// Cancel sends SIGUSR1 to the subprocess to set the cancel flag.
func (inst *Instance) Cancel() error {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.cmd != nil && inst.cmd.Process != nil {
		return inst.cmd.Process.Signal(syscall.SIGUSR1)
	}
	return nil
}

// Unload sends the unload command.
func (inst *Instance) Unload() error {
	resp, err := inst.sendAndReceive(map[string]any{"cmd": "unload"})
	if err != nil {
		return err
	}
	if resp.Status != "ok" {
		return fmt.Errorf("unload failed: %s", resp.Error)
	}
	inst.mu.Lock()
	inst.state = "unloaded"
	inst.mu.Unlock()
	slog.Info("model unloaded", "instance", inst.InstanceID)
	return nil
}

// Kill terminates the subprocess.
func (inst *Instance) Kill() {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.cmd != nil && inst.cmd.Process != nil {
		// Try graceful shutdown first
		if inst.stdin != nil {
			data, _ := json.Marshal(map[string]any{"cmd": "shutdown"})
			inst.stdin.Write(append(data, '\n'))
			inst.stdin.Close()
		}
		// Give it a moment, then force kill
		done := make(chan error, 1)
		go func() { done <- inst.cmd.Wait() }()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			inst.cmd.Process.Kill()
		}
	}
	inst.state = "stopped"
	inst.stdin = nil
	inst.cmd = nil
}

// InstanceManager manages all adapter instances for all models.
type InstanceManager struct {
	mu        sync.RWMutex
	instances map[string]*Instance  // instanceID -> Instance
	byModel   map[string][]string   // modelID -> []instanceID
	budgetGB  float64
	usedGB    float64
}

func NewInstanceManager(budgetGB float64) *InstanceManager {
	return &InstanceManager{
		instances: make(map[string]*Instance),
		byModel:   make(map[string][]string),
		budgetGB:  budgetGB,
	}
}

func (m *InstanceManager) Register(inst *Instance) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.instances[inst.InstanceID] = inst
	m.byModel[inst.ModelID] = append(m.byModel[inst.ModelID], inst.InstanceID)
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

// PickInstance finds the best instance for a new job.
// Returns nil if all instances are at capacity.
func (m *InstanceManager) PickInstance(modelID string) *Instance {
	m.mu.RLock()
	defer m.mu.RUnlock()

	ids := m.byModel[modelID]
	if len(ids) == 0 {
		return nil
	}

	// 1. Loaded with capacity (least busy first)
	var bestLoaded *Instance
	for _, id := range ids {
		inst := m.instances[id]
		if inst.State() == "loaded" && inst.HasCapacity() {
			if bestLoaded == nil || inst.ActiveJobs() < bestLoaded.ActiveJobs() {
				bestLoaded = inst
			}
		}
	}
	if bestLoaded != nil {
		return bestLoaded
	}

	// 2. Loading with capacity (job will wait for load to complete)
	for _, id := range ids {
		inst := m.instances[id]
		if inst.State() == "loading" && inst.HasCapacity() {
			return inst
		}
	}

	// 3. Unloaded or stopped with capacity (needs cold start)
	for _, id := range ids {
		inst := m.instances[id]
		s := inst.State()
		if s == "unloaded" || s == "stopped" {
			return inst
		}
	}

	// 4. All errored — return first for retry
	return m.instances[ids[0]]
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

func (m *InstanceManager) FreeGB() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.budgetGB - m.usedGB
}

// ReserveMemory adds to used VRAM. Returns false if would exceed budget.
func (m *InstanceManager) ReserveMemory(gb float64) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.usedGB+gb > m.budgetGB {
		return false
	}
	m.usedGB += gb
	return true
}

func (m *InstanceManager) ReleaseMemory(gb float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.usedGB -= gb
	if m.usedGB < 0 {
		m.usedGB = 0
	}
}

// EvictForGB tries to free enough memory by unloading idle models (LRU).
func (m *InstanceManager) EvictForGB(needed float64) error {
	m.mu.RLock()
	// Find evictable instances: loaded, no active jobs
	type candidate struct {
		inst     *Instance
		lastUsed time.Time
	}
	var candidates []candidate
	for _, inst := range m.instances {
		if inst.State() == "loaded" && inst.ActiveJobs() == 0 {
			candidates = append(candidates, candidate{inst, inst.LastActive()})
		}
	}
	m.mu.RUnlock()

	// Sort by last active (oldest first = LRU)
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].lastUsed.Before(candidates[i].lastUsed) {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	freed := 0.0
	for _, c := range candidates {
		if freed >= needed {
			break
		}
		slog.Info("evicting model", "instance", c.inst.InstanceID, "memory_gb", c.inst.memoryGB)
		if err := c.inst.Unload(); err != nil {
			slog.Error("eviction unload failed", "instance", c.inst.InstanceID, "error", err)
			continue
		}
		m.ReleaseMemory(c.inst.memoryGB)
		freed += c.inst.memoryGB
	}

	if freed < needed {
		return fmt.Errorf("need %.1fGB but can only free %.1fGB", needed, freed)
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
		if totalActive > 0 {
			stateStr = "active"
		} else if loadedCount > 0 {
			stateStr = "loaded"
		} else if g.instances[0].State() == "loading" {
			stateStr = "loading"
		}

		entry := map[string]any{
			"id":          modelID,
			"state":       stateStr,
			"memory_gb":   totalMem,
			"active_jobs": totalActive,
		}

		if totalMem == 0 {
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
			}
			inst.mu.Lock()
			if inst.cmd != nil && inst.cmd.Process != nil {
				pid := inst.cmd.Process.Pid
				ie["pid"] = pid
				if vram, ok := pidVRAM[pid]; ok {
					vramGB := float64(vram) / (1024 * 1024 * 1024)
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
		if len(g.instances) > 1 {
			entry["instances"] = instList
			entry["loaded_instances"] = loadedCount
			entry["total_instances"] = len(g.instances)
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
	var totalActualVRAM float64
	for _, inst := range m.instances {
		inst.mu.Lock()
		if inst.cmd != nil && inst.cmd.Process != nil {
			if vram, ok := pidVRAM[inst.cmd.Process.Pid]; ok {
				totalActualVRAM += float64(vram) / (1024 * 1024 * 1024)
			}
		}
		inst.mu.Unlock()
	}

	return map[string]any{
		"vram_budget_gb":    m.budgetGB,
		"vram_used_gb":      totalActualVRAM,
		"vram_configured_gb": m.usedGB,
		"models":            models,
	}
}

// KillAll shuts down all subprocesses.
func (m *InstanceManager) KillAll() {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, inst := range m.instances {
		inst.Kill()
	}
}
