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

	mu         sync.Mutex
	state      string // "stopped", "starting", "loading", "loaded", "unloading", "error"
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

	pythonBin   string
	projectRoot string
	workerCmd   []string // custom worker command (overrides python)
	workerEnv   []string // extra env vars for worker
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

	var cmd *exec.Cmd
	if len(inst.workerCmd) > 0 {
		cmd = exec.Command(inst.workerCmd[0], inst.workerCmd[1:]...)
	} else {
		cmd = exec.Command(inst.pythonBin, "-m", "arbiter.worker_main", inst.ModelID)
	}
	cmd.Dir = inst.projectRoot
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	for _, e := range inst.workerEnv {
		cmd.Env = append(cmd.Env, e)
	}

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

	// Subprocess died — unblock all pending requests with an error response
	slog.Warn("readLoop exited, subprocess died", "instance", inst.InstanceID)
	inst.mu.Lock()
	inst.state = "error"
	inst.mu.Unlock()

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
	inst.lastActive = time.Now() // start keepalive timer from load time
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

// Unload sends the unload command, waits for response, then kills the process.
// State transitions: loaded -> unloading -> stopped. Never leaves a ghost process.
func (inst *Instance) Unload() error {
	oldState := inst.State()
	if oldState == "unloading" || oldState == "stopped" {
		slog.Info("unload: already in state", "instance", inst.InstanceID, "state", oldState)
		return nil
	}
	if inst.ActiveJobs() > 0 {
		return fmt.Errorf("refusing to unload %s: %d active jobs", inst.InstanceID, inst.ActiveJobs())
	}
	if oldState == "loading" {
		return fmt.Errorf("refusing to unload %s: currently loading", inst.InstanceID)
	}

	inst.mu.Lock()
	inst.state = "unloading"
	inst.mu.Unlock()
	slog.Info("unloading model", "instance", inst.InstanceID, "old_state", oldState)

	// Send unload command with timeout
	done := make(chan error, 1)
	go func() {
		resp, err := inst.sendAndReceive(map[string]any{"cmd": "unload"})
		if err != nil {
			done <- err
		} else if resp.Status != "ok" {
			done <- fmt.Errorf("unload response: %s", resp.Error)
		} else {
			done <- nil
		}
	}()

	select {
	case err := <-done:
		if err != nil {
			slog.Warn("unload command failed, killing process", "instance", inst.InstanceID, "error", err)
		} else {
			slog.Info("unload command succeeded, killing process", "instance", inst.InstanceID)
		}
	case <-time.After(30 * time.Second):
		slog.Warn("unload command timed out after 30s, killing process", "instance", inst.InstanceID)
	}

	// Always kill the process to ensure no ghost workers
	inst.Kill()
	slog.Info("model unloaded and process stopped", "instance", inst.InstanceID)
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
}

func NewInstanceManager(budgetGB float64, pythonBin, projectRoot string) *InstanceManager {
	return &InstanceManager{
		instances:    make(map[string]*Instance),
		byModel:      make(map[string][]string),
		condemned:    make(map[string]bool),
		budgetGB:     budgetGB,
		reservations: make(map[string]*Reservation),
		pythonBin:    pythonBin,
		projectRoot:  projectRoot,
	}
}

func (m *InstanceManager) Register(inst *Instance) {
	m.mu.Lock()
	defer m.mu.Unlock()
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

// FreeGB returns free VRAM (budget minus usage minus reservations).
func (m *InstanceManager) FreeGB() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.budgetGB - m.usedGB - m.reservedGB
}

// ReserveMemory reserves VRAM if it fits under the budget.
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

func (m *InstanceManager) ReleaseMemory(gb float64) {
	m.mu.Lock()
	m.usedGB -= gb
	if m.usedGB < 0 {
		m.usedGB = 0
	}
	m.mu.Unlock()
	slog.Info("VRAM released", "gb", gb, "used_gb_now", m.usedGB)
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
		m.ReleaseMemory(c.inst.memoryGB)
		freed += c.inst.memoryGB
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

// CreateReservation reserves VRAM budget space, evicting models if needed.
// Smart eviction order:
//  1. Instances idle past keepalive (would be evicted anyway)
//  2. Excess instances (model with MOST loaded instances loses an idle one first)
//  3. Regular LRU among remaining idle instances
func (m *InstanceManager) CreateReservation(memoryGB float64, label string, keepAliveSecs map[string]int) (string, error) {
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
		m.ReleaseMemory(c.inst.memoryGB)
		freed += c.inst.memoryGB
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
		if totalActive > 0 {
			stateStr = "active"
		} else if loadedCount > 0 {
			stateStr = "loaded"
		} else if len(g.instances) > 0 && g.instances[0].State() == "loading" {
			stateStr = "loading"
		}

		entry := map[string]any{
			"id":          modelID,
			"state":       stateStr,
			"memory_gb":   totalMem,
			"active_jobs": totalActive,
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
			}
			if m.condemned[inst.InstanceID] {
				ie["condemned"] = true
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
		// Count condemned instances
		condemnedCount := 0
		for _, inst := range g.instances {
			if m.condemned[inst.InstanceID] {
				condemnedCount++
			}
		}

		if len(g.instances) > 1 || condemnedCount > 0 {
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

func buildWorkerEnv(cfg ModelConfig) []string {
	if len(cfg.AdapterParams) == 0 {
		return nil
	}
	env := make([]string, 0, len(cfg.AdapterParams))
	for k, v := range cfg.AdapterParams {
		env = append(env, k+"="+v)
	}
	return env
}

func (m *InstanceManager) newInstance(modelID, instanceID string, cfg ModelConfig) *Instance {
	inst := NewInstance(
		modelID, instanceID,
		cfg.MaxConcurrent,
		cfg.MemoryGB,
		m.pythonBin, m.projectRoot,
	)
	inst.workerCmd = cloneStrings(cfg.WorkerCmd)
	inst.workerEnv = buildWorkerEnv(cfg)
	return inst
}

func (m *InstanceManager) ApplyModelConfig(modelID string, cfg ModelConfig) {
	for _, inst := range m.GetModelInstances(modelID) {
		inst.mu.Lock()
		inst.MaxConcurrent = cfg.MaxConcurrent
		inst.workerCmd = cloneStrings(cfg.WorkerCmd)
		inst.workerEnv = buildWorkerEnv(cfg)
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

// ScaleModel changes the number of instances for a model at runtime.
// Scale up: creates new stopped instances.
// Scale down: removes idle instances immediately, condemns active ones.
func (m *InstanceManager) ScaleModel(modelID string, newCount int, cfg ModelConfig) map[string]any {
	m.mu.Lock()
	currentIDs := make([]string, len(m.byModel[modelID]))
	copy(currentIDs, m.byModel[modelID])
	m.mu.Unlock()

	currentCount := len(currentIDs)
	result := map[string]any{"added": 0, "removed": 0, "condemned": 0}

	if newCount == currentCount {
		return result
	}

	if newCount > currentCount {
		// Scale up
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

// ReloadModel replaces a model's worker processes without touching other models.
// New instances are registered first so only the target model is cycled.
func (m *InstanceManager) ReloadModel(modelID string, targetCount int, cfg ModelConfig) map[string]any {
	if targetCount < 0 {
		targetCount = 0
	}

	m.mu.Lock()
	currentIDs := make([]string, len(m.byModel[modelID]))
	copy(currentIDs, m.byModel[modelID])
	m.mu.Unlock()

	result := map[string]any{"added": 0, "removed": 0, "condemned": 0}
	nextIdx := m.nextInstanceIndex(modelID)
	for i := 0; i < targetCount; i++ {
		instanceID := fmt.Sprintf("%s#%d", modelID, nextIdx+i)
		inst := m.newInstance(modelID, instanceID, cfg)
		m.Register(inst)
		result["added"] = result["added"].(int) + 1
		slog.Info("replacement instance added", "model", modelID, "instance", instanceID)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	for _, iid := range currentIDs {
		m.retireInstanceLocked(modelID, iid, result)
	}
	return result
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
		m.ReleaseMemory(inst.memoryGB)
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
