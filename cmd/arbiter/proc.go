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
	"sort"
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
	mu              sync.RWMutex
	instances       map[string]*Instance  // instanceID -> Instance
	byModel         map[string][]string   // modelID -> []instanceID
	condemned       map[string]bool       // instance IDs pending removal (scale-down, permanent)
	softCondemned   map[string]bool       // instance IDs condemned for VRAM pressure (repriveable)
	softCondemnedGB float64               // total VRAM of soft-condemned instances
	softBudgetGB    float64               // soft VRAM limit (logical target)
	hardBudgetGB    float64               // hard VRAM limit (physical max during burst)
	usedGB          float64
	reservations    map[string]*Reservation
	reservedGB      float64
	pythonBin       string
	projectRoot     string
}

func NewInstanceManager(softBudgetGB, hardBudgetGB float64, pythonBin, projectRoot string) *InstanceManager {
	return &InstanceManager{
		instances:     make(map[string]*Instance),
		byModel:       make(map[string][]string),
		condemned:     make(map[string]bool),
		softCondemned: make(map[string]bool),
		softBudgetGB:  softBudgetGB,
		hardBudgetGB:  hardBudgetGB,
		reservations:  make(map[string]*Reservation),
		pythonBin:     pythonBin,
		projectRoot:   projectRoot,
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

// FreeGB returns logical free VRAM (soft budget minus logical usage).
func (m *InstanceManager) FreeGB() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	logicalUsed := m.usedGB - m.softCondemnedGB
	return m.softBudgetGB - logicalUsed - m.reservedGB
}

// HardFreeGB returns physical free VRAM (hard limit minus physical usage).
func (m *InstanceManager) HardFreeGB() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.hardBudgetGB - m.usedGB - m.reservedGB
}

// ReserveMemory reserves VRAM if it fits under the soft budget (logical).
// Also checks that physical usage stays under the hard limit.
func (m *InstanceManager) ReserveMemory(gb float64) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	logicalUsed := m.usedGB - m.softCondemnedGB
	if logicalUsed+gb > m.softBudgetGB-m.reservedGB {
		return false
	}
	if m.usedGB+gb > m.hardBudgetGB-m.reservedGB {
		return false
	}
	m.usedGB += gb
	return true
}

func (m *InstanceManager) ReleaseMemory(gb float64) {
	m.mu.Lock()
	m.usedGB -= gb
	if m.usedGB < 0 {
		m.usedGB = 0
	}
	m.mu.Unlock()
	m.TryReprieve()
}

// CondemnAndBurstReserve condemns active instances to free logical VRAM,
// then reserves physical VRAM under the hard limit.
// Used when the soft limit would be exceeded but we can temporarily burst.
//
// Condemned instances continue running their current jobs. When they finish,
// they are evicted automatically. If the pressure is relieved before then
// (e.g., the model that caused the burst is unloaded), they are reprieved.
func (m *InstanceManager) CondemnAndBurstReserve(needed float64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check hard limit (physical)
	if m.usedGB+needed > m.hardBudgetGB-m.reservedGB {
		return fmt.Errorf("need %.1fGB but only %.1fGB physical VRAM available (hard limit %.0fGB)",
			needed, m.hardBudgetGB-m.usedGB-m.reservedGB, m.hardBudgetGB)
	}

	// Special case: model itself exceeds soft limit (e.g., 85GB model with 80GB soft)
	// Just allow it — "suffer mode"
	if needed > m.softBudgetGB-m.reservedGB {
		slog.Warn("model exceeds soft VRAM budget, allowing under hard limit",
			"needed_gb", needed, "soft_budget_gb", m.softBudgetGB, "hard_budget_gb", m.hardBudgetGB)
		m.usedGB += needed
		return nil
	}

	// How much logical space do we need to free?
	logicalUsed := m.usedGB - m.softCondemnedGB
	logicalFree := m.softBudgetGB - logicalUsed - m.reservedGB
	logicalDeficit := needed - logicalFree
	if logicalDeficit <= 0 {
		m.usedGB += needed
		return nil
	}

	// Find instances to condemn
	// Priority: models with most active instances (most redundant), then fewest active jobs
	type candidate struct {
		instanceID    string
		modelID       string
		memoryGB      float64
		activeJobs    int
		instanceCount int
	}

	modelInstCount := make(map[string]int)
	for _, ids := range m.byModel {
		for _, id := range ids {
			inst := m.instances[id]
			if inst != nil && (inst.State() == "loaded" || inst.State() == "loading") {
				modelInstCount[inst.ModelID]++
			}
		}
	}

	var candidates []candidate
	for _, inst := range m.instances {
		if inst.State() != "loaded" || inst.ActiveJobs() == 0 {
			continue // only condemn active loaded instances
		}
		if m.condemned[inst.InstanceID] || m.softCondemned[inst.InstanceID] {
			continue // no double counting!
		}
		candidates = append(candidates, candidate{
			instanceID:    inst.InstanceID,
			modelID:       inst.ModelID,
			memoryGB:      inst.memoryGB,
			activeJobs:    inst.ActiveJobs(),
			instanceCount: modelInstCount[inst.ModelID],
		})
	}

	// Sort: most instances first (most redundant), then fewest active jobs
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].instanceCount != candidates[j].instanceCount {
			return candidates[i].instanceCount > candidates[j].instanceCount
		}
		return candidates[i].activeJobs < candidates[j].activeJobs
	})

	var toCondemn []candidate
	condemned := 0.0
	for _, c := range candidates {
		if condemned >= logicalDeficit {
			break
		}
		toCondemn = append(toCondemn, c)
		condemned += c.memoryGB
	}

	if condemned < logicalDeficit {
		return fmt.Errorf("need %.1fGB logical but can only condemn %.1fGB (%d candidates)",
			logicalDeficit, condemned, len(candidates))
	}

	// Commit the condemns
	for _, c := range toCondemn {
		m.softCondemned[c.instanceID] = true
		m.softCondemnedGB += c.memoryGB
		// Remove from dispatch list
		ids := m.byModel[c.modelID]
		for j, id := range ids {
			if id == c.instanceID {
				m.byModel[c.modelID] = append(ids[:j], ids[j+1:]...)
				break
			}
		}
		slog.Info("soft-condemned instance for burst load",
			"instance", c.instanceID, "memory_gb", c.memoryGB,
			"model_instances", c.instanceCount, "active_jobs", c.activeJobs)
	}

	m.usedGB += needed
	return nil
}

// TryReprieve un-condemns soft-condemned instances when VRAM pressure is relieved.
// Called automatically after any memory-freeing event.
func (m *InstanceManager) TryReprieve() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.softCondemned) == 0 {
		return
	}

	for instanceID := range m.softCondemned {
		inst := m.instances[instanceID]
		if inst == nil {
			delete(m.softCondemned, instanceID)
			continue
		}

		// Would un-condemning this push logical usage over soft limit?
		newLogicalUsed := (m.usedGB - m.softCondemnedGB) + inst.memoryGB
		if newLogicalUsed > m.softBudgetGB-m.reservedGB {
			continue
		}

		// Reprieve: remove from soft-condemned, add back to dispatch list
		delete(m.softCondemned, instanceID)
		m.softCondemnedGB -= inst.memoryGB
		if m.softCondemnedGB < 0 {
			m.softCondemnedGB = 0
		}
		m.byModel[inst.ModelID] = append(m.byModel[inst.ModelID], instanceID)
		slog.Info("reprieved soft-condemned instance",
			"instance", instanceID, "memory_gb", inst.memoryGB)
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
		if inst.State() != "loaded" || inst.ActiveJobs() > 0 {
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
	logicalUsed := m.usedGB - m.softCondemnedGB
	available := m.softBudgetGB - logicalUsed - m.reservedGB
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
	logicalUsed = m.usedGB - m.softCondemnedGB
	available = m.softBudgetGB - logicalUsed - m.reservedGB
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
		inst         *Instance
		lastUsed     time.Time
		expired      bool
		loadedCount  int // how many loaded instances this model has
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
		if inst.State() != "loaded" || inst.ActiveJobs() > 0 {
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
			if m.condemned[inst.InstanceID] {
				ie["condemned"] = true
			}
			if m.softCondemned[inst.InstanceID] {
				ie["soft_condemned"] = true
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
		"vram_budget_gb":      m.softBudgetGB,
		"vram_hard_limit_gb":  m.hardBudgetGB,
		"vram_used_gb":        totalActualVRAM,
		"vram_configured_gb":  m.usedGB,
		"vram_logical_gb":     m.usedGB - m.softCondemnedGB,
		"vram_reserved_gb":    m.reservedGB,
		"models":              models,
		"reservations":        reservations,
	}
	if m.softCondemnedGB > 0 {
		snap["vram_condemned_gb"] = m.softCondemnedGB
	}
	return snap
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
			inst := NewInstance(
				modelID, instanceID,
				cfg.MaxConcurrent,
				cfg.MemoryGB,
				m.pythonBin, m.projectRoot,
			)
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
		inst := m.instances[iid]
		if inst == nil {
			continue
		}

		// Remove from dispatch list
		ids := m.byModel[modelID]
		for j, id := range ids {
			if id == iid {
				m.byModel[modelID] = append(ids[:j], ids[j+1:]...)
				break
			}
		}

		state := inst.State()
		active := inst.ActiveJobs()

		if state == "stopped" || state == "unloaded" || state == "error" {
			// Not loaded — just remove
			if state == "unloaded" {
				inst.Kill()
			}
			delete(m.instances, iid)
			result["removed"] = result["removed"].(int) + 1
			slog.Info("instance removed", "model", modelID, "instance", iid, "was", state)
		} else if state == "loaded" && active == 0 {
			// Idle loaded — evict and remove
			slog.Info("evicting idle instance for scale-down", "instance", iid)
			if err := inst.Unload(); err != nil {
				slog.Error("scale-down unload failed", "instance", iid, "error", err)
			}
			m.usedGB -= inst.memoryGB
			if m.usedGB < 0 {
				m.usedGB = 0
			}
			inst.Kill()
			delete(m.instances, iid)
			result["removed"] = result["removed"].(int) + 1
			slog.Info("instance removed", "model", modelID, "instance", iid)
		} else {
			// Active or loading — condemn
			m.condemned[iid] = true
			result["condemned"] = result["condemned"].(int) + 1
			slog.Info("instance condemned", "model", modelID, "instance", iid,
				"state", state, "active_jobs", active)
		}
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
	isSoftCondemned := m.softCondemned[inst.InstanceID]
	m.mu.Unlock()

	if (isCondemned || isSoftCondemned) && inst.ActiveJobs() == 0 {
		go m.evictCondemnedOrSoft(inst, isSoftCondemned)
		return true
	}
	return false
}

func (m *InstanceManager) evictCondemnedOrSoft(inst *Instance, wasSoft bool) {
	m.mu.Lock()
	if wasSoft {
		if !m.softCondemned[inst.InstanceID] || inst.ActiveJobs() > 0 {
			m.mu.Unlock()
			return
		}
		delete(m.softCondemned, inst.InstanceID)
		m.softCondemnedGB -= inst.memoryGB
		if m.softCondemnedGB < 0 {
			m.softCondemnedGB = 0
		}
	} else {
		if !m.condemned[inst.InstanceID] || inst.ActiveJobs() > 0 {
			m.mu.Unlock()
			return
		}
		delete(m.condemned, inst.InstanceID)
	}
	m.mu.Unlock()

	slog.Info("evicting condemned instance", "instance", inst.InstanceID, "soft", wasSoft)
	if inst.State() == "loaded" {
		if err := inst.Unload(); err != nil {
			slog.Error("condemned unload failed", "instance", inst.InstanceID, "error", err)
		}
		m.ReleaseMemory(inst.memoryGB) // also calls TryReprieve
	}

	if wasSoft {
		// Soft condemned: put back as available (unloaded) instance
		m.mu.Lock()
		m.byModel[inst.ModelID] = append(m.byModel[inst.ModelID], inst.InstanceID)
		m.mu.Unlock()
		slog.Info("soft-condemned instance returned to pool", "instance", inst.InstanceID)
	} else {
		// Hard condemned (scale-down): remove entirely
		inst.Kill()
		m.mu.Lock()
		delete(m.instances, inst.InstanceID)
		m.mu.Unlock()
		slog.Info("condemned instance removed", "instance", inst.InstanceID)
	}
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
