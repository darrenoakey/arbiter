package main

import (
	"encoding/json"
	"fmt"
	"sync"
)

// Backend is the execution seam behind an Instance. Today every instance runs
// as a local Python/llm subprocess (LocalProcBackend); Phase 2 will let a
// model run on a remote HTTP endpoint (RemoteHTTPBackend) without the scheduler
// caring which. The method set is exactly the lifecycle the scheduler/api call
// on an Instance today: Spawn/Load/InferRaw/GetPort/Cancel/Unload/Kill.
//
// Phase 1 contract: LocalProcBackend MOVES no behavior — it delegates byte-for-
// byte to the existing Instance methods, so models behave identically. The
// interface exists so Phase 2 can add a remote implementation; nothing routes
// through it yet (Instance still exposes its own methods, which the local
// backend wraps).
type Backend interface {
	// Spawn starts the worker if not already running (local: subprocess).
	Spawn() error
	// Load brings the model into memory and blocks until ready.
	Load(device string) error
	// InferRaw runs one inference; the caller owns slot accounting.
	InferRaw(jobID, jobType string, params json.RawMessage, outputDir string) (*WorkerResponse, error)
	// GetPort returns the HTTP port of a chat-proxy worker (llm/vllm).
	GetPort() (int, error)
	// Cancel signals an in-flight job to abort.
	Cancel() error
	// Unload releases the model (local: process-tree obliteration).
	Unload() error
	// Kill force-terminates the worker and verifies it is gone.
	Kill()
	// IsRemote reports whether this backend runs off-host. Remote backends hold
	// ZERO audited VRAM on spark — they must never enter usedGB /
	// AuditVRAMConsistency.
	IsRemote() bool
}

// LocalProcBackend is the today-path backend: a local worker subprocess managed
// by the embedded *Instance. Every method delegates to the existing Instance
// implementation so behavior is byte-identical to before the seam existed.
type LocalProcBackend struct {
	inst *Instance
}

func (b *LocalProcBackend) Spawn() error { return b.inst.Spawn() }
func (b *LocalProcBackend) Load(device string) error {
	return b.inst.Load(device)
}
func (b *LocalProcBackend) InferRaw(jobID, jobType string, params json.RawMessage, outputDir string) (*WorkerResponse, error) {
	return b.inst.InferRaw(jobID, jobType, params, outputDir)
}
func (b *LocalProcBackend) GetPort() (int, error) { return b.inst.GetPort() }
func (b *LocalProcBackend) Cancel() error         { return b.inst.Cancel() }
func (b *LocalProcBackend) Unload() error         { return b.inst.Unload() }
func (b *LocalProcBackend) Kill()                 { b.inst.Kill() }
func (b *LocalProcBackend) IsRemote() bool        { return false }

// RemoteHTTPBackend is the Phase-2 backend skeleton: a model served by a remote
// HTTP endpoint (ollama/MLX on another box). Phase 1 only defines the type so
// it compiles and the seam is in place; every method returns "not implemented".
// A remote backend reports IsRemote()==true so its instance never touches
// spark's audited VRAM ledger.
type RemoteHTTPBackend struct {
	host     string // host id, e.g. "boringstack"
	addr     string // host:port of the remote backend
	modelTag string // remote model tag (e.g. ollama "gemma4:26b-mlx")
}

var errRemoteNotImplemented = fmt.Errorf("remote backend not implemented (Phase 2)")

func (b *RemoteHTTPBackend) Spawn() error { return errRemoteNotImplemented }
func (b *RemoteHTTPBackend) Load(device string) error {
	return errRemoteNotImplemented
}
func (b *RemoteHTTPBackend) InferRaw(jobID, jobType string, params json.RawMessage, outputDir string) (*WorkerResponse, error) {
	return nil, errRemoteNotImplemented
}
func (b *RemoteHTTPBackend) GetPort() (int, error) { return 0, errRemoteNotImplemented }
func (b *RemoteHTTPBackend) Cancel() error         { return errRemoteNotImplemented }
func (b *RemoteHTTPBackend) Unload() error         { return errRemoteNotImplemented }
func (b *RemoteHTTPBackend) Kill()                 {}
func (b *RemoteHTTPBackend) IsRemote() bool        { return true }

// remoteHostBudget is the per-host advisory memory accounting for a remote
// executor. It is deliberately SEPARATE from InstanceManager.usedGB: spark's
// audited VRAM ledger stays local-CUDA-only, while remote capacity is tracked
// here as an advisory number (Phase 2 polls the host's /api/ps to keep it
// truthful). Nothing in AuditVRAMConsistency ever consults this.
type remoteHostBudget struct {
	mu       sync.Mutex
	hostID   string
	addr     string
	budgetGB float64
	usedGB   float64 // advisory; populated by Phase 2 remote dispatch/polling
}

func newRemoteHostBudget(hostID, addr string, budgetGB float64) *remoteHostBudget {
	return &remoteHostBudget{hostID: hostID, addr: addr, budgetGB: budgetGB}
}

// FreeGB returns advisory free capacity on the remote host.
func (r *remoteHostBudget) FreeGB() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.budgetGB - r.usedGB
}

// SetUsedGB records observed/advisory usage on the remote host.
func (r *remoteHostBudget) SetUsedGB(gb float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if gb < 0 {
		gb = 0
	}
	r.usedGB = gb
}
