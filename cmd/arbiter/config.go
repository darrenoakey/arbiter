package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"
)

type ModelConfig struct {
	MemoryGB       float64           `json:"memory_gb"`
	MaxConcurrent  int               `json:"max_concurrent"`
	MaxInstances   *int              `json:"max_instances"`
	KeepAliveSec   int               `json:"keep_alive_seconds"`
	MaxRuntimeSec  int               `json:"max_runtime_seconds"`
	AvgInferenceMs float64           `json:"avg_inference_ms"`
	LoadMs         float64           `json:"load_ms"`
	AutoDownload   string            `json:"auto_download"`
	ModelPath      string            `json:"model_path"`
	Group          bool              `json:"group"`
	WorkerCmd      []string          `json:"worker_cmd,omitempty"`
	AdapterParams  map[string]string `json:"adapter_params,omitempty"`
	PressureIndex  *float64          `json:"pressure_index"` // 0..1 memory bandwidth fraction; sum across in-flight jobs must stay ≤ 1.0. Omitted/nil defaults to 1.0 (serialize). Explicit 0 means "no pressure" — runs alongside anything.
	// ConflictGroup names a hard mutual-exclusion set. Models sharing a
	// ConflictGroup never run at the same time (independent of pressure); models
	// in different/no groups are unconstrained by each other. GroupPriority
	// orders members within the group — a lower value runs first, and a member
	// is held while any higher-priority (lower-value) member of its group still
	// has pending work. Example: ltx2-denoise1 (group "ltx_denoise", prio 0) and
	// ltx2-denoise2 (group "ltx_denoise", prio 1) — all denoise1 drains before
	// any denoise2, and the two never co-load, while image-gen/encode run freely
	// alongside either.
	ConflictGroup string `json:"conflict_group,omitempty"`
	GroupPriority int    `json:"group_priority,omitempty"`
	// Placements is the ordered list of host ids this model may run on, most
	// preferred first. nil/empty means ["spark"] (local CUDA) — behaving
	// exactly as before the multi-machine seam existed. Phase 1 only records
	// the placement; routing across hosts arrives in Phase 2.
	Placements []string `json:"placements,omitempty"`
	// RemoteEnabled is a per-model kill switch for remote placement. nil means
	// enabled (the default). When false, the model is pinned to local hosts
	// regardless of its Placements. Phase 1 only stores the flag.
	RemoteEnabled *bool `json:"remote_enabled,omitempty"`
}

// PlacementsOrDefault returns the model's ordered host placements, defaulting
// to ["spark"] (the implicit local CUDA host) when none are configured. This
// keeps every existing model — which has no placements — behaving exactly as
// today.
func (m ModelConfig) PlacementsOrDefault() []string {
	if len(m.Placements) == 0 {
		return []string{LocalHost}
	}
	return m.Placements
}

// RemoteEnabledOrDefault reports whether remote placement is permitted for this
// model. Defaults to true (nil pointer).
func (m ModelConfig) RemoteEnabledOrDefault() bool {
	return m.RemoteEnabled == nil || *m.RemoteEnabled
}

// LocalHost is the implicit host id for spark's local CUDA backend. A model
// with no placements runs here, and an instance with host == LocalHost is the
// only kind that contributes to spark's audited VRAM ledger.
const LocalHost = "spark"

// HostConfig describes one executor in the fleet. The implicit host "spark"
// (LocalHost) is always present and local even when absent from this map.
type HostConfig struct {
	Addr     string  `json:"addr"`      // host:port of the remote backend (ollama/mlx); empty for the local host
	Kind     string  `json:"kind"`      // "cuda" (local spark) | "mlx" (remote Apple Silicon)
	BudgetGB float64 `json:"budget_gb"` // advisory memory budget on that host; not part of spark's audited ledger
}

// IsLocal reports whether a host id refers to spark's local CUDA backend.
func (c Config) HostIsLocal(hostID string) bool {
	if hostID == "" || hostID == LocalHost {
		return true
	}
	h, ok := c.Hosts[hostID]
	if !ok {
		// Unknown host id is treated as remote: it is not the local CUDA
		// backend, so it must never touch the audited VRAM ledger.
		return false
	}
	return h.Kind == "cuda" && h.Addr == ""
}

type Config struct {
	VRAMBudgetGB      float64 `json:"vram_budget_gb"`
	SystemRAMBudgetGB float64 `json:"system_ram_budget_gb"` // 0 = disabled. On unified-memory hardware (GB10), this caps total tree-RSS across all worker process trees so CPU-side allocations can't push the GPU driver into NV_ERR_NO_MEMORY.
	EmergencyFloorGB  float64 `json:"emergency_floor_gb"`   // MemAvailable floor below which the EmergencyGuardian force-kills the worst-offending instance (active jobs included). 0 = default 8GB. Must stay above earlyoom's SIGTERM threshold (~6GB at -m 5) so the informed kill happens before the host backstop.
	// SwapPatience scales the drain-before-swap guard. A loaded model with
	// queued work may only be evicted for a non-co-residable challenger once
	// the challenger's oldest queued job has waited longer than
	// victim.load_ms × SwapPatience. 0 = default (2.0). Negative disables the
	// guard entirely (restores the pre-2026-06-10 cost-blind behavior — the
	// live rollback knob). Without this guard a model with cheap fast jobs
	// (gemma, 5s) evicts a 7-minute-load model (ltx2 denoise) between every
	// single job, turning a 19-minute batch into 54+ minutes of load thrash.
	SwapPatience float64 `json:"swap_patience"`
	// EmergencyMemFreeFloorGB is the MemFree co-trigger for the
	// EmergencyGuardian. MemAvailable counts reclaimable page cache that the
	// NVIDIA driver cannot use (2026-06-10 host death: MemFree 8-12GB with
	// 25-65GB Cached while MemAvailable looked healthy and every
	// MemAvailable-keyed layer stayed silent). Below this MemFree floor the
	// guardian first drops page cache, and if MemFree stays under the floor it
	// force-kills the worst-offending instance. 0 = default 4GB; negative
	// disables the co-trigger.
	EmergencyMemFreeFloorGB float64 `json:"emergency_memfree_floor_gb"`
	Host                    string  `json:"host"`
	Port                    int     `json:"port"`
	OutputDir               string  `json:"output_dir"`
	ShareMount              string  `json:"share_mount"` // e.g. "/mnt/arbiter-store" — if set, monitored and remounted when unhealthy
	// AutoWakeSeconds is the grace period before a model that has queued jobs
	// but max_instances=0 is automatically scaled back to 1. A parked model
	// still accepts job submissions, so an operator who scales to 0 (to free
	// VRAM) and never restores it leaves the model silently dead with an
	// ever-growing queue — and the 0 is persisted, so restarts don't recover
	// it. 0 = default (300s); negative = guard disabled.
	AutoWakeSeconds int                    `json:"auto_wake_seconds"`
	Models          map[string]ModelConfig `json:"models"`
	// Hosts is the fleet of executors keyed by host id. The implicit host
	// "spark" (LocalHost) is always local CUDA and need not appear here. When
	// absent/empty, arbiter behaves exactly as today: a single local host.
	// Phase 1 only parses and stores this; cross-host routing is Phase 2.
	Hosts map[string]HostConfig `json:"hosts,omitempty"`
}

var mutableConfigMu sync.Mutex

// JobTypeToModel maps job type strings to model IDs.
var JobTypeToModel = map[string]string{
	"image-generate":          "flux2",
	"image-edit":              "flux2",
	"background-remove":       "birefnet",
	"caption":                 "moondream",
	"query":                   "moondream",
	"detect":                  "moondream",
	"point":                   "moondream",
	"transcribe":              "whisper-large",
	"tts-custom":              "tts-custom",
	"tts-clone":               "tts-clone",
	"tts-design":              "tts-design",
	"tts-kokoro":              "tts-kokoro",
	"talking-head":            "sonic",
	"talking-head-sadtalker":  "sadtalker",
	"lipsync":                 "latentsync",
	"video-generate":          "ltx2",
	"video-encode":            "ltx2-encode",
	"video-denoise1":          "ltx2-denoise1",
	"video-denoise2":          "ltx2-denoise2",
	"face-restore":            "face-restore",
	"face-restore-codeformer": "face-restore-codeformer",
	"face-embed":              "insightface",
	"aesthetic-score":         "aesthetic-scorer",
	"tts-voxtral":             "tts-voxtral",
	"lora-train":              "lora-train",
	"composite":               "composite",
	"embed-text":              "embed-text",
}

func LoadConfig(projectRoot string) (*Config, error) {
	cfgPath := filepath.Join(projectRoot, "local", "config.json")
	if _, err := os.Stat(cfgPath); os.IsNotExist(err) {
		cfgPath = filepath.Join(projectRoot, "local", "config.default.json")
	}

	data, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	cfg := &Config{
		// GB10 unified-memory safety ceiling. Each worker is hard-capped at
		// declared_memory_gb * 1.15 (see worker_main._apply_cuda_memory_cap),
		// eviction is synchronous (no transient co-residence), and the
		// scheduler keeps sum(loaded declared) <= this budget. So worst-case
		// real GPU usage is budget*1.15 + OS base (~6-8GB). The host has
		// 119.5GB and CUDA allocations escape cgroup accounting, so exceeding
		// it livelocks the machine with NO oom-kill (requires physical reset).
		// 90*1.15 = 103.5 + 8 = ~111.5GB leaves ~8GB headroom. Do NOT raise
		// above ~95 (95*1.15+8 = 117.3, the practical max) — 100/101 allowed
		// the documented double livelock of 2026-06-04.
		VRAMBudgetGB: 90,
		Host:         "0.0.0.0",
		Port:         8400,
	}
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Apply defaults
	for id, m := range cfg.Models {
		if m.MaxConcurrent < 1 {
			m.MaxConcurrent = 1
		}
		if m.MaxInstances == nil {
			one := 1
			m.MaxInstances = &one
		}
		if m.PressureIndex == nil {
			one := 1.0 // conservative default: serialize unknown models
			m.PressureIndex = &one
		}
		if m.KeepAliveSec == 0 {
			m.KeepAliveSec = 300
		}
		if m.MaxRuntimeSec == 0 {
			m.MaxRuntimeSec = 7200
		}
		cfg.Models[id] = m
	}

	// Environment overrides
	if v := os.Getenv("ARBITER_VRAM_BUDGET_GB"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.VRAMBudgetGB = f
		}
	}
	if v := os.Getenv("ARBITER_EMERGENCY_FLOOR_GB"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.EmergencyFloorGB = f
		}
	}
	if v := os.Getenv("ARBITER_SWAP_PATIENCE"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.SwapPatience = f
		}
	}
	if v := os.Getenv("ARBITER_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			cfg.Port = p
		}
	}
	if v := os.Getenv("ARBITER_HOST"); v != "" {
		cfg.Host = v
	}
	if v := os.Getenv("ARBITER_OUTPUT_DIR"); v != "" {
		cfg.OutputDir = v
	}

	return cfg, nil
}

func loadMutableConfigData(projectRoot string) (map[string]any, error) {
	cfgPath := filepath.Join(projectRoot, "local", "config.json")
	defaultPath := filepath.Join(projectRoot, "local", "config.default.json")

	var data map[string]any
	path := cfgPath
	raw, err := os.ReadFile(path)
	if err != nil {
		path = defaultPath
		raw, err = os.ReadFile(path)
		if err != nil {
			data = make(map[string]any)
			raw = nil
		}
	}
	if raw != nil {
		if err := json.Unmarshal(raw, &data); err != nil {
			return nil, fmt.Errorf("parse config: %w", err)
		}
	}
	if data == nil {
		data = make(map[string]any)
	}
	return data, nil
}

func writeConfigData(projectRoot string, data map[string]any) error {
	localDir := filepath.Join(projectRoot, "local")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		return fmt.Errorf("create local config dir: %w", err)
	}
	out, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}
	out = append(out, '\n')
	tmp, err := os.CreateTemp(localDir, ".config.*.tmp")
	if err != nil {
		return fmt.Errorf("create temp config: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(out); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write temp config: %w", err)
	}
	if err := tmp.Chmod(0o644); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("chmod temp config: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close temp config: %w", err)
	}
	return os.Rename(tmpName, filepath.Join(localDir, "config.json"))
}

func SaveModelConfig(projectRoot, modelID string, cfg ModelConfig) error {
	mutableConfigMu.Lock()
	defer mutableConfigMu.Unlock()
	data, err := loadMutableConfigData(projectRoot)
	if err != nil {
		return err
	}
	models, ok := data["models"].(map[string]any)
	if !ok {
		models = make(map[string]any)
		data["models"] = models
	}
	models[modelID] = cfg
	return writeConfigData(projectRoot, data)
}

// patchModelField updates a single field for a model in local/config.json,
// preserving every other key (including ones not in ModelConfig), so callers
// never clobber hand-edited fields.
func patchModelField(projectRoot, modelID, field string, value any) error {
	mutableConfigMu.Lock()
	defer mutableConfigMu.Unlock()
	data, err := loadMutableConfigData(projectRoot)
	if err != nil {
		return err
	}
	models, ok := data["models"].(map[string]any)
	if !ok {
		models = make(map[string]any)
		data["models"] = models
	}
	entry, ok := models[modelID].(map[string]any)
	if !ok {
		entry = make(map[string]any)
		models[modelID] = entry
	}
	entry[field] = value
	return writeConfigData(projectRoot, data)
}

// PatchModelMemoryGB is used by the drift watchdog to write back observed
// high-water marks.
func PatchModelMemoryGB(projectRoot, modelID string, newMemoryGB float64) error {
	return patchModelField(projectRoot, modelID, "memory_gb", newMemoryGB)
}

// PatchModelMaxInstances is used by the scheduler's auto-wake guard to undo a
// persisted scale-to-zero.
func PatchModelMaxInstances(projectRoot, modelID string, n int) error {
	return patchModelField(projectRoot, modelID, "max_instances", n)
}

func DeleteModelConfig(projectRoot, modelID string) error {
	mutableConfigMu.Lock()
	defer mutableConfigMu.Unlock()
	data, err := loadMutableConfigData(projectRoot)
	if err != nil {
		return err
	}
	models, ok := data["models"].(map[string]any)
	if ok {
		delete(models, modelID)
	}
	return writeConfigData(projectRoot, data)
}
