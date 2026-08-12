package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	maximumModelInstances           = 128
	maximumModelConcurrency         = 1024
	maximumDurationSeconds          = 604800
	maximumLatentSyncRuntimeSeconds = 4000000
	maximumMetricMillis             = 604800000
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
	WorkerCmd      []string          `json:"worker_cmd,omitempty"`     // Restricted to repository-owned identities; see API.md.
	AdapterParams  map[string]string `json:"adapter_params,omitempty"` // Closed, typed per-worker schema; never arbitrary subprocess environment.
	PressureIndex  *float64          `json:"pressure_index"`           // 0..1 memory bandwidth fraction; sum across in-flight jobs must stay ≤ 1.0. Omitted/nil defaults to 1.0 (serialize). Explicit 0 means "no pressure" — runs alongside anything.
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
	// NoRemoteSpill, when true, prevents jobs from spilling to a lower-preference
	// remote host when a higher-preference remote host is reachable but has no
	// capacity. The job waits for the preferred host instead. Failover to a lower-
	// preference remote host still happens when the preferred host is unreachable.
	// spark (LocalHost) remains the final fallback when no remote host is usable.
	NoRemoteSpill *bool `json:"no_remote_spill,omitempty"`
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

// NoRemoteSpillOrDefault reports whether remote capacity spill is disabled for
// this model. Defaults to false (nil pointer), preserving the historical spill
// behavior.
func (m ModelConfig) NoRemoteSpillOrDefault() bool {
	return m.NoRemoteSpill != nil && *m.NoRemoteSpill
}

// LocalHost is the implicit host id for spark's local CUDA backend. A model
// with no placements runs here, and an instance with host == LocalHost is the
// only kind that contributes to spark's audited VRAM ledger.
const LocalHost = "spark"

// HostConfig describes one executor in the fleet. The implicit host "spark"
// (LocalHost) is always present and local even when absent from this map.
type HostConfig struct {
	// Addr is the base URL for remote chat (and, when OllamaAddr is empty, for
	// embed/health/ps too). For kind "nativ" this is the Nativ mlx-vlm-server
	// (OpenAI-compatible, typically :8080). For kind "mlx"/legacy ollama this is
	// the ollama base (typically :11434).
	Addr string `json:"addr"`
	// Kind: "cuda" (local spark) | "mlx" (remote ollama/MLX) | "nativ" (remote
	// Nativ mlx-vlm-server). Empty is treated as "mlx" for back-compat.
	Kind string `json:"kind"`
	// BudgetGB is advisory memory budget on that host; not part of spark's audited ledger.
	BudgetGB float64 `json:"budget_gb"`
	// OllamaAddr is an optional second endpoint for Ollama-native routes
	// (embed-text /api/embed, and — when Kind is nativ — host health /api/version
	// and loaded-model /api/ps). When empty, Addr is used for those routes too.
	// Set this when chat has moved to Nativ but embeddings still run on Ollama
	// on the same box (different port).
	OllamaAddr string `json:"ollama_addr,omitempty"`
}

// KindOrDefault returns the host kind, defaulting empty to "mlx".
func (h HostConfig) KindOrDefault() string {
	if h.Kind == "" {
		return "mlx"
	}
	return h.Kind
}

// OllamaBase returns the base URL for Ollama-native HTTP (embed/health/ps).
// Nativ hosts that still run Ollama for embeddings set OllamaAddr; everyone
// else falls back to Addr.
func (h HostConfig) OllamaBase() string {
	if h.OllamaAddr != "" {
		return h.OllamaAddr
	}
	return h.Addr
}

// IsNativ reports whether this host speaks Nativ's mlx-vlm-server protocol
// (health at /health, unload at POST /unload, chat at /v1/chat/completions).
func (h HostConfig) IsNativ() bool {
	return h.KindOrDefault() == "nativ"
}

// HasLocalPlacement reports whether the model's placement chain includes the
// local host — i.e. whether a local "#N" subprocess pool should exist for it.
// A remote-only model (e.g. an ollama-served LLM placed on fleet hosts) must
// never get a local subprocess: there is no Python adapter for it, so a local
// worker would just crash-loop.
func (c *Config) HasLocalPlacement(mc ModelConfig) bool {
	for _, host := range mc.PlacementsOrDefault() {
		if c.HostIsLocal(host) {
			return true
		}
	}
	return false
}

// IsLocal reports whether a host id refers to spark's local CUDA backend.
func (c *Config) HostIsLocal(hostID string) bool {
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
	// mu guards Models. The arbiter shares one *Config across the HTTP
	// handlers, the scheduler goroutine, the /v1/ps cache goroutine, and the
	// emergency guardian goroutine. Every Models read/write MUST go through
	// the GetModel/SetModel/DeleteModel/RangeModels accessors below — a bare
	// map access races with the per-second Snapshot and aborts the process
	// with "concurrent map read and map write" (the 2026-08-11 arbiter
	// crash). Unexported so json.Unmarshal skips it (a zero sync.RWMutex is
	// ready to use).
	mu sync.RWMutex
	// Hosts is the fleet of executors keyed by host id. The implicit host
	// "spark" (LocalHost) is always local CUDA and need not appear here. When
	// absent/empty, arbiter behaves exactly as today: a single local host.
	// Phase 1 only parses and stores this; cross-host routing is Phase 2.
	Hosts map[string]HostConfig `json:"hosts,omitempty"`
	// RemoteDisabled is the GLOBAL remote kill-switch. When true, NO model uses
	// any remote placement — every job pins to spark regardless of per-model
	// remote_enabled / placements. Flip it via PATCH /v1/remote {"enabled":false}
	// for an instant, fleet-wide retreat to local (e.g. the whole LAN is flaky).
	// Persisted so it survives restarts. Defaults to false (remote allowed).
	RemoteDisabled bool `json:"remote_disabled,omitempty"`

	// LLMCacheDisabled turns OFF the content-addressed chat-completion cache.
	// The cache is ON by default: every identical chat call (any chat model)
	// returns a stored JSON result without touching a model, and a nightly
	// sweeper evicts entries not hit within the TTL. Set true only to disable.
	LLMCacheDisabled bool `json:"llm_cache_disabled,omitempty"`
	// LLMCacheTTLHours is the age (by mtime, refreshed on every hit) past which
	// a cache entry is swept. 0 = default 32h. Negative is treated as default.
	LLMCacheTTLHours float64 `json:"llm_cache_ttl_hours,omitempty"`
	// LLMAliases maps semantic category names (e.g. "local-chat") to canonical
	// llm:* model ids. Resolution is admission-time only: each request is resolved
	// to a concrete model before scheduling, so aliases create no extra queues,
	// instances, or caches. See API.md for precedence and management.
	LLMAliases map[string]string `json:"llm_aliases,omitempty"`
}

type configFileSnapshot struct {
	data   []byte
	mode   os.FileMode
	exists bool
}

// LLMCacheEnabledOrDefault reports whether the chat-completion cache is on.
// It is on unless explicitly disabled.
func (c *Config) LLMCacheEnabledOrDefault() bool {
	return !c.LLMCacheDisabled
}

// LLMCacheTTL is the sweep age for cache entries. Defaults to 32h.
func (c *Config) LLMCacheTTL() time.Duration {
	if c.LLMCacheTTLHours > 0 {
		return time.Duration(c.LLMCacheTTLHours * float64(time.Hour))
	}
	return 32 * time.Hour
}

// RemoteAllowedFor reports whether a given model may use remote placements right
// now, folding the global kill-switch over the per-model flag. The global switch
// dominates: if remote is globally disabled, no model goes remote.
func (c *Config) RemoteAllowedFor(modelID string) bool {
	if c.RemoteDisabled {
		return false
	}
	mc, ok := c.GetModel(modelID)
	if !ok {
		return true
	}
	return mc.RemoteEnabledOrDefault()
}

// GetModel returns the config for one model under the read lock.
func (c *Config) GetModel(id string) (ModelConfig, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	m, ok := c.Models[id]
	return m, ok
}

// SetModel stores or replaces a model config under the write lock.
func (c *Config) SetModel(id string, m ModelConfig) {
	c.mu.Lock()
	c.Models[id] = m
	c.mu.Unlock()
}

// DeleteModel removes a model config under the write lock.
func (c *Config) DeleteModel(id string) {
	c.mu.Lock()
	delete(c.Models, id)
	c.mu.Unlock()
}

// RangeModels calls f for each model while holding the read lock. f must not
// block or mutate config. Callers that need a stable snapshot for offline
// work should use CloneModels.
func (c *Config) RangeModels(f func(id string, m ModelConfig) bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for id, m := range c.Models {
		if !f(id, m) {
			return
		}
	}
}

// CloneModels returns a shallow copy of the model map. Use when a caller
// needs a consistent snapshot it can iterate without holding the lock.
func (c *Config) CloneModels() map[string]ModelConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make(map[string]ModelConfig, len(c.Models))
	for id, m := range c.Models {
		out[id] = m
	}
	return out
}

// ModelIDs returns a snapshot of the model ids in arbitrary order.
func (c *Config) ModelIDs() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	ids := make([]string, 0, len(c.Models))
	for id := range c.Models {
		ids = append(ids, id)
	}
	return ids
}

var mutableConfigMu sync.Mutex

// JobTypeToModel maps job type strings to model IDs.
var JobTypeToModel = map[string]string{
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
	"video-generate-h3":       "minimax-h3-local",
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
	"demucs":                  "demucs",
	"rvc-train":               "rvc-train",
	"rvc-convert":             "rvc-convert",
	"voice-fit":               "voice-fit",
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
	if err := rejectDuplicateLLMAliasNames(data); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
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

	// Owner policy is unconditional: persisted configs cannot resurrect a
	// still-image generator after restart. Keep the file intact for audit and
	// operator recovery, but exclude offenders from the runnable config.
	for id, modelCfg := range cfg.Models {
		policyErr := validateModelWorkerPolicy(projectRoot, id, modelCfg, cfg.HasLocalPlacement(modelCfg))
		if policyErr != nil {
			slog.Error("security policy: model omitted from runnable config",
				"model", id, "policy", policyErr)
			delete(cfg.Models, id)
		}
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
	if !finitePositive(cfg.VRAMBudgetGB) {
		return nil, fmt.Errorf("vram_budget_gb must be finite and > 0")
	}
	for id, modelConfig := range cfg.Models {
		if err := validateModelConfigNumbers(id, modelConfig, cfg.VRAMBudgetGB); err != nil {
			return nil, fmt.Errorf("model %q: %w", id, err)
		}
	}
	if cfg.LLMAliases == nil {
		cfg.LLMAliases = map[string]string{}
	}
	if err := validateLLMAliases(cfg.LLMAliases, cfg.Models); err != nil {
		return nil, fmt.Errorf("llm_aliases: %w", err)
	}

	return cfg, nil
}

func rejectDuplicateLLMAliasNames(data []byte) error {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(data, &envelope); err != nil {
		return err
	}
	raw, exists := envelope["llm_aliases"]
	if !exists || string(raw) == "null" {
		return nil
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	if delimiter, ok := token.(json.Delim); !ok || delimiter != '{' {
		return fmt.Errorf("llm_aliases must be an object")
	}
	seen := make(map[string]bool)
	for decoder.More() {
		nameToken, tokenErr := decoder.Token()
		if tokenErr != nil {
			return tokenErr
		}
		name, ok := nameToken.(string)
		if !ok {
			return fmt.Errorf("llm_aliases contains a non-string name")
		}
		normalized := strings.ToLower(strings.TrimSpace(name))
		if seen[normalized] {
			return fmt.Errorf("duplicate normalized llm alias name %q", name)
		}
		seen[normalized] = true
		var target json.RawMessage
		if err := decoder.Decode(&target); err != nil {
			return err
		}
	}
	_, err = decoder.Token()
	return err
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
	defer func() {
		if err := os.Remove(tmpName); err != nil && !os.IsNotExist(err) {
			slog.Warn("remove temporary config", "path", tmpName, "error", err)
		}
	}()
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
	hostCapacityGB, err := mutableHostCapacity(projectRoot)
	if err != nil {
		return err
	}
	if err := validateModelConfigNumbers(modelID, cfg, hostCapacityGB); err != nil {
		return err
	}
	requiresLocal := len(cfg.Placements) == 0
	for _, placement := range cfg.Placements {
		requiresLocal = requiresLocal || placement == "" || placement == LocalHost
	}
	if err := validateModelWorkerPolicy(projectRoot, modelID, cfg, requiresLocal); err != nil {
		return err
	}
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

func mutableHostCapacity(projectRoot string) (float64, error) {
	data, err := loadMutableConfigData(projectRoot)
	if err != nil {
		return 0, err
	}
	value, exists := data["vram_budget_gb"]
	if !exists {
		return 90, nil
	}
	hostCapacityGB, ok := value.(float64)
	if !ok || !finitePositive(hostCapacityGB) {
		return 0, fmt.Errorf("vram_budget_gb must be finite and > 0")
	}
	return hostCapacityGB, nil
}

func persistModelConfigTransaction(
	projectRoot, modelID string,
	config ModelConfig,
	hostCapacityGB float64,
	apply func() error,
	rollback func() error,
) error {
	if err := validatePersistedModelConfig(projectRoot, modelID, config, hostCapacityGB); err != nil {
		return err
	}
	mutableConfigMu.Lock()
	defer mutableConfigMu.Unlock()
	snapshot, err := captureConfigFile(projectRoot)
	if err != nil {
		return err
	}
	if err := saveModelConfigLocked(projectRoot, modelID, config); err != nil {
		return err
	}
	if applyErr := callModelRuntimeOperation("apply", apply); applyErr != nil {
		rollbackErr := callModelRuntimeOperation("rollback", rollback)
		restoreErr := restoreConfigFile(projectRoot, snapshot)
		return transactionRollbackError(applyErr, rollbackErr, restoreErr)
	}
	return nil
}

func callModelRuntimeOperation(name string, operation func() error) (operationError error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			operationError = fmt.Errorf("%s model runtime panic: %v", name, recovered)
		}
	}()
	return operation()
}

func validatePersistedModelConfig(projectRoot, modelID string, config ModelConfig, hostCapacityGB float64) error {
	if err := validateModelConfigNumbers(modelID, config, hostCapacityGB); err != nil {
		return err
	}
	requiresLocal := len(config.Placements) == 0
	for _, placement := range config.Placements {
		requiresLocal = requiresLocal || placement == "" || placement == LocalHost
	}
	return validateModelWorkerPolicy(projectRoot, modelID, config, requiresLocal)
}

func validateModelConfigNumbers(modelID string, config ModelConfig, hostCapacityGB float64) error {
	if err := validateModelAllocationNumbers(config, hostCapacityGB); err != nil {
		return err
	}
	if err := validateModelTimingNumbers(modelID, config); err != nil {
		return err
	}
	if config.PressureIndex != nil && !finiteRange(*config.PressureIndex, 0, 1) {
		return fmt.Errorf("pressure_index must be finite and between 0 and 1")
	}
	if config.GroupPriority < -1000000 || config.GroupPriority > 1000000 {
		return fmt.Errorf("group_priority must be between -1000000 and 1000000")
	}
	return nil
}

func validateModelAllocationNumbers(config ModelConfig, hostCapacityGB float64) error {
	if !finitePositive(hostCapacityGB) {
		return fmt.Errorf("host memory capacity must be finite and > 0")
	}
	if !finitePositive(config.MemoryGB) || config.MemoryGB > hostCapacityGB {
		return fmt.Errorf("memory_gb must be finite, > 0, and <= %.3g", hostCapacityGB)
	}
	if config.MaxInstances == nil || *config.MaxInstances < 0 || *config.MaxInstances > maximumModelInstances {
		return fmt.Errorf("max_instances must be between 0 and %d", maximumModelInstances)
	}
	if config.MaxConcurrent < 1 || config.MaxConcurrent > maximumModelConcurrency {
		return fmt.Errorf("max_concurrent must be between 1 and %d", maximumModelConcurrency)
	}
	return nil
}

func validateModelTimingNumbers(modelID string, config ModelConfig) error {
	if config.KeepAliveSec < 0 || config.KeepAliveSec > maximumDurationSeconds {
		return fmt.Errorf("keep_alive_seconds must be between 0 and %d", maximumDurationSeconds)
	}
	maximumRuntime := maximumRuntimeSecondsForModel(modelID)
	if config.MaxRuntimeSec < 1 || config.MaxRuntimeSec > maximumRuntime {
		return fmt.Errorf("max_runtime_seconds must be between 1 and %d", maximumRuntime)
	}
	if !finiteRange(config.AvgInferenceMs, 0, maximumMetricMillis) {
		return fmt.Errorf("avg_inference_ms must be finite and between 0 and %d", maximumMetricMillis)
	}
	if !finiteRange(config.LoadMs, 0, maximumMetricMillis) {
		return fmt.Errorf("load_ms must be finite and between 0 and %d", maximumMetricMillis)
	}
	return nil
}

func maximumRuntimeSecondsForModel(modelID string) int {
	if modelID == "latentsync" {
		return maximumLatentSyncRuntimeSeconds
	}
	return maximumDurationSeconds
}

func finitePositive(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value > 0
}

func finiteRange(value, minimum, maximum float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= minimum && value <= maximum
}

func saveModelConfigLocked(projectRoot, modelID string, config ModelConfig) error {
	data, err := loadMutableConfigData(projectRoot)
	if err != nil {
		return err
	}
	models, ok := data["models"].(map[string]any)
	if !ok {
		models = make(map[string]any)
		data["models"] = models
	}
	models[modelID] = config
	return writeConfigData(projectRoot, data)
}

func captureConfigFile(projectRoot string) (configFileSnapshot, error) {
	path := filepath.Join(projectRoot, "local", "config.json")
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return configFileSnapshot{}, nil
	}
	if err != nil {
		return configFileSnapshot{}, fmt.Errorf("inspect config snapshot: %w", err)
	}
	if !info.Mode().IsRegular() {
		return configFileSnapshot{}, fmt.Errorf("config snapshot path is not a regular file")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return configFileSnapshot{}, fmt.Errorf("read config snapshot: %w", err)
	}
	return configFileSnapshot{data: data, mode: info.Mode().Perm(), exists: true}, nil
}

func restoreConfigFile(projectRoot string, snapshot configFileSnapshot) error {
	path := filepath.Join(projectRoot, "local", "config.json")
	if !snapshot.exists {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove rolled-back config: %w", err)
		}
		return nil
	}
	if err := writeConfigBytes(projectRoot, snapshot.data, snapshot.mode); err != nil {
		return fmt.Errorf("restore rolled-back config: %w", err)
	}
	return nil
}

func writeConfigBytes(projectRoot string, data []byte, mode os.FileMode) error {
	localDirectory := filepath.Join(projectRoot, "local")
	temporary, err := os.CreateTemp(localDirectory, ".config.rollback.*.tmp")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if _, err := temporary.Write(data); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Chmod(mode); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(temporaryPath, filepath.Join(localDirectory, "config.json"))
}

func transactionRollbackError(applyErr, rollbackErr, restoreErr error) error {
	if rollbackErr == nil && restoreErr == nil {
		return fmt.Errorf("apply persisted model runtime: %w", applyErr)
	}
	return fmt.Errorf("apply persisted model runtime: %w; runtime rollback: %v; persistence rollback: %v", applyErr, rollbackErr, restoreErr)
}

// patchModelField updates a single field for a model in local/config.json,
// preserving every other key (including ones not in ModelConfig), so callers
// never clobber hand-edited fields.
func patchModelField(projectRoot, modelID, field string, value any) error {
	if isDisabledStillImageModel(modelID) {
		return fmt.Errorf("%s", stillImageDisabledMessage)
	}
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
	hostCapacityGB, err := mutableHostCapacity(projectRoot)
	if err != nil {
		return err
	}
	if !finitePositive(newMemoryGB) || newMemoryGB > hostCapacityGB {
		return fmt.Errorf("memory_gb must be finite, > 0, and <= %.3g", hostCapacityGB)
	}
	return patchModelField(projectRoot, modelID, "memory_gb", newMemoryGB)
}

// PatchModelMaxInstances is used by the scheduler's auto-wake guard to undo a
// persisted scale-to-zero.
func PatchModelMaxInstances(projectRoot, modelID string, n int) error {
	if n < 0 || n > maximumModelInstances {
		return fmt.Errorf("max_instances must be between 0 and %d", maximumModelInstances)
	}
	return patchModelField(projectRoot, modelID, "max_instances", n)
}

// PatchModelRemoteEnabled persists a per-model remote kill-switch flip. The
// flag must survive restarts so an operator who pins a model to spark (because
// a remote host is misbehaving) doesn't have it silently re-enabled on the next
// bounce.
func PatchModelRemoteEnabled(projectRoot, modelID string, enabled bool) error {
	return patchModelField(projectRoot, modelID, "remote_enabled", enabled)
}

// PatchRemoteDisabled persists the GLOBAL remote kill-switch. It is a top-level
// config field (not under models), so it gets written directly into the config
// map, preserving every other key.
func PatchRemoteDisabled(projectRoot string, disabled bool) error {
	mutableConfigMu.Lock()
	defer mutableConfigMu.Unlock()
	data, err := loadMutableConfigData(projectRoot)
	if err != nil {
		return err
	}
	if disabled {
		data["remote_disabled"] = true
	} else {
		// Keep the file clean: omit the flag entirely when remote is allowed.
		delete(data, "remote_disabled")
	}
	return writeConfigData(projectRoot, data)
}

// SaveLLMAliases atomically replaces the persisted alias map while preserving
// every unrelated mutable configuration key.
func SaveLLMAliases(projectRoot string, aliases map[string]string) error {
	mutableConfigMu.Lock()
	defer mutableConfigMu.Unlock()
	data, err := loadMutableConfigData(projectRoot)
	if err != nil {
		return err
	}
	if aliases == nil {
		aliases = map[string]string{}
	}
	data["llm_aliases"] = aliases
	return writeConfigData(projectRoot, data)
}

// DeleteModelConfig removes a model and any supplied dependent aliases in one
// atomic configuration-file replacement.
func DeleteModelConfig(projectRoot, modelID string, aliasesToDrop ...string) error {
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
	if len(aliasesToDrop) > 0 {
		rawAliases, ok := data["llm_aliases"].(map[string]any)
		if ok {
			for _, alias := range aliasesToDrop {
				delete(rawAliases, alias)
			}
		}
	}
	return writeConfigData(projectRoot, data)
}
