package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
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
}

type Config struct {
	VRAMBudgetGB      float64                `json:"vram_budget_gb"`
	SystemRAMBudgetGB float64                `json:"system_ram_budget_gb"` // 0 = disabled. On unified-memory hardware (GB10), this caps total tree-RSS across all worker process trees so CPU-side allocations can't push the GPU driver into NV_ERR_NO_MEMORY.
	Host              string                 `json:"host"`
	Port              int                    `json:"port"`
	OutputDir         string                 `json:"output_dir"`
	ShareMount        string                 `json:"share_mount"` // e.g. "/mnt/arbiter-store" — if set, monitored and remounted when unhealthy
	Models            map[string]ModelConfig `json:"models"`
}

// JobTypeToModel maps job type strings to model IDs.
var JobTypeToModel = map[string]string{
	"image-generate":         "flux-schnell",
	"image-edit":             "flux-schnell",
	"background-remove":      "birefnet",
	"caption":                "moondream",
	"query":                  "moondream",
	"detect":                 "moondream",
	"point":                  "moondream",
	"transcribe":             "whisper-large",
	"tts-custom":             "tts-custom",
	"tts-clone":              "tts-clone",
	"tts-design":             "tts-design",
	"talking-head":           "sonic",
	"talking-head-sadtalker": "sadtalker",
	"lipsync":                "latentsync",
	"video-generate":         "ltx2",
	"video-encode":           "ltx2-encode",
	"video-denoise1":         "ltx2-denoise1",
	"video-denoise2":         "ltx2-denoise2",
	"face-restore":             "face-restore",
	"face-restore-codeformer":  "face-restore-codeformer",
	"aesthetic-score":        "aesthetic-scorer",
	"tts-voxtral":            "tts-voxtral",
	"lora-train":             "lora-train",
	"composite":              "composite",
	"embed-text":             "embed-text",
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
		VRAMBudgetGB: 100,
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
	os.MkdirAll(filepath.Join(projectRoot, "local"), 0o755)
	out, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}
	out = append(out, '\n')
	return os.WriteFile(filepath.Join(projectRoot, "local", "config.json"), out, 0o644)
}

func SaveModelConfig(projectRoot, modelID string, cfg ModelConfig) error {
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

// PatchModelMemoryGB updates only the memory_gb field for a model in
// local/config.json, preserving every other key (including ones not in
// ModelConfig). Used by the drift watchdog to write back observed high-water
// marks without clobbering hand-edited fields.
func PatchModelMemoryGB(projectRoot, modelID string, newMemoryGB float64) error {
	data, err := loadMutableConfigData(projectRoot)
	if err != nil {
		return err
	}
	models, ok := data["models"].(map[string]any)
	if !ok {
		return fmt.Errorf("no models section in config")
	}
	entry, ok := models[modelID].(map[string]any)
	if !ok {
		entry = make(map[string]any)
		models[modelID] = entry
	}
	entry["memory_gb"] = newMemoryGB
	return writeConfigData(projectRoot, data)
}

func DeleteModelConfig(projectRoot, modelID string) error {
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
