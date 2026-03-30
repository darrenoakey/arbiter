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
	AvgInferenceMs float64           `json:"avg_inference_ms"`
	LoadMs         float64           `json:"load_ms"`
	AutoDownload   string            `json:"auto_download"`
	ModelPath      string            `json:"model_path"`
	Group          bool              `json:"group"`
	WorkerCmd      []string          `json:"worker_cmd,omitempty"`
	AdapterParams  map[string]string `json:"adapter_params,omitempty"`
}

type Config struct {
	VRAMBudgetGB float64                `json:"vram_budget_gb"`
	Host         string                 `json:"host"`
	Port         int                    `json:"port"`
	Models       map[string]ModelConfig `json:"models"`
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
	"aesthetic-score":        "aesthetic-scorer",
	"tts-voxtral":            "tts-voxtral",
	"lora-train":             "lora-train",
	"composite":              "composite",
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
		if m.KeepAliveSec == 0 {
			m.KeepAliveSec = 300
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
