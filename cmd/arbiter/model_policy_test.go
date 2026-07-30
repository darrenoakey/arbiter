package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStillImageModelClassification(t *testing.T) {
	for _, id := range []string{
		"flux", "FLUX_2", "flux-schnell-v2", "black-forest-labs/FLUX.1-dev",
		"kontext-pro", "Tongyi-MAI/Z_Image_Turbo", "stable_diffusion-xl", "sdxl-lightning",
		"pixart-sigma", "product-lora", "Qwen/Image-Generator", "hunyuan-image",
	} {
		if !isDisabledStillImageModel(id) {
			t.Errorf("%q was not classified as a disabled still-image model", id)
		}
	}
	for _, id := range []string{
		"birefnet", "ltx2", "ltx2-denoise2", "ltx2-dev-denoise1-lora", "lora-train",
		"moondream", "sonic", "whisper-large", "llm:qwen3.6-35b", "flora", "floral-voice",
	} {
		if isDisabledStillImageModel(id) {
			t.Errorf("%q was incorrectly classified as a still-image model", id)
		}
	}
}

func TestStillImageModelClassificationRejectsCanonicalLoraAliasesWithoutFloraFalsePositive(t *testing.T) {
	for _, id := range []string{"portrait-lora", "portrait_lora", "LORA/portrait", "flux-lora", "FLUX_KONTEXT.LoRA"} {
		if !isDisabledStillImageModel(id) {
			t.Errorf("%q LoRA alias bypassed still-image policy", id)
		}
	}
	for _, id := range []string{"flora", "floral", "florence-2", "llm:flora", "voice/flora-v2"} {
		if isDisabledStillImageModel(id) {
			t.Errorf("%q semantic model name was falsely classified", id)
		}
	}
}

func TestLoadConfigOmitsPersistedStillImageModels(t *testing.T) {
	root := t.TempDir()
	localDir := filepath.Join(root, "local")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"models":{
		"flux2":{"memory_gb":31},
		"opaque":{"memory_gb":20,"auto_download":"Tongyi-MAI/Z-Image-Turbo"},
		"birefnet":{"memory_gb":1},
		"ltx2-dev-denoise2":{"memory_gb":55,"model_path":"/models/ltx2/distilled-lora.safetensors"}
	}}`
	if err := os.WriteFile(filepath.Join(localDir, "config.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := cfg.Models["flux2"]; ok {
		t.Fatal("flux2 survived startup config filtering")
	}
	if _, ok := cfg.Models["opaque"]; ok {
		t.Fatal("metadata-disguised still generator survived startup filtering")
	}
	if _, ok := cfg.Models["birefnet"]; !ok {
		t.Fatal("birefnet was removed")
	}
	if _, ok := cfg.Models["ltx2-dev-denoise2"]; !ok {
		t.Fatal("LTX2 LoRA video variant was removed")
	}
}

func TestSetupInstancesRefusesInjectedStillImageConfig(t *testing.T) {
	zero := 0
	cfg := &Config{Models: map[string]ModelConfig{
		"flux2":    {MemoryGB: 1, MaxInstances: &zero},
		"birefnet": {MemoryGB: 1, MaxInstances: &zero},
	}}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())
	if got := len(mgr.GetModelInstances("flux2")); got != 0 {
		t.Fatalf("registered %d flux2 instances", got)
	}
	if _, ok := mgr.byModel["flux2"]; ok {
		t.Fatal("flux2 model shell was registered")
	}
	if _, ok := mgr.byModel["birefnet"]; !ok {
		t.Fatal("birefnet model shell was not registered")
	}
}

func TestWorkerPolicyAllowsEveryBuiltInAdapter(t *testing.T) {
	root := t.TempDir()
	for modelID := range trustedPythonAdapters {
		t.Run(modelID, func(t *testing.T) {
			if err := validateModelWorkerPolicy(root, modelID, ModelConfig{}, true); err != nil {
				t.Fatalf("built-in adapter rejected: %v", err)
			}
		})
	}
}

func TestWorkerPolicyAllowsDocumentedWorkerIdentities(t *testing.T) {
	root := t.TempDir()
	cases := []struct {
		modelID string
		config  ModelConfig
	}{
		{modelID: "tts-kokoro", config: pythonWorkerConfig(root, "kokoro", "tts-kokoro")},
		{modelID: "tts-custom", config: pythonWorkerConfig(root, "qwentts", "tts-custom")},
		{modelID: "tts-clone", config: pythonWorkerConfig(root, "qwentts", "tts-clone")},
		{modelID: "tts-design", config: pythonWorkerConfig(root, "qwentts", "tts-design")},
		{modelID: "demucs", config: pythonWorkerConfig(root, "demucs", "demucs")},
		{modelID: "rvc-train", config: pythonWorkerConfig(root, "rvc", "rvc-train")},
		{modelID: "rvc-convert", config: pythonWorkerConfig(root, "rvc", "rvc-convert")},
		{modelID: "voice-fit", config: pythonWorkerConfig(root, "voxsmith", "voice-fit")},
		{modelID: "llm:llama", config: repositoryWorkerConfig(root, "llm-worker", "llamacpp")},
		{modelID: "llm:qwen", config: repositoryWorkerConfig(root, "vllm-chat-worker", "vllm")},
		{modelID: "tts-voxtral", config: repositoryWorkerConfig(root, "vllm-worker", "")},
		{modelID: "llm:remote", config: ModelConfig{Placements: []string{"boringstack"}}},
	}
	for _, testCase := range cases {
		name := testCase.modelID
		if len(testCase.config.WorkerCmd) > 0 {
			name += filepath.Base(testCase.config.WorkerCmd[0])
		}
		t.Run(name, func(t *testing.T) {
			requiresLocal := len(testCase.config.Placements) == 0
			if err := validateModelWorkerPolicy(root, testCase.modelID, testCase.config, requiresLocal); err != nil {
				t.Fatalf("sanctioned worker rejected: %v", err)
			}
		})
	}
}

func TestLoadConfigAllowsMinimizedObservedProductionConfig(t *testing.T) {
	root := t.TempDir()
	models := map[string]ModelConfig{
		"aesthetic-scorer": productionPythonConfig(root, "aesthetic", "aesthetic-scorer"),
		"birefnet":         productionPythonConfig(root, "birefnet", "birefnet"),
		"embed-text":       productionPythonConfig(root, "embed", "embed-text"),
		"insightface":      productionPythonConfig(root, "insightface", "insightface"),
		"moondream":        productionPythonConfig(root, "moondream", "moondream"),
		"whisper-large":    productionPythonConfig(root, "whisper", "whisper-large"),
		"tts-voxtral": productionRepositoryConfig(root, "vllm-worker", map[string]string{
			"VLLM_MODE": "tts", "VLLM_MODEL": "mistralai/Voxtral-4B-TTS-2603",
		}),
		"llm:gemma4-26b": productionRepositoryConfig(root, "vllm-chat-worker", map[string]string{
			"LLM_BACKEND": "vllm", "LLM_CTX_SIZE": "8192", "VLLM_MODEL": "RedHatAI/gemma-4-26B-A4B-it-NVFP4",
			"VLLM_EXTRA_ARGS": "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --enforce-eager --speculative-config {\"method\":\"mtp\",\"model\":\"google/gemma-4-26B-A4B-it-assistant\",\"num_speculative_tokens\":4}",
		}),
		"llm:gemma4-26b-plain": productionRepositoryConfig(root, "vllm-chat-worker", map[string]string{
			"LLM_BACKEND": "vllm", "LLM_CTX_SIZE": "8192", "VLLM_MODEL": "RedHatAI/gemma-4-26B-A4B-it-NVFP4",
			"VLLM_EXTRA_ARGS": "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --enforce-eager",
		}),
		"llm:gemma4-26b-mtp": productionRepositoryConfig(root, "vllm-chat-worker", map[string]string{
			"LLM_BACKEND": "vllm", "LLM_CTX_SIZE": "8192", "VLLM_MODEL": "RedHatAI/gemma-4-26B-A4B-it-NVFP4",
			"VLLM_EXTRA_ARGS": "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --enforce-eager --speculative-config {\"method\":\"mtp\",\"model\":\"google/gemma-4-26B-A4B-it-assistant\",\"num_speculative_tokens\":4}",
		}),
		"llm:qwen3.6-35b": productionRepositoryConfig(root, "vllm-chat-worker", map[string]string{
			"LLM_BACKEND": "vllm", "LLM_CTX_SIZE": "8192", "VLLM_MODEL": "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
			"VLLM_EXTRA_ARGS": "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.25 --enforce-eager",
		}),
		"latentsync": {MemoryGB: 28, MaxRuntimeSec: 4000000},
	}
	writeModelConfigFixture(t, root, models)
	config, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("load minimized production config: %v", err)
	}
	for modelID := range models {
		if _, ok := config.Models[modelID]; !ok {
			t.Errorf("observed production model %q was omitted", modelID)
		}
	}
}

func TestLoadConfigAcceptsBothQwenMemoryBudgetTransitionVectors(t *testing.T) {
	for _, utilization := range []string{"0.25", "0.50"} {
		t.Run(utilization, func(t *testing.T) {
			root := t.TempDir()
			models := map[string]ModelConfig{
				"llm:qwen3.6-35b": productionRepositoryConfig(root, "vllm-chat-worker", map[string]string{
					"LLM_BACKEND":  "vllm",
					"LLM_CTX_SIZE": "32768",
					"VLLM_MODEL":   "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
					"VLLM_EXTRA_ARGS": "--max-model-len 32768 --max-num-batched-tokens 32768 " +
						"--gpu-memory-utilization " + utilization + " --enforce-eager",
				}),
			}
			writeModelConfigFixture(t, root, models)
			config, err := LoadConfig(root)
			if err != nil {
				t.Fatalf("load qwen transition config %s: %v", utilization, err)
			}
			if _, ok := config.Models["llm:qwen3.6-35b"]; !ok {
				t.Fatalf("qwen transition config %s was omitted by startup policy", utilization)
			}
		})
	}
}

func TestWorkerPolicyRejectsAdversarialCommands(t *testing.T) {
	root := t.TempDir()
	cases := []struct {
		name      string
		modelID   string
		command   []string
		wantImage bool
	}{
		{name: "FluxModule", modelID: "benign", command: []string{"python3", "-m", "arbiter.adapters.flux", "serve"}, wantImage: true},
		{name: "ZImageModule", modelID: "benign", command: []string{"python3", "-m", "z_image.worker", "serve"}, wantImage: true},
		{name: "Shell", modelID: "birefnet", command: []string{"sh", "-c", "true"}},
		{name: "ArbitraryExecutable", modelID: "birefnet", command: []string{"/tmp/worker"}},
		{name: "TrustedBasenameOutsideRoot", modelID: "llm:test", command: []string{"/tmp/llm-worker"}},
		{name: "NonCanonicalPathSpelling", modelID: "llm:test", command: []string{root + "/sub/../llm-worker"}},
		{name: "WrongSanctionedVenv", modelID: "birefnet", command: []string{filepath.Join(root, "venvs", "moondream", "bin", "python"), "-m", "arbiter.worker_main", "birefnet"}},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			cfg := ModelConfig{WorkerCmd: testCase.command, AdapterParams: map[string]string{"LLM_BACKEND": "llamacpp"}}
			err := validateModelWorkerPolicy(root, testCase.modelID, cfg, true)
			if err == nil {
				t.Fatal("adversarial command was accepted")
			}
			message := untrustedWorkerCommandMessage
			if testCase.wantImage {
				message = stillImageDisabledMessage
			}
			if !strings.Contains(err.Error(), message) {
				t.Fatalf("error = %q, want policy %q", err, message)
			}
		})
	}
}

func TestWorkerPolicyRejectsSymlinkExecutable(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "actual-worker")
	if err := os.WriteFile(target, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	worker := filepath.Join(root, "llm-worker")
	if err := os.Symlink(target, worker); err != nil {
		t.Fatal(err)
	}
	err := validateModelWorkerPolicy(root, "llm:test", repositoryWorkerConfig(root, "llm-worker", "llamacpp"), true)
	if err == nil || !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("symlink policy error = %v", err)
	}
}

func TestLoadConfigOmitsNestedUntrustedWorkerCommands(t *testing.T) {
	root := t.TempDir()
	localDir := filepath.Join(root, "local")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"models":{"birefnet":{"memory_gb":1,"worker_cmd":["sh","-c","true"]},"moondream":{"memory_gb":1}}}`
	if err := os.WriteFile(filepath.Join(localDir, "config.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := cfg.Models["birefnet"]; ok {
		t.Fatal("nested arbitrary command survived startup validation")
	}
	if _, ok := cfg.Models["moondream"]; !ok {
		t.Fatal("sanctioned sibling model was removed")
	}
}

func TestSpawnRevalidatesMutatedWorkerCommandBeforeExec(t *testing.T) {
	root := t.TempDir()
	marker := filepath.Join(root, "executed")
	instance := NewInstance("birefnet", "birefnet#0", 1, 1, "python3", root)
	instance.workerCmd = []string{"sh", "-c", "touch " + marker}
	err := instance.Spawn()
	if err == nil || !strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
		t.Fatalf("Spawn policy error = %v", err)
	}
	if _, statErr := os.Stat(marker); !os.IsNotExist(statErr) {
		t.Fatalf("untrusted command executed; marker stat error = %v", statErr)
	}
}

func pythonWorkerConfig(root, venv, modelID string) ModelConfig {
	return ModelConfig{WorkerCmd: []string{filepath.Join(root, "venvs", venv, "bin", "python"), "-m", "arbiter.worker_main", modelID}}
}

func repositoryWorkerConfig(root, worker, backend string) ModelConfig {
	one := 1
	params := map[string]string{}
	if backend != "" {
		params["LLM_BACKEND"] = backend
	}
	return ModelConfig{
		MemoryGB: 1, MaxConcurrent: 1, MaxInstances: &one, KeepAliveSec: 300,
		MaxRuntimeSec: 7200, WorkerCmd: []string{filepath.Join(root, worker)}, AdapterParams: params,
	}
}

func productionPythonConfig(root, venv, modelID string) ModelConfig {
	config := pythonWorkerConfig(root, venv, modelID)
	config.MemoryGB = 1
	return config
}

func productionRepositoryConfig(root, worker string, params map[string]string) ModelConfig {
	config := repositoryWorkerConfig(root, worker, "")
	config.AdapterParams = params
	return config
}

func writeModelConfigFixture(t *testing.T, root string, models map[string]ModelConfig) {
	t.Helper()
	data, err := json.Marshal(map[string]any{"models": models})
	if err != nil {
		t.Fatal(err)
	}
	localDirectory := filepath.Join(root, "local")
	if err := os.MkdirAll(localDirectory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(localDirectory, "config.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}
