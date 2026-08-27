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
		"minimax-h3", "moondream", "sonic", "whisper-large", "music-generate", "llm:qwen3.6-35b", "flora", "floral-voice",
	} {
		if isDisabledStillImageModel(id) {
			t.Errorf("%q was incorrectly classified as a still-image model", id)
		}
	}
}

func TestMiniMaxH3VideoAdmissionIsExactAndTopLevel(t *testing.T) {
	if err := validateJobModelCompatibility("video-generate", "minimax-h3"); err != nil {
		t.Fatalf("exact MiniMax H3 model rejected: %v", err)
	}
	for _, modelID := range []string{"minimax-h3-pro", "minimax-h3-copy", "minimax", "MiniMax-H3"} {
		if err := validateJobModelCompatibility("video-generate", modelID); err == nil {
			t.Fatalf("near-neighbor MiniMax model %q accepted", modelID)
		}
	}
	if nestedModelRoutesJob("video-generate") {
		t.Fatal("video-generate still routes params.model; MiniMax selection must be top-level")
	}
	if got := JobTypeToModel["video-generate"]; got != "ltx2" {
		t.Fatalf("omitted-model default changed: got %q want ltx2", got)
	}
}

func TestMusicGenerateAdmission(t *testing.T) {
	if err := validateJobModelCompatibility("music-generate", "music-generate"); err != nil {
		t.Fatalf("exact music-generate model rejected: %v", err)
	}
	if got := JobTypeToModel["music-generate"]; got != "music-generate" {
		t.Fatalf("JobTypeToModel[music-generate] = %q, want music-generate", got)
	}
	if venv, ok := trustedPythonAdapters["music-generate"]; !ok || venv != "music-generate" {
		t.Fatalf("trustedPythonAdapters[music-generate] = %q, want music-generate", venv)
	}
}

func TestMiniMaxH3TopLevelSubmissionAndLTX2Default(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.config.Models["minimax-h3"] = ModelConfig{}
	api.config.Models["ltx2"] = ModelConfig{}
	api.refreshAliasModels()

	explicit := performRequest(api, "POST", "/v1/jobs",
		`{"type":"video-generate","model":"minimax-h3","params":{"prompt":"shot","duration":4,"resolution":"768P"}}`)
	if explicit.Code != 200 {
		t.Fatalf("explicit MiniMax submission status=%d body=%s", explicit.Code, explicit.Body.String())
	}
	explicitJob, err := api.store.GetJob(decodeObject(t, explicit.Body.Bytes())["job_id"].(string))
	if err != nil || explicitJob.ModelID != "minimax-h3" {
		t.Fatalf("explicit MiniMax job=%+v error=%v", explicitJob, err)
	}

	defaulted := performRequest(api, "POST", "/v1/jobs",
		`{"type":"video-generate","params":{"segments":[],"audio_b64":""}}`)
	if defaulted.Code != 200 {
		t.Fatalf("default LTX2 submission status=%d body=%s", defaulted.Code, defaulted.Body.String())
	}
	defaultJob, err := api.store.GetJob(decodeObject(t, defaulted.Body.Bytes())["job_id"].(string))
	if err != nil || defaultJob.ModelID != "ltx2" {
		t.Fatalf("default LTX2 job=%+v error=%v", defaultJob, err)
	}

	nested := performRequest(api, "POST", "/v1/jobs",
		`{"type":"video-generate","params":{"model":"minimax-h3","prompt":"shot","duration":4,"resolution":"768P"}}`)
	if nested.Code != 400 {
		t.Fatalf("nested MiniMax selector status=%d body=%s", nested.Code, nested.Body.String())
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

func TestLoadConfigAllowsLaptopQwenDense27b38(t *testing.T) {
	root := t.TempDir()
	models := map[string]ModelConfig{
		"llm:qwen3.6-27b": {
			MemoryGB:      20,
			MaxRuntimeSec: 3600,
			KeepAliveSec:  3600,
			Placements:    []string{"boringstack"},
			AdapterParams: map[string]string{
				"remote_model_tag": "mlx-community/Qwen3.8-27B-4bit",
			},
		},
	}
	writeModelConfigFixture(t, root, models)
	config, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("load laptop qwen 27b 3.8 config: %v", err)
	}
	model, ok := config.Models["llm:qwen3.6-27b"]
	if !ok {
		t.Fatal("laptop qwen 27b 3.8 config was omitted by startup policy")
	}
	if got := model.AdapterParams["remote_model_tag"]; got != "mlx-community/Qwen3.8-27B-4bit" {
		t.Fatalf("remote_model_tag = %s, want mlx-community/Qwen3.8-27B-4bit", got)
	}
}

func TestLoadConfigAllowsLaptopNemotron30bA3b(t *testing.T) {
	root := t.TempDir()
	models := map[string]ModelConfig{
		"llm:nemotron-30b-a3b": {
			MemoryGB:      40,
			MaxRuntimeSec: 3600,
			KeepAliveSec:  3600,
			Placements:    []string{"boringstack"},
			AdapterParams: map[string]string{
				"remote_model_tag": "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit",
			},
		},
	}
	writeModelConfigFixture(t, root, models)
	config, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("load laptop nemotron config: %v", err)
	}
	model, ok := config.Models["llm:nemotron-30b-a3b"]
	if !ok {
		t.Fatal("laptop nemotron config was omitted by startup policy")
	}
	if got := model.AdapterParams["remote_model_tag"]; got != "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit" {
		t.Fatalf("remote_model_tag = %s, want mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit", got)
	}
}

func TestLoadConfigAcceptsQwenMemoryBudgetTransitionVectors(t *testing.T) {
	vectors := map[string]string{
		"combined_0.50_8G": "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 8G --enforce-eager",
		"legacy_0.25":      "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.25 --enforce-eager",
		"unsafe_0.50":      "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --enforce-eager",
		"explicit_8G":      "--max-model-len 32768 --max-num-batched-tokens 32768 --kv-cache-memory-bytes 8G --enforce-eager",
	}
	for name, vector := range vectors {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			models := map[string]ModelConfig{
				"llm:qwen3.6-35b": productionRepositoryConfig(root, "vllm-chat-worker", map[string]string{
					"LLM_BACKEND":     "vllm",
					"LLM_CTX_SIZE":    "32768",
					"VLLM_MODEL":      "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
					"VLLM_EXTRA_ARGS": vector,
				}),
			}
			writeModelConfigFixture(t, root, models)
			config, err := LoadConfig(root)
			if err != nil {
				t.Fatalf("load qwen transition config: %v", err)
			}
			if _, ok := config.Models["llm:qwen3.6-35b"]; !ok {
				t.Fatal("qwen transition config was omitted by startup policy")
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
