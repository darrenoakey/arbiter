package main

import (
	"bufio"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestAdapterParamsAllowEverySanctionedProductionKey(t *testing.T) {
	root := t.TempDir()
	modelPath := filepath.Join(root, "model.gguf")
	writeExecutableTestFile(t, modelPath, "model")
	llamaPath := filepath.Join(root, "local", "bin", "llama-server")
	writeExecutableTestFile(t, llamaPath, "#!/bin/sh\nexit 0\n")

	llama := repositoryWorkerConfig(root, "llm-worker", "llamacpp")
	llama.AdapterParams = map[string]string{
		"LLM_BACKEND": "llamacpp", "LLM_CTX_SIZE": "32768", "LLM_GPU_LAYERS": "-1",
		"LLM_HF_FILE": "weights/model.gguf", "LLM_HF_REPO": "org/model-GGUF",
		"LLM_MODEL_PATH": modelPath, "LLM_PARALLEL": "8", "LLAMA_ARG_CACHE_TYPE_K": "q8_0",
		"LLAMA_ARG_CACHE_TYPE_V": "f16", "LLAMA_ARG_FLASH_ATTN": "off",
		"LLAMA_ARG_JINJA": "on", "LLAMA_SERVER_BIN": llamaPath,
		"remote_model_tag": "model:latest",
	}
	if err := validateAdapterParams(root, "llm:test", llama); err != nil {
		t.Fatalf("llama production params rejected: %v", err)
	}

	vllm := repositoryWorkerConfig(root, "vllm-chat-worker", "vllm")
	vllm.AdapterParams = map[string]string{
		"LLM_BACKEND": "vllm", "LLM_CTX_SIZE": "32768", "VLLM_DTYPE": "bfloat16",
		"VLLM_GPU_MEMORY_UTILIZATION": "0.9", "VLLM_MAX_MODEL_LEN": "32768",
		"VLLM_MAX_NUM_SEQS": "16", "VLLM_MODEL": "org/model", "VLLM_QUANTIZATION": "awq",
		"VLLM_READY_TIMEOUT_SEC": "900", "VLLM_TENSOR_PARALLEL_SIZE": "2",
	}
	if err := validateAdapterParams(root, "llm:test", vllm); err != nil {
		t.Fatalf("vllm production params rejected: %v", err)
	}
}

func TestAdapterParamsAllowObservedProductionVllmCompatibilityValues(t *testing.T) {
	root := t.TempDir()
	gemmaMTP := `--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --enforce-eager --speculative-config {"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}`
	tests := map[string]string{
		"llm:gemma4-26b":       gemmaMTP,
		"llm:gemma4-26b-mtp":   gemmaMTP,
		"llm:gemma4-26b-plain": `--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --enforce-eager`,
		"llm:qwen3.6-35b":      `--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.25 --enforce-eager`,
	}
	for modelID, legacy := range tests {
		t.Run(modelID, func(t *testing.T) {
			chat := repositoryWorkerConfig(root, "vllm-chat-worker", "vllm")
			chat.AdapterParams["VLLM_EXTRA_ARGS"] = legacy
			if err := validateAdapterParams(root, modelID, chat); err != nil {
				t.Fatalf("observed vllm chat settings rejected: %v", err)
			}
		})
	}

	tts := repositoryWorkerConfig(root, "vllm-worker", "")
	tts.AdapterParams = map[string]string{"VLLM_MODE": "tts", "VLLM_MODEL": "mistralai/Voxtral-4B-TTS-2603"}
	if err := validateAdapterParams(root, "tts-voxtral", tts); err != nil {
		t.Fatalf("observed vllm TTS settings rejected: %v", err)
	}
}

func TestQwenMemoryBudgetTransitionAcceptsOnlySanctionedExactVectors(t *testing.T) {
	root := t.TempDir()
	accepted := map[string]string{
		"combined_0.50_8G": "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 8G --enforce-eager",
		"legacy_0.25":      "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.25 --enforce-eager",
		"unsafe_0.50":      "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --enforce-eager",
		"explicit_8G":      "--max-model-len 32768 --max-num-batched-tokens 32768 --kv-cache-memory-bytes 8G --enforce-eager",
	}
	for name, vector := range accepted {
		t.Run(name, func(t *testing.T) {
			chat := repositoryWorkerConfig(root, "vllm-chat-worker", "vllm")
			chat.AdapterParams["VLLM_EXTRA_ARGS"] = vector
			if err := validateAdapterParams(root, "llm:qwen3.6-35b", chat); err != nil {
				t.Fatalf("qwen transition vector rejected: %v", err)
			}
		})
	}
	rejected := map[string]string{
		"alternate_decimal":   "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.5 --enforce-eager",
		"utilization_0.49":    "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.49 --enforce-eager",
		"utilization_0.51":    "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.51 --enforce-eager",
		"kv_reordered":        "--max-num-batched-tokens 32768 --max-model-len 32768 --kv-cache-memory-bytes 8G --enforce-eager",
		"kv_7G":               "--max-model-len 32768 --max-num-batched-tokens 32768 --kv-cache-memory-bytes 7G --enforce-eager",
		"kv_9G":               "--max-model-len 32768 --max-num-batched-tokens 32768 --kv-cache-memory-bytes 9G --enforce-eager",
		"kv_alt_spelling":     "--max-model-len 32768 --max-num-batched-tokens 32768 --kv-cache-memory-bytes 8GiB --enforce-eager",
		"kv_extra_flag":       "--max-model-len 32768 --max-num-batched-tokens 32768 --kv-cache-memory-bytes 8G --enforce-eager --served-model-name injected",
		"combined_reordered":  "--max-model-len 32768 --max-num-batched-tokens 32768 --kv-cache-memory-bytes 8G --gpu-memory-utilization 0.50 --enforce-eager",
		"combined_gpu_0.49":   "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.49 --kv-cache-memory-bytes 8G --enforce-eager",
		"combined_gpu_0.51":   "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.51 --kv-cache-memory-bytes 8G --enforce-eager",
		"combined_kv_7G":      "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 7G --enforce-eager",
		"combined_kv_9G":      "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 9G --enforce-eager",
		"combined_no_eager":   "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 8G",
		"combined_no_batch":   "--max-model-len 32768 --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 8G --enforce-eager",
		"combined_no_context": "--max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 8G --enforce-eager",
		"combined_extra":      "--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 8G --enforce-eager --served-model-name injected",
	}
	for name, vector := range rejected {
		t.Run(name, func(t *testing.T) {
			chat := repositoryWorkerConfig(root, "vllm-chat-worker", "vllm")
			chat.AdapterParams["VLLM_EXTRA_ARGS"] = vector
			if err := validateAdapterParams(root, "llm:qwen3.6-35b", chat); err == nil {
				t.Fatalf("unsanctioned qwen transition vector accepted: %q", vector)
			}
		})
	}
}

func TestAdapterParamsRejectVllmCompatibilityNearNeighbors(t *testing.T) {
	root := t.TempDir()
	gemmaMTP := `--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --enforce-eager --speculative-config {"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}`
	gemmaPlain := `--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --enforce-eager`
	qwen := `--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.25 --enforce-eager`
	tests := []struct {
		name    string
		modelID string
		worker  string
		key     string
		value   string
	}{
		{name: "unlisted model", modelID: "llm:gemma4-26b-copy", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: gemmaMTP},
		{name: "reordered", modelID: "llm:gemma4-26b", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: `--max-num-batched-tokens 32768 --max-model-len 32768 --gpu-memory-utilization 0.50 --enforce-eager --speculative-config {"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}`},
		{name: "missing", modelID: "llm:gemma4-26b-plain", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: strings.TrimSuffix(gemmaPlain, " --enforce-eager")},
		{name: "duplicated", modelID: "llm:qwen3.6-35b", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: qwen + " --enforce-eager"},
		{name: "alternate integer spelling", modelID: "llm:gemma4-26b-plain", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: strings.Replace(gemmaPlain, "32768", "032768", 1)},
		{name: "alternate decimal spelling", modelID: "llm:gemma4-26b", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: strings.Replace(gemmaMTP, "0.50", "0.5", 1)},
		{name: "alternate json layout", modelID: "llm:gemma4-26b-mtp", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: strings.Replace(gemmaMTP, `{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}`, `{"model":"google/gemma-4-26B-A4B-it-assistant","method":"mtp","num_speculative_tokens":4}`, 1)},
		{name: "alternate speculative model", modelID: "llm:gemma4-26b", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: strings.Replace(gemmaMTP, "google/gemma-4-26B-A4B-it-assistant", "org/model", 1)},
		{name: "alternate speculative count", modelID: "llm:gemma4-26b-mtp", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: strings.Replace(gemmaMTP, `"num_speculative_tokens":4`, `"num_speculative_tokens":5`, 1)},
		{name: "extra flag", modelID: "llm:gemma4-26b-plain", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: gemmaPlain + " --served-model-name injected"},
		{name: "unsanctioned qwen utilization", modelID: "llm:qwen3.6-35b", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: strings.Replace(gemmaPlain, "0.50", "0.51", 1)},
		{name: "speculation on plain", modelID: "llm:gemma4-26b-plain", worker: "vllm-chat-worker", key: "VLLM_EXTRA_ARGS", value: gemmaMTP},
		{name: "chat mode on TTS", modelID: "tts-voxtral", worker: "vllm-worker", key: "VLLM_MODE", value: "chat"},
		{name: "mode on chat worker", modelID: "llm:gemma4-26b", worker: "vllm-chat-worker", key: "VLLM_MODE", value: "tts"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := repositoryWorkerConfig(root, test.worker, "")
			config.AdapterParams[test.key] = test.value
			if err := validateAdapterParams(root, test.modelID, config); err == nil {
				t.Fatalf("near-neighbor %s=%q was accepted", test.key, test.value)
			}
		})
	}
}

func TestLLMAliasesDoNotCreateAdapterPolicyModels(t *testing.T) {
	aliases := []string{
		"local-chat", "local-summariser", "local-extract", "local-coder", "local-vision",
	}
	for _, alias := range aliases {
		if _, exists := vllmLegacyTuningByModel["llm:"+alias]; exists {
			t.Fatalf("alias %q became an adapter-policy model", alias)
		}
	}
	expected := `--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 8G --enforce-eager`
	if got := vllmLegacyTuningByModel["llm:qwen3.6-35b"]; got != expected {
		t.Fatalf("qwen adapter exception changed: %q", got)
	}
}

func TestAdapterParamsRejectVoxtralLlmBackendSelector(t *testing.T) {
	root := t.TempDir()
	config := repositoryWorkerConfig(root, "vllm-worker", "vllm")
	config.AdapterParams["VLLM_MODE"] = "tts"
	if err := validateAdapterParams(root, "tts-voxtral", config); err == nil {
		t.Fatal("Voxtral accepted LLM_BACKEND=vllm")
	}
}

func TestAdapterParamsRejectOverlappingStructuredAndLegacyVllmTuning(t *testing.T) {
	root := t.TempDir()
	legacy := `--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.25 --enforce-eager`
	overlaps := map[string]string{"VLLM_MAX_MODEL_LEN": "32768", "VLLM_GPU_MEMORY_UTILIZATION": "0.25"}
	for key, value := range overlaps {
		t.Run(key, func(t *testing.T) {
			config := repositoryWorkerConfig(root, "vllm-chat-worker", "vllm")
			config.AdapterParams["VLLM_EXTRA_ARGS"] = legacy
			config.AdapterParams[key] = value
			if err := validateAdapterParams(root, "llm:qwen3.6-35b", config); err == nil {
				t.Fatalf("overlapping structured key %q accepted", key)
			}
		})
	}

	config := repositoryWorkerConfig(root, "vllm-chat-worker", "vllm")
	config.AdapterParams["LLM_CTX_SIZE"] = "8192"
	config.AdapterParams["VLLM_MODEL"] = "RedHatAI/Qwen3.6-35B-A3B-NVFP4"
	config.AdapterParams["VLLM_DTYPE"] = "bfloat16"
	config.AdapterParams["VLLM_MAX_NUM_SEQS"] = "16"
	config.AdapterParams["VLLM_EXTRA_ARGS"] = legacy
	if err := validateAdapterParams(root, "llm:qwen3.6-35b", config); err != nil {
		t.Fatalf("non-overlapping production settings rejected: %v", err)
	}
}

func TestAdapterParamsRejectInjectionSensitiveKeysAndSpellings(t *testing.T) {
	root := t.TempDir()
	keys := []string{
		"LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
		"PATH", "PYTHONPATH", "PYTHONHOME", "SHELL", "BASH_ENV", "ENV", "NODE_OPTIONS",
		"VLLM_EXTRA_ARGS", "LLM_FLAGS", "llm_ctx_size", "LLM-CTX-SIZE", "ＬＬＭ_CTX_SIZE",
		"LLM_CTX_SIZE ", "LLM_CTX_SİZE",
	}
	for _, key := range keys {
		t.Run(key, func(t *testing.T) {
			config := repositoryWorkerConfig(root, "llm-worker", "llamacpp")
			config.AdapterParams[key] = "harmless"
			err := validateAdapterParams(root, "llm:test", config)
			if err == nil || !strings.Contains(err.Error(), untrustedAdapterParamsMessage) {
				t.Fatalf("key %q policy error = %v", key, err)
			}
		})
	}
}

func TestAdapterParamsRejectNonCanonicalValuesAndPathAliases(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "target.gguf")
	writeExecutableTestFile(t, target, "model")
	alias := filepath.Join(root, "alias.gguf")
	if err := os.Symlink(target, alias); err != nil {
		t.Fatal(err)
	}
	cases := map[string]string{
		"LLM_CTX_SIZE":   "8192\nLD_PRELOAD=x",
		"LLM_GPU_LAYERS": "+1",
		"LLM_HF_REPO":    "../model",
		"LLM_HF_FILE":    "../model.gguf",
		"LLM_MODEL_PATH": alias,
		"LLM_PARALLEL":   "1 --flag",
	}
	for key, value := range cases {
		t.Run(key, func(t *testing.T) {
			config := repositoryWorkerConfig(root, "llm-worker", "llamacpp")
			config.AdapterParams[key] = value
			if err := validateAdapterParams(root, "llm:test", config); err == nil {
				t.Fatalf("non-canonical %s value %q accepted", key, value)
			}
		})
	}
}

func TestAdapterParamsRejectNonFiniteDecimal(t *testing.T) {
	root := t.TempDir()
	config := repositoryWorkerConfig(root, "vllm-chat-worker", "vllm")
	config.AdapterParams["VLLM_GPU_MEMORY_UTILIZATION"] = "NaN"
	if err := validateAdapterParams(root, "llm:test", config); err == nil {
		t.Fatal("NaN GPU memory utilization accepted")
	}
}

func TestWorkerEnvironmentStripsLoaderInterpreterShellAndInheritedPath(t *testing.T) {
	root := t.TempDir()
	workerPath := filepath.Join(root, "llm-worker")
	capturePath := filepath.Join(root, "environment.txt")
	writeEnvironmentCaptureWorker(t, workerPath, capturePath)
	libraryPath := compileHarmlessSharedLibrary(t)

	t.Setenv("LD_PRELOAD", libraryPath)
	t.Setenv("DYLD_INSERT_LIBRARIES", libraryPath)
	t.Setenv("PYTHONPATH", "/tmp/injected-python")
	t.Setenv("PYTHONHOME", "/tmp/injected-home")
	t.Setenv("BASH_ENV", "/tmp/injected-shell")
	t.Setenv("NODE_OPTIONS", "--require=/tmp/injected.js")
	t.Setenv("PATH", "/tmp/injected-bin")

	instance := NewInstance("llm:test", "llm:test#0", 1, 1, "/usr/bin/python3", root)
	instance.workerCmd = []string{workerPath}
	instance.adapterParams = map[string]string{"LLM_BACKEND": "llamacpp", "LLM_CTX_SIZE": "8192"}
	if err := instance.Load("cuda"); err != nil {
		t.Fatalf("load capture worker: %v", err)
	}
	t.Cleanup(instance.Kill)

	environment := readEnvironmentFile(t, capturePath)
	for _, forbidden := range []string{"LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "PYTHONHOME", "BASH_ENV", "NODE_OPTIONS"} {
		if _, ok := environment[forbidden]; ok {
			t.Errorf("forbidden inherited variable reached worker: %s", forbidden)
		}
	}
	// PYTHONPATH is forbidden as a caller-supplied adapter param, but the
	// server sets its own repo-owned value so the default system-python
	// worker (whose .venv interpreter resolves to /usr/bin/python3.12 and
	// loses venv site-packages activation) can still `import arbiter`.
	// The caller-injected PYTHONPATH must never survive; only the repo path.
	if got := environment["PYTHONPATH"]; got != filepath.Join(root, "src") {
		t.Fatalf("PYTHONPATH must be the server-set repo src path, got %q; injected value should not survive", got)
	}
	if strings.Contains(environment["PATH"], "/tmp/injected-bin") {
		t.Fatalf("inherited PATH reached worker: %q", environment["PATH"])
	}
	if environment["LLM_CTX_SIZE"] != "8192" {
		t.Fatalf("sanctioned setting missing: %v", environment)
	}
}

// TestDefaultPythonWorkerExposesArbiterPackage regresses the 2026-07-19
// outage: commit 0289046 hardened the worker environment by dropping the
// inherited PYTHONPATH, but resolveTrustedPythonExecutable collapses the
// repo's .venv/bin/python symlink chain to /usr/bin/python3.12 — losing the
// venv's editable-install .pth. Default system-python workers (ltx2-encode,
// ltx2, latentsync, composite, sadtalker, …) then died instantly with
// "No module named 'arbiter'". The server must set its own repo-owned
// PYTHONPATH so import resolution is independent of interpreter resolution.
// buildCleanWorkerEnvironment is shared by every worker path (repository
// worker, sanctioned venv, and default system-python), so we exercise it via
// the same llm-worker capture harness as the strip test above.
func TestDefaultPythonWorkerExposesArbiterPackage(t *testing.T) {
	root := t.TempDir()
	workerPath := filepath.Join(root, "llm-worker")
	capturePath := filepath.Join(root, "environment.txt")
	writeEnvironmentCaptureWorker(t, workerPath, capturePath)

	instance := NewInstance("llm:test", "llm:test#0", 1, 1, "/usr/bin/python3", root)
	instance.workerCmd = []string{workerPath}
	if err := instance.Load("cuda"); err != nil {
		t.Fatalf("load capture worker: %v", err)
	}
	t.Cleanup(instance.Kill)

	environment := readEnvironmentFile(t, capturePath)
	if got := environment["PYTHONPATH"]; got != filepath.Join(root, "src") {
		t.Fatalf("worker PYTHONPATH = %q, want %q (regression: system-python workers must import arbiter)", got, filepath.Join(root, "src"))
	}
}

// TestRepoVenvPythonSymlinkCollapsesToSystemExecutable documents the second
// half of the 2026-07-19 outage and pins the deploy-to-spark.sh guard.
// resolveTrustedPythonExecutable deliberately collapses the interpreter via
// EvalSymlinks (defence against symlink-swap TOCTOU). A repo .venv created
// with `python -m venv` (the default, symlinked) has .venv/bin/python →
// python3 → /usr/bin/python3.12; the collapse returns the SYSTEM python,
// which loses venv activation (pyvenv.cfg lookup) and every site-packages
// dependency (torch, diffusers, ltx_core). The PYTHONPATH fix above restores
// `import arbiter`, but site-packages deps require venv activation — so the
// deploy converts the repo .venv to real binary copies (--copies semantics),
// exactly like the sanctioned per-adapter venvs. This test proves the trap:
// a symlinked .venv python collapses to its underlying target and would
// break venv-only imports, which is why the deploy step is mandatory.
func TestRepoVenvPythonSymlinkCollapsesToSystemExecutable(t *testing.T) {
	// EvalSymlinks resolves macOS's /var → /private/var, so evaluate the
	// projectRoot the same way to keep both sides of trustedPythonLocation
	// in agreement.
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	venvBin := filepath.Join(root, ".venv", "bin")
	if err := os.MkdirAll(venvBin, 0o755); err != nil {
		t.Fatal(err)
	}
	// The symlink target lives inside the repo root (a sanctioned location
	// for the test) and stands in for /usr/bin/python3.12.
	underlying := filepath.Join(root, "python3.12-real")
	writeExecutableTestFile(t, underlying, "#!/bin/sh\nexit 0\n")
	// Mirror the default `python -m venv` layout: python → python3 → target.
	if err := os.Symlink(underlying, filepath.Join(venvBin, "python3")); err != nil {
		t.Fatal(err)
	}
	venvPython := filepath.Join(venvBin, "python")
	if err := os.Symlink("python3", venvPython); err != nil {
		t.Fatal(err)
	}
	resolved, err := resolveTrustedPythonExecutable(root, venvPython)
	if err != nil {
		t.Fatalf("resolve symlinked .venv python: %v", err)
	}
	if resolved == venvPython {
		t.Fatal("symlinked .venv python resolved to itself; a symlinked venv interpreter must collapse to its target — if that ever changes, revisit the deploy's binary-copy guard")
	}
	if resolved != underlying {
		t.Fatalf("symlinked .venv python resolved to %q, want the underlying target %q", resolved, underlying)
	}
}

func TestSpawnRejectsAdapterParamMutationImmediatelyBeforeExec(t *testing.T) {
	root := t.TempDir()
	workerPath := filepath.Join(root, "llm-worker")
	marker := filepath.Join(root, "executed")
	writeExecutableTestFile(t, workerPath, "#!/bin/sh\ntouch "+marker+"\n")
	instance := NewInstance("llm:test", "llm:test#0", 1, 1, "/usr/bin/python3", root)
	instance.workerCmd = []string{workerPath}
	instance.adapterParams = map[string]string{"LLM_BACKEND": "llamacpp"}
	instance.adapterParams["LD_PRELOAD"] = filepath.Join(root, "payload.so")
	if err := instance.Spawn(); err == nil || !strings.Contains(err.Error(), untrustedAdapterParamsMessage) {
		t.Fatalf("Spawn policy error = %v", err)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("mutated worker executed; marker stat error = %v", err)
	}
}

func TestLoadConfigRejectsPersistedInjectedAdapterParams(t *testing.T) {
	root := t.TempDir()
	localDirectory := filepath.Join(root, "local")
	if err := os.MkdirAll(localDirectory, 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"models":{"llm:poisoned":{"memory_gb":1,"worker_cmd":["` + filepath.Join(root, "llm-worker") + `"],"adapter_params":{"LLM_BACKEND":"llamacpp","LD_PRELOAD":"/tmp/payload.so"}},"birefnet":{"memory_gb":1}}}`
	if err := os.WriteFile(filepath.Join(localDirectory, "config.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	config, err := LoadConfig(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := config.Models["llm:poisoned"]; ok {
		t.Fatal("persisted injected model survived startup policy")
	}
	if _, ok := config.Models["birefnet"]; !ok {
		t.Fatal("valid sibling model was removed")
	}
}

func TestSanctionedVenvInterpreterSymlinkChainExecutesResolvedRegularFile(t *testing.T) {
	root := t.TempDir()
	bin := filepath.Join(root, "venvs", "qwentts", "bin")
	realPython := filepath.Join(bin, "python-real")
	marker := filepath.Join(root, "executed")
	writeExecutableTestFile(t, realPython, "#!/bin/sh\nprintf trusted > "+marker+"\n")
	if err := os.Symlink("python-real", filepath.Join(bin, "python3")); err != nil {
		t.Fatal(err)
	}
	interpreter := filepath.Join(bin, "python")
	if err := os.Symlink("python3", interpreter); err != nil {
		t.Fatal(err)
	}
	command := []string{interpreter, "-m", "arbiter.worker_main", "tts-custom"}
	resolved, err := resolveWorkerExecutable(root, command, "python3")
	if err != nil {
		t.Fatalf("resolve sanctioned venv chain: %v", err)
	}
	if resolved != realPython {
		t.Fatalf("resolved interpreter = %q, want %q", resolved, realPython)
	}
	if output, err := exec.Command(resolved).CombinedOutput(); err != nil {
		t.Fatalf("execute resolved interpreter: %v: %s", err, output)
	}
	if contents, err := os.ReadFile(marker); err != nil || string(contents) != "trusted" {
		t.Fatalf("trusted process marker = %q, error = %v", contents, err)
	}
}

func TestSanctionedVenvInterpreterRejectsEscapesCyclesAndTraversal(t *testing.T) {
	root := t.TempDir()
	bin := filepath.Join(root, "venvs", "rvc", "bin")
	outside := filepath.Join(root, "outside")
	writeExecutableTestFile(t, outside, "#!/bin/sh\nexit 0\n")
	cases := map[string]func(string) error{
		"absolute escape":    func(path string) error { return os.Symlink(outside, path) },
		"relative traversal": func(path string) error { return os.Symlink("../../../outside", path) },
		"cycle": func(path string) error {
			if err := os.MkdirAll(bin, 0o755); err != nil {
				return err
			}
			if err := os.Symlink("python", filepath.Join(bin, "python3")); err != nil {
				return err
			}
			return os.Symlink("python3", path)
		},
	}
	for name, create := range cases {
		t.Run(name, func(t *testing.T) {
			caseRoot := filepath.Join(root, strings.ReplaceAll(name, " ", "-"))
			caseBin := filepath.Join(caseRoot, "venvs", "rvc", "bin")
			bin = caseBin
			if err := os.MkdirAll(caseBin, 0o755); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(caseBin, "python")
			if err := create(path); err != nil {
				t.Fatal(err)
			}
			command := []string{path, "-m", "arbiter.worker_main", "rvc-convert"}
			if _, err := resolveWorkerExecutable(caseRoot, command, "python3"); err == nil {
				t.Fatal("adversarial interpreter chain was accepted")
			}
		})
	}
}

func TestResolvedVenvInterpreterCannotBeReplacedThroughOriginalSymlink(t *testing.T) {
	root := t.TempDir()
	bin := filepath.Join(root, "venvs", "kokoro", "bin")
	trusted := filepath.Join(bin, "python-real")
	trustedMarker := filepath.Join(root, "trusted")
	malicious := filepath.Join(root, "malicious")
	maliciousMarker := filepath.Join(root, "malicious-ran")
	writeExecutableTestFile(t, trusted, "#!/bin/sh\ntouch "+trustedMarker+"\n")
	writeExecutableTestFile(t, malicious, "#!/bin/sh\ntouch "+maliciousMarker+"\n")
	interpreter := filepath.Join(bin, "python")
	if err := os.Symlink("python-real", interpreter); err != nil {
		t.Fatal(err)
	}
	resolved, err := resolveWorkerExecutable(root, []string{interpreter, "-m", "arbiter.worker_main", "tts-kokoro"}, "python3")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(interpreter); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(malicious, interpreter); err != nil {
		t.Fatal(err)
	}
	if output, err := exec.Command(resolved).CombinedOutput(); err != nil {
		t.Fatalf("execute pinned resolved interpreter: %v: %s", err, output)
	}
	if _, err := os.Stat(trustedMarker); err != nil {
		t.Fatalf("trusted interpreter did not run: %v", err)
	}
	if _, err := os.Stat(maliciousMarker); !os.IsNotExist(err) {
		t.Fatalf("replacement interpreter ran; marker error = %v", err)
	}
}

func TestEverySanctionedCustomVenvInterpreterChainResolves(t *testing.T) {
	for adapter, venv := range trustedPythonAdapters {
		if venv == "" {
			continue
		}
		t.Run(adapter, func(t *testing.T) {
			root := t.TempDir()
			bin := filepath.Join(root, "venvs", venv, "bin")
			writeExecutableTestFile(t, filepath.Join(bin, "python-real"), "#!/bin/sh\nexit 0\n")
			if err := os.Symlink("python-real", filepath.Join(bin, "python3")); err != nil {
				t.Fatal(err)
			}
			interpreter := filepath.Join(bin, "python")
			if err := os.Symlink("python3", interpreter); err != nil {
				t.Fatal(err)
			}
			command := []string{interpreter, "-m", "arbiter.worker_main", adapter}
			if _, err := resolveWorkerExecutable(root, command, "python3"); err != nil {
				t.Fatalf("sanctioned %s interpreter rejected: %v", venv, err)
			}
		})
	}
}

func TestSanctionedVenvInterpreterRejectsSymlinkedParentEscape(t *testing.T) {
	root := t.TempDir()
	bin := filepath.Join(root, "venvs", "demucs", "bin")
	outside := filepath.Join(root, "outside")
	writeExecutableTestFile(t, filepath.Join(outside, "python-real"), "#!/bin/sh\nexit 0\n")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(bin, "redirect")); err != nil {
		t.Fatal(err)
	}
	interpreter := filepath.Join(bin, "python")
	if err := os.Symlink("redirect/python-real", interpreter); err != nil {
		t.Fatal(err)
	}
	command := []string{interpreter, "-m", "arbiter.worker_main", "demucs"}
	if _, err := resolveWorkerExecutable(root, command, "python3"); err == nil {
		t.Fatal("interpreter escaped through a symlinked parent directory")
	}
}

func writeExecutableTestFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o755); err != nil {
		t.Fatal(err)
	}
}

func writeEnvironmentCaptureWorker(t *testing.T, path, capturePath string) {
	t.Helper()
	script := "#!/bin/sh\nenv > " + capturePath + "\nwhile IFS= read -r line; do printf '{\"status\":\"ok\"}\\n'; done\n"
	writeExecutableTestFile(t, path, script)
}

func readEnvironmentFile(t *testing.T, path string) map[string]string {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = file.Close() })
	result := make(map[string]string)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		key, value, found := strings.Cut(scanner.Text(), "=")
		if found {
			result[key] = value
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return result
}

func compileHarmlessSharedLibrary(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	source := filepath.Join(directory, "harmless.c")
	library := filepath.Join(directory, "harmless.so")
	if err := os.WriteFile(source, []byte("int arbiter_harmless(void) { return 0; }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	arguments := []string{"-shared", "-fPIC", source, "-o", library}
	if runtime.GOOS == "darwin" {
		library = filepath.Join(directory, "harmless.dylib")
		arguments = []string{"-dynamiclib", source, "-o", library}
	}
	command := exec.Command("cc", arguments...)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("compile harmless shared library: %v: %s", err, output)
	}
	return library
}
