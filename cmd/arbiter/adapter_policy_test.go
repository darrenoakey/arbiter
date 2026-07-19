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
	for _, forbidden := range []string{"LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "PYTHONPATH", "PYTHONHOME", "BASH_ENV", "NODE_OPTIONS"} {
		if _, ok := environment[forbidden]; ok {
			t.Errorf("forbidden inherited variable reached worker: %s", forbidden)
		}
	}
	if strings.Contains(environment["PATH"], "/tmp/injected-bin") {
		t.Fatalf("inherited PATH reached worker: %q", environment["PATH"])
	}
	if environment["LLM_CTX_SIZE"] != "8192" {
		t.Fatalf("sanctioned setting missing: %v", environment)
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
	root := t.TempDir()
	for adapter, venv := range map[string]string{
		"tts-custom": "qwentts", "tts-kokoro": "kokoro", "demucs": "demucs", "rvc-convert": "rvc",
	} {
		t.Run(adapter, func(t *testing.T) {
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
