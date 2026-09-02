package main

import (
	"fmt"
	"maps"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const untrustedAdapterParamsMessage = "adapter_params contains an unsanctioned worker setting"

type adapterValueValidator func(string, string) error

var adapterKeyPattern = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_]*$`)
var modelReferencePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+:/-]*$`)

var llamaAdapterParams = map[string]adapterValueValidator{
	"LLM_BACKEND":            enumAdapterValue("llamacpp"),
	"LLM_CTX_SIZE":           integerAdapterValue(128, 1048576),
	"LLM_GPU_LAYERS":         integerAdapterValue(-1, 10000),
	"LLM_HF_FILE":            relativePathAdapterValue,
	"LLM_HF_REPO":            modelReferenceAdapterValue,
	"LLM_MODEL_PATH":         absoluteModelPathAdapterValue,
	"LLM_PARALLEL":           integerAdapterValue(1, 1024),
	"LLAMA_ARG_CACHE_TYPE_K": enumAdapterValue("f32", "f16", "bf16", "q8_0", "q4_0"),
	"LLAMA_ARG_CACHE_TYPE_V": enumAdapterValue("f32", "f16", "bf16", "q8_0", "q4_0"),
	"LLAMA_ARG_FLASH_ATTN":   enumAdapterValue("on", "off"),
	"LLAMA_ARG_JINJA":        enumAdapterValue("on", "off"),
	"LLAMA_SERVER_BIN":       trustedLlamaServerPath,
}

var vllmAdapterParams = map[string]adapterValueValidator{
	"LLM_BACKEND":                 enumAdapterValue("vllm"),
	"LLM_CTX_SIZE":                integerAdapterValue(128, 1048576),
	"VLLM_DTYPE":                  enumAdapterValue("auto", "half", "float16", "bfloat16", "float32"),
	"VLLM_GPU_MEMORY_UTILIZATION": decimalAdapterValue(0.01, 0.99),
	"VLLM_MAX_MODEL_LEN":          integerAdapterValue(128, 1048576),
	"VLLM_MAX_NUM_SEQS":           integerAdapterValue(1, 4096),
	"VLLM_MODEL":                  modelOrAbsolutePathAdapterValue,
	"VLLM_QUANTIZATION":           enumAdapterValue("awq", "gptq", "bitsandbytes", "fp8", "compressed-tensors"),
	"VLLM_READY_TIMEOUT_SEC":      integerAdapterValue(1, 3600),
	"VLLM_TENSOR_PARALLEL_SIZE":   integerAdapterValue(1, 128),
}

// torchWorkerAdapterParams is the sanctioned allocator tuning surface for
// python adapter workers (sanctioned venv or repo default). The CUDA caching
// allocator's fixed-size block strategy strands a fragmented high-water far
// above live tensors; expandable segments let the large transient generation
// buffers (the h3 video pipeline peaks ~31 GB above its 49 GB of weights)
// map and unmap instead of pinning the box below the EmergencyGuardian and
// earlyoom floors mid-render.
var torchWorkerAdapterParams = map[string]adapterValueValidator{
	"PYTORCH_CUDA_ALLOC_CONF": exactAdapterValues("expandable_segments:True"),
}

var vllmLegacyTuningByModel = map[string]string{
	"llm:qwen3.6-35b": `--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 8G --enforce-eager --enable-auto-tool-choice --tool-call-parser hermes`,
}

// The qwen memory-budget transition promotes the combined admission/KV limit
// while preserving every deployed vector for rollback. This policy change
// does not edit local/config.json; production can migrate its authoritative
// config only after this release is deployed.
//
// The primary vector additionally carries vLLM's native OpenAI tool-calling
// flags, because production's authoritative config adopted them on 2026-08-03.
// Sanctioning them is load-bearing, not cosmetic: an unsanctioned value drops
// the model from the runnable config, which in turn makes the `local-vision`
// alias unresolvable and aborts config load, so arbiter refuses to start at
// all. Keep the pre-tool-calling vector below so a rollback release still
// validates the same config.
var vllmLegacyTuningAlternatesByModel = map[string][]string{
	"llm:qwen3.6-35b": {
		`--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.25 --enforce-eager`,
		`--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --enforce-eager`,
		`--max-model-len 32768 --max-num-batched-tokens 32768 --kv-cache-memory-bytes 8G --enforce-eager`,
		`--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 8G --enforce-eager`,
	},
}

var vllmLegacyOverlappingParams = []string{"VLLM_MAX_MODEL_LEN", "VLLM_GPU_MEMORY_UTILIZATION"}

var inheritedWorkerEnvironment = []string{
	"CUDA_VISIBLE_DEVICES", "HOME", "HTTP_PROXY", "HTTPS_PROXY", "LANG", "LC_ALL", "LC_CTYPE",
	"LOGNAME", "NVIDIA_VISIBLE_DEVICES", "NO_PROXY", "SSL_CERT_DIR", "SSL_CERT_FILE", "TMPDIR",
	"TORCH_CUDA_ARCH_LIST", "TZ", "USER", "http_proxy", "https_proxy", "no_proxy",
}

var forbiddenAdapterKeys = map[string]bool{
	"BASH_ENV": true, "CDPATH": true, "CFLAGS": true, "CLASSPATH": true, "CPPFLAGS": true,
	"ENV": true, "GEM_HOME": true, "GLOBIGNORE": true, "GOPATH": true, "GOROOT": true,
	"IFS": true, "JAVA_TOOL_OPTIONS": true, "JDK_JAVA_OPTIONS": true, "LDFLAGS": true,
	"NODE_OPTIONS": true, "PATH": true, "PERL5LIB": true, "PROMPT_COMMAND": true,
	"PS4": true, "PYTHONBREAKPOINT": true, "PYTHONHOME": true, "PYTHONINSPECT": true,
	"PYTHONPATH": true, "PYTHONSTARTUP": true, "PYTHONWARNINGS": true, "RUBYLIB": true,
	"SHELL": true,
}

func validateAdapterParams(projectRoot, modelID string, config ModelConfig) error {
	allowed := adapterParamPolicy(modelID, config)
	if err := validateVllmLegacyTuningOverlap(config); err != nil {
		return err
	}
	for key, value := range config.AdapterParams {
		if key == "remote_model_tag" {
			if err := modelReferenceAdapterValue(projectRoot, value); err != nil {
				return adapterParamError(key, err)
			}
			continue
		}
		if !adapterKeyPattern.MatchString(key) {
			return adapterParamError(key, fmt.Errorf("key must use exact ASCII spelling"))
		}
		validator, ok := allowed[key]
		if forbiddenAdapterKey(key) && (key != "VLLM_EXTRA_ARGS" || !ok) {
			return adapterParamError(key, fmt.Errorf("loader, path, shell, interpreter, and command-option variables are forbidden"))
		}
		if !ok {
			return adapterParamError(key, fmt.Errorf("key is not allowlisted for model %q", modelID))
		}
		if err := validator(projectRoot, value); err != nil {
			return adapterParamError(key, err)
		}
	}
	return nil
}

func forbiddenAdapterKey(key string) bool {
	return strings.HasPrefix(key, "LD_") || strings.HasPrefix(key, "DYLD_") ||
		strings.HasSuffix(key, "_EXTRA_ARGS") || strings.HasSuffix(key, "_FLAGS") ||
		forbiddenAdapterKeys[key]
}

func adapterParamPolicy(modelID string, config ModelConfig) map[string]adapterValueValidator {
	if len(config.WorkerCmd) == 1 {
		switch filepath.Base(config.WorkerCmd[0]) {
		case "llm-worker":
			return llamaAdapterParams
		case "vllm-chat-worker":
			allowed := maps.Clone(vllmAdapterParams)
			if legacy, ok := vllmLegacyTuningByModel[modelID]; ok {
				values := append([]string{legacy}, vllmLegacyTuningAlternatesByModel[modelID]...)
				allowed["VLLM_EXTRA_ARGS"] = exactAdapterValues(values...)
			}
			return allowed
		case "vllm-worker":
			allowed := maps.Clone(vllmAdapterParams)
			delete(allowed, "LLM_BACKEND")
			allowed["VLLM_MODE"] = enumAdapterValue("tts")
			return allowed
		}
	}
	if strings.HasPrefix(modelID, "llm:") && config.AdapterParams["LLM_BACKEND"] == "llamacpp" {
		return llamaAdapterParams
	}
	return torchWorkerAdapterParams
}

func validateVllmLegacyTuningOverlap(config ModelConfig) error {
	if _, ok := config.AdapterParams["VLLM_EXTRA_ARGS"]; !ok {
		return nil
	}
	for _, key := range vllmLegacyOverlappingParams {
		if _, ok := config.AdapterParams[key]; ok {
			return adapterParamError(key, fmt.Errorf("must not overlap with VLLM_EXTRA_ARGS"))
		}
	}
	return nil
}

func exactAdapterValues(expected ...string) adapterValueValidator {
	return func(_ string, value string) error {
		for _, candidate := range expected {
			if value == candidate {
				return nil
			}
		}
		return fmt.Errorf("value must exactly match a sanctioned production vector")
	}
}

func adapterParamError(key string, cause error) error {
	return fmt.Errorf("%s %q: %w", untrustedAdapterParamsMessage, key, cause)
}

func enumAdapterValue(allowed ...string) adapterValueValidator {
	return func(_ string, value string) error {
		for _, candidate := range allowed {
			if value == candidate {
				return nil
			}
		}
		return fmt.Errorf("value %q is not one of %s", value, strings.Join(allowed, ", "))
	}
}

func integerAdapterValue(minimum, maximum int64) adapterValueValidator {
	return func(_ string, value string) error {
		parsed, err := strconv.ParseInt(value, 10, 64)
		if err != nil || strconv.FormatInt(parsed, 10) != value {
			return fmt.Errorf("value must be a canonical base-10 integer")
		}
		if parsed < minimum || parsed > maximum {
			return fmt.Errorf("value must be between %d and %d", minimum, maximum)
		}
		return nil
	}
}

func decimalAdapterValue(minimum, maximum float64) adapterValueValidator {
	return func(_ string, value string) error {
		parsed, err := strconv.ParseFloat(value, 64)
		if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) || parsed < minimum || parsed > maximum {
			return fmt.Errorf("value must be a decimal between %g and %g", minimum, maximum)
		}
		if strconv.FormatFloat(parsed, 'f', -1, 64) != value {
			return fmt.Errorf("value must use canonical plain decimal notation")
		}
		return nil
	}
}

func modelReferenceAdapterValue(_ string, value string) error {
	if !modelReferencePattern.MatchString(value) || strings.Contains(value, "..") || strings.Contains(value, "//") {
		return fmt.Errorf("value must be a canonical model identifier")
	}
	return nil
}

func relativePathAdapterValue(_ string, value string) error {
	if value == "" || filepath.IsAbs(value) || filepath.Clean(value) != value || strings.HasPrefix(value, "..") {
		return fmt.Errorf("value must be a canonical relative path")
	}
	if strings.ContainsAny(value, "\x00\r\n") {
		return fmt.Errorf("value contains control characters")
	}
	return nil
}

func absoluteModelPathAdapterValue(_ string, value string) error {
	if err := validateResolvedPath(value, false); err != nil {
		return err
	}
	info, err := os.Lstat(value)
	if err != nil {
		return fmt.Errorf("inspect model path: %w", err)
	}
	if !info.Mode().IsRegular() || strings.ToLower(filepath.Ext(value)) != ".gguf" {
		return fmt.Errorf("llama.cpp model path must be a regular .gguf file")
	}
	return nil
}

func modelOrAbsolutePathAdapterValue(projectRoot, value string) error {
	if filepath.IsAbs(value) {
		return validateResolvedPath(value, false)
	}
	return modelReferenceAdapterValue(projectRoot, value)
}

func trustedLlamaServerPath(projectRoot, value string) error {
	if err := validateResolvedPath(value, true); err != nil {
		return err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("resolve home directory: %w", err)
	}
	allowed := []string{
		filepath.Join(projectRoot, "local", "bin", "llama-server"),
		filepath.Join(home, "src", "llama.cpp", "build", "bin", "llama-server"),
		"/usr/local/bin/llama-server",
		"/opt/homebrew/bin/llama-server",
	}
	for _, candidate := range allowed {
		if value == candidate {
			return nil
		}
	}
	return fmt.Errorf("executable is outside the sanctioned llama-server paths")
}

func validateResolvedPath(value string, executable bool) error {
	if !filepath.IsAbs(value) || filepath.Clean(value) != value || strings.ContainsAny(value, "\x00\r\n") {
		return fmt.Errorf("path must be absolute and canonical")
	}
	info, err := os.Lstat(value)
	if err != nil {
		return fmt.Errorf("inspect path: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("path must not be a symlink")
	}
	if executable && (!info.Mode().IsRegular() || info.Mode().Perm()&0o111 == 0) {
		return fmt.Errorf("path must be an executable regular file")
	}
	return nil
}

func resolveWorkerExecutable(projectRoot string, workerCommand []string, pythonBin string) (string, error) {
	if len(workerCommand) > 0 {
		if isSanctionedVenvWorkerCommand(projectRoot, workerCommand) {
			return resolveSanctionedVenvPython(projectRoot, workerCommand[0])
		}
		if err := validateResolvedPath(workerCommand[0], true); err != nil {
			return "", fmt.Errorf("worker executable: %w", err)
		}
		return workerCommand[0], nil
	}
	return resolveTrustedPythonExecutable(projectRoot, pythonBin)
}

func isSanctionedVenvWorkerCommand(projectRoot string, command []string) bool {
	if len(command) != 4 || command[1] != "-m" || command[2] != "arbiter.worker_main" {
		return false
	}
	venv, ok := trustedPythonAdapters[command[3]]
	return ok && venv != "" && command[0] == filepath.Join(projectRoot, "venvs", venv, "bin", "python")
}

func resolveSanctionedVenvPython(projectRoot, interpreter string) (string, error) {
	venvRoot := filepath.Dir(filepath.Dir(interpreter))
	if err := rejectSymlinkPathComponents(projectRoot, venvRoot); err != nil {
		return "", fmt.Errorf("venv interpreter parent: %w", err)
	}
	resolved, err := resolveTrustedInterpreterChain(venvRoot, interpreter)
	if err != nil {
		return "", err
	}
	return resolved, nil
}

func rejectSymlinkPathComponents(projectRoot, path string) error {
	relative, err := filepath.Rel(projectRoot, path)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(os.PathSeparator)) {
		return fmt.Errorf("venv path is outside the project root")
	}
	current := projectRoot
	for _, component := range strings.Split(relative, string(os.PathSeparator)) {
		current = filepath.Join(current, component)
		info, inspectErr := os.Lstat(current)
		if inspectErr != nil {
			return fmt.Errorf("inspect %q: %w", current, inspectErr)
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("parent path %q must not be a symlink", current)
		}
	}
	return nil
}

func resolveTrustedInterpreterChain(venvRoot, interpreter string) (string, error) {
	current := interpreter
	visited := make(map[string]bool)
	for {
		if visited[current] {
			return "", fmt.Errorf("python executable symlink cycle at %q", current)
		}
		visited[current] = true
		info, err := os.Lstat(current)
		if err != nil {
			return "", fmt.Errorf("inspect Python executable %q: %w", current, err)
		}
		if info.Mode()&os.ModeSymlink == 0 {
			if !info.Mode().IsRegular() || info.Mode().Perm()&0o111 == 0 {
				return "", fmt.Errorf("resolved Python executable must be an executable regular file")
			}
			return current, nil
		}
		target, err := os.Readlink(current)
		if err != nil {
			return "", fmt.Errorf("read Python executable symlink %q: %w", current, err)
		}
		next, err := trustedInterpreterTarget(venvRoot, current, target)
		if err != nil {
			return "", err
		}
		current = next
	}
}

func trustedInterpreterTarget(venvRoot, current, target string) (string, error) {
	if target == "" || strings.ContainsAny(target, "\x00\r\n") {
		return "", fmt.Errorf("python executable symlink has an invalid target")
	}
	if !filepath.IsAbs(target) {
		if filepath.Clean(target) != target || target == ".." || strings.HasPrefix(target, ".."+string(os.PathSeparator)) {
			return "", fmt.Errorf("python executable symlink uses relative traversal")
		}
		target = filepath.Join(filepath.Dir(current), target)
	} else if filepath.Clean(target) != target {
		return "", fmt.Errorf("python executable symlink target must be canonical")
	}
	if !trustedInterpreterLocation(venvRoot, target) {
		return "", fmt.Errorf("python executable symlink escapes sanctioned venv/toolchain locations")
	}
	trustedRoot, ok := trustedInterpreterRoot(venvRoot, target)
	if !ok {
		return "", fmt.Errorf("python executable symlink has no sanctioned trust root")
	}
	if err := rejectSymlinkPathComponents(trustedRoot, filepath.Dir(target)); err != nil {
		return "", fmt.Errorf("python executable symlink parent: %w", err)
	}
	return target, nil
}

func trustedInterpreterLocation(venvRoot, value string) bool {
	_, ok := trustedInterpreterRoot(venvRoot, value)
	return ok
}

func trustedInterpreterRoot(venvRoot, value string) (string, bool) {
	for _, root := range []string{venvRoot, "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "/Library/Frameworks/Python.framework"} {
		if pathWithin(root, value) {
			return root, true
		}
	}
	return "", false
}

func pathWithin(root, value string) bool {
	relative, err := filepath.Rel(root, value)
	return err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(os.PathSeparator))
}

func resolveTrustedPythonExecutable(projectRoot, configured string) (string, error) {
	original := configured
	if !filepath.IsAbs(original) {
		if original != "python3" {
			return "", fmt.Errorf("python executable %q is not sanctioned", original)
		}
		resolved, err := exec.LookPath(original)
		if err != nil {
			return "", fmt.Errorf("resolve Python executable: %w", err)
		}
		original = resolved
	}
	if filepath.Clean(original) != original {
		return "", fmt.Errorf("python executable path must be canonical")
	}
	resolved, err := filepath.EvalSymlinks(original)
	if err != nil {
		return "", fmt.Errorf("resolve Python executable symlinks: %w", err)
	}
	if err := validateResolvedPath(resolved, true); err != nil {
		return "", fmt.Errorf("resolved Python executable: %w", err)
	}
	if !trustedPythonLocation(projectRoot, original) || !trustedPythonLocation(projectRoot, resolved) {
		return "", fmt.Errorf("python executable resolves outside sanctioned locations")
	}
	return resolved, nil
}

func trustedPythonLocation(projectRoot, value string) bool {
	allowedRoots := []string{
		projectRoot,
		"/usr/bin",
		"/usr/local/bin",
		"/opt/homebrew/bin",
		"/Library/Frameworks/Python.framework",
	}
	for _, root := range allowedRoots {
		relative, err := filepath.Rel(root, value)
		if err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(os.PathSeparator)) {
			return true
		}
	}
	return false
}

func buildCleanWorkerEnvironment(projectRoot, executable string, memoryGB float64, params map[string]string) []string {
	values := make(map[string]string)
	for _, key := range inheritedWorkerEnvironment {
		if value, ok := os.LookupEnv(key); ok {
			values[key] = value
		}
	}
	values["PATH"] = trustedWorkerPath(projectRoot, executable)
	values["PYTHONUNBUFFERED"] = "1"
	// Worker subprocesses import `arbiter.worker_main` via `python -m`. The
	// trusted interpreter may be the repo's .venv symlink, which
	// resolveTrustedPythonExecutable collapses via EvalSymlinks to the
	// underlying system python (/usr/bin/python3.12) — losing the venv's
	// site-packages activation and the editable-install .pth that exposes
	// the `arbiter` package. PYTHONPATH is forbidden as a caller-supplied
	// adapter param (see forbiddenAdapterKeys), but the server sets its own
	// repo-owned value here so the default system-python worker path (used by
	// ltx2-encode, ltx2, latentsync, composite, sadtalker, …) resolves the
	// package regardless of interpreter resolution. A later caller-supplied
	// PYTHONPATH in `params` cannot reach here: validateAdapterParams rejects it.
	values["PYTHONPATH"] = filepath.Join(projectRoot, "src")
	if memoryGB > 0 {
		values["ARBITER_MEMORY_GB"] = strconv.FormatFloat(memoryGB, 'g', -1, 64)
	}
	for key, value := range params {
		if key != "remote_model_tag" {
			values[key] = value
		}
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	environment := make([]string, 0, len(keys))
	for _, key := range keys {
		environment = append(environment, key+"="+values[key])
	}
	return environment
}

func trustedWorkerPath(projectRoot, executable string) string {
	directories := []string{filepath.Dir(executable), filepath.Join(projectRoot, ".venv", "bin"), "/usr/local/cuda/bin", "/usr/local/bin", "/usr/bin", "/bin"}
	seen := make(map[string]bool)
	result := make([]string, 0, len(directories))
	for _, directory := range directories {
		if directory != "." && !seen[directory] {
			seen[directory] = true
			result = append(result, directory)
		}
	}
	return strings.Join(result, string(os.PathListSeparator))
}

func cloneAdapterParams(params map[string]string) map[string]string {
	return maps.Clone(params)
}
