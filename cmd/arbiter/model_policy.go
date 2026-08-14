package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

const stillImageDisabledMessage = "still-image generation is actively disabled in Arbiter; callers must use the Mac mini Codex image service"

const untrustedWorkerCommandMessage = "worker_cmd is not a trusted repository-owned Arbiter adapter/worker identity"

var disabledStillImageMarkers = []string{
	"flux", "kontext", "z-image", "zimage", "stable-diffusion", "stablediffusion",
	"sdxl", "sd3", "sd-3", "pixart", "kandinsky", "aura-flow", "auraflow",
	"playground-v", "ideogram", "recraft", "hidream", "hunyuan-image", "qwen-image",
	"kolors", "omnigen", "dreamshaper", "realvis", "juggernaut", "image-generator",
}

var trustedPythonAdapters = map[string]string{
	"aesthetic-scorer":        "aesthetic",
	"birefnet":                "birefnet",
	"composite":               "",
	"demucs":                  "demucs",
	"echomimic":               "",
	"embed-text":              "embed",
	"face-restore":            "",
	"face-restore-codeformer": "",
	"insightface":             "insightface",
	"latentsync":              "",
	"lora-train":              "",
	"ltx2":                    "",
	"ltx2-denoise1":           "",
	"ltx2-denoise2":           "",
	"ltx2-dev-denoise1":       "",
	"ltx2-dev-denoise2":       "",
	"ltx2-encode":             "",
	"minimax-h3-local":        "minimax-h3",
	"moondream":               "moondream",
	"minimax-h3":              "",
	"rvc-convert":             "rvc",
	"rvc-train":               "rvc",
	"voice-fit":               "voxsmith",
	"sadtalker":               "",
	"sonic":                   "",
	"tts-clone":               "qwentts",
	"tts-custom":              "qwentts",
	"tts-design":              "qwentts",
	"tts-kokoro":              "kokoro",
	"wan-s2v":                 "",
	"whisper-large":           "whisper",
}

var trustedRepositoryWorkers = map[string]string{
	"llm-worker":       "llamacpp",
	"vllm-chat-worker": "vllm-chat",
	"vllm-worker":      "vllm-tts",
}

func normalizedPolicyText(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	return strings.Map(func(r rune) rune {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r):
			return r
		default:
			return '-'
		}
	}, value)
}

// isDisabledStillImageModel identifies every still-image adapter retained in
// this repository plus common aliases/repository IDs. LoRA is denied unless it
// is Arbiter's language-model trainer or an LTX2 video component.
func isDisabledStillImageModel(modelID string) bool {
	normalized := normalizedPolicyText(modelID)
	if normalized == "" {
		return false
	}
	if normalized == "lora-train" || normalized == "ltx2" || strings.HasPrefix(normalized, "ltx2-") {
		return false
	}
	return disabledStillImageText(normalized, false)
}

func disabledStillImageText(normalized string, allowLora bool) bool {
	if !allowLora && hasPolicyToken(normalized, "lora") {
		return true
	}
	for _, marker := range disabledStillImageMarkers {
		if strings.Contains(normalized, marker) {
			return true
		}
	}
	return false
}

func hasPolicyToken(normalized, token string) bool {
	for _, candidate := range strings.FieldsFunc(normalized, func(character rune) bool {
		return character == '-'
	}) {
		if candidate == token {
			return true
		}
	}
	return false
}

func nestedModelRoutesJob(jobType string) bool {
	switch jobType {
	case "background-remove", "caption", "query", "detect", "point", "transcribe",
		"talking-head", "talking-head-sadtalker", "lipsync", "video-generate-h3", "video-encode",
		"video-denoise1", "video-denoise2", "face-restore", "face-restore-codeformer",
		"face-embed", "aesthetic-score", "composite", "demucs", "chat-completion",
		"chat-completion-stream":
		return true
	default:
		return false
	}
}

func disabledStillImageConfig(modelID string, cfg ModelConfig) bool {
	if isDisabledStillImageModel(modelID) {
		return true
	}
	videoLora := normalizedPolicyText(modelID) == "ltx2" || strings.HasPrefix(normalizedPolicyText(modelID), "ltx2-")
	values := []string{cfg.AutoDownload, cfg.ModelPath}
	values = append(values, cfg.WorkerCmd...)
	for key, value := range cfg.AdapterParams {
		values = append(values, key, value)
	}
	for _, value := range values {
		if disabledStillImageText(normalizedPolicyText(value), videoLora) {
			return true
		}
	}
	return false
}

func validateModelWorkerPolicy(projectRoot, modelID string, cfg ModelConfig, requiresLocal bool) error {
	if disabledStillImageConfig(modelID, cfg) {
		return fmt.Errorf("%s", stillImageDisabledMessage)
	}
	if len(cfg.WorkerCmd) > 0 {
		if err := validateWorkerCommand(projectRoot, modelID, cfg); err != nil {
			return err
		}
	}
	if requiresLocal && len(cfg.WorkerCmd) == 0 && len(cfg.Placements) == 0 {
		if _, ok := trustedPythonAdapters[modelID]; !ok {
			return untrustedWorkerPolicyError(modelID, "no trusted built-in adapter")
		}
	}
	return validateAdapterParams(projectRoot, modelID, cfg)
}

func validateWorkerCommand(projectRoot, modelID string, cfg ModelConfig) error {
	command := cfg.WorkerCmd
	if len(command) == 4 && command[1] == "-m" && command[2] == "arbiter.worker_main" {
		return validatePythonWorkerCommand(projectRoot, modelID, command)
	}
	if len(command) != 1 {
		return untrustedWorkerPolicyError(modelID, "expected an exact worker identity without arbitrary arguments")
	}
	return validateRepositoryWorkerCommand(projectRoot, modelID, cfg)
}

func validatePythonWorkerCommand(projectRoot, modelID string, command []string) error {
	adapterID := command[3]
	venv, ok := trustedPythonAdapters[adapterID]
	if !ok || venv == "" {
		return untrustedWorkerPolicyError(modelID, fmt.Sprintf("adapter %q has no sanctioned custom Python environment", adapterID))
	}
	if modelID != adapterID {
		return untrustedWorkerPolicyError(modelID, fmt.Sprintf("command selects incompatible adapter %q", adapterID))
	}
	expected := filepath.Join(projectRoot, "venvs", venv, "bin", "python")
	if command[0] != expected {
		return untrustedWorkerPolicyError(modelID, fmt.Sprintf("Python executable must be %q", expected))
	}
	return nil
}

func validateRepositoryWorkerCommand(projectRoot, modelID string, cfg ModelConfig) error {
	workerName := filepath.Base(cfg.WorkerCmd[0])
	identity, ok := trustedRepositoryWorkers[workerName]
	expected := filepath.Join(projectRoot, workerName)
	if !ok || cfg.WorkerCmd[0] != expected {
		return untrustedWorkerPolicyError(modelID, "executable is outside the repository worker set")
	}
	if !repositoryWorkerCompatible(identity, modelID) {
		return untrustedWorkerPolicyError(modelID, fmt.Sprintf("worker %q is incompatible with this model", workerName))
	}
	return rejectSymlinkWorker(cfg.WorkerCmd[0], modelID)
}

func repositoryWorkerCompatible(identity, modelID string) bool {
	switch identity {
	case "llamacpp", "vllm-chat":
		return strings.HasPrefix(modelID, "llm:")
	case "vllm-tts":
		return modelID == "tts-voxtral"
	default:
		return false
	}
}

func rejectSymlinkWorker(path, modelID string) error {
	info, err := os.Lstat(path)
	if err != nil && !os.IsNotExist(err) {
		return untrustedWorkerPolicyError(modelID, fmt.Sprintf("inspect executable: %v", err))
	}
	if err == nil && info.Mode()&os.ModeSymlink != 0 {
		return untrustedWorkerPolicyError(modelID, "executable path must not be a symlink")
	}
	return nil
}

func untrustedWorkerPolicyError(modelID, detail string) error {
	return fmt.Errorf("%s for model %q: %s", untrustedWorkerCommandMessage, modelID, detail)
}

func rejectDisabledStillImage(jobType, modelID string) error {
	if jobType == "image-generate" || jobType == "image-edit" || isDisabledStillImageModel(modelID) {
		return fmt.Errorf("%s", stillImageDisabledMessage)
	}
	return nil
}

// validateJobModelCompatibility prevents an override from turning a harmless
// job type into a different adapter invocation. Variants are explicit: LTX2
// video stages and the supported talking-head adapters remain available.
func validateJobModelCompatibility(jobType, modelID string) error {
	if err := rejectDisabledStillImage(jobType, modelID); err != nil {
		return err
	}
	defaultModel, known := JobTypeToModel[jobType]
	if !known {
		if (jobType == "chat-completion" || jobType == "chat-completion-stream") && strings.HasPrefix(modelID, "llm:") {
			return nil
		}
		return fmt.Errorf("unknown job type: %s", jobType)
	}
	if modelID == defaultModel {
		return nil
	}

	compatible := false
	switch jobType {
	case "talking-head":
		compatible = modelID == "sonic" || modelID == "echomimic" || modelID == "wan-s2v"
	case "video-generate":
		normalized := normalizedPolicyText(modelID)
		compatible = modelID == "minimax-h3" || strings.HasPrefix(normalized, "ltx2") &&
			!strings.Contains(normalized, "denoise") && !strings.Contains(normalized, "encode")
	case "video-encode":
		normalized := normalizedPolicyText(modelID)
		compatible = strings.HasPrefix(normalized, "ltx2") && strings.Contains(normalized, "encode")
	case "video-denoise1":
		normalized := normalizedPolicyText(modelID)
		compatible = strings.HasPrefix(normalized, "ltx2") && strings.Contains(normalized, "denoise1")
	case "video-denoise2":
		normalized := normalizedPolicyText(modelID)
		compatible = strings.HasPrefix(normalized, "ltx2") && strings.Contains(normalized, "denoise2")
	default:
		compatible = strings.HasPrefix(modelID, defaultModel+"-")
	}
	if !compatible {
		return fmt.Errorf("model %q is not compatible with job type %q", modelID, jobType)
	}
	return nil
}
