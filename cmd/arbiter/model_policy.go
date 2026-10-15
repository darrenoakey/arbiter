package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

const stillImageDisabledMessage = "still-image generation is actively disabled in Arbiter; callers must use the Mac mini Codex image service"

// referenceImageEditModel is sanctioned exception #1 to the still-image
// policy (owner decision, 2026-09-16). It is a reference-conditioned local
// editor (FLUX.2-klein-9B) used to produce *reference renders* for other
// pipelines — e.g. a "what would a pro DSLR shot of this photo look like"
// target that a per-segment grading loop then steers toward. It is NOT an
// image generator for end users: it always requires an input image, it is
// never wired into generate_image / daz-agent-sdk / the Mac mini Codex IGS
// route, and it must never be used as a fallback when IGS is unavailable.
// The legacy image-generate / image-edit job types and every other still-image
// marker stay disabled.
const referenceImageEditModel = "reference-image-edit"

// qwenImage21Model is sanctioned exception #2 to the still-image policy
// (owner decision, 2026-09-21): a local unified text-to-image +
// reference-image editing adapter (Qwen/Qwen-Image-2.1, venvs/qwenimage)
// exposed only through the `qwen-image` job type. Unlike the reference-only
// FLUX editor it legitimately generates images from text. It is not wired
// into generate_image / daz-agent-sdk / the Mac mini Codex IGS route, and
// every other qwen-image* id (LoRA variants, renamed copies) stays disabled.
const qwenImage21Model = "qwen-image-2.1"
const qwenImageJobType = "qwen-image"

// qwenImage21HereticModel is sanctioned exception #3 to the still-image
// policy (owner decision, 2026-09-22): the identical Qwen-Image-2.1 pipeline
// (same DiT/VAE, same venvs/qwenimage interpreter) with the stock Qwen3-VL
// text encoder swapped for the community abliterated ("heretic") text
// encoder, exposed only through the `qwen-image-heretic` job type. Every
// other qwen-image* id (LoRA variants, renamed copies) stays disabled.
const qwenImage21HereticModel = "qwen-image-2.1-heretic"
const qwenImageHereticJobType = "qwen-image-heretic"

// qwenImage21SglangModel is sanctioned exception #4 (owner decision,
// 2026-09-29): the same Qwen-Image-2.1 checkpoint served by SGLang-Diffusion
// (venvs/qwenimage-sglang) through the `qwen-image-sglang` job type only.
const qwenImage21SglangModel = "qwen-image-2.1-sglang"
const qwenImageSglangJobType = "qwen-image-sglang"

// qwenImage21TurboModel is sanctioned exception #5 (owner decision,
// 2026-10-10): the official Qwen-Image-2.1-Turbo 8-step checkpoint
// (venvs/qwenimage-turbo) through the `qwen-image-turbo` job type only.
const qwenImage21TurboModel = "qwen-image-2.1-turbo"
const qwenImageTurboJobType = "qwen-image-turbo"

// iris3BModel and mingImageDesignModel are owner-requested comparison
// adapters (2026-10-11). Exact ids only; not IGS fallbacks.
const iris3BModel = "iris-3b"
const iris3BJobType = "iris-3b"
const mingImageDesignModel = "ming-image-0.1-design"
const mingImageDesignJobType = "ming-image-design"

// flux3KleinModels are the owner-directed FLUX 3 Image Klein benchmark
// adapters (2026-10-16): the distilled DiT at three exact precisions, each
// served only through its identically named job type from venvs/flux3. They
// are benchmark adapters, never wired into generate_image / the SDK / IGS and
// never a fallback. Every other flux* id (flux3, flux3-klein, base, LoRA,
// renamed copies) stays denied by the generic markers.
const flux3KleinBF16Model = "flux3-klein-bf16"
const flux3KleinFP8Model = "flux3-klein-fp8"
const flux3KleinNVFP4Model = "flux3-klein-nvfp4"

var flux3KleinModels = map[string]bool{
	flux3KleinBF16Model:  true,
	flux3KleinFP8Model:   true,
	flux3KleinNVFP4Model: true,
}

// flux3WeightsRootParam is the only extra adapter param the flux3 workers
// accept: the absolute snapshot root holding flux-3-image-klein/.
const flux3WeightsRootParam = "FLUX3_WEIGHTS_ROOT"

func isReferenceImageEditModel(modelID string) bool {
	return normalizedPolicyText(modelID) == referenceImageEditModel
}

func isQwenImage21Model(modelID string) bool {
	return normalizedPolicyText(modelID) == normalizedPolicyText(qwenImage21Model)
}

func isQwenImage21HereticModel(modelID string) bool {
	return normalizedPolicyText(modelID) == normalizedPolicyText(qwenImage21HereticModel)
}

func isQwenImage21SglangModel(modelID string) bool {
	return normalizedPolicyText(modelID) == normalizedPolicyText(qwenImage21SglangModel)
}

func isQwenImage21TurboModel(modelID string) bool {
	return normalizedPolicyText(modelID) == normalizedPolicyText(qwenImage21TurboModel)
}

func isIris3BModel(modelID string) bool {
	return normalizedPolicyText(modelID) == normalizedPolicyText(iris3BModel)
}

func isMingImageDesignModel(modelID string) bool {
	return normalizedPolicyText(modelID) == normalizedPolicyText(mingImageDesignModel)
}

func isFlux3KleinModel(modelID string) bool {
	return flux3KleinModels[normalizedPolicyText(modelID)]
}

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
	"vocal-stem":              "demucs",
	"echomimic":               "",
	"embed-text":              "embed",
	"face-restore":            "",
	"face-restore-codeformer": "",
	"insightface":             "insightface",
	"latentsync":              "",
	"lora-train":              "",
	"fine-tune":               "",
	"ltx2":                    "",
	"ltx2-denoise1":           "",
	"ltx2-denoise2":           "",
	"ltx2-dev-denoise1":       "",
	"ltx2-dev-denoise2":       "",
	"ltx2-encode":             "",
	"ltx25-denoise1":          "ltx25",
	"ltx25-encode":            "ltx25",
	"minimax-h3-local":        "minimax-h3",
	"minimax-fast-h3":         "minimax-h3",
	"moondream":               "moondream",
	"mini-agi":                "",
	"rvc-convert":             "rvc",
	"routing-decide":          "routing-decide",
	"nimble-scorer":           "nimble-scorer",
	"clef-scorer":             "clef-scorer",
	"clef-flash-scorer":       "clef-scorer",
	"rvc-train":               "rvc",
	"voice-fit":               "voxsmith",
	"music-generate":          "music-generate",
	"yue2":                    "yue2",
	"photo-enhance":           "photo-enhance",
	"sadtalker":               "",
	"sonic":                   "",
	"tts-clone":               "qwentts",
	"tts-custom":              "qwentts",
	"tts-design":              "qwentts",
	"tts-kokoro":              "kokoro",
	"tts-breeze":              "breeze",
	"wan-s2v":                 "",
	"whisper-large":           "whisper",
	referenceImageEditModel:   "flux2",
	qwenImage21Model:          "qwenimage",
	qwenImage21HereticModel:   "qwenimage",
	qwenImage21SglangModel:    "qwenimage-sglang",
	qwenImage21TurboModel:     "qwenimage-turbo",
	iris3BModel:               "iris3b",
	mingImageDesignModel:      "mingimage",
	flux3KleinBF16Model:       "flux3",
	flux3KleinFP8Model:        "flux3",
	flux3KleinNVFP4Model:      "flux3",
	"trellis2":                "trellis2",
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
	if normalized == referenceImageEditModel {
		return false
	}
	if isQwenImage21Model(normalized) {
		return false
	}
	if isQwenImage21HereticModel(normalized) {
		return false
	}
	if isQwenImage21SglangModel(normalized) {
		return false
	}
	if isQwenImage21TurboModel(normalized) {
		return false
	}
	if isIris3BModel(normalized) {
		return false
	}
	if isMingImageDesignModel(normalized) {
		return false
	}
	if isFlux3KleinModel(normalized) {
		return false
	}
	if normalized == "lora-train" || normalized == "fine-tune" || normalized == "ltx2" || strings.HasPrefix(normalized, "ltx2-") ||
		normalized == "ltx25" || strings.HasPrefix(normalized, "ltx25-") {
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
		"talking-head", "talking-head-sadtalker", "lipsync", "video-generate-h3", "video-generate-fast-h3", "video-encode",
		"video-denoise1", "video-denoise2", "face-restore", "face-restore-codeformer",
		"face-embed", "aesthetic-score", "composite", "demucs", "chat-completion",
		"chat-completion-stream", "music-generate":
		return true
	default:
		return false
	}
}

func disabledStillImageConfig(modelID string, cfg ModelConfig) bool {
	if isDisabledStillImageModel(modelID) {
		return true
	}
	if isReferenceImageEditModel(modelID) {
		// The sanctioned reference editor legitimately names a FLUX checkpoint
		// and the venvs/flux2 interpreter; the worker command itself is still
		// pinned by validateWorkerCommand to the trusted adapter identity.
		return false
	}
	if isQwenImage21Model(modelID) || isQwenImage21HereticModel(modelID) || isQwenImage21SglangModel(modelID) || isQwenImage21TurboModel(modelID) || isIris3BModel(modelID) || isMingImageDesignModel(modelID) || isFlux3KleinModel(modelID) {
		// The sanctioned Qwen adapters legitimately name the Qwen checkpoint
		// (or its local merged heretic / turbo dir) and their pinned interpreter.
		return false
	}
	videoLora := normalizedPolicyText(modelID) == "ltx2" || strings.HasPrefix(normalizedPolicyText(modelID), "ltx2-") ||
		normalizedPolicyText(modelID) == "ltx25" || strings.HasPrefix(normalizedPolicyText(modelID), "ltx25-")
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

func cloudVideoRendererDenied(modelID string, cfg ModelConfig) error {
	if modelID == "minimax-h3" {
		return fmt.Errorf("cloud MiniMax-H3 is removed; cloud video renderers must not be attempted; use minimax-h3-local")
	}
	needle := "minimax." + "io"
	parts := []string{modelID, cfg.AutoDownload, cfg.ModelPath}
	parts = append(parts, cfg.WorkerCmd...)
	parts = append(parts, cfg.Placements...)
	for key, value := range cfg.AdapterParams {
		parts = append(parts, key, value)
	}
	if strings.Contains(strings.ToLower(strings.Join(parts, "\n")), needle) {
		return fmt.Errorf("cloud video renderers must not be attempted")
	}
	return nil
}

func validateModelWorkerPolicy(projectRoot, modelID string, cfg ModelConfig, requiresLocal bool) error {
	if err := cloudVideoRendererDenied(modelID, cfg); err != nil {
		return err
	}
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
	if isReferenceImageEditModel(modelID) && jobType != "reference-image-edit" {
		// The reference editor cannot be smuggled in under another job type.
		return fmt.Errorf("%s", stillImageDisabledMessage)
	}
	if isQwenImage21Model(modelID) && jobType != qwenImageJobType {
		// Same narrowing for the Qwen unified adapter.
		return fmt.Errorf("%s", stillImageDisabledMessage)
	}
	if isQwenImage21HereticModel(modelID) && jobType != qwenImageHereticJobType {
		// The heretic variant is narrowed to its own job type, and the stock
		// adapter above is already barred from claiming it.
		return fmt.Errorf("%s", stillImageDisabledMessage)
	}
	if isQwenImage21SglangModel(modelID) && jobType != qwenImageSglangJobType {
		return fmt.Errorf("%s", stillImageDisabledMessage)
	}
	if isQwenImage21TurboModel(modelID) && jobType != qwenImageTurboJobType {
		return fmt.Errorf("%s", stillImageDisabledMessage)
	}
	if isIris3BModel(modelID) && jobType != iris3BJobType {
		return fmt.Errorf("%s", stillImageDisabledMessage)
	}
	if isMingImageDesignModel(modelID) && jobType != mingImageDesignJobType {
		return fmt.Errorf("%s", stillImageDisabledMessage)
	}
	if isFlux3KleinModel(modelID) && jobType != normalizedPolicyText(modelID) {
		// Each precision is reachable only through its own job type, so a
		// benchmark row can never be served by another variant.
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
		compatible = strings.HasPrefix(normalized, "ltx2") &&
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
