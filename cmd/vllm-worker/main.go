// vLLM worker for Arbiter — wraps `vllm serve --omni` as a subprocess.
//
// Speaks the Arbiter adapter protocol on stdin/stdout:
//   {"cmd": "load", "device": "cuda"}       → start vllm serve --omni
//   {"cmd": "infer", "req_id": "x", ...}    → proxy to vllm API
//   {"cmd": "unload"}                        → stop vllm serve
//   {"cmd": "shutdown"}                      → exit
//
// Environment:
//   VLLM_MODEL       — HuggingFace model ID (e.g., "mistralai/Voxtral-4B-TTS-2603")
//   VLLM_MODE        — endpoint mode: "tts" or "chat" (default: "tts")
//   VLLM_EXTRA_ARGS  — additional args for vllm serve (space-separated)
//   VLLM_PYTHON      — python binary to use (default: python from arbiter .venv)
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

type Request struct {
	Cmd       string          `json:"cmd"`
	ReqID     string          `json:"req_id,omitempty"`
	Device    string          `json:"device,omitempty"`
	Params    json.RawMessage `json:"params,omitempty"`
	JobType   string          `json:"job_type,omitempty"`
	OutputDir string          `json:"output_dir,omitempty"`
}

type Response struct {
	Status string          `json:"status"`
	ReqID  string          `json:"req_id,omitempty"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  string          `json:"error,omitempty"`
}

var (
	vllmCmd    *exec.Cmd
	vllmPort   int
	cancelFlag bool
	mode       string
)

func respond(r Response) {
	data, _ := json.Marshal(r)
	fmt.Println(string(data))
}

func findFreePort() int {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 18090
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port
}

func envGet(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func venvRoot() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "src", "arbiter", ".venv")
}

func findVllmBin() string {
	venvBin := filepath.Join(venvRoot(), "bin", "vllm")
	if _, err := os.Stat(venvBin); err == nil {
		return venvBin
	}
	return "vllm"
}

func startVLLM() error {
	model := os.Getenv("VLLM_MODEL")
	if model == "" {
		return fmt.Errorf("VLLM_MODEL not set")
	}

	vllmPort = findFreePort()
	vllmBin := findVllmBin()

	args := []string{
		"serve", model,
		"--omni",
		"--port", fmt.Sprintf("%d", vllmPort),
		"--host", "127.0.0.1",
	}

	if extra := os.Getenv("VLLM_EXTRA_ARGS"); extra != "" {
		args = append(args, strings.Fields(extra)...)
	}

	log.Printf("Starting vllm serve --omni on port %d: %s %s", vllmPort, vllmBin, strings.Join(args, " "))

	vllmCmd = exec.Command(vllmBin, args...)
	vllmCmd.Stderr = os.Stderr
	vllmCmd.Stdout = os.Stderr

	// Build env: inherit, strip CLAUDECODE, add CUDA paths
	filtered := make([]string, 0, len(os.Environ())+4)
	hasCudaPath := false
	hasTorchArch := false
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "CLAUDECODE=") {
			continue
		}
		if strings.HasPrefix(e, "PATH=") {
			// Ensure /usr/local/cuda/bin is in PATH for nvcc
			if !strings.Contains(e, "/usr/local/cuda/bin") {
				e = "PATH=/usr/local/cuda/bin:" + e[5:]
			}
			hasCudaPath = true
		}
		if strings.HasPrefix(e, "TORCH_CUDA_ARCH_LIST=") {
			hasTorchArch = true
		}
		filtered = append(filtered, e)
	}
	if !hasCudaPath {
		filtered = append(filtered, "PATH=/usr/local/cuda/bin:"+os.Getenv("PATH"))
	}
	if !hasTorchArch {
		filtered = append(filtered, "TORCH_CUDA_ARCH_LIST=12.0")
	}
	filtered = append(filtered, "CUDA_HOME=/usr/local/cuda")
	vllmCmd.Env = filtered

	if err := vllmCmd.Start(); err != nil {
		return fmt.Errorf("failed to start vllm: %w", err)
	}

	// Wait for vllm to be ready (health check) — omni models take longer
	deadline := time.Now().Add(15 * time.Minute)
	url := fmt.Sprintf("http://127.0.0.1:%d/health", vllmPort)
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				log.Printf("vllm ready on port %d", vllmPort)
				return nil
			}
		}
		if vllmCmd.ProcessState != nil {
			return fmt.Errorf("vllm exited prematurely")
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("vllm did not become ready within 15 minutes")
}

func stopVLLM() {
	if vllmCmd != nil && vllmCmd.Process != nil {
		vllmCmd.Process.Signal(syscall.SIGTERM)
		done := make(chan error, 1)
		go func() { done <- vllmCmd.Wait() }()
		select {
		case <-done:
		case <-time.After(15 * time.Second):
			vllmCmd.Process.Kill()
		}
		vllmCmd = nil
		log.Printf("vllm stopped")
	}
}

func proxyTTS(reqID string, params json.RawMessage, outputDir string) Response {
	var p struct {
		Text        string  `json:"text"`
		Voice       string  `json:"voice"`
		Language    string  `json:"language"`
		Temperature float64 `json:"temperature"`
		Speed       float64 `json:"speed"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return Response{Status: "error", ReqID: reqID, Error: fmt.Sprintf("invalid params: %s", err)}
	}
	if p.Text == "" {
		return Response{Status: "error", ReqID: reqID, Error: "missing required param: text"}
	}
	if p.Voice == "" {
		p.Voice = "casual_male"
	}

	model := os.Getenv("VLLM_MODEL")

	speechReq := map[string]any{
		"input":           p.Text,
		"model":           model,
		"voice":           p.Voice,
		"response_format": "wav",
	}
	if p.Speed > 0 {
		speechReq["speed"] = p.Speed
	}

	body, _ := json.Marshal(speechReq)
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/audio/speech", vllmPort)

	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Post(url, "application/json", strings.NewReader(string(body)))
	if err != nil {
		return Response{Status: "error", ReqID: reqID, Error: fmt.Sprintf("vllm request failed: %s", err)}
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		errBody, _ := io.ReadAll(resp.Body)
		return Response{Status: "error", ReqID: reqID, Error: fmt.Sprintf("vllm %d: %s", resp.StatusCode, string(errBody))}
	}

	outPath := filepath.Join(outputDir, "result.wav")
	f, err := os.Create(outPath)
	if err != nil {
		return Response{Status: "error", ReqID: reqID, Error: fmt.Sprintf("create output: %s", err)}
	}
	n, err := io.Copy(f, resp.Body)
	f.Close()
	if err != nil {
		return Response{Status: "error", ReqID: reqID, Error: fmt.Sprintf("write output: %s", err)}
	}

	log.Printf("TTS output: %s (%d bytes)", outPath, n)

	result, _ := json.Marshal(map[string]any{
		"format":      "wav",
		"sample_rate": 24000,
	})
	return Response{Status: "ok", ReqID: reqID, Result: result}
}

func proxyChat(reqID string, params json.RawMessage) Response {
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", vllmPort)

	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Post(url, "application/json", strings.NewReader(string(params)))
	if err != nil {
		return Response{Status: "error", ReqID: reqID, Error: fmt.Sprintf("proxy error: %s", err)}
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return Response{Status: "error", ReqID: reqID, Error: fmt.Sprintf("read error: %s", err)}
	}

	if resp.StatusCode != 200 {
		return Response{Status: "error", ReqID: reqID, Error: fmt.Sprintf("vllm %d: %s", resp.StatusCode, string(body))}
	}

	var chatResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
				Role    string `json:"role"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	json.Unmarshal(body, &chatResp)

	resultMap := map[string]any{
		"format":   "json",
		"response": json.RawMessage(body),
	}
	if len(chatResp.Choices) > 0 {
		resultMap["text"] = chatResp.Choices[0].Message.Content
		resultMap["finish_reason"] = chatResp.Choices[0].FinishReason
	}
	resultMap["usage"] = chatResp.Usage

	resultJSON, _ := json.Marshal(resultMap)
	return Response{Status: "ok", ReqID: reqID, Result: resultJSON}
}

func main() {
	log.SetOutput(os.Stderr)
	log.SetPrefix("[vllm-worker] ")

	mode = envGet("VLLM_MODE", "tts")

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGUSR1)
	go func() {
		for range sigCh {
			cancelFlag = true
			log.Println("Cancel signal received")
		}
	}()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 10*1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var req Request
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			respond(Response{Status: "error", Error: "invalid JSON"})
			continue
		}

		switch req.Cmd {
		case "load":
			if err := startVLLM(); err != nil {
				respond(Response{Status: "error", Error: err.Error()})
			} else {
				respond(Response{Status: "ok"})
			}

		case "infer":
			cancelFlag = false
			var resp Response
			inferMode := mode
			if strings.Contains(req.JobType, "tts") {
				inferMode = "tts"
			} else if strings.Contains(req.JobType, "chat") {
				inferMode = "chat"
			}

			switch inferMode {
			case "tts":
				resp = proxyTTS(req.ReqID, req.Params, req.OutputDir)
			case "chat":
				resp = proxyChat(req.ReqID, req.Params)
			default:
				resp = Response{Status: "error", ReqID: req.ReqID, Error: fmt.Sprintf("unknown mode: %s", inferMode)}
			}

			if cancelFlag {
				resp = Response{Status: "cancelled", ReqID: req.ReqID}
			}
			respond(resp)

		case "unload":
			stopVLLM()
			respond(Response{Status: "ok"})

		case "shutdown":
			stopVLLM()
			respond(Response{Status: "ok"})
			return

		case "ping":
			respond(Response{Status: "ok"})

		default:
			respond(Response{Status: "error", Error: fmt.Sprintf("unknown command: %s", req.Cmd)})
		}
	}

	stopVLLM()
}
