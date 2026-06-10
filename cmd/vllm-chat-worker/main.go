// vLLM chat worker for Arbiter — wraps plain `vllm serve` as a subprocess.
//
// This is a separate worker from `vllm-worker` (which uses vllm_omni with
// `--omni` for TTS/multimodal). This worker uses stock vLLM from an isolated
// venv (`~/src/arbiter/.venv-vllm`) so the two installs cannot conflict.
//
// Speaks the Arbiter adapter protocol on stdin/stdout:
//
//	{"cmd": "load", "device": "cuda"}       → start vllm serve
//	{"cmd": "infer", "req_id": "x", ...}    → proxy chat completion
//	{"cmd": "get_port"}                      → return vllm server port
//	{"cmd": "unload"}                        → stop vllm serve
//	{"cmd": "shutdown"}                      → exit
//
// Environment:
//
//	VLLM_MODEL       — HuggingFace model ID or GGUF repo:file spec
//	                   (e.g., "Qwen/Qwen3.6-35B-A3B" or
//	                    "unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-Q4_K_S.gguf")
//	VLLM_EXTRA_ARGS  — additional args for vllm serve (space-separated)
//	VLLM_BIN         — path to vllm CLI (default: ~/src/arbiter/.venv-vllm/bin/vllm)
//	VLLM_READY_TIMEOUT_SEC — how long to wait for health (default 900)
package main

import (
	"bufio"
	"bytes"
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
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

type Request struct {
	Cmd     string          `json:"cmd"`
	ReqID   string          `json:"req_id,omitempty"`
	Device  string          `json:"device,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	JobType string          `json:"job_type,omitempty"`
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
	stdoutMu   sync.Mutex
)

// respond serialises one Response line on stdout. Multiple in-flight infer
// goroutines call this concurrently; the mutex prevents JSON line interleaving
// from corrupting the arbiter's stdout parser.
func respond(r Response) {
	data, _ := json.Marshal(r)
	stdoutMu.Lock()
	defer stdoutMu.Unlock()
	fmt.Println(string(data))
}

func findFreePort() int {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 18100
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port
}

func findVllmBin() string {
	if v := os.Getenv("VLLM_BIN"); v != "" {
		return v
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "src", "arbiter", ".venv-vllm", "bin", "vllm")
}

func startVLLM() error {
	model := os.Getenv("VLLM_MODEL")
	if model == "" {
		return fmt.Errorf("VLLM_MODEL not set")
	}

	vllmPort = findFreePort()
	vllmBin := findVllmBin()
	if _, err := os.Stat(vllmBin); err != nil {
		return fmt.Errorf("vllm binary not found at %s: %w", vllmBin, err)
	}

	args := []string{
		"serve", model,
		"--port", strconv.Itoa(vllmPort),
		"--host", "127.0.0.1",
	}
	if extra := os.Getenv("VLLM_EXTRA_ARGS"); extra != "" {
		args = append(args, strings.Fields(extra)...)
	}

	log.Printf("Starting vllm serve on port %d: %s %s", vllmPort, vllmBin, strings.Join(args, " "))

	vllmCmd = exec.Command(vllmBin, args...)
	vllmCmd.Stderr = os.Stderr
	vllmCmd.Stdout = os.Stderr

	// Build env: inherit, strip CLAUDECODE, ensure CUDA paths and the vllm
	// venv's bin (so JIT-spawned tools like ninja are on PATH).
	venvBin := filepath.Dir(vllmBin) // e.g. ~/src/arbiter/.venv-vllm/bin
	pathPrefix := venvBin + ":/usr/local/cuda/bin"
	filtered := make([]string, 0, len(os.Environ())+4)
	hasCudaPath := false
	hasTorchArch := false
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "CLAUDECODE=") {
			continue
		}
		if strings.HasPrefix(e, "PATH=") {
			e = "PATH=" + pathPrefix + ":" + e[5:]
			hasCudaPath = true
		}
		if strings.HasPrefix(e, "TORCH_CUDA_ARCH_LIST=") {
			hasTorchArch = true
		}
		filtered = append(filtered, e)
	}
	if !hasCudaPath {
		filtered = append(filtered, "PATH="+pathPrefix+":"+os.Getenv("PATH"))
	}
	if !hasTorchArch {
		filtered = append(filtered, "TORCH_CUDA_ARCH_LIST=12.0")
	}
	filtered = append(filtered, "CUDA_HOME=/usr/local/cuda")
	vllmCmd.Env = filtered

	if err := vllmCmd.Start(); err != nil {
		return fmt.Errorf("failed to start vllm: %w", err)
	}

	readySec := 900
	if v := os.Getenv("VLLM_READY_TIMEOUT_SEC"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			readySec = n
		}
	}
	deadline := time.Now().Add(time.Duration(readySec) * time.Second)
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
	return fmt.Errorf("vllm did not become ready within %d seconds", readySec)
}

func stopVLLM() {
	if vllmCmd != nil && vllmCmd.Process != nil {
		vllmCmd.Process.Signal(syscall.SIGTERM)
		done := make(chan error, 1)
		go func() { done <- vllmCmd.Wait() }()
		select {
		case <-done:
		case <-time.After(30 * time.Second):
			vllmCmd.Process.Kill()
		}
		vllmCmd = nil
		log.Printf("vllm stopped")
	}
}

// stripStreamAndRewriteModel cleans stream flags AND overwrites params.model
// with the canonical VLLM_MODEL env value. vLLM is strict about the model
// field — it must match the path/HF id passed to `vllm serve`. Callers (the
// arbiter dispatcher) routinely pass friendly names like "gemma4-26b", which
// vllm rejects with 404. llama-server is lenient and ignores model field, so
// the same bug doesn't surface there.
func stripStreamAndRewriteModel(params json.RawMessage) json.RawMessage {
	var m map[string]any
	if err := json.Unmarshal(params, &m); err != nil {
		return params
	}
	delete(m, "stream")
	delete(m, "stream_options")
	if canonical := os.Getenv("VLLM_MODEL"); canonical != "" {
		m["model"] = canonical
	}
	out, _ := json.Marshal(m)
	return out
}

func proxyChat(reqID string, params json.RawMessage) Response {
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", vllmPort)
	cleanParams := stripStreamAndRewriteModel(params)

	client := &http.Client{Timeout: 30 * time.Minute}
	resp, err := client.Post(url, "application/json", bytes.NewReader(cleanParams))
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
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
				Role             string `json:"role"`
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

	result := map[string]any{
		"format":   "json",
		"response": json.RawMessage(body),
	}
	if len(chatResp.Choices) > 0 {
		text := chatResp.Choices[0].Message.Content
		if text == "" && chatResp.Choices[0].Message.ReasoningContent != "" {
			text = chatResp.Choices[0].Message.ReasoningContent
			result["reasoning"] = true
		}
		result["text"] = text
		result["finish_reason"] = chatResp.Choices[0].FinishReason
	}
	result["usage"] = chatResp.Usage

	resultJSON, _ := json.Marshal(result)
	return Response{Status: "ok", ReqID: reqID, Result: resultJSON}
}

func main() {
	log.SetOutput(os.Stderr)
	log.SetPrefix("[vllm-chat-worker] ")

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
			// Run inference in a goroutine so the main loop can keep accepting
			// new infer commands. vLLM batches concurrent HTTP requests
			// internally; the worker just needs to not serialise them at the
			// stdin protocol layer.
			cancelFlag = false
			reqID := req.ReqID
			params := req.Params
			go func() {
				resp := proxyChat(reqID, params)
				respond(resp)
			}()
			continue

		case "get_port":
			portResult, _ := json.Marshal(map[string]any{"port": vllmPort})
			respond(Response{Status: "ok", Result: portResult})

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
