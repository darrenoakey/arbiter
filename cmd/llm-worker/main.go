// LLM worker for Arbiter — wraps llama-server as a subprocess.
//
// Speaks the Arbiter adapter protocol on stdin/stdout:
//   {"cmd": "load", "device": "cuda"}       → start llama-server
//   {"cmd": "infer", "req_id": "x", ...}    → proxy chat completion
//   {"cmd": "unload"}                        → stop llama-server
//   {"cmd": "shutdown"}                      → exit
//
// Environment:
//   LLM_HF_REPO     — HuggingFace repo for GGUF model (e.g., "unsloth/gpt-oss-20b-GGUF")
//   LLM_HF_FILE     — specific GGUF file in the repo (e.g., "gpt-oss-20b-Q8_0.gguf")
//   LLM_MODEL_PATH  — local path to GGUF file (alternative to HF download)
//   LLM_GPU_LAYERS  — number of GPU layers (-1 = all, default)
//   LLM_CTX_SIZE    — context size (default: 8192)
//   LLAMA_SERVER_BIN — path to llama-server binary
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
	"strconv"
	"strings"
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
	llamaCmd    *exec.Cmd
	llamaPort   int
	cancelFlag  bool
)

func respond(r Response) {
	data, _ := json.Marshal(r)
	fmt.Println(string(data))
}

func findFreePort() int {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 18080
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func startLlamaServer() error {
	bin := env("LLAMA_SERVER_BIN", "llama-server")

	// Find the binary
	if _, err := exec.LookPath(bin); err != nil {
		// Try common locations
		for _, path := range []string{
			os.ExpandEnv("$HOME/src/llama.cpp/build/bin/llama-server"),
			"/usr/local/bin/llama-server",
		} {
			if _, err := os.Stat(path); err == nil {
				bin = path
				break
			}
		}
	}

	llamaPort = findFreePort()

	args := []string{
		"--port", strconv.Itoa(llamaPort),
		"--host", "127.0.0.1",
		"-ngl", env("LLM_GPU_LAYERS", "-1"), // all layers on GPU by default
		"-c", env("LLM_CTX_SIZE", "8192"),
		"--no-warmup",
	}

	// Model source: HF repo or local path
	if modelPath := os.Getenv("LLM_MODEL_PATH"); modelPath != "" {
		args = append(args, "-m", modelPath)
	} else if hfRepo := os.Getenv("LLM_HF_REPO"); hfRepo != "" {
		args = append(args, "--hf-repo", hfRepo)
		if hfFile := os.Getenv("LLM_HF_FILE"); hfFile != "" {
			args = append(args, "--hf-file", hfFile)
		}
	} else {
		return fmt.Errorf("no model specified: set LLM_HF_REPO or LLM_MODEL_PATH")
	}

	log.Printf("Starting llama-server on port %d: %s %s", llamaPort, bin, strings.Join(args, " "))

	llamaCmd = exec.Command(bin, args...)
	llamaCmd.Stderr = os.Stderr // llama-server logs go to our stderr → arbiter captures them
	llamaCmd.Stdout = os.Stderr // don't mix with our protocol stdout

	if err := llamaCmd.Start(); err != nil {
		return fmt.Errorf("failed to start llama-server: %w", err)
	}

	// Wait for llama-server to be ready (health check)
	deadline := time.Now().Add(10 * time.Minute) // model download + load can take a while
	url := fmt.Sprintf("http://127.0.0.1:%d/health", llamaPort)
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				log.Printf("llama-server ready on port %d", llamaPort)
				return nil
			}
		}
		// Check if process died
		if llamaCmd.ProcessState != nil {
			return fmt.Errorf("llama-server exited prematurely")
		}
		time.Sleep(time.Second)
	}
	return fmt.Errorf("llama-server did not become ready within 10 minutes")
}

func stopLlamaServer() {
	if llamaCmd != nil && llamaCmd.Process != nil {
		llamaCmd.Process.Signal(syscall.SIGTERM)
		done := make(chan error, 1)
		go func() { done <- llamaCmd.Wait() }()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			llamaCmd.Process.Kill()
		}
		llamaCmd = nil
		log.Printf("llama-server stopped")
	}
}

func proxyChat(reqID string, params json.RawMessage) Response {
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", llamaPort)

	// params should contain messages, temperature, max_tokens, etc.
	// Pass through as-is to llama-server's OpenAI-compatible endpoint
	resp, err := http.Post(url, "application/json", bytes.NewReader(params))
	if err != nil {
		return Response{Status: "error", ReqID: reqID, Error: fmt.Sprintf("proxy error: %s", err)}
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return Response{Status: "error", ReqID: reqID, Error: fmt.Sprintf("read error: %s", err)}
	}

	if resp.StatusCode != 200 {
		return Response{Status: "error", ReqID: reqID, Error: fmt.Sprintf("llama-server %d: %s", resp.StatusCode, string(body))}
	}

	// Parse the OpenAI response to extract the text for the result
	var chatResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
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

	// Build result with both the full OpenAI response and extracted fields
	result := map[string]any{
		"format":   "json",
		"response": json.RawMessage(body), // full OpenAI response
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
	log.SetPrefix("[llm-worker] ")

	// Handle cancel signal (SIGUSR1 from arbiter)
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
			if err := startLlamaServer(); err != nil {
				respond(Response{Status: "error", Error: err.Error()})
			} else {
				respond(Response{Status: "ok"})
			}

		case "infer":
			cancelFlag = false
			resp := proxyChat(req.ReqID, req.Params)
			if cancelFlag {
				resp = Response{Status: "cancelled", ReqID: req.ReqID}
			}
			respond(resp)

		case "unload":
			stopLlamaServer()
			respond(Response{Status: "ok"})

		case "shutdown":
			stopLlamaServer()
			respond(Response{Status: "ok"})
			return

		case "ping":
			respond(Response{Status: "ok"})

		default:
			respond(Response{Status: "error", Error: fmt.Sprintf("unknown command: %s", req.Cmd)})
		}
	}

	stopLlamaServer()
}
