package main

import (
	"fmt"
	"net"
	"net/http"
	"os/exec"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestAppendVllmTuningUsesStructuredArgumentBoundaries(t *testing.T) {
	t.Setenv("VLLM_MAX_MODEL_LEN", "32768")
	t.Setenv("VLLM_QUANTIZATION", "awq")
	t.Setenv("VLLM_DTYPE", "bfloat16")
	t.Setenv("VLLM_TENSOR_PARALLEL_SIZE", "2")
	t.Setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")
	t.Setenv("VLLM_MAX_NUM_SEQS", "16")
	want := []string{
		"serve", "model", "--max-model-len", "32768", "--quantization", "awq",
		"--dtype", "bfloat16", "--tensor-parallel-size", "2",
		"--gpu-memory-utilization", "0.9", "--max-num-seqs", "16",
	}
	if got := appendVllmTuning([]string{"serve", "model"}); !reflect.DeepEqual(got, want) {
		t.Fatalf("structured arguments = %#v, want %#v", got, want)
	}
}

// A child that dies before serving health must fail the load promptly instead
// of waiting out the readiness deadline (the tts-voxtral load used to hang 15
// minutes on a zombie vllm that crashed at import).
func TestWaitReadyReturnsPromptlyWhenProcessExits(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	url := fmt.Sprintf("http://%s/health", listener.Addr())
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("sh", "-c", "exit 3")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	var exitErr error
	exited := reapInBackground(cmd, &exitErr)
	start := time.Now()
	err = waitReady(url, exited, &exitErr, time.Minute, 50*time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "exited prematurely") || !strings.Contains(err.Error(), "exit status 3") {
		t.Fatalf("waitReady error = %v, want premature exit with status 3", err)
	}
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Fatalf("waitReady took %s to notice the exit", elapsed)
	}
}

func TestWaitReadySucceedsWhenHealthServes(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})}
	go func() { _ = server.Serve(listener) }()
	defer func() { _ = server.Close() }()
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	var exitErr error
	exited := reapInBackground(cmd, &exitErr)
	defer func() {
		_ = cmd.Process.Kill()
		<-exited
	}()
	url := fmt.Sprintf("http://%s/health", listener.Addr())
	if err := waitReady(url, exited, &exitErr, 10*time.Second, 50*time.Millisecond); err != nil {
		t.Fatalf("waitReady = %v, want ready", err)
	}
}
