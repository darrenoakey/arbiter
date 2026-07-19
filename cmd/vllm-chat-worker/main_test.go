package main

import (
	"reflect"
	"testing"
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
