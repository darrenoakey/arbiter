package main

import (
	"encoding/json"
	"testing"
)

func unmarshal(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return m
}

func TestPrepareParams_InjectsDefaultMaxTokensWhenMissing(t *testing.T) {
	in := json.RawMessage(`{"model":"x","messages":[{"role":"user","content":"hi"}]}`)
	out := prepareParams(in)
	m := unmarshal(t, out)
	got, ok := m["max_tokens"]
	if !ok {
		t.Fatalf("expected max_tokens to be injected, got %v", m)
	}
	if got != float64(defaultMaxTokensCap) {
		t.Fatalf("expected max_tokens=%d, got %v", defaultMaxTokensCap, got)
	}
}

func TestPrepareParams_RespectsExistingMaxTokens(t *testing.T) {
	in := json.RawMessage(`{"model":"x","max_tokens":50,"messages":[]}`)
	out := prepareParams(in)
	m := unmarshal(t, out)
	if m["max_tokens"].(float64) != 50 {
		t.Fatalf("client max_tokens overwritten: %v", m["max_tokens"])
	}
}

func TestPrepareParams_RespectsExistingNPredict(t *testing.T) {
	in := json.RawMessage(`{"model":"x","n_predict":75,"messages":[]}`)
	out := prepareParams(in)
	m := unmarshal(t, out)
	if _, hasMax := m["max_tokens"]; hasMax {
		t.Fatalf("max_tokens should NOT be added when n_predict is set, got %v", m)
	}
	if m["n_predict"].(float64) != 75 {
		t.Fatalf("n_predict overwritten: %v", m["n_predict"])
	}
}

func TestPrepareParams_StripsStream(t *testing.T) {
	in := json.RawMessage(`{"model":"x","stream":true,"stream_options":{"include_usage":true},"messages":[]}`)
	out := prepareParams(in)
	m := unmarshal(t, out)
	if _, ok := m["stream"]; ok {
		t.Fatalf("stream should be stripped: %v", m)
	}
	if _, ok := m["stream_options"]; ok {
		t.Fatalf("stream_options should be stripped: %v", m)
	}
}

func TestPrepareParams_PassesThroughOnInvalidJSON(t *testing.T) {
	in := json.RawMessage(`not json`)
	out := prepareParams(in)
	if string(out) != "not json" {
		t.Fatalf("expected pass-through on invalid input, got %s", out)
	}
}
