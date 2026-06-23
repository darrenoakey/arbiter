package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSubmitJobRejectsSparkImageGeneration(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.config.Models["flux2"] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1}

	req := httptest.NewRequest(
		http.MethodPost,
		"/v1/jobs",
		bytes.NewBufferString(`{"type":"image-generate","params":{"prompt":"test"}}`),
	)
	rec := httptest.NewRecorder()
	api.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "image server/Codex") {
		t.Fatalf("body does not explain codex image-server path: %s", rec.Body.String())
	}
}

func TestSubmitJobRejectsZImageTurboModelOverride(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.config.Models["z-image-turbo"] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1}

	for name, body := range map[string]string{
		"top-level":    `{"type":"background-remove","model":"z-image-turbo","params":{"image":"x"}}`,
		"params-model": `{"type":"background-remove","params":{"model":"z-image-turbo","image":"x"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/jobs", bytes.NewBufferString(body))
			rec := httptest.NewRecorder()
			api.Handler().ServeHTTP(rec, req)

			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), "z-image-turbo is permanently disabled") {
				t.Fatalf("body does not explain z-image-turbo ban: %s", rec.Body.String())
			}
		})
	}
}
