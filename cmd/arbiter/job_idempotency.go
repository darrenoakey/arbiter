package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"
)

const maxIdempotencyKeyBytes = 256

type submitJobRequest struct {
	Type           string          `json:"type"`
	Model          string          `json:"model"`
	Params         json.RawMessage `json:"params"`
	IdempotencyKey json.RawMessage `json:"idempotency_key"`
}

func validateIdempotencyKey(raw json.RawMessage) (string, bool, error) {
	if len(raw) == 0 {
		return "", false, nil
	}
	var key string
	if string(raw) == "null" || json.Unmarshal(raw, &key) != nil {
		return "", true, fmt.Errorf("idempotency_key must be a string")
	}
	if !utf8.ValidString(key) || strings.TrimSpace(key) == "" {
		return "", true, fmt.Errorf("idempotency_key must be a non-empty UTF-8 string")
	}
	if len(key) > maxIdempotencyKeyBytes {
		return "", true, fmt.Errorf("idempotency_key must be at most %d bytes", maxIdempotencyKeyBytes)
	}
	return key, true, nil
}

func normalizedJobRequestHash(jobType, modelID string, params json.RawMessage) (string, error) {
	var normalizedParams any
	decoder := json.NewDecoder(bytes.NewReader(params))
	decoder.UseNumber()
	if err := decoder.Decode(&normalizedParams); err != nil {
		return "", fmt.Errorf("normalize job params: %w", err)
	}
	canonical, err := json.Marshal(struct {
		Type   string `json:"type"`
		Model  string `json:"model"`
		Params any    `json:"params"`
	}{Type: jobType, Model: modelID, Params: normalizedParams})
	if err != nil {
		return "", fmt.Errorf("marshal normalized job request: %w", err)
	}
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:]), nil
}
