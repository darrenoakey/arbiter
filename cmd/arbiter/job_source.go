package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"
)

// JobSource is caller-supplied provenance for a job: WHO submitted it and,
// when a root task is driving a chain of tools, the human-meaningful WHY all
// the way at the top (e.g. who="waggler", why="hang williams video"). It is
// deliberately NOT part of the dedup/idempotency hash or the LLM cache key —
// identical work from two callers still shares one canonical job; the source
// is recorded per submission row and surfaces in /v1/jobs, /v1/ps and the
// event log for attribution, not for identity.
type JobSource struct {
	Who string `json:"who,omitempty"`
	Why string `json:"why,omitempty"`
}

const (
	maxSourceWhoBytes = 128
	maxSourceWhyBytes = 256
)

// validateJobSource normalizes and validates a decoded job source. A nil or
// empty source is valid (jobs without provenance stay first-class). Both
// fields are trimmed; over-length or control-character values are rejected so
// logs and the dashboard can never be poisoned by a caller string.
func validateJobSource(src *JobSource) (*JobSource, error) {
	if src == nil {
		return nil, nil
	}
	who := strings.TrimSpace(src.Who)
	why := strings.TrimSpace(src.Why)
	if who == "" && why == "" {
		return nil, nil
	}
	if err := validateSourceField("source.who", who, maxSourceWhoBytes); err != nil {
		return nil, err
	}
	if err := validateSourceField("source.why", why, maxSourceWhyBytes); err != nil {
		return nil, err
	}
	return &JobSource{Who: who, Why: why}, nil
}

func validateSourceField(name, value string, maxBytes int) error {
	if value == "" {
		return nil
	}
	if !utf8.ValidString(value) {
		return fmt.Errorf("%s must be valid UTF-8", name)
	}
	if len(value) > maxBytes {
		return fmt.Errorf("%s must be at most %d bytes", name, maxBytes)
	}
	for _, r := range value {
		if r < 0x20 && r != '\t' {
			return fmt.Errorf("%s must not contain control characters", name)
		}
	}
	return nil
}

// decodeJobSource parses a raw JSON source value (accepts null, absent, or an
// object with who/why strings). Anything else is a 400.
func decodeJobSource(raw json.RawMessage) (*JobSource, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var src JobSource
	if err := json.Unmarshal(raw, &src); err != nil {
		return nil, fmt.Errorf("source must be an object with optional string fields who/why")
	}
	return validateJobSource(&src)
}

// logFields renders a job source for the event log: absent source logs no
// keys, so existing log consumers see no shape change.
func (s *JobSource) logFields() map[string]any {
	if s == nil {
		return nil
	}
	fields := map[string]any{}
	if s.Who != "" {
		fields["source_who"] = s.Who
	}
	if s.Why != "" {
		fields["source_why"] = s.Why
	}
	return fields
}

// mergeLogFields copies src's fields into base (base wins on conflict).
func mergeLogFields(base map[string]any, src *JobSource) map[string]any {
	for k, v := range src.logFields() {
		if _, exists := base[k]; !exists {
			base[k] = v
		}
	}
	return base
}

// extractChatBodySource pops the non-standard top-level "who"/"why" fields
// from an OpenAI-compatible chat body and returns the provenance plus the
// stripped body. Stripping BEFORE canonicalization keeps the LLM cache key and
// dedup hash independent of provenance (identical content from two callers
// still shares one entry) and keeps the extra fields out of the worker payload.
func extractChatBodySource(body []byte) (*JobSource, []byte, error) {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, nil, err
	}
	rawWho, hasWho := m["who"]
	rawWhy, hasWhy := m["why"]
	if !hasWho && !hasWhy {
		return nil, body, nil
	}
	src := &JobSource{}
	if hasWho {
		s, ok := rawWho.(string)
		if !ok {
			return nil, nil, fmt.Errorf("who must be a string")
		}
		src.Who = s
	}
	if hasWhy {
		s, ok := rawWhy.(string)
		if !ok {
			return nil, nil, fmt.Errorf("why must be a string")
		}
		src.Why = s
	}
	delete(m, "who")
	delete(m, "why")
	stripped, err := json.Marshal(m)
	if err != nil {
		return nil, nil, err
	}
	normed, err := validateJobSource(src)
	if err != nil {
		return nil, nil, err
	}
	return normed, stripped, nil
}
