package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// EventLogger writes structured JSONL event logs, one file per day.
type EventLogger struct {
	dir         string
	mu          sync.Mutex
	file        *os.File
	currentDate string
}

func NewEventLogger(dir string) *EventLogger {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		slog.Error("create event log directory", "dir", dir, "error", err)
	}
	return &EventLogger{dir: dir}
}

func (l *EventLogger) Log(event string, fields map[string]any) {
	entry := map[string]any{
		"ts":    float64(time.Now().UnixNano()) / 1e9,
		"event": event,
	}
	for k, v := range fields {
		entry[k] = v
	}
	data, _ := json.Marshal(entry)

	l.mu.Lock()
	defer l.mu.Unlock()

	today := time.Now().UTC().Format("2006-01-02")
	if today != l.currentDate {
		if l.file != nil {
			if err := l.file.Close(); err != nil {
				slog.Warn("close rotated event log", "error", err)
			}
		}
		path := filepath.Join(l.dir, fmt.Sprintf("arbiter-%s.jsonl", today))
		l.file, _ = os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		l.currentDate = today
	}

	if l.file != nil {
		if _, err := l.file.Write(append(data, '\n')); err != nil {
			slog.Error("write event log", "error", err)
		}
	}
}

func (l *EventLogger) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.file != nil {
		if err := l.file.Close(); err != nil {
			slog.Warn("close event log", "error", err)
		}
		l.file = nil
	}
}
