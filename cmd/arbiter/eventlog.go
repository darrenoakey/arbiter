package main

import (
	"encoding/json"
	"fmt"
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
	os.MkdirAll(dir, 0o755)
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
			l.file.Close()
		}
		path := filepath.Join(l.dir, fmt.Sprintf("arbiter-%s.jsonl", today))
		l.file, _ = os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		l.currentDate = today
	}

	if l.file != nil {
		l.file.Write(data)
		l.file.Write([]byte("\n"))
	}
}

func (l *EventLogger) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.file != nil {
		l.file.Close()
		l.file = nil
	}
}
