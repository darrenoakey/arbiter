package main

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
)

func TestNewStoreAppliesConcurrencyPragmasToEveryConnection(t *testing.T) {
	store, err := NewStore(filepath.Join(t.TempDir(), "store with spaces.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	t.Cleanup(func() { _ = store.db.Close() })

	ctx := context.Background()
	connections := make([]*sql.Conn, 0, 8)
	t.Cleanup(func() {
		for _, conn := range connections {
			_ = conn.Close()
		}
	})
	for i := 0; i < 8; i++ {
		conn, err := store.db.Conn(ctx)
		if err != nil {
			t.Fatalf("open connection %d: %v", i, err)
		}
		connections = append(connections, conn)

		var journalMode string
		if err := conn.QueryRowContext(ctx, "PRAGMA journal_mode").Scan(&journalMode); err != nil {
			t.Fatalf("connection %d journal mode: %v", i, err)
		}
		if journalMode != "wal" {
			t.Fatalf("connection %d journal mode = %q, want wal", i, journalMode)
		}

		var busyTimeout int
		if err := conn.QueryRowContext(ctx, "PRAGMA busy_timeout").Scan(&busyTimeout); err != nil {
			t.Fatalf("connection %d busy timeout: %v", i, err)
		}
		if busyTimeout != 30000 {
			t.Fatalf("connection %d busy timeout = %d, want 30000", i, busyTimeout)
		}
	}
	for _, conn := range connections {
		if err := conn.Close(); err != nil {
			t.Fatalf("close connection: %v", err)
		}
	}
	connections = nil
}
