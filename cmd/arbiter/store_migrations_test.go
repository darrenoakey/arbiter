package main

import (
	"database/sql"
	"path/filepath"
	"strings"
	"testing"
)

func TestStoreMigrationIsChecksummedAndFailsClosedOnDrift(t *testing.T) {
	databasePath := filepath.Join(t.TempDir(), "store.db")
	store, err := NewStore(databasePath)
	if err != nil {
		t.Fatal(err)
	}
	var count int
	var checksum string
	if err := store.db.QueryRow(
		"SELECT COUNT(*), MIN(sha256) FROM database_version WHERE chain = 'schema' AND version = 1",
	).Scan(&count, &checksum); err != nil {
		t.Fatal(err)
	}
	if count != 1 || len(checksum) != 64 {
		t.Fatalf("migration record count=%d checksum=%q", count, checksum)
	}
	store.Close()

	database, err := sql.Open("sqlite", databasePath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := database.Exec("UPDATE database_version SET sha256 = ? WHERE chain = 'schema' AND version = 1", strings.Repeat("0", 64)); err != nil {
		t.Fatal(err)
	}
	if err := database.Close(); err != nil {
		t.Fatal(err)
	}
	if drifted, err := NewStore(databasePath); err == nil {
		drifted.Close()
		t.Fatal("checksum drift did not prevent store startup")
	}
}

func TestStoreMigrationRejectsUnknownAppliedVersion(t *testing.T) {
	databasePath := filepath.Join(t.TempDir(), "store.db")
	store, err := NewStore(databasePath)
	if err != nil {
		t.Fatal(err)
	}
	_, err = store.db.Exec(
		"INSERT INTO database_version (chain, version, name, sha256, applied_at, duration_ms) VALUES ('schema',2,'unknown.sql',?,0,0)",
		strings.Repeat("a", 64),
	)
	store.Close()
	if err != nil {
		t.Fatal(err)
	}
	if reopened, err := NewStore(databasePath); err == nil {
		reopened.Close()
		t.Fatal("unknown applied migration did not prevent store startup")
	}
}
