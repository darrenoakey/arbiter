package main

import (
	"crypto/sha256"
	"database/sql"
	"embed"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

//go:embed migrations/*.sql
var storeMigrationFiles embed.FS

type storeMigration struct {
	version  int
	name     string
	contents string
	checksum string
}

func applyStoreMigrations(db *sql.DB) error {
	if err := ensureStoreVersionTable(db); err != nil {
		return err
	}
	migrations, err := readStoreMigrations()
	if err != nil {
		return err
	}
	if err := verifyAppliedStoreMigrations(db, migrations); err != nil {
		return err
	}
	for _, migration := range migrations {
		if err := applyStoreMigration(db, migration); err != nil {
			return err
		}
	}
	return nil
}

func verifyAppliedStoreMigrations(db *sql.DB, migrations []storeMigration) error {
	known := make(map[int]storeMigration, len(migrations))
	for _, migration := range migrations {
		known[migration.version] = migration
	}
	rows, err := db.Query("SELECT version, name, sha256 FROM database_version WHERE chain = 'schema' ORDER BY version")
	if err != nil {
		return fmt.Errorf("read applied store migrations: %w", err)
	}
	defer func() { _ = rows.Close() }()
	applied := make(map[int]bool)
	maxVersion := 0
	for rows.Next() {
		var version int
		var name, checksum string
		if err := rows.Scan(&version, &name, &checksum); err != nil {
			return err
		}
		migration, ok := known[version]
		if !ok || migration.name != name || migration.checksum != checksum {
			return fmt.Errorf("store migration %04d is unknown or modified; restore the migration chain", version)
		}
		applied[version], maxVersion = true, version
	}
	for _, migration := range migrations {
		if migration.version <= maxVersion && !applied[migration.version] {
			return fmt.Errorf("store migration %04d is missing below applied version %04d", migration.version, maxVersion)
		}
	}
	return rows.Err()
}

func ensureStoreVersionTable(db *sql.DB) error {
	_, err := db.Exec(`CREATE TABLE IF NOT EXISTS database_version (
        chain TEXT NOT NULL,
        version INTEGER NOT NULL,
        name TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        applied_at REAL NOT NULL,
        duration_ms INTEGER NOT NULL,
        PRIMARY KEY (chain, version)
    )`)
	return err
}

func readStoreMigrations() ([]storeMigration, error) {
	entries, err := storeMigrationFiles.ReadDir("migrations")
	if err != nil {
		return nil, fmt.Errorf("read store migrations: %w", err)
	}
	var migrations []storeMigration
	for _, entry := range entries {
		migration, err := readStoreMigration(entry.Name())
		if err != nil {
			return nil, err
		}
		migrations = append(migrations, migration)
	}
	sort.Slice(migrations, func(i, j int) bool { return migrations[i].version < migrations[j].version })
	return migrations, nil
}

func readStoreMigration(name string) (storeMigration, error) {
	versionText, _, ok := strings.Cut(name, "_")
	if !ok {
		return storeMigration{}, fmt.Errorf("invalid store migration name %q", name)
	}
	version, err := strconv.Atoi(versionText)
	if err != nil {
		return storeMigration{}, fmt.Errorf("invalid store migration version %q: %w", name, err)
	}
	contents, err := storeMigrationFiles.ReadFile("migrations/" + name)
	if err != nil {
		return storeMigration{}, fmt.Errorf("read store migration %q: %w", name, err)
	}
	sum := sha256.Sum256(contents)
	return storeMigration{version: version, name: name, contents: string(contents), checksum: hex.EncodeToString(sum[:])}, nil
}

func applyStoreMigration(db *sql.DB, migration storeMigration) error {
	var name, checksum string
	err := db.QueryRow(
		"SELECT name, sha256 FROM database_version WHERE chain = 'schema' AND version = ?",
		migration.version,
	).Scan(&name, &checksum)
	if err == nil {
		if name != migration.name || checksum != migration.checksum {
			return fmt.Errorf("store migration %04d checksum mismatch; restore %s and add a new migration", migration.version, migration.name)
		}
		return nil
	}
	if err != sql.ErrNoRows {
		return fmt.Errorf("read store migration %04d: %w", migration.version, err)
	}
	return executeStoreMigration(db, migration)
}

func executeStoreMigration(db *sql.DB, migration storeMigration) error {
	started := nowTS()
	tx, err := db.Begin()
	if err != nil {
		return fmt.Errorf("begin store migration %04d: %w", migration.version, err)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.Exec(migration.contents); err != nil {
		return fmt.Errorf("apply store migration %04d: %w", migration.version, err)
	}
	duration := int((nowTS() - started) * 1000)
	_, err = tx.Exec(
		"INSERT INTO database_version (chain, version, name, sha256, applied_at, duration_ms) VALUES ('schema',?,?,?,?,?)",
		migration.version, migration.name, migration.checksum, nowTS(), duration,
	)
	if err != nil {
		return fmt.Errorf("record store migration %04d: %w", migration.version, err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit store migration %04d: %w", migration.version, err)
	}
	return nil
}
