package main

import (
	"encoding/json"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

// shareJobDir returns the path under the share mount where a job's output should
// live after completion, or "" if no share mount is configured.
func shareJobDir(cfg *Config, jobID string) string {
	if cfg.ShareMount == "" {
		return ""
	}
	return filepath.Join(cfg.ShareMount, "output", "jobs", jobID)
}

// resolveJobDir returns the actual on-disk location of a job's output directory.
// Prefers the share mount if it exists there, otherwise falls back to the
// local output directory. Used by API result lookups so that relocated jobs
// remain readable.
func resolveJobDir(cfg *Config, outputDir, jobID string) string {
	if share := shareJobDir(cfg, jobID); share != "" {
		if _, err := os.Stat(share); err == nil {
			return share
		}
	}
	return filepath.Join(outputDir, "jobs", jobID)
}

// relocateJobOutput moves the local job output directory onto the share mount.
// Returns the post-move path, which equals localDir on no-op. Errors are
// logged and surfaced as no-ops — the job is already complete and the local
// path is still readable.
func relocateJobOutput(cfg *Config, jobID, localDir string) string {
	share := shareJobDir(cfg, jobID)
	if share == "" {
		return localDir
	}
	// Same-path no-op: when output_dir is itself under share_mount the worker
	// already wrote the final file in the share location. Falling through to
	// rename/copy+RemoveAll deletes the just-written file (CIFS rename A→A is
	// not POSIX-clean, copyTree(A,A) opens dst with O_TRUNC and wipes it,
	// then RemoveAll(localDir) deletes the dir).
	if absLocal, errLocal := filepath.Abs(localDir); errLocal == nil {
		if absShare, errShare := filepath.Abs(share); errShare == nil && absLocal == absShare {
			return share
		}
	}
	info, err := os.Stat(localDir)
	if err != nil || !info.IsDir() {
		return localDir
	}
	if err := os.MkdirAll(filepath.Dir(share), 0o755); err != nil {
		slog.Warn("relocate: mkdir parent failed", "job", jobID, "dst", share, "err", err)
		return localDir
	}
	// CIFS is a different device — Rename will EXDEV, but try cheap path first.
	if err := os.Rename(localDir, share); err == nil {
		return share
	}
	if err := copyTree(localDir, share); err != nil {
		slog.Warn("relocate: copy failed", "job", jobID, "src", localDir, "dst", share, "err", err)
		_ = os.RemoveAll(share)
		return localDir
	}
	if err := os.RemoveAll(localDir); err != nil {
		slog.Warn("relocate: remove local failed", "job", jobID, "src", localDir, "err", err)
	}
	return share
}

func copyTree(src, dst string) error {
	return filepath.WalkDir(src, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(src, p)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		return copyFile(p, target)
	})
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func() {
		if err := in.Close(); err != nil {
			slog.Debug("close relocation source", "path", src, "error", err)
		}
	}()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		if closeErr := out.Close(); closeErr != nil {
			slog.Debug("close failed relocation destination", "path", dst, "error", closeErr)
		}
		return err
	}
	return out.Close()
}

// rewriteResultPaths walks a result JSON value and replaces any string
// occurrence of oldPrefix with newPrefix. Returns the original bytes on
// unmarshal/marshal error or when no rewrite is needed.
func rewriteResultPaths(result json.RawMessage, oldPrefix, newPrefix string) json.RawMessage {
	if len(result) == 0 || oldPrefix == "" || newPrefix == "" || oldPrefix == newPrefix {
		return result
	}
	var v any
	if err := json.Unmarshal(result, &v); err != nil {
		return result
	}
	walked := walkReplace(v, oldPrefix, newPrefix)
	out, err := json.Marshal(walked)
	if err != nil {
		return result
	}
	return out
}

func walkReplace(v any, oldP, newP string) any {
	switch x := v.(type) {
	case string:
		if strings.Contains(x, oldP) {
			return strings.ReplaceAll(x, oldP, newP)
		}
		return x
	case []any:
		for i, e := range x {
			x[i] = walkReplace(e, oldP, newP)
		}
		return x
	case map[string]any:
		for k, e := range x {
			x[k] = walkReplace(e, oldP, newP)
		}
		return x
	}
	return v
}
