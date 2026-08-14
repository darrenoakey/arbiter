package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func (a *API) validateMiniMaxFramePaths(jobType, modelID string, params json.RawMessage, inboxDir string) error {
	if jobType != "video-generate" || modelID != "minimax-h3" {
		return nil
	}
	var frames struct {
		First *string `json:"first_frame_path"`
		Last  *string `json:"last_frame_path"`
	}
	if err := json.Unmarshal(params, &frames); err != nil {
		return fmt.Errorf("invalid MiniMax H3 params")
	}
	for name, value := range map[string]*string{"first_frame_path": frames.First, "last_frame_path": frames.Last} {
		if value != nil {
			if err := validateCanonicalStagedFile(*value, a.stagingRoots(inboxDir)); err != nil {
				return fmt.Errorf("invalid file path for param %q: %w", name, err)
			}
		}
	}
	return nil
}

func (a *API) stagingRoots(inboxDir string) []string {
	var roots []string
	if inboxDir != "" {
		roots = append(roots, inboxDir)
	}
	if a.scheduler != nil && a.scheduler.config != nil && a.scheduler.config.ShareMount != "" {
		roots = append(roots, filepath.Join(a.scheduler.config.ShareMount, "inbox"))
		roots = append(roots, filepath.Join(a.scheduler.config.ShareMount, "output"))
	}
	if a.outputDir != "" {
		roots = append(roots, a.outputDir)
	}
	return roots
}

func validateCanonicalStagedFile(value string, roots []string) error {
	if value == "" || !filepath.IsAbs(value) {
		return fmt.Errorf("path must be absolute and staged")
	}
	if containsParentTraversal(value) {
		return fmt.Errorf("path traversal is forbidden")
	}
	for _, root := range roots {
		if err := validateFileBelowRoot(value, root); err == nil {
			return nil
		}
	}
	return fmt.Errorf("%q is not a canonical staged file", value)
}

func containsParentTraversal(path string) bool {
	for _, part := range strings.Split(filepath.ToSlash(path), "/") {
		if part == ".." {
			return true
		}
	}
	return false
}

func validateFileBelowRoot(value, root string) error {
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return err
	}
	valueAbs, err := filepath.Abs(value)
	if err != nil {
		return err
	}
	relative, err := filepath.Rel(rootAbs, valueAbs)
	if err != nil || relative == "." || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return fmt.Errorf("outside staging root")
	}
	current := rootAbs
	for _, part := range strings.Split(relative, string(filepath.Separator)) {
		current = filepath.Join(current, part)
		info, err := os.Lstat(current)
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("symlinks are forbidden")
		}
	}
	info, err := os.Stat(valueAbs)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() == 0 {
		return fmt.Errorf("staged path must be a non-empty regular file")
	}
	return nil
}
