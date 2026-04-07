package main

import (
	"context"
	"log/slog"
	"os"
	"os/exec"
	"time"
)

// RunShareWatchdog periodically checks that the SMB/CIFS share at mountPath is
// accessible.  If a stat on probeDir hangs for more than 5 seconds or returns
// an error, it force-unmounts the share and remounts it via sudo.
//
// probeDir is any path that lives on the share (typically outputDir).
// mountPath is the mount-point itself (e.g. /mnt/arbiter-store).
func RunShareWatchdog(ctx context.Context, probeDir, mountPath string, logger *EventLogger) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	consecutiveFailures := 0

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		if shareHealthy(probeDir) {
			if consecutiveFailures > 0 {
				slog.Info("share watchdog: share recovered", "mount", mountPath)
				logger.Log("share.recovered", map[string]any{"mount": mountPath})
				consecutiveFailures = 0
			}
			continue
		}

		consecutiveFailures++
		slog.Warn("share watchdog: share unhealthy", "mount", mountPath, "consecutive_failures", consecutiveFailures)
		logger.Log("share.unhealthy", map[string]any{"mount": mountPath, "consecutive_failures": consecutiveFailures})

		// Only attempt remount after 2 consecutive failures to avoid flapping
		// on a momentary hiccup.
		if consecutiveFailures < 2 {
			continue
		}

		slog.Warn("share watchdog: attempting remount", "mount", mountPath)
		if err := remountShare(mountPath); err != nil {
			slog.Error("share watchdog: remount failed", "mount", mountPath, "error", err)
			logger.Log("share.remount_failed", map[string]any{"mount": mountPath, "error": err.Error()})
		} else {
			slog.Info("share watchdog: remount succeeded", "mount", mountPath)
			logger.Log("share.remounted", map[string]any{"mount": mountPath})
			consecutiveFailures = 0
		}
	}
}

// shareHealthy reads the directory listing of probeDir inside a 5-second timeout.
//
// Simply stat-ing the mount point is not sufficient: on CIFS, the kernel can
// cache the directory entry so stat succeeds even when the server is gone.
// Reading the directory contents forces a real round-trip and will return
// ESTALE, EIO, or ENOENT when the share is stale or unreachable — exactly
// the "everything says No such file or directory" symptom seen in practice.
func shareHealthy(probeDir string) bool {
	result := make(chan error, 1)
	go func() {
		_, err := os.ReadDir(probeDir)
		result <- err
	}()
	select {
	case err := <-result:
		if err != nil {
			slog.Debug("share watchdog: health check failed", "path", probeDir, "error", err)
		}
		return err == nil
	case <-time.After(5 * time.Second):
		slog.Debug("share watchdog: health check timed out", "path", probeDir)
		return false
	}
}

// remountShare force-unmounts mountPath (lazy, so stuck operations don't
// block), waits briefly for the kernel to clean up, then remounts using the
// fstab entry for that path.
func remountShare(mountPath string) error {
	// -f = force even if the server is unreachable
	// -l = lazy: detach the mount now, clean up when no longer busy
	if out, err := exec.Command("sudo", "umount", "-f", "-l", mountPath).CombinedOutput(); err != nil {
		slog.Warn("share watchdog: umount returned error (continuing)", "output", string(out), "error", err)
		// Don't abort — the mount might already be detached; proceed to mount.
	}

	// Give the kernel a moment to release the old mount.
	time.Sleep(2 * time.Second)

	out, err := exec.Command("sudo", "mount", mountPath).CombinedOutput()
	if err != nil {
		return &remountError{output: string(out), cause: err}
	}
	return nil
}

type remountError struct {
	output string
	cause  error
}

func (e *remountError) Error() string {
	if e.output != "" {
		return e.cause.Error() + ": " + e.output
	}
	return e.cause.Error()
}
