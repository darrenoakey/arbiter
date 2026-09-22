package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	gpuIdleDefaultSeconds         = 180
	gpuIdleDefaultLowUtilPct      = 5
	gpuIdleDefaultHighIntervalSec = 30
	gpuIdleDefaultLowIntervalSec  = 5
	gpuIdleDefaultCooldownSec     = 120
	gpuIdleDefaultAgentd3URL      = "http://10.0.0.44:8655/investigate"
	gpuIdleDefaultAgentd3Model    = "agentic-high"
	gpuIdleDefaultAgentd3Cwd      = "/Users/darrenoakey/src/arbiter"
	gpuIdleDefaultAgentd3Policy   = "yolo"
	gpuIdleMaxPayloadBytes        = 4096
	gpuIdleSampleCap              = 48
	gpuIdleProgressLogEvery       = time.Minute
)

// GPUIdleConfig is the hung-job GPU-idle watchdog. Zero values mean defaults.
type GPUIdleConfig struct {
	Disabled        bool    `json:"disabled,omitempty"`
	IdleSeconds     float64 `json:"idle_seconds,omitempty"`
	LowUtilPct      int     `json:"low_util_pct,omitempty"`
	HighIntervalSec float64 `json:"high_interval_sec,omitempty"`
	LowIntervalSec  float64 `json:"low_interval_sec,omitempty"`
	CooldownSec     float64 `json:"cooldown_sec,omitempty"`
	Agentd3URL      string  `json:"agentd3_url,omitempty"`
	Agentd3Model    string  `json:"agentd3_model,omitempty"`
	Agentd3Cwd      string  `json:"agentd3_cwd,omitempty"`
	Agentd3Policy   string  `json:"agentd3_policy,omitempty"`
}

// GPUIdleWatchdog kills local workers that still look running while the GPU
// has been continuously idle AND the worker itself has gone silent, then starts
// an agentd3 investigation.
//
// The worker-silence condition is not optional garnish. GPU utilization alone is
// NOT a liveness signal: a healthy worker can legitimately sit at 0% for many
// minutes. Observed 2026-09-18 on job 0c2b2db53140 (photo-enhance): the SeedVR2
// tile pass finished, and the reference/climb phases that follow are CPU
// PIL/numpy work on a 5292x3969 frame whose vision turns were answered from the
// on-disk LLMCache (qwen3-vl had just been evicted and never reloaded). nvidia-smi
// read 0% for 181s while the worker was accepting a climb step every ~35s, and
// the watchdog destroyed 16 minutes of finished GPU work. A worker that is still
// emitting stdout/stderr is making progress and must never be killed here.
type GPUIdleWatchdog struct {
	cfg       *Config
	mgr       *InstanceManager
	store     *Store
	logger    *EventLogger
	outputDir string

	gpuUtil            func() int
	gpuStatus          func() string
	now                func() time.Time
	killInstance       func(inst *Instance)
	startInvestigation func(report GPUIdleKillReport) error

	mu              sync.Mutex
	idleSince       time.Time
	lastKill        time.Time
	lastProgressLog time.Time
	samples         []gpuIdleSample
}

type gpuIdleSample struct {
	At   time.Time `json:"at"`
	Util int       `json:"util_pct"`
}

type gpuIdleVictim struct {
	Inst *Instance
	Jobs []*Job
	// OutputAge is how long the worker has been silent on stdout/stderr at the
	// sample time. Negative means the worker has never emitted anything.
	OutputAge time.Duration
}

// GPUIdleKillReport is the payload written to disk and POSTed to the laptop hook.
type GPUIdleKillReport struct {
	KilledAt       time.Time           `json:"killed_at"`
	IdleForSeconds float64             `json:"idle_for_seconds"`
	GPUUtilPct     int                 `json:"gpu_util_pct"`
	LowUtilPct     int                 `json:"low_util_pct"`
	GPUStatus      string              `json:"gpu_status,omitempty"`
	Samples        []gpuIdleSample     `json:"samples"`
	Jobs           []gpuIdleJobDetail  `json:"jobs"`
	Instances      []gpuIdleInstDetail `json:"instances"`
	LogRefs        []string            `json:"log_refs"`
	Prompt         string              `json:"prompt"`
	Conversation   GPUIdleConversation `json:"conversation"`
	BundlePath     string              `json:"bundle_path,omitempty"`
}

// GPUIdleConversation is the agentd3 create body the laptop hook forwards.
type GPUIdleConversation struct {
	Model          string            `json:"model"`
	Cwd            string            `json:"cwd"`
	Policy         string            `json:"policy"`
	Title          string            `json:"title"`
	Source         string            `json:"source"`
	IdempotencyKey string            `json:"idempotency_key"`
	Origin         map[string]string `json:"origin"`
}

type gpuIdleJobDetail struct {
	JobID     string          `json:"job_id"`
	ModelID   string          `json:"model_id"`
	JobType   string          `json:"job_type"`
	State     string          `json:"state"`
	CreatedAt float64         `json:"created_at"`
	StartedAt *float64        `json:"started_at,omitempty"`
	ElapsedS  float64         `json:"elapsed_seconds,omitempty"`
	Who       string          `json:"who,omitempty"`
	Why       string          `json:"why,omitempty"`
	Payload   json.RawMessage `json:"payload,omitempty"`
}

type gpuIdleInstDetail struct {
	InstanceID string   `json:"instance_id"`
	ModelID    string   `json:"model_id"`
	State      string   `json:"state"`
	PID        int      `json:"pid"`
	ActiveJobs int      `json:"active_jobs"`
	PendingIDs []string `json:"pending_job_ids,omitempty"`
	// SilentForSeconds is how long the worker had produced no stdout/stderr when
	// it was killed; -1 means it never produced any output at all.
	SilentForSeconds float64 `json:"silent_for_seconds"`
}

// NewGPUIdleWatchdog constructs a watchdog. Caller starts it with Run.
func NewGPUIdleWatchdog(cfg *Config, mgr *InstanceManager, store *Store, logger *EventLogger, outputDir string) *GPUIdleWatchdog {
	w := &GPUIdleWatchdog{
		cfg:       cfg,
		mgr:       mgr,
		store:     store,
		logger:    logger,
		outputDir: outputDir,
		gpuUtil:   GetGPUUtilization,
		gpuStatus: GetGPUStatusLine,
		now:       time.Now,
	}
	w.killInstance = w.forceKill
	w.startInvestigation = w.postInvestigation
	return w
}

func (c GPUIdleConfig) idleFor() time.Duration {
	if c.IdleSeconds > 0 {
		return time.Duration(c.IdleSeconds * float64(time.Second))
	}
	return gpuIdleDefaultSeconds * time.Second
}

func (c GPUIdleConfig) lowUtilPct() int {
	if c.LowUtilPct > 0 {
		return c.LowUtilPct
	}
	return gpuIdleDefaultLowUtilPct
}

func (c GPUIdleConfig) highInterval() time.Duration {
	if c.HighIntervalSec > 0 {
		return time.Duration(c.HighIntervalSec * float64(time.Second))
	}
	return gpuIdleDefaultHighIntervalSec * time.Second
}

func (c GPUIdleConfig) lowInterval() time.Duration {
	if c.LowIntervalSec > 0 {
		return time.Duration(c.LowIntervalSec * float64(time.Second))
	}
	return gpuIdleDefaultLowIntervalSec * time.Second
}

func (c GPUIdleConfig) cooldown() time.Duration {
	if c.CooldownSec > 0 {
		return time.Duration(c.CooldownSec * float64(time.Second))
	}
	return gpuIdleDefaultCooldownSec * time.Second
}

func (c GPUIdleConfig) agentd3URL() string {
	if strings.TrimSpace(c.Agentd3URL) != "" {
		return strings.TrimSpace(c.Agentd3URL)
	}
	return gpuIdleDefaultAgentd3URL
}

func (c GPUIdleConfig) agentd3Model() string {
	if strings.TrimSpace(c.Agentd3Model) != "" {
		return strings.TrimSpace(c.Agentd3Model)
	}
	return gpuIdleDefaultAgentd3Model
}

func (c GPUIdleConfig) agentd3Cwd() string {
	if strings.TrimSpace(c.Agentd3Cwd) != "" {
		return strings.TrimSpace(c.Agentd3Cwd)
	}
	return gpuIdleDefaultAgentd3Cwd
}

func (c GPUIdleConfig) agentd3Policy() string {
	if strings.TrimSpace(c.Agentd3Policy) != "" {
		return strings.TrimSpace(c.Agentd3Policy)
	}
	return gpuIdleDefaultAgentd3Policy
}

// Run polls GPU utilization until ctx is done. Interval shrinks while util is low.
func (w *GPUIdleWatchdog) Run(ctx context.Context) {
	if w.cfg.GPUIdle.Disabled {
		slog.Info("gpu idle watchdog disabled")
		return
	}
	slog.Info("gpu idle watchdog up",
		"idle_seconds", w.cfg.GPUIdle.idleFor().Seconds(),
		"low_util_pct", w.cfg.GPUIdle.lowUtilPct(),
		"agentd3_url", w.cfg.GPUIdle.agentd3URL())
	timer := time.NewTimer(w.cfg.GPUIdle.highInterval())
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
			next := w.tick()
			timer.Reset(next)
		}
	}
}

func (w *GPUIdleWatchdog) tick() time.Duration {
	cfg := w.cfg.GPUIdle
	queryStart := w.now()
	util := w.gpuUtil()
	at := w.now()
	w.recordSample(util, at)
	// A hung nvidia-smi call is not continuous observation. Counting the
	// blackout as idle made one 0% sample after a 14-minute driver lock
	// satisfy the kill window immediately (2026-09-22, job 633b27a52464).
	if at.Sub(queryStart) > cfg.lowInterval() {
		w.resetIdle()
		return cfg.lowInterval()
	}
	if util < 0 || util > cfg.lowUtilPct() {
		w.resetIdle()
		return cfg.highInterval()
	}
	next := cfg.lowInterval()
	victims := w.listLocalRunning(at)
	if len(victims) == 0 {
		w.resetIdle()
		return next
	}
	w.mu.Lock()
	if w.idleSince.IsZero() {
		w.idleSince = at
	}
	idleFor := at.Sub(w.idleSince)
	sinceKill := time.Duration(0)
	if !w.lastKill.IsZero() {
		sinceKill = at.Sub(w.lastKill)
	}
	ready := idleFor >= cfg.idleFor() && (w.lastKill.IsZero() || sinceKill >= cfg.cooldown())
	w.mu.Unlock()
	if !ready {
		return next
	}
	// Second gate: only workers that have ALSO stopped emitting output for the
	// whole idle window are hung. A chatty worker keeps its instance alive and
	// leaves the GPU-idle clock running, so the kill lands the moment it truly
	// goes quiet — no extra window is added.
	silent := silentVictims(victims, cfg.idleFor())
	if len(silent) == 0 {
		w.noteProgressSkip(at, victims)
		return next
	}
	w.fire(silent, util, at, idleFor)
	return next
}

// silentVictims keeps only the victims whose worker has emitted nothing for at
// least the idle window. Never-spawned / never-spoken instances (zero last
// output) count as silent.
func silentVictims(victims []gpuIdleVictim, window time.Duration) []gpuIdleVictim {
	var out []gpuIdleVictim
	for _, v := range victims {
		if v.OutputAge < 0 || v.OutputAge >= window {
			out = append(out, v)
		}
	}
	return out
}

// noteProgressSkip logs (at most once per minute) that the GPU is idle but every
// candidate worker is still producing output, so nothing was killed.
func (w *GPUIdleWatchdog) noteProgressSkip(at time.Time, victims []gpuIdleVictim) {
	w.mu.Lock()
	due := w.lastProgressLog.IsZero() || at.Sub(w.lastProgressLog) >= gpuIdleProgressLogEvery
	if due {
		w.lastProgressLog = at
	}
	w.mu.Unlock()
	if !due {
		return
	}
	ages := make([]string, 0, len(victims))
	for _, v := range victims {
		ages = append(ages, fmt.Sprintf("%s=%.0fs", v.Inst.InstanceID, v.OutputAge.Seconds()))
	}
	slog.Info("gpu idle watchdog: GPU idle but workers still emitting progress; not killing",
		"instances", strings.Join(ages, " "))
}

func (w *GPUIdleWatchdog) resetIdle() {
	w.mu.Lock()
	w.idleSince = time.Time{}
	w.mu.Unlock()
}

func (w *GPUIdleWatchdog) recordSample(util int, at time.Time) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.samples = append(w.samples, gpuIdleSample{At: at, Util: util})
	if len(w.samples) > gpuIdleSampleCap {
		w.samples = append([]gpuIdleSample(nil), w.samples[len(w.samples)-gpuIdleSampleCap:]...)
	}
}

func (w *GPUIdleWatchdog) listLocalRunning(at time.Time) []gpuIdleVictim {
	var victims []gpuIdleVictim
	running, _ := w.store.GetRunningJobs()
	byModel := map[string][]*Job{}
	byID := map[string]*Job{}
	for _, job := range running {
		byModel[job.ModelID] = append(byModel[job.ModelID], job)
		byID[job.ID] = job
	}
	for _, inst := range w.mgr.AllInstances() {
		if inst == nil || inst.isRemote() {
			continue
		}
		state := inst.State()
		if state == "loading" || state == "starting" {
			continue
		}
		if inst.ActiveJobs() == 0 {
			continue
		}
		// No outstanding worker request means inference already returned
		// (readLoop clears pending before the dispatch goroutine resumes).
		// The stall, if any, is server-side bookkeeping — Stat, relocate,
		// sqlite — not a hung GPU kernel. Killing the process destroys a
		// loaded model that is waiting for the next command.
		// 2026-09-22 ltx25-denoise1 job 633b27a52464: result.mp4 was on disk
		// at 13:06Z, the worker was idle, and the watchdog killed it at 13:24Z
		// while completion bookkeeping was still blocked.
		if len(inst.PendingJobIDs()) == 0 {
			continue
		}
		jobs := collectVictimJobs(inst, byID, byModel)
		victims = append(victims, gpuIdleVictim{Inst: inst, Jobs: jobs, OutputAge: outputAge(inst, at)})
	}
	return victims
}

// outputAge is how long the worker has been silent, or -1 when it has never
// emitted anything (which the silence gate treats as hung, not as fresh).
func outputAge(inst *Instance, at time.Time) time.Duration {
	last := inst.LastOutputAt()
	if last.IsZero() {
		return -1
	}
	age := at.Sub(last)
	if age < 0 {
		return 0
	}
	return age
}

func collectVictimJobs(inst *Instance, byID map[string]*Job, byModel map[string][]*Job) []*Job {
	seen := map[string]bool{}
	var jobs []*Job
	for _, id := range inst.PendingJobIDs() {
		if job := byID[id]; job != nil && !seen[job.ID] {
			seen[job.ID] = true
			jobs = append(jobs, job)
		}
	}
	for _, job := range byModel[inst.ModelID] {
		if !seen[job.ID] {
			seen[job.ID] = true
			jobs = append(jobs, job)
		}
	}
	return jobs
}

func (w *GPUIdleWatchdog) fire(victims []gpuIdleVictim, util int, at time.Time, idleFor time.Duration) {
	report := w.buildReport(victims, util, at, idleFor)
	bundlePath := w.writeBundle(report, at)
	report.BundlePath = bundlePath
	report.LogRefs = append([]string{bundlePath}, report.LogRefs...)
	report.Prompt = renderGPUIdlePrompt(report)

	for _, v := range victims {
		w.killInstance(v.Inst)
		for _, job := range v.Jobs {
			w.failJob(job, idleFor)
		}
	}
	w.mu.Lock()
	w.lastKill = at
	w.idleSince = time.Time{}
	w.mu.Unlock()

	fields := map[string]any{
		"idle_for_seconds": idleFor.Seconds(),
		"gpu_util_pct":     util,
		"instance_ids":     instanceIDs(victims),
		"job_ids":          jobIDs(victims),
		"bundle_path":      bundlePath,
	}
	w.logger.Log("gpu.idle_kill", fields)
	slog.Error("gpu idle watchdog: killed hung local workers", "idle_for_seconds", idleFor.Seconds(), "gpu_util_pct", util, "jobs", jobIDs(victims))

	go func() {
		if err := w.startInvestigation(report); err != nil {
			slog.Error("gpu idle watchdog: investigation dispatch failed", "error", err)
			w.logger.Log("gpu.idle_investigate_failed", map[string]any{"error": err.Error(), "bundle_path": bundlePath})
		}
	}()
}

func (w *GPUIdleWatchdog) forceKill(inst *Instance) {
	if inst == nil {
		return
	}
	slog.Error("gpu idle watchdog: killing instance", "instance", inst.InstanceID, "model", inst.ModelID, "pid", inst.PID(), "active_jobs", inst.ActiveJobs())
	inst.Kill()
	w.mgr.ReleaseMemoryFor(inst)
}

func (w *GPUIdleWatchdog) failJob(job *Job, idleFor time.Duration) {
	if job == nil {
		return
	}
	errMsg := fmt.Sprintf("killed by gpu-idle-watchdog: GPU utilization stayed at or below %d%% for %.0fs and the worker produced no output for that whole window while this job was running", w.cfg.GPUIdle.lowUtilPct(), idleFor.Seconds())
	if err := w.store.UpdateState(job.ID, "failed", WithError(errMsg), WithFinishedAt(nowTS())); err != nil {
		slog.Warn("gpu idle watchdog: fail job", "job_id", job.ID, "error", err)
		return
	}
	if n := w.store.ResolveFollowers(job.ID, "failed", nil, errMsg, w.outputDir); n > 0 {
		slog.Info("gpu idle watchdog: resolved followers", "original", job.ID, "followers", n)
	}
}

func (w *GPUIdleWatchdog) buildReport(victims []gpuIdleVictim, util int, at time.Time, idleFor time.Duration) GPUIdleKillReport {
	cfg := w.cfg.GPUIdle
	jobs := make([]gpuIdleJobDetail, 0)
	insts := make([]gpuIdleInstDetail, 0, len(victims))
	seenJob := map[string]bool{}
	for _, v := range victims {
		insts = append(insts, gpuIdleInstDetail{
			InstanceID:       v.Inst.InstanceID,
			ModelID:          v.Inst.ModelID,
			State:            v.Inst.State(),
			PID:              v.Inst.PID(),
			ActiveJobs:       v.Inst.ActiveJobs(),
			PendingIDs:       v.Inst.PendingJobIDs(),
			SilentForSeconds: v.OutputAge.Seconds(),
		})
		for _, job := range v.Jobs {
			if job == nil || seenJob[job.ID] {
				continue
			}
			seenJob[job.ID] = true
			jobs = append(jobs, jobDetailFrom(job, at))
		}
	}
	sort.Slice(jobs, func(i, j int) bool { return jobs[i].JobID < jobs[j].JobID })
	title := "GPU idle kill"
	if len(jobs) > 0 {
		title = fmt.Sprintf("GPU idle kill: %s %s", jobs[0].JobID, jobs[0].ModelID)
	}
	ref := strings.Join(jobIDs(victims), ",")
	w.mu.Lock()
	samples := append([]gpuIdleSample(nil), w.samples...)
	w.mu.Unlock()
	report := GPUIdleKillReport{
		KilledAt:       at.UTC(),
		IdleForSeconds: idleFor.Seconds(),
		GPUUtilPct:     util,
		LowUtilPct:     cfg.lowUtilPct(),
		GPUStatus:      w.gpuStatus(),
		Samples:        samples,
		Jobs:           jobs,
		Instances:      insts,
		LogRefs:        gpuIdleLogRefs(at, jobs),
		Conversation: GPUIdleConversation{
			Model:          cfg.agentd3Model(),
			Cwd:            cfg.agentd3Cwd(),
			Policy:         cfg.agentd3Policy(),
			Title:          title,
			Source:         "arbiter-gpu-idle-watchdog",
			IdempotencyKey: fmt.Sprintf("gpu-idle-%s-%d", ref, at.Unix()),
			Origin: map[string]string{
				"kind":   "service",
				"actor":  "arbiter-gpu-idle-watchdog",
				"ref":    ref,
				"detail": fmt.Sprintf("killed after %.0fs idle GPU", idleFor.Seconds()),
			},
		},
	}
	report.Prompt = renderGPUIdlePrompt(report)
	return report
}

func jobDetailFrom(job *Job, at time.Time) gpuIdleJobDetail {
	d := gpuIdleJobDetail{
		JobID:     job.ID,
		ModelID:   job.ModelID,
		JobType:   job.JobType,
		State:     job.State,
		CreatedAt: job.CreatedAt,
		StartedAt: job.StartedAt,
		Payload:   truncatePayload(job.Payload),
	}
	if job.Source != nil {
		d.Who = job.Source.Who
		d.Why = job.Source.Why
	}
	if job.StartedAt != nil {
		d.ElapsedS = float64(at.Unix()) - *job.StartedAt
		if d.ElapsedS < 0 {
			d.ElapsedS = 0
		}
	}
	return d
}

func truncatePayload(raw json.RawMessage) json.RawMessage {
	if len(raw) <= gpuIdleMaxPayloadBytes {
		return raw
	}
	return json.RawMessage(fmt.Sprintf(`{"_truncated":true,"bytes":%d}`, len(raw)))
}

func gpuIdleLogRefs(at time.Time, jobs []gpuIdleJobDetail) []string {
	day := at.UTC().Format("2006-01-02")
	month := at.UTC().Format("2006/01")
	refs := []string{
		fmt.Sprintf("/mnt/arbiter-store/output/logs/arbiter-%s.jsonl", day),
		fmt.Sprintf("/Volumes/ssd_4/arbiter/output/logs/arbiter-%s.jsonl", day),
		fmt.Sprintf("/home/darren/local/auto/output/logs/arbiter/%s/", month),
		"/home/darren/local/blackbox/",
		"http://10.0.0.254:8400/v1/ps",
		"ssh darren@10.0.0.254",
	}
	for _, job := range jobs {
		refs = append(refs,
			fmt.Sprintf("/mnt/arbiter-store/output/jobs/%s/", job.JobID),
			fmt.Sprintf("http://10.0.0.254:8400/v1/jobs/%s", job.JobID),
		)
	}
	return refs
}

func renderGPUIdlePrompt(report GPUIdleKillReport) string {
	body, _ := json.MarshalIndent(reportWithoutPrompt(report), "", "  ")
	var b strings.Builder
	b.WriteString("Arbiter on spark (10.0.0.254) killed hung GPU work because nvidia-smi utilization stayed at or below ")
	b.WriteString(fmt.Sprintf("%d%% for %.0f continuous seconds AND the worker emitted no stdout/stderr for that whole window while local jobs still appeared running (per-instance silence is in `silent_for_seconds`; -1 means the worker never spoke).\n\n", report.LowUtilPct, report.IdleForSeconds))
	b.WriteString("Investigate WHY the worker was stuck (deadlock, waiting on IO/network, infinite CPU loop, adapter bug, silent CUDA stall). ")
	b.WriteString("Read the logs and job records below. If it is a code bug in arbiter, fix it. ")
	b.WriteString("Do not cancel other people's jobs. Do not bounce arbiter while unrelated work is in flight.\n\n")
	b.WriteString("Spark access: ssh darren@10.0.0.254 then curl 127.0.0.1:8400. Laptop mount of the same output tree: /Volumes/ssd_4/arbiter/output/\n\n")
	b.WriteString("Kill report (JSON):\n```json\n")
	b.Write(body)
	b.WriteString("\n```\n")
	return b.String()
}

func reportWithoutPrompt(report GPUIdleKillReport) GPUIdleKillReport {
	cp := report
	cp.Prompt = ""
	return cp
}

func (w *GPUIdleWatchdog) writeBundle(report GPUIdleKillReport, at time.Time) string {
	dir := filepath.Join(w.outputDir, "gpu-idle-kills")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		slog.Warn("gpu idle watchdog: mkdir bundle", "error", err)
		return ""
	}
	name := fmt.Sprintf("%s.json", at.UTC().Format("20060102T150405Z"))
	if len(report.Jobs) > 0 {
		name = fmt.Sprintf("%s-%s.json", at.UTC().Format("20060102T150405Z"), report.Jobs[0].JobID)
	}
	path := filepath.Join(dir, name)
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return ""
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		slog.Warn("gpu idle watchdog: write bundle", "error", err)
		return ""
	}
	return path
}

func (w *GPUIdleWatchdog) postInvestigation(report GPUIdleKillReport) error {
	url := w.cfg.GPUIdle.agentd3URL()
	if url == "" {
		return fmt.Errorf("gpu idle watchdog: agentd3 url empty")
	}
	data, err := json.Marshal(report)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 15 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("investigation POST %s: HTTP %d", url, resp.StatusCode)
	}
	slog.Info("gpu idle watchdog: investigation dispatched", "url", url, "status", resp.StatusCode)
	w.logger.Log("gpu.idle_investigate", map[string]any{
		"url":         url,
		"job_ids":     jobIDsFromReport(report),
		"bundle_path": report.BundlePath,
	})
	return nil
}

func instanceIDs(victims []gpuIdleVictim) []string {
	out := make([]string, 0, len(victims))
	for _, v := range victims {
		out = append(out, v.Inst.InstanceID)
	}
	return out
}

func jobIDs(victims []gpuIdleVictim) []string {
	seen := map[string]bool{}
	var out []string
	for _, v := range victims {
		for _, job := range v.Jobs {
			if job != nil && !seen[job.ID] {
				seen[job.ID] = true
				out = append(out, job.ID)
			}
		}
	}
	sort.Strings(out)
	return out
}

func jobIDsFromReport(report GPUIdleKillReport) []string {
	out := make([]string, 0, len(report.Jobs))
	for _, job := range report.Jobs {
		out = append(out, job.JobID)
	}
	return out
}
