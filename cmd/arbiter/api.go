package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// portCacheEntry is one row of the per-instance worker-port cache.
type portCacheEntry struct {
	port     int
	loadedAt time.Time
}

type API struct {
	portCacheMu sync.Mutex
	portCache   map[string]portCacheEntry // instance_id -> port
	config      *Config
	store       *Store
	mgr         *InstanceManager
	scheduler   *Scheduler
	logger      *EventLogger
	outputDir   string
	projectRoot string
	startTime   time.Time
	// hostMonitor is the Phase-3 per-host liveness monitor. /v1/ps consults it
	// for the SEPARATE remote_hosts panel. nil when no remote hosts are
	// configured (single-host arbiter) — the panel is simply omitted then.
	hostMonitor *HostMonitor

	// Cached /v1/ps response — updated every second by background goroutine
	psCache atomic.Value // json.RawMessage

	// Cached DB-derived /v1/ps aggregates. These scan all job history
	// (O(rows), and rows only grow), so they are refreshed on a slow cadence
	// rather than every tick — recomputing all-time averages 65 times a second
	// (global + per-model) pinned a CPU core indefinitely.
	statsMu       sync.Mutex
	statsAt       time.Time
	statsCounts   map[string]map[string]int // model_id -> state -> count
	statsGlobalCt map[string]int            // state -> count (global)
	statsModel    map[string]JobStats       // model_id -> completed stats
	statsGlobal   JobStats                  // global completed stats

	// requestShutdown triggers a graceful process shutdown. Set by main once
	// the HTTP server exists. Invoked by the drain monitor when shutdown_when_idle
	// was requested and the last in-flight job has finished.
	requestShutdown   func()
	drainShutdownOnce sync.Once
}

// SetShutdownFunc wires the graceful-shutdown callback used by drain
// (shutdown_when_idle). Called once from main after the HTTP server is built.
func (a *API) SetShutdownFunc(fn func()) {
	a.requestShutdown = fn
}

// SetHostMonitor wires the Phase-3 liveness monitor so /v1/ps can render the
// remote_hosts panel. Called once from main after the monitor is built.
func (a *API) SetHostMonitor(hm *HostMonitor) {
	a.hostMonitor = hm
}

type modelConfigRequest struct {
	ModelID        string             `json:"model_id"`
	MemoryGB       *float64           `json:"memory_gb"`
	MaxConcurrent  *int               `json:"max_concurrent"`
	MaxInstances   *int               `json:"max_instances"`
	KeepAliveSec   *int               `json:"keep_alive_seconds"`
	AvgInferenceMs *float64           `json:"avg_inference_ms"`
	LoadMs         *float64           `json:"load_ms"`
	AutoDownload   *string            `json:"auto_download"`
	ModelPath      *string            `json:"model_path"`
	Group          *bool              `json:"group"`
	WorkerCmd      *[]string          `json:"worker_cmd"`
	AdapterParams  *map[string]string `json:"adapter_params"`
	PressureIndex  *float64           `json:"pressure_index"`
	MaxRuntimeSec  *int               `json:"max_runtime_seconds"`
	ConflictGroup  *string            `json:"conflict_group"`
	GroupPriority  *int               `json:"group_priority"`
	// RemoteEnabled is the per-model remote kill-switch. PATCH with
	// {"remote_enabled":false} pins the model to spark instantly — in ONE curl,
	// working even if the remote host is unreachable. The handler also drains any
	// in-flight remote job for the model back to spark.
	RemoteEnabled *bool `json:"remote_enabled"`
	ReloadWorkers bool  `json:"reload_workers"`
}

type llmRegisterRequest struct {
	HFModel        string            `json:"hf_model"`
	HFFile         string            `json:"hf_file"`
	ModelPath      string            `json:"model_path"`
	Name           string            `json:"name"`
	MemoryGB       float64           `json:"memory_gb"`
	CtxSize        int               `json:"ctx_size"`
	GPULayers      int               `json:"gpu_layers"`
	WorkerCmd      []string          `json:"worker_cmd"`
	AdapterParams  map[string]string `json:"adapter_params"`
	LlamaServerBin string            `json:"llama_server_bin"`
	MaxConcurrent  *int              `json:"max_concurrent"`
	MaxInstances   *int              `json:"max_instances"`
	KeepAliveSec   *int              `json:"keep_alive_seconds"`
	MaxRuntimeSec  *int              `json:"max_runtime_seconds"`
	AvgInferenceMs *float64          `json:"avg_inference_ms"`
	LoadMs         *float64          `json:"load_ms"`
	// Backend selects the inference engine: "llamacpp" (default, uses
	// llm-worker → llama-server) or "vllm" (uses vllm-chat-worker → vllm serve).
	Backend       string `json:"backend"`
	VllmModel     string `json:"vllm_model"`      // VLLM_MODEL env (HF id or repo:file GGUF spec); defaults from HFModel
	VllmExtraArgs string `json:"vllm_extra_args"` // VLLM_EXTRA_ARGS env (e.g., "--max-model-len 32768 --quantization awq")
}

func NewAPI(cfg *Config, store *Store, mgr *InstanceManager, sched *Scheduler, logger *EventLogger, outputDir, projectRoot string) *API {
	a := &API{
		config:      cfg,
		store:       store,
		mgr:         mgr,
		scheduler:   sched,
		logger:      logger,
		outputDir:   outputDir,
		projectRoot: projectRoot,
		startTime:   time.Now(),
	}
	return a
}

func (a *API) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/jobs", a.submitJob)
	mux.HandleFunc("GET /v1/jobs/{id}", a.getJob)
	mux.HandleFunc("DELETE /v1/jobs/{id}", a.cancelJob)
	mux.HandleFunc("POST /v1/jobs/status", a.bulkStatus)
	mux.HandleFunc("GET /v1/jobs", a.listJobs)
	mux.HandleFunc("GET /v1/ps", a.systemStatus)
	mux.HandleFunc("POST /v1/refs", a.uploadRef)
	mux.HandleFunc("GET /v1/refs", a.listRefs)
	mux.HandleFunc("GET /v1/refs/{id}", a.getRef)
	mux.HandleFunc("DELETE /v1/refs/{id}", a.deleteRef)
	mux.HandleFunc("POST /v1/reserve", a.createReservation)
	mux.HandleFunc("GET /v1/reserve", a.listReservations)
	mux.HandleFunc("DELETE /v1/reserve/{id}", a.releaseReservation)
	mux.HandleFunc("POST /v1/models", a.registerModel)
	mux.HandleFunc("GET /v1/models", a.listModels)
	mux.HandleFunc("GET /v1/models/{model_id}", a.getModel)
	mux.HandleFunc("PATCH /v1/models/{model_id}", a.updateModel)
	mux.HandleFunc("DELETE /v1/models/{model_id}", a.removeModel)
	mux.HandleFunc("POST /v1/models/{model_id}/reload", a.reloadModel)
	mux.HandleFunc("DELETE /v1/models/{model_id}/queue", a.clearModelQueue)
	mux.HandleFunc("DELETE /v1/models/{model_id}/running", a.killModelRunning)
	mux.HandleFunc("DELETE /v1/models/{model_id}/workers", a.hardKillModelWorkers)
	mux.HandleFunc("POST /v1/llm/models", a.registerLLM)
	mux.HandleFunc("GET /v1/llm/models", a.listLLMs)
	mux.HandleFunc("DELETE /v1/llm/models/{name}", a.deregisterLLM)
	mux.HandleFunc("POST /v1/chat/completions", a.chatCompletion)
	mux.HandleFunc("PATCH /v1/remote", a.setGlobalRemote)
	mux.HandleFunc("GET /v1/remote", a.getGlobalRemote)
	mux.HandleFunc("POST /v1/drain", a.drain)
	mux.HandleFunc("POST /v1/admin/models/unload_all", a.adminUnloadAll)
	mux.HandleFunc("POST /v1/admin/models/preload", a.adminPreload)
	mux.HandleFunc("GET /v1/health", a.health)
	return withLogging(mux)
}

// RunPSCache updates the cached ps response every second.
func (a *API) RunPSCache(done <-chan struct{}) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			a.updatePSCache()
		}
	}
}

// psStatsInterval bounds how often the DB-derived /v1/ps aggregates (queue
// counts + completed-job stats) are recomputed. They aggregate over all job
// history, so per-second recomputation is pure waste; a status dashboard
// polling every few seconds does not need fresher-than-this historical data.
const psStatsInterval = 5 * time.Second

// refreshStats recomputes the DB-derived /v1/ps aggregates if the cached copy
// is older than psStatsInterval, in two grouped scans (one for counts, one for
// completed stats) covering every model at once. On query error it keeps the
// last good values rather than blanking the dashboard.
func (a *API) refreshStats() {
	a.statsMu.Lock()
	defer a.statsMu.Unlock()
	if !a.statsAt.IsZero() && time.Since(a.statsAt) < psStatsInterval {
		return
	}
	perModelCounts, globalCounts, err := a.store.CountByStateGrouped()
	if err != nil {
		return
	}
	perModelStats, globalStats, err := a.store.CompletedJobStatsGrouped()
	if err != nil {
		return
	}
	a.statsCounts = perModelCounts
	a.statsGlobalCt = globalCounts
	a.statsModel = perModelStats
	a.statsGlobal = globalStats
	a.statsAt = time.Now()
}

func (a *API) updatePSCache() {
	snap := a.mgr.Snapshot()

	// Add GPU utilization
	snap["gpu_utilization_pct"] = GetGPUUtilization()

	// Drain state + total in-flight jobs — so a deploy can poll for
	// "draining && active_jobs == 0" to know it is safe to bounce.
	snap["draining"] = a.scheduler.IsDraining()
	snap["active_jobs"] = a.mgr.TotalActiveJobs()

	// Queue counts + completed-job stats come from the throttled cache (see
	// refreshStats) so the per-second tick stays cheap.
	a.refreshStats()
	a.statsMu.Lock()
	globalCounts := a.statsGlobalCt
	globalStats := a.statsGlobal
	perModelCounts := a.statsCounts
	perModelStats := a.statsModel
	a.statsMu.Unlock()

	snap["queue"] = globalCounts
	snap["job_stats"] = map[string]any{
		"completed_jobs":        globalStats.Count,
		"avg_total_seconds":     globalStats.AvgTotal,
		"avg_execution_seconds": globalStats.AvgExec,
		"avg_waiting_seconds":   math.Max(globalStats.AvgTotal-globalStats.AvgExec, 0),
	}

	if models, ok := snap["models"].([]map[string]any); ok {
		for _, m := range models {
			if id, ok := m["id"].(string); ok {
				m["queued_jobs"] = perModelCounts[id]["queued"]
				st := perModelStats[id]
				m["completed_jobs"] = st.Count
				m["avg_total_seconds"] = st.AvgTotal
				m["avg_execution_seconds"] = st.AvgExec
				m["avg_waiting_seconds"] = math.Max(st.AvgTotal-st.AvgExec, 0)
				if cfg, ok := a.config.Models[id]; ok {
					m["max_instances"] = *cfg.MaxInstances
					m["max_concurrent"] = cfg.MaxConcurrent
				}
			}
		}
	}

	// SEPARATE remote-hosts panel — advisory used/budget, liveness, and the
	// models each host's ollama reports loaded. Deliberately disjoint from the
	// audited local VRAM numbers above: remote hosts hold ZERO bytes in spark's
	// usedGB / AuditVRAMConsistency ledger.
	if a.hostMonitor != nil {
		if panel := a.hostMonitor.RemoteHostsPanel(); len(panel) > 0 {
			snap["remote_hosts"] = panel
		}
	}

	data, _ := json.Marshal(snap)
	a.psCache.Store(data)
}

func (a *API) submitJob(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Type   string          `json:"type"`
		Model  string          `json:"model"`
		Params json.RawMessage `json:"params"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, "invalid request body")
		return
	}

	// If explicit model provided and configured, use it; otherwise fall back to type mapping
	var modelID string
	var ok bool
	if req.Model != "" {
		if _, exists := a.config.Models[req.Model]; exists {
			modelID = req.Model
			ok = true
		} else {
			writeError(w, 404, fmt.Sprintf("model not configured: %s", req.Model))
			return
		}
	} else {
		modelID, ok = JobTypeToModel[req.Type]
		// Also check for model inside params (callers may put it there)
		if ok {
			var pm struct {
				Model string `json:"model"`
			}
			json.Unmarshal(req.Params, &pm)
			if pm.Model != "" {
				if _, exists := a.config.Models[pm.Model]; exists {
					modelID = pm.Model
				}
			}
		}
	}
	if !ok && req.Type == "chat-completion" {
		// Generic chat-completion: resolve model from params
		var chatParams struct {
			Model string `json:"model"`
		}
		json.Unmarshal(req.Params, &chatParams)
		if chatParams.Model == "" {
			writeError(w, 400, "chat-completion requires model in params")
			return
		}
		modelID = llmModelID(chatParams.Model)
		if _, exists := a.config.Models[modelID]; !exists {
			writeError(w, 404, fmt.Sprintf("LLM not registered: %s", chatParams.Model))
			return
		}
		ok = true
	}
	if !ok {
		writeError(w, 400, fmt.Sprintf("unknown job type: %s", req.Type))
		return
	}
	if _, ok := a.config.Models[modelID]; !ok {
		writeError(w, 400, fmt.Sprintf("model not configured: %s", modelID))
		return
	}

	if req.Params == nil {
		req.Params = json.RawMessage("{}")
	}

	// --- Validate staged file paths ---
	// Any *_file param must either use the "ref:" prefix (managed reference file)
	// or point to a path inside ARBITER_INBOX_PATH (the shared NFS/SMB inbox).
	// Direct SCP to spark's local /tmp is not permitted — all files must be staged
	// via the arbiter-client service (which writes to the shared inbox mount).
	if inboxDir := strings.TrimRight(os.Getenv("ARBITER_INBOX_PATH"), "/"); inboxDir != "" {
		var params map[string]json.RawMessage
		if err := json.Unmarshal(req.Params, &params); err == nil {
			for key, raw := range params {
				if !strings.HasSuffix(key, "_file") {
					continue
				}
				var val string
				if err := json.Unmarshal(raw, &val); err != nil || val == "" || strings.HasPrefix(val, "ref:") {
					continue
				}
				if !strings.HasPrefix(filepath.Clean(val), inboxDir) && !strings.HasPrefix(filepath.Clean(val), a.outputDir) {
					writeError(w, 400, fmt.Sprintf(
						"invalid file path for param %q: %q is not inside the shared inbox %q. "+
							"All files must be staged via the arbiter-client service "+
							"(set ARBITER_CLIENT_URL=http://localhost:8401 and use arbiter_client.stage_file). "+
							"Direct SCP to spark is not permitted.",
						key, val, inboxDir,
					))
					return
				}
			}
		}
	}

	// --- Reject jobs whose declared input files don't exist ---
	// Bad paths must never reach the queue: a queued job with missing inputs
	// would trigger a model load, get dispatched, fail, and (before
	// isClientError) trip the inference circuit breaker, starving every
	// other job in the same model's queue. Reject at the door instead.
	if err := a.scheduler.ValidateJobInputs(req.Params); err != nil {
		writeError(w, 400, err.Error())
		return
	}

	// --- Dedup check ---
	var forceNew bool
	{
		var f struct {
			Force bool `json:"force"`
		}
		json.Unmarshal(req.Params, &f)
		forceNew = f.Force
	}
	var dedupHash string
	if !forceNew {
		dedupHash = computeJobHash(req.Type, req.Params)
		hash := dedupHash
		if origID, err := a.store.DedupLookup(hash, 86400); err == nil && origID != "" {
			origJob, _ := a.store.GetJob(origID)
			if origJob != nil {
				switch origJob.State {
				case "completed":
					// Instant cache hit — create pre-completed job. Store the
					// canonical job id in the DB instead of creating a
					// filesystem symlink: previous os.Symlink(origDir, newDir)
					// left dangling pointers on the CIFS mount when the orig
					// dir became unreachable, causing downstream jobs that
					// referenced /output/jobs/<new_id>/file to get EINVAL
					// from the broken symlink and fail their entire model's
					// queue via the inference circuit breaker.
					//
					// The result JSON we copy in already references the orig's
					// canonical output paths directly, so no aliasing is
					// needed for correctness — and now CountCanonicalReferences
					// gives output cleanup a way to know "don't delete this
					// orig dir, N followers depend on it".
					newJob, err := a.store.CreateJob(modelID, req.Type, req.Params, 0)
					if err == nil {
						if err := a.store.SetCanonicalJobID(newJob.ID, origID); err != nil {
							slog.Warn("dedup: failed to set canonical_job_id",
								"job", newJob.ID, "orig", origID, "error", err)
						}
						if origJob.Result != nil {
							a.store.UpdateState(newJob.ID, "completed", WithResult(*origJob.Result), WithFinishedAt(nowTS()))
						}
						a.logger.Log("job.dedup_hit", map[string]any{
							"job_id": newJob.ID, "original_id": origID, "type": "cached",
						})
						writeJSON(w, 200, map[string]any{
							"job_id": newJob.ID, "status": "completed",
							"model": modelID, "cached": true,
							"original_job_id": origID,
						})
						return
					}
				case "queued", "scheduled", "running":
					// In-flight — create follower
					follower, err := a.store.CreateFollowerJob(modelID, req.Type, req.Params, origID)
					if err == nil {
						a.logger.Log("job.dedup_hit", map[string]any{
							"job_id": follower.ID, "original_id": origID, "type": "following",
						})
						writeJSON(w, 200, map[string]any{
							"job_id": follower.ID, "status": "following",
							"model":           modelID,
							"original_job_id": origID,
						})
						return
					}
					// failed/cancelled: fall through to create new job
				}
			}
		}
		// hash saved for dedup registration after job creation
	}

	priority := a.scheduler.computePriority(modelID)
	job, err := a.store.CreateJob(modelID, req.Type, req.Params, priority)
	if err != nil {
		writeError(w, 500, fmt.Sprintf("create job: %s", err))
		return
	}

	cfg := a.config.Models[modelID]
	estimated := cfg.AvgInferenceMs
	if !a.mgr.IsLoaded(modelID) {
		estimated += cfg.LoadMs
	}

	if dedupHash != "" {
		a.store.DedupRegister(dedupHash, job.ID)
	}

	a.logger.Log("job.submitted", map[string]any{
		"job_id":   job.ID,
		"model_id": modelID,
		"job_type": req.Type,
		"priority": priority,
	})

	a.scheduler.Wake()

	writeJSON(w, 200, map[string]any{
		"job_id":            job.ID,
		"status":            "queued",
		"model":             modelID,
		"estimated_seconds": estimated / 1000,
	})
}

func (a *API) getJob(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("id")
	job, err := a.store.GetJob(jobID)
	if err != nil || job == nil {
		writeError(w, 404, fmt.Sprintf("job not found: %s", jobID))
		return
	}

	resp := map[string]any{
		"job_id":     job.ID,
		"status":     job.State,
		"model":      job.ModelID,
		"created_at": job.CreatedAt,
	}
	if job.StartedAt != nil {
		resp["started_at"] = *job.StartedAt
	}
	if job.FinishedAt != nil {
		resp["finished_at"] = *job.FinishedAt
	}
	if job.Error != "" {
		resp["error"] = job.Error
	}
	if job.Result != nil {
		var result map[string]any
		json.Unmarshal(*job.Result, &result)

		// Inline result file as base64 if present
		if job.State == "completed" && result != nil {
			if fmt, ok := result["format"].(string); ok && fmt != "" {
				resultFile := filepath.Join(resolveJobDir(a.config, a.outputDir, job.ID), "result."+fmt)
				result["result_path"] = resultFile
				skipData := r.URL.Query().Get("no_data") == "1"
				if !skipData {
					if data, err := os.ReadFile(resultFile); err == nil {
						result["data"] = base64.StdEncoding.EncodeToString(data)
					}
				}
			}
		}
		resp["result"] = result
	}

	writeJSON(w, 200, resp)
}

func (a *API) cancelJob(w http.ResponseWriter, r *http.Request) {
	jobID := r.PathValue("id")
	job, err := a.store.GetJob(jobID)
	if err != nil || job == nil {
		writeError(w, 404, fmt.Sprintf("job not found: %s", jobID))
		return
	}

	if job.State == "completed" || job.State == "failed" || job.State == "cancelled" {
		writeJSON(w, 200, map[string]any{
			"job_id":  jobID,
			"status":  job.State,
			"message": "job already finished",
		})
		return
	}

	// Try to cancel in store (queued/scheduled)
	cancelled, _ := a.store.CancelJob(jobID)
	if cancelled {
		if job.State != "following" {
			a.store.ResolveFollowers(job.ID, "cancelled", nil, "original cancelled by operator", a.outputDir)
		}
		writeJSON(w, 200, map[string]any{"job_id": jobID, "status": "cancelled"})
		return
	}

	// If running, find the instance and send cancel signal
	for _, inst := range a.mgr.GetModelInstances(job.ModelID) {
		inst.Cancel()
	}
	writeJSON(w, 200, map[string]any{"job_id": jobID, "status": "cancelling"})
}

func (a *API) bulkStatus(w http.ResponseWriter, r *http.Request) {
	var req struct {
		JobIDs []string `json:"job_ids"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, "invalid request body")
		return
	}
	if len(req.JobIDs) == 0 {
		writeJSON(w, 200, map[string]any{"jobs": []any{}})
		return
	}
	if len(req.JobIDs) > 1000 {
		writeError(w, 400, "max 1000 job IDs per request")
		return
	}

	jobs, err := a.store.GetJobs(req.JobIDs)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}

	// Return in request order, null for missing
	out := make([]any, len(req.JobIDs))
	for i, id := range req.JobIDs {
		j, ok := jobs[id]
		if !ok {
			out[i] = nil
			continue
		}
		entry := map[string]any{
			"job_id":     j.ID,
			"status":     j.State,
			"model":      j.ModelID,
			"type":       j.JobType,
			"created_at": j.CreatedAt,
		}
		if j.StartedAt != nil {
			entry["started_at"] = *j.StartedAt
		}
		if j.FinishedAt != nil {
			entry["finished_at"] = *j.FinishedAt
		}
		if j.Error != "" {
			entry["error"] = j.Error
		}
		if j.State == "completed" && j.Result != nil {
			var result map[string]any
			json.Unmarshal(*j.Result, &result)
			// Include result metadata but NOT file data (use GET /v1/jobs/{id} for that)
			delete(result, "data")
			entry["result"] = result
		}
		out[i] = entry
	}

	writeJSON(w, 200, map[string]any{"jobs": out})
}
func (a *API) listJobs(w http.ResponseWriter, r *http.Request) {
	state := r.URL.Query().Get("state")
	model := r.URL.Query().Get("model")
	limit := 100
	if l := r.URL.Query().Get("limit"); l != "" {
		if n, err := strconv.Atoi(l); err == nil {
			limit = n
		}
	}

	jobs, err := a.store.ListJobs(state, model, limit)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}

	var out []map[string]any
	for _, j := range jobs {
		entry := map[string]any{
			"job_id":     j.ID,
			"type":       j.JobType,
			"model":      j.ModelID,
			"status":     j.State,
			"created_at": j.CreatedAt,
		}
		if j.StartedAt != nil {
			entry["started_at"] = *j.StartedAt
		}
		if j.FinishedAt != nil {
			entry["finished_at"] = *j.FinishedAt
		}
		out = append(out, entry)
	}
	if out == nil {
		out = []map[string]any{}
	}
	writeJSON(w, 200, out)
}

func (a *API) systemStatus(w http.ResponseWriter, r *http.Request) {
	if cached := a.psCache.Load(); cached != nil {
		w.Header().Set("Content-Type", "application/json")
		w.Write(cached.([]byte))
		return
	}
	// Fallback before first cache update
	a.updatePSCache()
	if cached := a.psCache.Load(); cached != nil {
		w.Header().Set("Content-Type", "application/json")
		w.Write(cached.([]byte))
		return
	}
	writeJSON(w, 200, map[string]any{})
}

func (a *API) refsDir() string {
	return filepath.Join(a.outputDir, "refs")
}

func (a *API) uploadRef(w http.ResponseWriter, r *http.Request) {
	// Accept multipart (file field) or raw body (with ?filename= query param)
	var data []byte
	var filename string

	if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/") {
		if err := r.ParseMultipartForm(100 << 20); err != nil { // 100MB
			writeError(w, 400, fmt.Sprintf("parse multipart: %s", err))
			return
		}
		f, header, err := r.FormFile("file")
		if err != nil {
			writeError(w, 400, fmt.Sprintf("missing file field: %s", err))
			return
		}
		defer f.Close()
		filename = header.Filename
		data, err = io.ReadAll(f)
		if err != nil {
			writeError(w, 500, fmt.Sprintf("read file: %s", err))
			return
		}
	} else {
		var err error
		data, err = io.ReadAll(io.LimitReader(r.Body, 100<<20))
		if err != nil {
			writeError(w, 400, fmt.Sprintf("read body: %s", err))
			return
		}
		filename = r.URL.Query().Get("filename")
		if filename == "" {
			writeError(w, 400, "raw upload requires ?filename= query param")
			return
		}
	}

	if len(data) == 0 {
		writeError(w, 400, "empty file")
		return
	}

	ext := filepath.Ext(filename)
	refID := genID() + ext
	dst := filepath.Join(a.refsDir(), refID)
	if err := os.WriteFile(dst, data, 0o644); err != nil {
		writeError(w, 500, fmt.Sprintf("write ref: %s", err))
		return
	}

	slog.Info("ref uploaded", "ref_id", refID, "size", len(data), "filename", filename)
	writeJSON(w, 200, map[string]any{
		"ref_id":     refID,
		"size_bytes": len(data),
		"filename":   filename,
	})
}

func (a *API) getRef(w http.ResponseWriter, r *http.Request) {
	refID := r.PathValue("id")
	path := filepath.Join(a.refsDir(), refID)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		writeError(w, 404, "ref not found")
		return
	}
	http.ServeFile(w, r, path)
}

func (a *API) deleteRef(w http.ResponseWriter, r *http.Request) {
	refID := r.PathValue("id")
	path := filepath.Join(a.refsDir(), refID)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		writeError(w, 404, "ref not found")
		return
	}
	os.Remove(path)
	slog.Info("ref deleted", "ref_id", refID)
	writeJSON(w, 200, map[string]any{"ref_id": refID, "status": "deleted"})
}

func (a *API) listRefs(w http.ResponseWriter, r *http.Request) {
	entries, err := os.ReadDir(a.refsDir())
	if err != nil {
		writeJSON(w, 200, []any{})
		return
	}
	var refs []map[string]any
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		refs = append(refs, map[string]any{
			"ref_id":     e.Name(),
			"size_bytes": info.Size(),
			"created_at": info.ModTime().Unix(),
		})
	}
	if refs == nil {
		refs = []map[string]any{}
	}
	writeJSON(w, 200, refs)
}

func (a *API) createReservation(w http.ResponseWriter, r *http.Request) {
	var req struct {
		MemoryGB float64 `json:"memory_gb"`
		Label    string  `json:"label"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, "invalid request body")
		return
	}
	if req.MemoryGB <= 0 {
		writeError(w, 400, "memory_gb must be > 0")
		return
	}

	// Build keepalive map from config for smart eviction
	keepAliveSecs := make(map[string]int)
	for id, cfg := range a.config.Models {
		keepAliveSecs[id] = cfg.KeepAliveSec
	}

	id, err := a.mgr.CreateReservation(req.MemoryGB, req.Label, keepAliveSecs)
	if err != nil {
		writeError(w, 409, err.Error())
		return
	}

	a.logger.Log("reservation.create", map[string]any{
		"id":        id,
		"memory_gb": req.MemoryGB,
		"label":     req.Label,
	})

	writeJSON(w, 200, map[string]any{
		"reservation_id": id,
		"memory_gb":      req.MemoryGB,
		"label":          req.Label,
	})
}

func (a *API) releaseReservation(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !a.mgr.ReleaseReservation(id) {
		writeError(w, 404, fmt.Sprintf("reservation not found: %s", id))
		return
	}

	a.logger.Log("reservation.release", map[string]any{"id": id})
	writeJSON(w, 200, map[string]any{"reservation_id": id, "released": true})
}

func (a *API) listReservations(w http.ResponseWriter, r *http.Request) {
	reservations := a.mgr.ListReservations()
	out := make([]map[string]any, 0, len(reservations))
	for _, r := range reservations {
		out = append(out, map[string]any{
			"id":         r.ID,
			"memory_gb":  r.MemoryGB,
			"label":      r.Label,
			"created_at": r.CreatedAt.Unix(),
		})
	}
	writeJSON(w, 200, out)
}

func validateModelConfigRequest(req modelConfigRequest) error {
	if req.MemoryGB != nil && *req.MemoryGB <= 0 {
		return fmt.Errorf("memory_gb must be > 0")
	}
	if req.MaxConcurrent != nil && *req.MaxConcurrent < 1 {
		return fmt.Errorf("max_concurrent must be >= 1")
	}
	if req.MaxInstances != nil && *req.MaxInstances < 0 {
		return fmt.Errorf("max_instances must be >= 0")
	}
	if req.KeepAliveSec != nil && *req.KeepAliveSec < 0 {
		return fmt.Errorf("keep_alive_seconds must be >= 0")
	}
	if req.AvgInferenceMs != nil && *req.AvgInferenceMs < 0 {
		return fmt.Errorf("avg_inference_ms must be >= 0")
	}
	if req.LoadMs != nil && *req.LoadMs < 0 {
		return fmt.Errorf("load_ms must be >= 0")
	}
	if req.WorkerCmd != nil && len(*req.WorkerCmd) == 0 {
		return fmt.Errorf("worker_cmd must not be empty")
	}
	return nil
}

func (a *API) resolveConfiguredModelID(id string) (string, bool) {
	if _, ok := a.config.Models[id]; ok {
		return id, true
	}
	llmID := llmModelID(id)
	if _, ok := a.config.Models[llmID]; ok {
		return llmID, true
	}
	return "", false
}

func serializeModelConfig(modelID string, cfg ModelConfig) map[string]any {
	resp := map[string]any{
		"model_id":            modelID,
		"memory_gb":           cfg.MemoryGB,
		"max_concurrent":      cfg.MaxConcurrent,
		"keep_alive_seconds":  cfg.KeepAliveSec,
		"avg_inference_ms":    cfg.AvgInferenceMs,
		"load_ms":             cfg.LoadMs,
		"auto_download":       cfg.AutoDownload,
		"model_path":          cfg.ModelPath,
		"group":               cfg.Group,
		"worker_cmd":          cfg.WorkerCmd,
		"adapter_params":      cfg.AdapterParams,
		"pressure_index":      cfg.PressureIndex,
		"max_runtime_seconds": cfg.MaxRuntimeSec,
		"remote_enabled":      cfg.RemoteEnabledOrDefault(),
	}
	if len(cfg.Placements) > 0 {
		resp["placements"] = cfg.Placements
	}
	if cfg.MaxInstances != nil {
		resp["max_instances"] = *cfg.MaxInstances
	}
	if strings.HasPrefix(modelID, "llm:") {
		resp["llm_name"] = strings.TrimPrefix(modelID, "llm:")
	}
	return resp
}

func applyModelConfigRequest(cfg ModelConfig, req modelConfigRequest) ModelConfig {
	if req.MemoryGB != nil {
		cfg.MemoryGB = *req.MemoryGB
	}
	if req.MaxConcurrent != nil {
		cfg.MaxConcurrent = *req.MaxConcurrent
	}
	if req.MaxInstances != nil {
		n := *req.MaxInstances
		cfg.MaxInstances = &n
	}
	if req.KeepAliveSec != nil {
		cfg.KeepAliveSec = *req.KeepAliveSec
	}
	if req.AvgInferenceMs != nil {
		cfg.AvgInferenceMs = *req.AvgInferenceMs
	}
	if req.LoadMs != nil {
		cfg.LoadMs = *req.LoadMs
	}
	if req.AutoDownload != nil {
		cfg.AutoDownload = *req.AutoDownload
	}
	if req.ModelPath != nil {
		cfg.ModelPath = *req.ModelPath
	}
	if req.Group != nil {
		cfg.Group = *req.Group
	}
	if req.WorkerCmd != nil {
		cfg.WorkerCmd = cloneStrings(*req.WorkerCmd)
	}
	if req.AdapterParams != nil {
		merged := maps.Clone(cfg.AdapterParams)
		if merged == nil {
			merged = map[string]string{}
		}
		for k, v := range *req.AdapterParams {
			merged[k] = v
		}
		cfg.AdapterParams = merged
	}
	if req.PressureIndex != nil {
		v := *req.PressureIndex
		cfg.PressureIndex = &v
	}
	if req.MaxRuntimeSec != nil {
		cfg.MaxRuntimeSec = *req.MaxRuntimeSec
	}
	if req.ConflictGroup != nil {
		cfg.ConflictGroup = *req.ConflictGroup
	}
	if req.GroupPriority != nil {
		cfg.GroupPriority = *req.GroupPriority
	}
	if req.RemoteEnabled != nil {
		v := *req.RemoteEnabled
		cfg.RemoteEnabled = &v
	}
	return cfg
}

func (a *API) registerModel(w http.ResponseWriter, r *http.Request) {
	var req modelConfigRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, "invalid request body")
		return
	}
	if err := validateModelConfigRequest(req); err != nil {
		writeError(w, 400, err.Error())
		return
	}
	if req.ModelID == "" {
		writeError(w, 400, "model_id is required")
		return
	}
	if _, exists := a.config.Models[req.ModelID]; exists {
		writeError(w, 409, fmt.Sprintf("model already configured: %s", req.ModelID))
		return
	}

	one := 1
	cfg := ModelConfig{
		MaxConcurrent: 1,
		MaxInstances:  &one,
		KeepAliveSec:  300,
	}
	cfg = applyModelConfigRequest(cfg, req)
	a.config.Models[req.ModelID] = cfg
	a.mgr.EnsureModel(req.ModelID)

	scaleResult := a.mgr.ScaleModel(req.ModelID, *cfg.MaxInstances, cfg)
	a.mgr.ApplyModelConfig(req.ModelID, cfg)
	if err := SaveModelConfig(a.projectRoot, req.ModelID, cfg); err != nil {
		writeError(w, 500, fmt.Sprintf("persist model config: %s", err))
		return
	}

	a.scheduler.rescoreModel(req.ModelID)
	a.scheduler.Wake()
	a.logger.Log("model.registered", map[string]any{
		"model_id":       req.ModelID,
		"max_instances":  *cfg.MaxInstances,
		"max_concurrent": cfg.MaxConcurrent,
		"worker_cmd":     cfg.WorkerCmd,
	})

	writeJSON(w, 200, map[string]any{
		"model_id":       req.ModelID,
		"max_instances":  *cfg.MaxInstances,
		"max_concurrent": cfg.MaxConcurrent,
		"added":          scaleResult["added"],
		"status":         "registered",
	})
}

func (a *API) listModels(w http.ResponseWriter, r *http.Request) {
	models := make([]map[string]any, 0, len(a.config.Models))
	for modelID, cfg := range a.config.Models {
		models = append(models, serializeModelConfig(modelID, cfg))
	}
	if models == nil {
		models = []map[string]any{}
	}
	writeJSON(w, 200, models)
}

func (a *API) getModel(w http.ResponseWriter, r *http.Request) {
	modelID, ok := a.resolveConfiguredModelID(r.PathValue("model_id"))
	if !ok {
		writeError(w, 404, fmt.Sprintf("model not configured: %s", r.PathValue("model_id")))
		return
	}
	writeJSON(w, 200, serializeModelConfig(modelID, a.config.Models[modelID]))
}

func (a *API) updateModel(w http.ResponseWriter, r *http.Request) {
	modelID, ok := a.resolveConfiguredModelID(r.PathValue("model_id"))
	if !ok {
		writeError(w, 404, fmt.Sprintf("model not configured: %s", r.PathValue("model_id")))
		return
	}
	current, ok := a.config.Models[modelID]

	var req modelConfigRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, "invalid request body")
		return
	}
	if err := validateModelConfigRequest(req); err != nil {
		writeError(w, 400, err.Error())
		return
	}
	if req.MemoryGB != nil && *req.MemoryGB != current.MemoryGB && !req.ReloadWorkers && a.mgr.IsLoaded(modelID) {
		writeError(w, 400, "memory_gb changes require reload_workers=true while the model is loaded")
		return
	}

	updated := applyModelConfigRequest(current, req)
	a.config.Models[modelID] = updated
	a.mgr.ApplyModelConfig(modelID, updated)

	result := map[string]any{
		"model_id":                modelID,
		"max_instances":           *updated.MaxInstances,
		"max_concurrent":          updated.MaxConcurrent,
		"reload_workers":          req.ReloadWorkers,
		"previous_max_instances":  *current.MaxInstances,
		"previous_max_concurrent": current.MaxConcurrent,
	}

	var scaleResult map[string]any
	if req.ReloadWorkers {
		scaleResult = a.mgr.ReloadModel(modelID, *updated.MaxInstances, updated)
		a.logger.Log("model.reloaded", map[string]any{
			"model_id":  modelID,
			"added":     scaleResult["added"],
			"removed":   scaleResult["removed"],
			"condemned": scaleResult["condemned"],
		})
	} else if req.MaxInstances != nil && *req.MaxInstances != *current.MaxInstances {
		scaleResult = a.mgr.ScaleModel(modelID, *updated.MaxInstances, updated)
		a.logger.Log("model.scaled", map[string]any{
			"model_id":          modelID,
			"old_max_instances": *current.MaxInstances,
			"new_max_instances": *updated.MaxInstances,
			"added":             scaleResult["added"],
			"removed":           scaleResult["removed"],
			"condemned":         scaleResult["condemned"],
		})
	}

	if scaleResult != nil {
		result["added"] = scaleResult["added"]
		result["removed"] = scaleResult["removed"]
		result["condemned"] = scaleResult["condemned"]
	}
	if err := SaveModelConfig(a.projectRoot, modelID, updated); err != nil {
		writeError(w, 500, fmt.Sprintf("persist model config: %s", err))
		return
	}

	a.scheduler.rescoreModel(modelID)
	a.scheduler.Wake()
	if req.MaxConcurrent != nil && *req.MaxConcurrent != current.MaxConcurrent {
		a.logger.Log("model.concurrency_changed", map[string]any{
			"model_id": modelID, "old": current.MaxConcurrent, "new": updated.MaxConcurrent,
		})
	}
	// Remote kill-switch: when remote_enabled flips to false, pin the model to
	// spark. New jobs already route to spark (PickInstanceForJob honors the
	// persisted flag above); drain any IN-FLIGHT remote job back to spark via the
	// transparent-failover path. Instant, ONE curl, works even if the remote host
	// is unreachable (Cancel is a local channel close, no network round-trip).
	if req.RemoteEnabled != nil && !*req.RemoteEnabled {
		drained := a.scheduler.DrainRemoteJobsForModel(modelID)
		result["remote_enabled"] = false
		result["drained_remote_jobs"] = drained
		a.logger.Log("model.remote_disabled", map[string]any{
			"model_id":            modelID,
			"drained_remote_jobs": drained,
		})
	} else if req.RemoteEnabled != nil && *req.RemoteEnabled {
		result["remote_enabled"] = true
		a.logger.Log("model.remote_enabled", map[string]any{"model_id": modelID})
		a.scheduler.Wake()
	}
	if !req.ReloadWorkers && (req.WorkerCmd != nil || req.AdapterParams != nil) {
		result["message"] = "config updated; existing loaded workers keep running until this model is reloaded"
	} else if scaleResult == nil && req.MaxConcurrent == nil && req.MemoryGB == nil &&
		req.KeepAliveSec == nil && req.AvgInferenceMs == nil && req.LoadMs == nil &&
		req.WorkerCmd == nil && req.AdapterParams == nil && req.AutoDownload == nil &&
		req.ModelPath == nil && req.Group == nil && req.MaxInstances == nil &&
		req.ConflictGroup == nil && req.GroupPriority == nil && req.RemoteEnabled == nil {
		result["message"] = "no changes"
	}
	writeJSON(w, 200, result)
}

func (a *API) reloadModel(w http.ResponseWriter, r *http.Request) {
	modelID, ok := a.resolveConfiguredModelID(r.PathValue("model_id"))
	if !ok {
		writeError(w, 404, fmt.Sprintf("model not configured: %s", r.PathValue("model_id")))
		return
	}
	cfg := a.config.Models[modelID]
	scaleResult := a.mgr.ReloadModel(modelID, *cfg.MaxInstances, cfg)
	a.scheduler.rescoreModel(modelID)
	a.scheduler.Wake()
	a.logger.Log("model.reloaded", map[string]any{
		"model_id":  modelID,
		"added":     scaleResult["added"],
		"removed":   scaleResult["removed"],
		"condemned": scaleResult["condemned"],
	})
	writeJSON(w, 200, map[string]any{
		"model_id":  modelID,
		"added":     scaleResult["added"],
		"removed":   scaleResult["removed"],
		"condemned": scaleResult["condemned"],
		"status":    "reloaded",
	})
}

func (a *API) clearModelQueue(w http.ResponseWriter, r *http.Request) {
	modelID, ok := a.resolveConfiguredModelID(r.PathValue("model_id"))
	if !ok {
		writeError(w, 404, fmt.Sprintf("model not configured: %s", r.PathValue("model_id")))
		return
	}

	cancelled, err := a.store.CancelQueuedForModel(modelID)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	cancelledFollowing, err := a.store.CancelFollowingForModel(modelID, "cancelled by operator while waiting on deduped original")
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}

	a.logger.Log("queue.cleared", map[string]any{
		"model_id":            modelID,
		"cancelled":           cancelled,
		"cancelled_following": cancelledFollowing,
	})

	writeJSON(w, 200, map[string]any{
		"model_id":            modelID,
		"cancelled":           cancelled,
		"cancelled_following": cancelledFollowing,
	})
}

func (a *API) hardKillModelWorkers(w http.ResponseWriter, r *http.Request) {
	modelID, ok := a.resolveConfiguredModelID(r.PathValue("model_id"))
	if !ok {
		writeError(w, 404, fmt.Sprintf("model not configured: %s", r.PathValue("model_id")))
		return
	}
	cfg, ok := a.config.Models[modelID]

	cancelledQueued, err := a.store.CancelQueuedForModel(modelID)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	cancelledFollowing, err := a.store.CancelFollowingForModel(modelID, "adapter hard-killed by operator while waiting on deduped original")
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	failedActive, err := a.store.FailActiveForModel(modelID, "adapter hard-killed by operator")
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}

	killResult := a.mgr.HardKillModel(modelID, true, &cfg)
	a.scheduler.rescoreModel(modelID)
	a.scheduler.Wake()

	a.logger.Log("model.hard_killed", map[string]any{
		"model_id":            modelID,
		"cancelled_queued":    cancelledQueued,
		"cancelled_following": cancelledFollowing,
		"failed_active":       failedActive,
		"killed":              killResult["killed"],
		"recreated":           killResult["recreated"],
	})

	writeJSON(w, 200, map[string]any{
		"model_id":            modelID,
		"cancelled_queued":    cancelledQueued,
		"cancelled_following": cancelledFollowing,
		"failed_active":       failedActive,
		"killed_workers":      killResult["killed"],
		"recreated":           killResult["recreated"],
		"status":              "hard_killed",
	})
}

func (a *API) killModelRunning(w http.ResponseWriter, r *http.Request) {
	modelID, ok := a.resolveConfiguredModelID(r.PathValue("model_id"))
	if !ok {
		writeError(w, 404, fmt.Sprintf("model not configured: %s", r.PathValue("model_id")))
		return
	}

	// Cancel queued/scheduled jobs in the store
	cancelledQueued, _ := a.store.CancelQueuedForModel(modelID)
	cancelledFollowing, _ := a.store.CancelFollowingForModel(modelID, "cancelled by operator while waiting on deduped original")

	// Send cancel signal to all instances (kills running inference)
	instances := a.mgr.GetModelInstances(modelID)
	cancelledRunning := 0
	for _, inst := range instances {
		if inst.ActiveJobs() > 0 {
			inst.Cancel()
			cancelledRunning += inst.ActiveJobs()
		}
	}

	a.logger.Log("model.killed", map[string]any{
		"model_id":            modelID,
		"cancelled_queued":    cancelledQueued,
		"cancelled_following": cancelledFollowing,
		"cancelled_running":   cancelledRunning,
	})

	writeJSON(w, 200, map[string]any{
		"model_id":            modelID,
		"cancelled_queued":    cancelledQueued,
		"cancelled_following": cancelledFollowing,
		"cancelled_running":   cancelledRunning,
	})
}

func removeJobTypeMappings(modelID string) []string {
	var removed []string
	for jobType, mappedModelID := range JobTypeToModel {
		if mappedModelID == modelID {
			delete(JobTypeToModel, jobType)
			removed = append(removed, jobType)
		}
	}
	return removed
}

func (a *API) removeModel(w http.ResponseWriter, r *http.Request) {
	modelID, ok := a.resolveConfiguredModelID(r.PathValue("model_id"))
	if !ok {
		writeError(w, 404, fmt.Sprintf("model not configured: %s", r.PathValue("model_id")))
		return
	}
	cfg, ok := a.config.Models[modelID]

	force := r.URL.Query().Get("force") == "1" || r.URL.Query().Get("force") == "true"
	counts, err := a.store.CountByState(modelID)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	activeOrQueued := counts["queued"] + counts["scheduled"] + counts["running"] + counts["following"]
	if activeOrQueued > 0 && !force {
		writeError(w, 409, "model has queued or active jobs; retry with ?force=1 to remove it")
		return
	}

	cancelledQueued := 0
	cancelledFollowing := 0
	failedActive := 0
	if force {
		cancelledQueued, err = a.store.CancelQueuedForModel(modelID)
		if err != nil {
			writeError(w, 500, err.Error())
			return
		}
		cancelledFollowing, err = a.store.CancelFollowingForModel(modelID, "adapter removed by operator while waiting on deduped original")
		if err != nil {
			writeError(w, 500, err.Error())
			return
		}
		failedActive, err = a.store.FailActiveForModel(modelID, "adapter removed by operator")
		if err != nil {
			writeError(w, 500, err.Error())
			return
		}
	}

	killResult := a.mgr.HardKillModel(modelID, false, &cfg)
	delete(a.config.Models, modelID)
	removedJobTypes := removeJobTypeMappings(modelID)
	if err := DeleteModelConfig(a.projectRoot, modelID); err != nil {
		writeError(w, 500, fmt.Sprintf("delete model config: %s", err))
		return
	}

	a.logger.Log("model.removed", map[string]any{
		"model_id":            modelID,
		"force":               force,
		"removed_job_types":   removedJobTypes,
		"cancelled_queued":    cancelledQueued,
		"cancelled_following": cancelledFollowing,
		"failed_active":       failedActive,
		"killed":              killResult["killed"],
	})

	writeJSON(w, 200, map[string]any{
		"model_id":            modelID,
		"force":               force,
		"removed_job_types":   removedJobTypes,
		"cancelled_queued":    cancelledQueued,
		"cancelled_following": cancelledFollowing,
		"failed_active":       failedActive,
		"killed_workers":      killResult["killed"],
		"status":              "removed",
	})
}

// --- LLM Management ---

func llmModelID(name string) string {
	return "llm:" + name
}

func llmWorkerBin(projectRoot string) string {
	// Try built binary first
	bin := filepath.Join(projectRoot, "llm-worker")
	if _, err := os.Stat(bin); err == nil {
		return bin
	}
	return "llm-worker" // hope it's in PATH
}

func defaultLLMWorkerBin(backend, projectRoot string) string {
	if backend == "vllm" {
		return vllmChatWorkerBin(projectRoot)
	}
	return llmWorkerBin(projectRoot)
}

func vllmChatWorkerBin(projectRoot string) string {
	bin := filepath.Join(projectRoot, "vllm-chat-worker")
	if _, err := os.Stat(bin); err == nil {
		return bin
	}
	return "vllm-chat-worker"
}

func estimateMemoryGB(totalParams int64) float64 {
	// fp16: 2 bytes per param + 20% overhead, rounded up to nearest 5GB
	gb := float64(totalParams) * 2.0 / (1024 * 1024 * 1024) * 1.2
	return math.Ceil(gb/5) * 5
}

func (a *API) registerLLM(w http.ResponseWriter, r *http.Request) {
	var req llmRegisterRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, "invalid request body")
		return
	}
	backend := strings.ToLower(strings.TrimSpace(req.Backend))
	if backend == "" {
		backend = "llamacpp"
	}
	if backend != "llamacpp" && backend != "vllm" {
		writeError(w, 400, "backend must be 'llamacpp' or 'vllm'")
		return
	}
	if backend == "llamacpp" && req.HFModel == "" && req.ModelPath == "" {
		writeError(w, 400, "hf_model or model_path required")
		return
	}
	if backend == "vllm" && req.VllmModel == "" && req.HFModel == "" && req.ModelPath == "" {
		writeError(w, 400, "vllm_model, hf_model, or model_path required for vllm backend")
		return
	}

	// Derive name
	name := req.Name
	if name == "" {
		if req.HFModel != "" {
			parts := strings.Split(req.HFModel, "/")
			name = parts[len(parts)-1]
			// Strip -GGUF suffix
			name = strings.TrimSuffix(name, "-GGUF")
			name = strings.TrimSuffix(name, "-gguf")
		} else {
			// Use filename without extension
			base := filepath.Base(req.ModelPath)
			name = strings.TrimSuffix(base, filepath.Ext(base))
		}
	}

	modelID := llmModelID(name)

	// Check if already registered
	if _, ok := a.config.Models[modelID]; ok {
		writeJSON(w, 200, map[string]any{
			"model_id": modelID,
			"name":     name,
			"status":   "already_registered",
		})
		return
	}

	// Estimate memory if not provided
	memGB := req.MemoryGB
	if memGB == 0 {
		// Default conservative estimate: 45GB for a 20B model
		memGB = 45
		slog.Warn("no memory_gb specified for LLM, using default", "model", name, "memory_gb", memGB)
	}

	// Build adapter params (env vars for the worker). The set of env vars
	// depends on the backend — llama.cpp consumes LLM_*, vllm consumes VLLM_*.
	adapterParams := make(map[string]string)
	ctx := req.CtxSize
	if ctx == 0 {
		ctx = 8192
	}
	if backend == "llamacpp" {
		if req.HFModel != "" {
			adapterParams["LLM_HF_REPO"] = req.HFModel
		}
		if req.HFFile != "" {
			adapterParams["LLM_HF_FILE"] = req.HFFile
		}
		if req.ModelPath != "" {
			adapterParams["LLM_MODEL_PATH"] = req.ModelPath
		}
		adapterParams["LLM_CTX_SIZE"] = strconv.Itoa(ctx)
		gpuLayers := req.GPULayers
		if gpuLayers == 0 {
			gpuLayers = -1
		}
		adapterParams["LLM_GPU_LAYERS"] = strconv.Itoa(gpuLayers)
		if req.LlamaServerBin != "" {
			adapterParams["LLAMA_SERVER_BIN"] = req.LlamaServerBin
		}
	} else { // vllm
		vmodel := req.VllmModel
		if vmodel == "" {
			if req.HFModel != "" && req.HFFile != "" {
				vmodel = req.HFModel + ":" + req.HFFile
			} else if req.HFModel != "" {
				vmodel = req.HFModel
			} else {
				vmodel = req.ModelPath
			}
		}
		adapterParams["VLLM_MODEL"] = vmodel
		extra := strings.TrimSpace(req.VllmExtraArgs)
		// Inject context size if caller didn't explicitly set --max-model-len.
		if !strings.Contains(extra, "--max-model-len") {
			if extra == "" {
				extra = "--max-model-len " + strconv.Itoa(ctx)
			} else {
				extra = "--max-model-len " + strconv.Itoa(ctx) + " " + extra
			}
		}
		adapterParams["VLLM_EXTRA_ARGS"] = extra
		adapterParams["LLM_CTX_SIZE"] = strconv.Itoa(ctx) // for visibility/inspection
	}
	adapterParams["LLM_BACKEND"] = backend
	for k, v := range req.AdapterParams {
		adapterParams[k] = v
	}

	// Register in config
	one := 1
	cfg := ModelConfig{
		MemoryGB:       memGB,
		MaxConcurrent:  1,
		MaxInstances:   &one,
		KeepAliveSec:   3600,
		MaxRuntimeSec:  600,
		AvgInferenceMs: 5000,
		LoadMs:         120000, // LLMs can take a while to download + load
		WorkerCmd:      []string{defaultLLMWorkerBin(backend, a.projectRoot)},
		AdapterParams:  adapterParams,
	}
	if len(req.WorkerCmd) > 0 {
		cfg.WorkerCmd = cloneStrings(req.WorkerCmd)
	}
	if req.MaxConcurrent != nil {
		cfg.MaxConcurrent = *req.MaxConcurrent
	}
	if req.MaxInstances != nil {
		n := *req.MaxInstances
		cfg.MaxInstances = &n
	}
	if req.KeepAliveSec != nil {
		cfg.KeepAliveSec = *req.KeepAliveSec
	}
	if req.MaxRuntimeSec != nil {
		cfg.MaxRuntimeSec = *req.MaxRuntimeSec
	}
	if req.AvgInferenceMs != nil {
		cfg.AvgInferenceMs = *req.AvgInferenceMs
	}
	if req.LoadMs != nil {
		cfg.LoadMs = *req.LoadMs
	}
	a.config.Models[modelID] = cfg

	// Register job type mapping
	JobTypeToModel["chat-completion:"+name] = modelID

	// Create instance
	result := a.mgr.ScaleModel(modelID, *cfg.MaxInstances, cfg)

	// Persist
	if err := SaveModelConfig(a.projectRoot, modelID, cfg); err != nil {
		slog.Error("failed to persist LLM config", "error", err)
	}

	a.scheduler.rescoreModel(modelID)

	a.logger.Log("llm.registered", map[string]any{
		"model_id":  modelID,
		"name":      name,
		"hf_model":  req.HFModel,
		"memory_gb": memGB,
	})

	writeJSON(w, 200, map[string]any{
		"model_id":  modelID,
		"name":      name,
		"memory_gb": memGB,
		"status":    "registered",
		"added":     result["added"],
	})
}

func (a *API) listLLMs(w http.ResponseWriter, r *http.Request) {
	var llms []map[string]any
	for id, cfg := range a.config.Models {
		if !strings.HasPrefix(id, "llm:") {
			continue
		}
		entry := serializeModelConfig(id, cfg)
		entry["name"] = strings.TrimPrefix(id, "llm:")
		llms = append(llms, entry)
	}
	if llms == nil {
		llms = []map[string]any{}
	}
	writeJSON(w, 200, llms)
}

func (a *API) deregisterLLM(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	modelID := llmModelID(name)

	cfg, ok := a.config.Models[modelID]
	if !ok {
		writeError(w, 404, fmt.Sprintf("LLM not registered: %s", name))
		return
	}

	killResult := a.mgr.HardKillModel(modelID, false, &cfg)
	delete(a.config.Models, modelID)
	delete(JobTypeToModel, "chat-completion:"+name)

	if err := DeleteModelConfig(a.projectRoot, modelID); err != nil {
		slog.Error("failed to delete LLM config", "model_id", modelID, "error", err)
	}

	a.logger.Log("llm.deregistered", map[string]any{"model_id": modelID, "name": name, "killed": killResult["killed"]})
	writeJSON(w, 200, map[string]any{"model_id": modelID, "name": name, "killed_workers": killResult["killed"], "status": "deregistered"})
}

func (a *API) chatCompletion(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	body, _ := io.ReadAll(r.Body)
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, 400, "invalid request body")
		return
	}
	if req.Model == "" {
		writeError(w, 400, "model field required")
		return
	}

	modelID := llmModelID(req.Model)
	if _, ok := a.config.Models[modelID]; !ok {
		writeError(w, 404, fmt.Sprintf("LLM not registered: %s (register via POST /v1/llm/models)", req.Model))
		return
	}

	// Reject if any declared input files don't exist (no-op for typical
	// chat payloads; safety net for tool-using flows that reference files).
	if err := a.scheduler.ValidateJobInputs(json.RawMessage(body)); err != nil {
		writeError(w, 400, err.Error())
		return
	}

	// Streaming: proxy directly to llama-server for SSE support
	if req.Stream {
		a.chatCompletionStream(w, r, modelID, body)
		return
	}

	// Non-streaming: submit as a regular arbiter job and wait synchronously
	priority := a.scheduler.computePriority(modelID)
	job, err := a.store.CreateJob(modelID, "chat-completion", json.RawMessage(body), priority)
	if err != nil {
		writeError(w, 500, fmt.Sprintf("create job: %s", err))
		return
	}
	a.scheduler.Wake()

	// Wait for completion (synchronous)
	timeout := time.After(15 * time.Minute)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			writeError(w, 504, "chat completion timed out")
			return
		case <-ticker.C:
			j, _ := a.store.GetJob(job.ID)
			if j == nil {
				continue
			}
			switch j.State {
			case "completed":
				if j.Result != nil {
					var result map[string]any
					json.Unmarshal(*j.Result, &result)
					// Return the OpenAI response directly
					if resp, ok := result["response"]; ok {
						w.Header().Set("Content-Type", "application/json")
						if raw, ok := resp.(json.RawMessage); ok {
							w.Write(raw)
						} else {
							json.NewEncoder(w).Encode(resp)
						}
						return
					}
					writeJSON(w, 200, result)
					return
				}
				writeJSON(w, 200, map[string]any{"error": "no result"})
				return
			case "failed":
				writeError(w, 500, j.Error)
				return
			case "cancelled":
				writeError(w, 499, "request cancelled")
				return
			}
		}
	}
}

// chatCompletionStream enqueues a chat-completion-stream job and waits for
// the scheduler to dispatch it. The scheduler calls dispatchStreamHandoff,
// which hands the picked instance back to this handler via the handoff
// registry. This handler then proxies SSE from the worker to the client and
// signals completion. There is one queue and one MaxConcurrent — streaming
// and non-streaming jobs share it.
func (a *API) chatCompletionStream(w http.ResponseWriter, r *http.Request, modelID string, body []byte) {
	priority := a.scheduler.computePriority(modelID)
	job, err := a.store.CreateJob(modelID, "chat-completion-stream", json.RawMessage(body), priority)
	if err != nil {
		writeError(w, 500, fmt.Sprintf("create job: %s", err))
		return
	}
	handoff := a.scheduler.RegisterStreamHandoff(job.ID)
	a.scheduler.Wake()

	// Wait for the scheduler to dispatch (i.e. for a slot to open under
	// MaxConcurrent). Honour client cancellation so we don't hold a slot
	// for a client that gave up.
	var inst *Instance
	select {
	case inst = <-handoff.instCh:
	case <-r.Context().Done():
		// Client gave up before we got a slot. The scheduler's handoff wait
		// will time out; mark the job cancelled now so it doesn't sit as
		// "running" forever once dispatched.
		a.scheduler.UnregisterStreamHandoff(job.ID)
		a.store.UpdateState(job.ID, "cancelled", WithFinishedAt(nowTS()))
		return
	case <-time.After(15 * time.Minute):
		a.scheduler.UnregisterStreamHandoff(job.ID)
		a.store.UpdateState(job.ID, "failed",
			WithError("queued >15min waiting for slot"), WithFinishedAt(nowTS()))
		writeError(w, 504, "chat completion timed out waiting for slot")
		return
	}

	// Got a slot. Proxy SSE from the worker to the client, then signal done.
	streamErr := proxyStreamFromInstance(w, r, inst, body)
	handoff.doneCh <- streamErr
}

// proxyStreamFromInstance pipes a chat completion to the client as SSE. For a
// LOCAL instance it proxies live SSE from the worker's llama-server/vLLM port.
// For a REMOTE instance it uses the buffer-on-remote/replay-from-spark path:
// the full completion is fetched over HTTP (non-streamed, on a detached
// context), then replayed locally as SSE chunks. Returns any error so the
// scheduler can mark the job appropriately — critically, a remote absence error
// is returned BEFORE any client byte is written, so the scheduler can fail the
// job over to the next box invisibly to the client.
func proxyStreamFromInstance(w http.ResponseWriter, r *http.Request, inst *Instance, body []byte) error {
	if inst.isRemote() {
		return replayRemoteStream(w, inst, body)
	}
	port, err := inst.GetPort()
	if err != nil {
		writeError(w, 500, fmt.Sprintf("get worker port: %s", err))
		return err
	}
	target := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port)
	proxyReq, err := http.NewRequestWithContext(r.Context(), "POST", target, bytes.NewReader(body))
	if err != nil {
		writeError(w, 500, fmt.Sprintf("create proxy request: %s", err))
		return err
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Minute}
	resp, err := client.Do(proxyReq)
	if err != nil {
		writeError(w, 502, fmt.Sprintf("worker error: %s", err))
		return err
	}
	defer resp.Body.Close()

	for k, vv := range resp.Header {
		for _, v := range vv {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)

	flusher, ok := w.(http.Flusher)
	if !ok {
		_, copyErr := io.Copy(w, resp.Body)
		return copyErr
	}
	buf := make([]byte, 4096)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
			flusher.Flush()
		}
		if rerr != nil {
			if rerr == io.EOF {
				return nil
			}
			return rerr
		}
	}
}

// replayRemoteStream implements the buffer-on-remote/replay-from-spark path. It
// fetches the FULL completion from the remote backend (non-streamed, detached
// context — a slow call drains, it is never cancelled by the client), then emits
// it to the client as a valid OpenAI SSE stream from spark.
//
// Failover safety: the remote call is awaited BEFORE any byte is written to w.
// If it errors (absence), we return the error without touching w, so the
// scheduler's stream-handoff path can requeue to the next box and the client —
// which has received nothing — sees no break. Only once we have the full,
// successful completion do we start writing SSE.
func replayRemoteStream(w http.ResponseWriter, inst *Instance, body []byte) error {
	resp, err := inst.backend.InferRaw("stream-"+genID(), "chat-completion-stream", json.RawMessage(body), "")
	if err != nil {
		// No client bytes written yet → caller can fail this over transparently.
		return err
	}
	if resp == nil || resp.Status != "ok" {
		errMsg := "remote stream produced no result"
		if resp != nil && resp.Error != "" {
			errMsg = resp.Error
		}
		return fmt.Errorf("%s", errMsg)
	}

	// Pull the full OpenAI response out of the worker-shaped result.
	var result struct {
		Response json.RawMessage `json:"response"`
	}
	json.Unmarshal(resp.Result, &result)
	full := result.Response
	if len(full) == 0 {
		full = resp.Result
	}

	var chatResp struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	json.Unmarshal(full, &chatResp)

	content := ""
	finish := "stop"
	if len(chatResp.Choices) > 0 {
		content = chatResp.Choices[0].Message.Content
		if chatResp.Choices[0].FinishReason != "" {
			finish = chatResp.Choices[0].FinishReason
		}
	}
	id := chatResp.ID
	if id == "" {
		id = "chatcmpl-" + genID()
	}
	created := chatResp.Created
	if created == 0 {
		created = time.Now().Unix()
	}
	model := chatResp.Model

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)

	writeChunk := func(delta map[string]any, finishReason any) {
		chunk := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         delta,
				"finish_reason": finishReason,
			}},
		}
		data, _ := json.Marshal(chunk)
		w.Write([]byte("data: "))
		w.Write(data)
		w.Write([]byte("\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
	}

	// Role chunk, then the content as a single delta, then the finish chunk and
	// the SSE terminator — a valid OpenAI stream the client can consume normally.
	writeChunk(map[string]any{"role": "assistant"}, nil)
	if content != "" {
		writeChunk(map[string]any{"content": content}, nil)
	}
	writeChunk(map[string]any{}, finish)
	w.Write([]byte("data: [DONE]\n\n"))
	if flusher != nil {
		flusher.Flush()
	}
	return nil
}

// --- Admin ---
//
// Admin endpoints let the operator unload all models (clean baseline) and
// preload a single model without running a job (fair load-time measurement).

// getGlobalRemote reports the global remote kill-switch state.
func (a *API) getGlobalRemote(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{"enabled": !a.config.RemoteDisabled})
}

// setGlobalRemote implements PATCH /v1/remote — the GLOBAL remote kill-switch.
//
//	{"enabled":false} → no model uses any remote placement; ALL jobs pin to
//	                    spark, and every in-flight remote job is drained to spark.
//	{"enabled":true}  → restore remote routing (per-model flags still apply).
//
// Instant, ONE curl, works even when remote hosts are unreachable: new routing
// honors the flag immediately and the drain is a local channel-close per job.
func (a *API) setGlobalRemote(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Enabled *bool `json:"enabled"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Enabled == nil {
		writeError(w, 400, "body must be {\"enabled\":true|false}")
		return
	}
	disabled := !*req.Enabled
	a.config.RemoteDisabled = disabled
	if err := PatchRemoteDisabled(a.projectRoot, disabled); err != nil {
		writeError(w, 500, fmt.Sprintf("persist global remote flag: %s", err))
		return
	}

	drained := 0
	if disabled {
		// Drain in-flight remote work across EVERY model to spark.
		for modelID := range a.config.Models {
			drained += a.scheduler.DrainRemoteJobsForModel(modelID)
		}
	}
	a.scheduler.Wake()
	a.logger.Log("remote.global_toggle", map[string]any{
		"enabled":             *req.Enabled,
		"drained_remote_jobs": drained,
	})
	writeJSON(w, 200, map[string]any{
		"enabled":             *req.Enabled,
		"drained_remote_jobs": drained,
	})
}

// drain implements POST /v1/drain — graceful, job-safe wind-down.
//
//	{}                          → enter drain mode: start no new jobs, let
//	                              in-flight jobs finish, keep queued work for
//	                              after a restart. Never kills a running job.
//	{"shutdown_when_idle":true} → also exit the process gracefully once the
//	                              last in-flight job completes (one-shot safe
//	                              shutdown for redeploys).
//	{"resume":true}             → leave drain mode, resume normal dispatch.
//
// Poll GET /v1/ps for {"draining":true,"active_jobs":0} to know it is safe to
// bounce.
func (a *API) drain(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Resume           bool `json:"resume"`
		ShutdownWhenIdle bool `json:"shutdown_when_idle"`
	}
	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&req) // body is optional
	}

	if req.Resume {
		a.scheduler.SetDraining(false)
		a.logger.Log("scheduler.drain", map[string]any{"draining": false})
		writeJSON(w, 200, map[string]any{"draining": false})
		return
	}

	a.scheduler.SetDraining(true)
	active := a.mgr.TotalActiveJobs()
	a.logger.Log("scheduler.drain", map[string]any{
		"draining": true, "active_jobs": active, "shutdown_when_idle": req.ShutdownWhenIdle,
	})

	if req.ShutdownWhenIdle {
		a.startDrainShutdownMonitor()
	}

	writeJSON(w, 200, map[string]any{
		"draining":           true,
		"active_jobs":        active,
		"shutdown_when_idle": req.ShutdownWhenIdle,
		"message":            "no new jobs will start; in-flight jobs will finish. poll GET /v1/ps for draining && active_jobs==0",
	})
}

// startDrainShutdownMonitor launches (at most once) a goroutine that waits for
// all in-flight jobs to finish, then triggers a graceful process shutdown.
// Aborts if drain mode is resumed before idle.
func (a *API) startDrainShutdownMonitor() {
	a.drainShutdownOnce.Do(func() {
		go func() {
			for {
				time.Sleep(2 * time.Second)
				if !a.scheduler.IsDraining() {
					slog.Info("drain resumed before idle — cancelling shutdown monitor")
					return
				}
				if a.mgr.TotalActiveJobs() == 0 {
					slog.Info("drain complete — no in-flight jobs, shutting down")
					a.logger.Log("scheduler.drain_shutdown", map[string]any{})
					if a.requestShutdown != nil {
						a.requestShutdown()
					}
					return
				}
			}
		}()
	})
}

func (a *API) adminUnloadAll(w http.ResponseWriter, r *http.Request) {
	totalKilled := 0
	models := make([]string, 0, len(a.config.Models))
	for id := range a.config.Models {
		models = append(models, id)
	}
	// recreate=true preserves the instance shells (so subsequent preload still
	// finds an instance to load into) but kills the running workers and frees
	// VRAM/RSS — exactly the "clean baseline" benchmark mode wants.
	for _, id := range models {
		cfg := a.config.Models[id]
		res := a.mgr.HardKillModel(id, true, &cfg)
		if k, ok := res["killed"].(int); ok {
			totalKilled += k
		}
	}
	a.logger.Log("admin.unload_all", map[string]any{"killed": totalKilled})
	writeJSON(w, 200, map[string]any{"killed_workers": totalKilled, "models_count": len(models)})
}

// cachedWorkerPort returns the worker's HTTP port, looking it up via the
// stdin/stdout protocol on first access and caching subsequent lookups.
// Concurrent benchmark requests would otherwise race on the protocol's
// non-unique default reqID and corrupt each other's responses.
func (a *API) cachedWorkerPort(inst *Instance) (int, error) {
	a.portCacheMu.Lock()
	if a.portCache == nil {
		a.portCache = make(map[string]portCacheEntry)
	}
	if entry, ok := a.portCache[inst.InstanceID]; ok {
		a.portCacheMu.Unlock()
		return entry.port, nil
	}
	a.portCacheMu.Unlock()
	port, err := inst.GetPort()
	if err != nil {
		return 0, err
	}
	a.portCacheMu.Lock()
	a.portCache[inst.InstanceID] = portCacheEntry{port: port, loadedAt: time.Now()}
	a.portCacheMu.Unlock()
	return port, nil
}

// adminPreload loads a model without running a job. Returns load time and a
// memory snapshot taken immediately after readiness — fair input for comparing
// load cost across backends.
func (a *API) adminPreload(w http.ResponseWriter, r *http.Request) {
	var body struct {
		ModelID string `json:"model_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.ModelID == "" {
		writeError(w, 400, "model_id required")
		return
	}
	if _, ok := a.config.Models[body.ModelID]; !ok {
		writeError(w, 404, "unknown model: "+body.ModelID)
		return
	}
	insts := a.mgr.GetModelInstances(body.ModelID)
	if len(insts) == 0 {
		writeError(w, 500, "no instances for model")
		return
	}
	inst := insts[0]
	start := time.Now()
	if err := a.scheduler.ensureLoaded(inst); err != nil {
		writeError(w, 503, fmt.Sprintf("load failed: %s", err))
		return
	}
	loadMs := time.Since(start).Milliseconds()
	a.logger.Log("admin.preload", map[string]any{"model_id": body.ModelID, "load_ms": loadMs})
	writeJSON(w, 200, map[string]any{
		"model_id":       body.ModelID,
		"instance_id":    inst.InstanceID,
		"load_ms":        loadMs,
		"already_loaded": loadMs < 100, // heuristic — sub-100ms means it was already up
	})
}

func (a *API) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{
		"status":         "ok",
		"uptime_seconds": time.Since(a.startTime).Seconds(),
	})
}

// Logging middleware
type responseWriter struct {
	http.ResponseWriter
	status int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.status = code
	rw.ResponseWriter.WriteHeader(code)
}

var requestPool = sync.Pool{
	New: func() any { return &responseWriter{} },
}

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rw := &responseWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(rw, r)
		slog.Info("http",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rw.status,
			"remote", r.RemoteAddr,
		)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"detail": msg})
}
