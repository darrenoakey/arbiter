package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

type API struct {
	config      *Config
	store       *Store
	mgr         *InstanceManager
	scheduler   *Scheduler
	logger      *EventLogger
	outputDir   string
	projectRoot string
	startTime   time.Time

	// Cached /v1/ps response — updated every second by background goroutine
	psCache atomic.Value // json.RawMessage
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
	mux.HandleFunc("PATCH /v1/models/{model_id}", a.updateModel)
	mux.HandleFunc("DELETE /v1/models/{model_id}/queue", a.clearModelQueue)
	mux.HandleFunc("DELETE /v1/models/{model_id}/running", a.killModelRunning)
	mux.HandleFunc("POST /v1/llm/models", a.registerLLM)
	mux.HandleFunc("GET /v1/llm/models", a.listLLMs)
	mux.HandleFunc("DELETE /v1/llm/models/{name}", a.deregisterLLM)
	mux.HandleFunc("POST /v1/chat/completions", a.chatCompletion)
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

func (a *API) updatePSCache() {
	snap := a.mgr.Snapshot()

	// Add GPU utilization
	snap["gpu_utilization_pct"] = GetGPUUtilization()

	// Add queue counts
	counts, _ := a.store.CountByState("")
	snap["queue"] = counts

	if models, ok := snap["models"].([]map[string]any); ok {
		for _, m := range models {
			if id, ok := m["id"].(string); ok {
				modelCounts, _ := a.store.CountByState(id)
				m["queued_jobs"] = modelCounts["queued"]
				if cfg, ok := a.config.Models[id]; ok {
					m["max_instances"] = *cfg.MaxInstances
				m["max_concurrent"] = cfg.MaxConcurrent
				}
			}
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

	// --- Dedup check ---
	var forceNew bool
	{
		var f struct{ Force bool `json:"force"` }
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
					// Instant cache hit — create pre-completed job
					newJob, err := a.store.CreateJob(modelID, req.Type, req.Params, 0)
					if err == nil {
						origDir := filepath.Join(a.outputDir, "jobs", origID)
						newDir := filepath.Join(a.outputDir, "jobs", newJob.ID)
						os.Symlink(origDir, newDir)
						if origJob.Result != nil {
							a.store.UpdateState(newJob.ID, "completed", WithResult(*origJob.Result), WithFinishedAt(nowTS()))
						}
						a.logger.Log("job.dedup_hit", map[string]any{
							"job_id": newJob.ID, "original_id": origID, "type": "cached",
						})
						writeJSON(w, 202, map[string]any{
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
						writeJSON(w, 202, map[string]any{
							"job_id": follower.ID, "status": "following",
							"model": modelID,
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

	writeJSON(w, 202, map[string]any{
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
				resultFile := filepath.Join(a.outputDir, "jobs", job.ID, "result."+fmt)
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
	writeJSON(w, 201, map[string]any{
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

	writeJSON(w, 201, map[string]any{
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

func (a *API) updateModel(w http.ResponseWriter, r *http.Request) {
	modelID := r.PathValue("model_id")

	cfg, ok := a.config.Models[modelID]
	if !ok {
		writeError(w, 404, fmt.Sprintf("model not configured: %s", modelID))
		return
	}

	var req struct {
		MaxInstances *int `json:"max_instances"`
		MaxConcurrent *int `json:"max_concurrent"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, "invalid request body")
		return
	}
	if req.MaxInstances != nil && *req.MaxInstances < 0 {
		writeError(w, 400, "max_instances must be >= 0")
		return
	}
	if req.MaxConcurrent != nil && *req.MaxConcurrent < 1 {
		writeError(w, 400, "max_concurrent must be >= 1")
		return
	}

	result := map[string]any{"model_id": modelID}
	changed := false

	// Handle max_concurrent change
	if req.MaxConcurrent != nil {
		oldConc := cfg.MaxConcurrent
		newConc := *req.MaxConcurrent
		if newConc != oldConc {
			cfg.MaxConcurrent = newConc
			a.config.Models[modelID] = cfg
			// Update all live instances
			for _, inst := range a.mgr.GetModelInstances(modelID) {
				inst.MaxConcurrent = newConc
			}
			SaveModelConfigField(a.projectRoot, modelID, "max_concurrent", newConc)
			result["max_concurrent"] = newConc
			result["previous_max_concurrent"] = oldConc
			changed = true
			a.logger.Log("model.concurrency_changed", map[string]any{
				"model_id": modelID, "old": oldConc, "new": newConc,
			})
		}
	}

	// Handle max_instances change
	if req.MaxInstances != nil {
		oldMax := *cfg.MaxInstances
		newMax := *req.MaxInstances
		if newMax != oldMax {
			cfg.MaxInstances = req.MaxInstances
			a.config.Models[modelID] = cfg
			scaleResult := a.mgr.ScaleModel(modelID, newMax, cfg)
			SaveModelConfigField(a.projectRoot, modelID, "max_instances", newMax)
			a.scheduler.rescoreModel(modelID)
			result["max_instances"] = newMax
			result["previous_max_instances"] = oldMax
			result["added"] = scaleResult["added"]
			result["removed"] = scaleResult["removed"]
			result["condemned"] = scaleResult["condemned"]
			changed = true
			a.logger.Log("model.scaled", map[string]any{
				"model_id":          modelID,
				"old_max_instances": oldMax,
				"new_max_instances": newMax,
				"added":             scaleResult["added"],
				"removed":           scaleResult["removed"],
				"condemned":         scaleResult["condemned"],
			})
		}
	}

	if !changed {
		result["message"] = "no changes"
	}
	writeJSON(w, 200, result)
}

func (a *API) clearModelQueue(w http.ResponseWriter, r *http.Request) {
	modelID := r.PathValue("model_id")
	if _, ok := a.config.Models[modelID]; !ok {
		writeError(w, 404, fmt.Sprintf("model not configured: %s", modelID))
		return
	}

	cancelled, err := a.store.CancelQueuedForModel(modelID)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}

	a.logger.Log("queue.cleared", map[string]any{
		"model_id":  modelID,
		"cancelled": cancelled,
	})

	writeJSON(w, 200, map[string]any{
		"model_id":  modelID,
		"cancelled": cancelled,
	})
}

func (a *API) killModelRunning(w http.ResponseWriter, r *http.Request) {
	modelID := r.PathValue("model_id")
	if _, ok := a.config.Models[modelID]; !ok {
		writeError(w, 404, fmt.Sprintf("model not configured: %s", modelID))
		return
	}

	// Cancel queued/scheduled jobs in the store
	cancelledQueued, _ := a.store.CancelQueuedForModel(modelID)

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
		"model_id":          modelID,
		"cancelled_queued":  cancelledQueued,
		"cancelled_running": cancelledRunning,
	})

	writeJSON(w, 200, map[string]any{
		"model_id":          modelID,
		"cancelled_queued":  cancelledQueued,
		"cancelled_running": cancelledRunning,
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

func estimateMemoryGB(totalParams int64) float64 {
	// fp16: 2 bytes per param + 20% overhead, rounded up to nearest 5GB
	gb := float64(totalParams) * 2.0 / (1024 * 1024 * 1024) * 1.2
	return math.Ceil(gb/5) * 5
}

func (a *API) registerLLM(w http.ResponseWriter, r *http.Request) {
	var req struct {
		HFModel    string  `json:"hf_model"`    // e.g., "unsloth/gpt-oss-20b-GGUF"
		HFFile     string  `json:"hf_file"`     // e.g., "gpt-oss-20b-Q8_0.gguf"
		ModelPath  string  `json:"model_path"`  // alternative: local GGUF path
		Name       string  `json:"name"`        // short name (auto-derived if empty)
		MemoryGB   float64 `json:"memory_gb"`   // VRAM estimate (auto if 0)
		CtxSize    int     `json:"ctx_size"`    // context size (default 8192)
		GPULayers  int     `json:"gpu_layers"`  // -1 = all (default)
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, "invalid request body")
		return
	}
	if req.HFModel == "" && req.ModelPath == "" {
		writeError(w, 400, "hf_model or model_path required")
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

	// Build adapter params (env vars for the worker)
	adapterParams := make(map[string]string)
	if req.HFModel != "" {
		adapterParams["LLM_HF_REPO"] = req.HFModel
	}
	if req.HFFile != "" {
		adapterParams["LLM_HF_FILE"] = req.HFFile
	}
	if req.ModelPath != "" {
		adapterParams["LLM_MODEL_PATH"] = req.ModelPath
	}
	ctx := req.CtxSize
	if ctx == 0 {
		ctx = 8192
	}
	adapterParams["LLM_CTX_SIZE"] = strconv.Itoa(ctx)
	gpuLayers := req.GPULayers
	if gpuLayers == 0 {
		gpuLayers = -1
	}
	adapterParams["LLM_GPU_LAYERS"] = strconv.Itoa(gpuLayers)

	// Register in config
	one := 1
	cfg := ModelConfig{
		MemoryGB:       memGB,
		MaxConcurrent:  1,
		MaxInstances:   &one,
		KeepAliveSec:   3600,
		AvgInferenceMs: 5000,
		LoadMs:         120000, // LLMs can take a while to download + load
		WorkerCmd:      []string{llmWorkerBin(a.projectRoot)},
		AdapterParams:  adapterParams,
	}
	a.config.Models[modelID] = cfg

	// Register job type mapping
	JobTypeToModel["chat-completion:"+name] = modelID

	// Create instance
	result := a.mgr.ScaleModel(modelID, 1, cfg)

	// Persist
	if err := SaveModelConfigField(a.projectRoot, modelID, "memory_gb", memGB); err != nil {
		slog.Error("failed to persist LLM config", "error", err)
	}
	// Save all config fields for the LLM
	cfgMap := map[string]any{
		"memory_gb":        memGB,
		"max_concurrent":   1,
		"max_instances":    1,
		"keep_alive_seconds": 3600,
		"avg_inference_ms": 5000,
		"load_ms":          120000,
		"worker_cmd":       cfg.WorkerCmd,
		"adapter_params":   adapterParams,
	}
	// Write full model config
	data := make(map[string]any)
	cfgPath := filepath.Join(a.projectRoot, "local", "config.json")
	if raw, err := os.ReadFile(cfgPath); err == nil {
		json.Unmarshal(raw, &data)
	}
	models, _ := data["models"].(map[string]any)
	if models == nil {
		models = make(map[string]any)
		data["models"] = models
	}
	models[modelID] = cfgMap
	out, _ := json.MarshalIndent(data, "", "  ")
	os.WriteFile(cfgPath, append(out, '\n'), 0o644)

	a.scheduler.rescoreModel(modelID)

	a.logger.Log("llm.registered", map[string]any{
		"model_id":  modelID,
		"name":      name,
		"hf_model":  req.HFModel,
		"memory_gb": memGB,
	})

	writeJSON(w, 201, map[string]any{
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
		name := strings.TrimPrefix(id, "llm:")
		entry := map[string]any{
			"model_id":      id,
			"name":          name,
			"memory_gb":     cfg.MemoryGB,
			"max_instances": *cfg.MaxInstances,
		}
		if cfg.AdapterParams != nil {
			if hf, ok := cfg.AdapterParams["LLM_HF_REPO"]; ok {
				entry["hf_model"] = hf
			}
			if hf, ok := cfg.AdapterParams["LLM_HF_FILE"]; ok {
				entry["hf_file"] = hf
			}
		}
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

	if _, ok := a.config.Models[modelID]; !ok {
		writeError(w, 404, fmt.Sprintf("LLM not registered: %s", name))
		return
	}

	// Scale to 0 (kills instances)
	zero := 0
	cfg := a.config.Models[modelID]
	cfg.MaxInstances = &zero
	a.config.Models[modelID] = cfg
	a.mgr.ScaleModel(modelID, 0, cfg)

	// Remove from config
	delete(a.config.Models, modelID)
	delete(JobTypeToModel, "chat-completion:"+name)

	// Remove from config file
	cfgPath := filepath.Join(a.projectRoot, "local", "config.json")
	if raw, err := os.ReadFile(cfgPath); err == nil {
		var data map[string]any
		if json.Unmarshal(raw, &data) == nil {
			if models, ok := data["models"].(map[string]any); ok {
				delete(models, modelID)
				out, _ := json.MarshalIndent(data, "", "  ")
				os.WriteFile(cfgPath, append(out, '\n'), 0o644)
			}
		}
	}

	a.logger.Log("llm.deregistered", map[string]any{"model_id": modelID, "name": name})
	writeJSON(w, 200, map[string]any{"model_id": modelID, "name": name, "status": "deregistered"})
}

func (a *API) chatCompletion(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
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

	// Thin wrapper: submit as a regular arbiter job and wait synchronously
	priority := a.scheduler.computePriority(modelID)
	job, err := a.store.CreateJob(modelID, "chat-completion", json.RawMessage(body), priority)
	if err != nil {
		writeError(w, 500, fmt.Sprintf("create job: %s", err))
		return
	}
	a.scheduler.Wake()

	// Wait for completion (synchronous)
	timeout := time.After(5 * time.Minute)
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
