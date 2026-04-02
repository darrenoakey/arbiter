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
	ReloadWorkers  bool               `json:"reload_workers"`
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
	AvgInferenceMs *float64          `json:"avg_inference_ms"`
	LoadMs         *float64          `json:"load_ms"`
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
	completedJobs, avgTotalSeconds, avgExecSeconds, _ := a.store.CompletedJobStats("")
	snap["job_stats"] = map[string]any{
		"completed_jobs":        completedJobs,
		"avg_total_seconds":     avgTotalSeconds,
		"avg_execution_seconds": avgExecSeconds,
		"avg_waiting_seconds":   math.Max(avgTotalSeconds-avgExecSeconds, 0),
	}

	if models, ok := snap["models"].([]map[string]any); ok {
		for _, m := range models {
			if id, ok := m["id"].(string); ok {
				modelCounts, _ := a.store.CountByState(id)
				m["queued_jobs"] = modelCounts["queued"]
				completedJobs, avgTotalSeconds, avgExecSeconds, _ := a.store.CompletedJobStats(id)
				m["completed_jobs"] = completedJobs
				m["avg_total_seconds"] = avgTotalSeconds
				m["avg_execution_seconds"] = avgExecSeconds
				m["avg_waiting_seconds"] = math.Max(avgTotalSeconds-avgExecSeconds, 0)
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
		"model_id":           modelID,
		"memory_gb":          cfg.MemoryGB,
		"max_concurrent":     cfg.MaxConcurrent,
		"keep_alive_seconds": cfg.KeepAliveSec,
		"avg_inference_ms":   cfg.AvgInferenceMs,
		"load_ms":            cfg.LoadMs,
		"auto_download":      cfg.AutoDownload,
		"model_path":         cfg.ModelPath,
		"group":              cfg.Group,
		"worker_cmd":         cfg.WorkerCmd,
		"adapter_params":     cfg.AdapterParams,
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

	writeJSON(w, 201, map[string]any{
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
	if !req.ReloadWorkers && (req.WorkerCmd != nil || req.AdapterParams != nil) {
		result["message"] = "config updated; existing loaded workers keep running until this model is reloaded"
	} else if scaleResult == nil && req.MaxConcurrent == nil && req.MemoryGB == nil &&
		req.KeepAliveSec == nil && req.AvgInferenceMs == nil && req.LoadMs == nil &&
		req.WorkerCmd == nil && req.AdapterParams == nil && req.AutoDownload == nil &&
		req.ModelPath == nil && req.Group == nil && req.MaxInstances == nil {
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
	if req.LlamaServerBin != "" {
		adapterParams["LLAMA_SERVER_BIN"] = req.LlamaServerBin
	}
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
		AvgInferenceMs: 5000,
		LoadMs:         120000, // LLMs can take a while to download + load
		WorkerCmd:      []string{llmWorkerBin(a.projectRoot)},
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

// chatCompletionStream handles streaming chat completions by proxying SSE
// directly from llama-server to the client, bypassing the job system.
func (a *API) chatCompletionStream(w http.ResponseWriter, r *http.Request, modelID string, body []byte) {
	// Ensure model is loaded
	instances := a.scheduler.mgr.GetModelInstances(modelID)
	if len(instances) == 0 {
		writeError(w, 500, "no instances for model")
		return
	}
	inst := instances[0]

	if err := a.scheduler.ensureLoaded(inst); err != nil {
		writeError(w, 503, fmt.Sprintf("model not ready: %s", err))
		return
	}

	// Get the llama-server port from the worker
	port, err := inst.GetPort()
	if err != nil {
		writeError(w, 500, fmt.Sprintf("get llama-server port: %s", err))
		return
	}

	// Proxy the request directly to llama-server
	llamaURL := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port)
	proxyReq, err := http.NewRequestWithContext(r.Context(), "POST", llamaURL, bytes.NewReader(body))
	if err != nil {
		writeError(w, 500, fmt.Sprintf("create proxy request: %s", err))
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(proxyReq)
	if err != nil {
		writeError(w, 502, fmt.Sprintf("llama-server error: %s", err))
		return
	}
	defer resp.Body.Close()

	// Copy headers from llama-server response
	for k, vv := range resp.Header {
		for _, v := range vv {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)

	// Stream the response body directly to the client
	flusher, ok := w.(http.Flusher)
	if !ok {
		io.Copy(w, resp.Body)
		return
	}

	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
			flusher.Flush()
		}
		if err != nil {
			break
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
