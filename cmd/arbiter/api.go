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
					m["max_instances"] = cfg.MaxInstances
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
		Params json.RawMessage `json:"params"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, "invalid request body")
		return
	}

	modelID, ok := JobTypeToModel[req.Type]
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
		MaxInstances int `json:"max_instances"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, "invalid request body")
		return
	}
	if req.MaxInstances < 0 {
		writeError(w, 400, "max_instances must be >= 0")
		return
	}

	oldMax := cfg.MaxInstances
	newMax := req.MaxInstances

	if newMax == oldMax {
		writeJSON(w, 200, map[string]any{
			"model_id":               modelID,
			"max_instances":          newMax,
			"previous_max_instances": oldMax,
			"added": 0, "removed": 0, "condemned": 0,
		})
		return
	}

	// Update in-memory config FIRST so scheduler sees new capacity immediately
	cfg.MaxInstances = newMax
	a.config.Models[modelID] = cfg

	// Scale instances
	result := a.mgr.ScaleModel(modelID, newMax, cfg)

	// Persist to config file
	if err := SaveModelConfigField(a.projectRoot, modelID, "max_instances", newMax); err != nil {
		slog.Error("failed to persist config", "error", err)
		// Non-fatal: in-memory change is already applied
	}

	// Rescore
	a.scheduler.rescoreModel(modelID)

	a.logger.Log("model.scaled", map[string]any{
		"model_id":          modelID,
		"old_max_instances": oldMax,
		"new_max_instances": newMax,
		"added":             result["added"],
		"removed":           result["removed"],
		"condemned":         result["condemned"],
	})

	result["model_id"] = modelID
	result["max_instances"] = newMax
	result["previous_max_instances"] = oldMax
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
