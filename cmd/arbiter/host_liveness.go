package main

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

// HostMonitor runs one liveness poll per remote host and is the authoritative
// source of "is this box reachable right now". It exists for ONE hard reason:
// detection must be fast and active. A slept laptop's established TCP socket can
// hang silently with no RST, so we cannot wait for the inference timeout to
// discover a host is gone. The monitor polls a cheap endpoint ({addr}/api/version)
// every few seconds and, on N CONSECUTIVE failures, flips the host absent AND
// fires the per-backend Cancel() hook on every remote instance placed there — so
// the Phase-2 transparent failover fires in seconds, not minutes.
//
// CRITICAL invariant (overrides any "cancel in-flight" wording): Cancel() is
// ONLY ever fired for a host the poll has CONFIRMED dead. A slow-but-alive host
// (poll still succeeding) is NEVER cancelled — aborting a live ollama request
// wedges that model's runner (every later /api/chat hangs). So the cancel path
// runs strictly inside the "consecutive failures crossed the threshold" branch.
type HostMonitor struct {
	mgr    *InstanceManager
	logger *EventLogger
	sched  *Scheduler

	// pollInterval / failThreshold tune detection latency vs flap-resistance.
	// At 4s × 3 a host is declared absent ~12s after it stops answering — fast
	// enough that failover beats a human noticing, slow enough that a single
	// dropped poll (GC pause on the Mac, a momentary Wi-Fi blip) doesn't trip it.
	pollInterval  time.Duration
	failThreshold int

	mu     sync.RWMutex
	states map[string]*hostState // host id -> liveness state

	// httpClient is a dedicated client for the cheap version poll. It uses a
	// short timeout and DisableKeepAlives so a flapped LAN route can't brick a
	// pooled connection (the documented Go net/http EHOSTUNREACH wedge) — every
	// poll is a fresh dial, exactly like RemoteHTTPBackend.
	httpClient *http.Client
}

type hostState struct {
	hostID      string
	addr        string // chat / primary base (nativ or ollama)
	healthAddr  string // base used for liveness poll (/health or /api/version)
	psAddr      string // base used for loaded-model listing
	kind        string // nativ | mlx
	apiKey      string // nativ management-endpoint auth, from HostConfig.ApiKey
	reachable   bool
	failStreak  int
	lastChecked time.Time
	lastOK      time.Time
	// probed is set on the first successful poll. A monitor (re)start resets
	// reachable=true but loses flap history, so the first CONFIRMED-reachable
	// probe — distinct from an absent→reachable recovery — is the moment to
	// forgive stale excluded_hosts entries left over from before the restart.
	probed bool
}

// NewHostMonitor builds a monitor over the remote hosts in cfg. Local (spark)
// is never polled — it is the always-up coordinator. Remote hosts start as
// reachable=true so a model isn't spuriously pinned-to-spark before the first
// poll completes (the first poll corrects it within pollInterval if wrong).
func NewHostMonitor(cfg *Config, mgr *InstanceManager, logger *EventLogger, sched *Scheduler) *HostMonitor {
	states := make(map[string]*hostState)
	for id, h := range cfg.Hosts {
		if cfg.HostIsLocal(id) {
			continue
		}
		kind := h.KindOrDefault()
		healthAddr := h.Addr
		psAddr := h.Addr
		// Nativ exposes /health on its own port; Ollama-native /api/ps (and
		// optional liveness via /api/version) live on OllamaBase when set.
		// Prefer Nativ /health for chat-host liveness so a dead Nativ is
		// detected even if Ollama is still up for embeds.
		if kind == "nativ" {
			healthAddr = h.Addr
			if h.OllamaAddr != "" {
				psAddr = h.OllamaAddr
			}
		} else {
			// legacy ollama: health + ps on the same base
			healthAddr = h.OllamaBase()
			psAddr = h.OllamaBase()
		}
		states[id] = &hostState{
			hostID:     id,
			addr:       h.Addr,
			healthAddr: healthAddr,
			psAddr:     psAddr,
			kind:       kind,
			apiKey:     h.ApiKey,
			reachable:  true,
		}
	}
	return &HostMonitor{
		mgr:           mgr,
		logger:        logger,
		sched:         sched,
		pollInterval:  4 * time.Second,
		failThreshold: 3,
		states:        states,
		httpClient: &http.Client{
			Timeout: 3 * time.Second,
			Transport: &http.Transport{
				DisableKeepAlives:   true,
				MaxIdleConns:        1,
				IdleConnTimeout:     time.Second,
				TLSHandshakeTimeout: 3 * time.Second,
			},
		},
	}
}

// IsReachable reports whether a host id is currently reachable. Unknown ids and
// the local host are always reachable (spark is the always-up final link). A
// host with no monitor entry (e.g. tests that don't run the poll) is treated as
// reachable so routing isn't blocked by a never-started monitor.
func (hm *HostMonitor) IsReachable(hostID string) bool {
	if hostID == "" || hostID == LocalHost {
		return true
	}
	hm.mu.RLock()
	defer hm.mu.RUnlock()
	st, ok := hm.states[hostID]
	if !ok {
		return true
	}
	return st.reachable
}

// Run drives the per-host poll loop until ctx is cancelled. Hosts are polled
// concurrently each tick so one slow/dead host's dial timeout never delays the
// others' detection.
func (hm *HostMonitor) Run(ctx context.Context) {
	if len(hm.states) == 0 {
		return // no remote hosts → nothing to poll
	}
	slog.Info("host liveness monitor started", "hosts", len(hm.states),
		"interval", hm.pollInterval.String(), "fail_threshold", hm.failThreshold)
	ticker := time.NewTicker(hm.pollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			slog.Info("host liveness monitor stopped")
			return
		case <-ticker.C:
			hm.pollAll(ctx)
		}
	}
}

// pollAll polls every monitored host concurrently and applies the result.
func (hm *HostMonitor) pollAll(ctx context.Context) {
	hm.mu.RLock()
	type target struct {
		id, healthAddr, kind, apiKey string
	}
	targets := make([]target, 0, len(hm.states))
	for id, st := range hm.states {
		targets = append(targets, target{id, st.healthAddr, st.kind, st.apiKey})
	}
	hm.mu.RUnlock()

	var wg sync.WaitGroup
	for _, t := range targets {
		wg.Add(1)
		go func(id, healthAddr, kind, apiKey string) {
			defer wg.Done()
			ok := hm.pollOne(ctx, healthAddr, kind, apiKey)
			hm.applyResult(id, ok)
		}(t.id, t.healthAddr, t.kind, t.apiKey)
	}
	wg.Wait()
}

// pollOne hits a cheap health endpoint on the host. Nativ: GET /health.
// Ollama/MLX: GET /api/version. Reachable == a 2xx response. apiKey, when
// non-empty, is sent as "Authorization: Bearer <apiKey>" — required by a
// Nativ mlx-vlm-server bound to a non-localhost address, which gates /health
// (and /metrics, /apc/*, /unload) behind --api-key even though the actual
// chat completions endpoint stays open. Without this header a key-protected
// host always 401s and gets permanently flagged absent regardless of whether
// it can actually serve inference.
func (hm *HostMonitor) pollOne(ctx context.Context, healthAddr, kind, apiKey string) bool {
	reqCtx, cancel := context.WithTimeout(ctx, hm.httpClient.Timeout)
	defer cancel()
	path := "/api/version"
	if kind == "nativ" {
		path = "/health"
	}
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, healthAddr+path, nil)
	if err != nil {
		return false
	}
	if kind == "nativ" && apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := hm.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			slog.Debug("close host liveness response", "error", err)
		}
	}()
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		return false
	}
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

// applyResult folds one poll outcome into the host's state, emitting
// host.absent / host.recovered exactly once per transition and firing the
// active-cancel hook on the absent transition.
func (hm *HostMonitor) applyResult(hostID string, ok bool) {
	hm.mu.Lock()
	st := hm.states[hostID]
	if st == nil {
		hm.mu.Unlock()
		return
	}
	st.lastChecked = time.Now()

	var becameAbsent, recovered bool
	firstProbe := false
	if ok {
		st.lastOK = time.Now()
		if !st.probed {
			st.probed = true
			firstProbe = true
		}
		if !st.reachable {
			st.reachable = true
			recovered = true
		}
		st.failStreak = 0
	} else {
		st.failStreak++
		if st.reachable && st.failStreak >= hm.failThreshold {
			st.reachable = false
			becameAbsent = true
		}
	}
	streak := st.failStreak
	hm.mu.Unlock()

	switch {
	case becameAbsent:
		slog.Warn("host liveness: host flipped ABSENT — cancelling in-flight remote work",
			"host", hostID, "fail_streak", streak)
		if hm.logger != nil {
			hm.logger.Log("host.absent", map[string]any{
				"host_id":     hostID,
				"fail_streak": streak,
			})
		}
		// Fire the Phase-2 active-cancel hook on EVERY remote instance placed on
		// this host. This is the ONLY place Cancel() is driven, and it runs ONLY
		// after a CONFIRMED-absence transition — never on a slow-but-alive host.
		// Cancel() makes an in-flight Infer return INFRA (→ transparent failover)
		// while leaving the detached upstream call to drain harmlessly.
		hm.cancelHostInstances(hostID)
		if hm.sched != nil {
			hm.sched.Wake() // let the scheduler re-pick down the chain immediately
		}
	case recovered:
		slog.Info("host liveness: host RECOVERED — eligible for placement again", "host", hostID)
		if hm.logger != nil {
			hm.logger.Log("host.recovered", map[string]any{"host_id": hostID})
		}
		fallthrough
	case firstProbe:
		// Host is confirmed reachable (recovered, or first successful probe
		// after a monitor restart). Forgive stale excluded_hosts entries so a
		// transient flap doesn't strand jobs forever; dampened by minAge so an
		// active flap isn't re-forgiven at the liveness cadence.
		if hm.sched != nil {
			if healed, err := hm.sched.ClearStaleExclusionsForHost(hostID, staleExclusionMinAge); err != nil {
				slog.Warn("host liveness: clear stale exclusions failed", "host", hostID, "error", err)
			} else if healed > 0 {
				slog.Info("host liveness: forgave stale excluded host for active jobs", "host", hostID, "healed", healed)
				if hm.logger != nil {
					hm.logger.Log("host.exclusion_cleared", map[string]any{"host_id": hostID, "healed": healed})
				}
			}
			hm.sched.Wake() // a parked/queued model may now place on / drain to this host
		}
	}
}

// cancelHostInstances fires Cancel() on every remote instance whose host just
// flipped absent. It is idempotent (Cancel is epoch-based) and never touches a
// local instance.
func (hm *HostMonitor) cancelHostInstances(hostID string) {
	for _, inst := range hm.mgr.AllInstances() {
		if inst.host == hostID && inst.backend != nil && inst.backend.IsRemote() {
			if err := inst.backend.Cancel(); err != nil {
				slog.Warn("host liveness: cancel failed", "host", hostID,
					"instance", inst.InstanceID, "error", err)
			}
			// A cancelled remote instance is no longer serving — flip its state so
			// pickFromPool won't hand it new work until a recovery cold-warm.
			inst.setState("error")
		}
	}
}

// RemoteHostsPanel builds the SEPARATE /v1/ps remote-hosts panel. It is advisory
// and DELIBERATELY disjoint from the audited local VRAM ledger (usedGB /
// AuditVRAMConsistency) — remote hosts hold ZERO bytes there. Each entry carries
// the advisory used/budget (from remoteHostBudget), the liveness flag, and the
// models the host reports loaded. Never call this on the hot path: it does a
// network round-trip per host.
func (hm *HostMonitor) RemoteHostsPanel() []map[string]any {
	hm.mu.RLock()
	type snap struct {
		id, addr, psAddr, kind, apiKey string
		reachable                      bool
		failStreak                     int
		lastOK                         time.Time
	}
	snaps := make([]snap, 0, len(hm.states))
	for _, st := range hm.states {
		snaps = append(snaps, snap{st.hostID, st.addr, st.psAddr, st.kind, st.apiKey, st.reachable, st.failStreak, st.lastOK})
	}
	hm.mu.RUnlock()

	panel := make([]map[string]any, 0, len(snaps))
	for _, s := range snaps {
		entry := map[string]any{
			"host_id":   s.id,
			"addr":      s.addr,
			"kind":      s.kind,
			"reachable": s.reachable,
		}
		if s.failStreak > 0 {
			entry["fail_streak"] = s.failStreak
		}
		if !s.lastOK.IsZero() {
			entry["last_ok_seconds_ago"] = time.Since(s.lastOK).Seconds()
		}
		if hb := hm.mgr.RemoteHostBudget(s.id); hb != nil {
			entry["budget_gb"] = hb.budgetGB
			entry["used_gb"] = hb.usedGB // advisory only — NOT spark's audited VRAM
		}
		// Live loaded-model list, best-effort. Only attempt when reachable so a
		// dead host doesn't add a dial-timeout to the request.
		if s.reachable {
			if loaded := hm.queryLoadedModels(s.psAddr, s.kind, s.addr, s.apiKey); loaded != nil {
				entry["models_loaded"] = loaded
			}
		}
		panel = append(panel, entry)
	}
	return panel
}

// queryLoadedModels returns the models currently loaded on a remote host.
// Nativ: GET {nativAddr}/health → loaded_model. Ollama: GET {psAddr}/api/ps.
func (hm *HostMonitor) queryLoadedModels(psAddr, kind, nativAddr, apiKey string) []string {
	if kind == "nativ" {
		return hm.queryNativLoaded(nativAddr, apiKey)
	}
	return hm.queryOllamaPS(psAddr)
}

func (hm *HostMonitor) queryNativLoaded(addr, apiKey string) []string {
	req, err := http.NewRequest(http.MethodGet, addr+"/health", nil)
	if err != nil {
		return nil
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := hm.httpClient.Do(req)
	if err != nil {
		return nil
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			slog.Debug("close nativ health response", "error", err)
		}
	}()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil
	}
	var body struct {
		LoadedModel  string `json:"loaded_model"`
		LoadedModels struct {
			TextGeneration string `json:"text_generation"`
		} `json:"loaded_models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil
	}
	names := make([]string, 0, 2)
	if body.LoadedModel != "" {
		names = append(names, body.LoadedModel)
	} else if body.LoadedModels.TextGeneration != "" {
		names = append(names, body.LoadedModels.TextGeneration)
	}
	if len(names) == 0 {
		return nil
	}
	return names
}

// queryOllamaPS asks a host's ollama which models are currently loaded
// (GET /api/ps). Best-effort: returns nil on any error so the panel still
// renders. This is the remote analogue of spark's /v1/ps "loaded models", kept
// SEPARATE from the audited local ledger.
func (hm *HostMonitor) queryOllamaPS(addr string) []string {
	req, err := http.NewRequest(http.MethodGet, addr+"/api/ps", nil)
	if err != nil {
		return nil
	}
	resp, err := hm.httpClient.Do(req)
	if err != nil {
		return nil
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			slog.Debug("close ollama process response", "error", err)
		}
	}()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil
	}
	var body struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil
	}
	names := make([]string, 0, len(body.Models))
	for _, m := range body.Models {
		names = append(names, m.Name)
	}
	return names
}
