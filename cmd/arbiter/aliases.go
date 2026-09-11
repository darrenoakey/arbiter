package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"regexp"
	"slices"
	"strings"
)

// aliasNameRE matches allowed alias names. The local- prefix is deliberate
// policy baked into the server; other namespaces require a code change.
var aliasNameRE = regexp.MustCompile(`^local-[a-z0-9][a-z0-9-]*$`)

// validateLLMAliases checks a proposed alias map against the current concrete
// models. It enforces naming, target existence, no shadowing of model ids/names,
// no duplicate normalized names, and no alias chains.
func validateLLMAliases(aliases map[string]string, models map[string]ModelConfig) error {
	seen := make(map[string]bool)
	for alias, target := range aliases {
		lower := strings.ToLower(alias)
		if !aliasNameRE.MatchString(alias) {
			return fmt.Errorf("alias %q does not match %s", alias, aliasNameRE.String())
		}
		if seen[lower] {
			return fmt.Errorf("duplicate alias name (case-insensitive): %s", alias)
		}
		seen[lower] = true

		// Refuse to shadow any configured model id or bare LLM name.
		if _, ok := models[alias]; ok {
			return fmt.Errorf("alias %q shadows an existing model id", alias)
		}
		if _, ok := models[llmModelID(alias)]; ok {
			return fmt.Errorf("alias %q shadows an existing LLM bare name", alias)
		}

		// Target must be an exact concrete configured LLM id. Bare names are
		// deliberately rejected so persisted configuration is unambiguous.
		if !strings.HasPrefix(target, "llm:") {
			return fmt.Errorf("alias %q target %q must be a canonical llm:* model id", alias, target)
		}
		if _, ok := models[target]; !ok {
			return fmt.Errorf("alias %q target %q is not a registered model", alias, target)
		}

		// No chains: target must not itself be an alias.
		if _, isAlias := aliases[strings.TrimPrefix(target, "llm:")]; isAlias {
			return fmt.Errorf("alias %q target %q is itself an alias (chains forbidden)", alias, target)
		}
	}
	return nil
}

// validateLLMAliasFallbacks checks proposed per-alias fallback lists. Every
// entry must be a registered concrete llm:* model id that is neither an alias
// (no chains) nor the alias's own primary target, and each list must name an
// alias that actually exists. Duplicate entries are rejected so the ordered
// list reads as the operator wrote it.
func validateLLMAliasFallbacks(fallbacks map[string][]string, aliases map[string]string, models map[string]ModelConfig) error {
	for alias, chain := range fallbacks {
		primary, exists := aliases[alias]
		if !exists {
			return fmt.Errorf("fallbacks for %q name no configured alias", alias)
		}
		if len(chain) == 0 {
			return fmt.Errorf("alias %q has an empty fallback list; omit the entry instead", alias)
		}
		seen := make(map[string]bool, len(chain))
		for _, target := range chain {
			if !strings.HasPrefix(target, "llm:") {
				return fmt.Errorf("alias %q fallback %q must be a canonical llm:* model id", alias, target)
			}
			if _, ok := models[target]; !ok {
				return fmt.Errorf("alias %q fallback %q is not a registered model", alias, target)
			}
			if _, isAlias := aliases[strings.TrimPrefix(target, "llm:")]; isAlias {
				return fmt.Errorf("alias %q fallback %q is itself an alias (chains forbidden)", alias, target)
			}
			if target == primary {
				return fmt.Errorf("alias %q fallback %q repeats the primary target", alias, target)
			}
			if seen[target] {
				return fmt.Errorf("alias %q lists fallback %q twice", alias, target)
			}
			seen[target] = true
		}
	}
	return nil
}

// resolveLLMModelID resolves a requested model string to a canonical model id,
// reporting whether an alias was used. Resolution order:
//  1. Exact configured model id (including llm:* ids).
//  2. Bare LLM name -> llm:<name> if configured.
//  3. Alias -> configured target, or the first servable fallback when that
//     target currently has no servable placement.
//  4. Not found.
func (a *API) resolveLLMModelID(requested string) (canonicalModelID string, aliasUsed string, ok bool) {
	if requested == "" {
		return "", "", false
	}
	a.configMutationMu.RLock()
	defer a.configMutationMu.RUnlock()
	aliases, models := a.aliasStateSnapshot()

	// 1. Exact model id.
	if _, exists := models[requested]; exists {
		return requested, "", true
	}
	// 2. Bare LLM name.
	llmID := llmModelID(requested)
	if _, exists := models[llmID]; exists {
		return llmID, "", true
	}
	// 3. Alias.
	if target, exists := aliases[requested]; exists {
		if _, exists := models[target]; exists {
			return a.aliasTargetOrFallback(requested, target, models), requested, true
		}
	}
	return "", "", false
}

// aliasTargetOrFallback returns the alias target to admit against. The
// configured target wins whenever it is servable. When it is not — every host
// it may run on is an unreachable remote — the first servable configured
// fallback is used instead, because admitting against an unservable model
// parks the job in "queued" indefinitely with nothing failing and nothing
// running (live 2026-09-10 incident). With no servable fallback the configured
// target is kept, so behaviour is unchanged for aliases without fallbacks.
func (a *API) aliasTargetOrFallback(alias, target string, models map[string]struct{}) string {
	if a.mgr == nil || a.mgr.ModelHasServablePlacement(target) {
		return target
	}
	for _, candidate := range a.aliasFallbackSnapshot(alias) {
		if _, exists := models[candidate]; !exists {
			continue
		}
		if !a.mgr.ModelHasServablePlacement(candidate) {
			continue
		}
		slog.Warn("llm.alias_fallback: primary target has no servable placement",
			"alias", alias, "primary", target, "resolved", candidate,
			"reason", "every placement of the primary is an unreachable remote host")
		a.logger.Log("llm.alias_fallback", map[string]any{
			"alias":    alias,
			"primary":  target,
			"resolved": candidate,
			"reason":   "primary has no servable placement",
		})
		return candidate
	}
	return target
}

// bareModelName returns the bare model name for a canonical llm:* id.
func bareModelName(modelID string) string {
	return strings.TrimPrefix(modelID, "llm:")
}

// canonicalizeChatBody rewrites the "model" field in a chat body to the bare
// canonical model name. This must happen before cache lookup and before dedup
// hashing so that identical content via alias or concrete name collides.
func canonicalizeChatBody(body []byte, canonicalModelID string) ([]byte, error) {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, err
	}
	bare := bareModelName(canonicalModelID)
	if requested, ok := m["model"].(string); ok && requested == bare {
		return slices.Clone(body), nil
	}
	m["model"] = bare
	out, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// canonicalizeChatParams is canonicalizeChatBody for the params object inside a
// chat-completion job request.
func canonicalizeChatParams(params json.RawMessage, canonicalModelID string) (json.RawMessage, error) {
	canon, err := canonicalizeChatBody(params, canonicalModelID)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(canon), nil
}

// rewriteOpenAIResponseModel returns response bytes with the top-level "model"
// field replaced by requestedModel. If requestedModel is empty the body is
// returned unchanged.
func rewriteOpenAIResponseModel(resp []byte, requestedModel string) []byte {
	if requestedModel == "" {
		return resp
	}
	var m map[string]any
	if err := json.Unmarshal(resp, &m); err != nil {
		return resp
	}
	m["model"] = requestedModel
	out, err := json.Marshal(m)
	if err != nil {
		return resp
	}
	return out
}

func rewriteChatResultMap(result map[string]any, requestedModel string) map[string]any {
	if requestedModel == "" || result == nil {
		return result
	}
	response, exists := result["response"]
	if !exists {
		return result
	}
	responseBytes, err := json.Marshal(response)
	if err != nil {
		return result
	}
	rewrittenBytes := rewriteOpenAIResponseModel(responseBytes, requestedModel)
	var rewritten any
	if err := json.Unmarshal(rewrittenBytes, &rewritten); err != nil {
		return result
	}
	result["response"] = rewritten
	return result
}

// setModelIdentityHeaders writes the standard request/resolved/alias headers on
// a response writer.
func setModelIdentityHeaders(w http.ResponseWriter, requested, resolved, alias string) {
	if requested != "" {
		w.Header().Set("X-Arbiter-Requested-Model", requested)
	}
	if resolved != "" {
		w.Header().Set("X-Arbiter-Resolved-Model", resolved)
	}
	if alias != "" {
		w.Header().Set("X-Arbiter-Alias", alias)
	}
}

func aliasForRequest(requested, resolved string) string {
	if requested != resolved && llmModelID(requested) != resolved {
		return requested
	}
	return ""
}

// aliasesTargeting returns the aliases that currently point at canonicalModelID.
func (a *API) aliasesTargeting(canonicalModelID string) []string {
	var out []string
	for alias, target := range a.aliasSnapshot() {
		if target == canonicalModelID {
			out = append(out, alias)
		}
	}
	slices.Sort(out)
	return out
}

// listAliases handles GET /v1/llm/aliases.
func (a *API) listAliases(w http.ResponseWriter, r *http.Request) {
	a.configMutationMu.RLock()
	defer a.configMutationMu.RUnlock()
	aliases, models := a.aliasStateSnapshot()
	keys := make([]string, 0, len(aliases))
	for k := range aliases {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	out := make(map[string]any, len(keys))
	fallbacks := a.aliasFallbacksSnapshot()
	for _, alias := range keys {
		target := aliases[alias]
		_, configured := models[target]
		entry := map[string]any{
			"target":            target,
			"resolved":          target,
			"target_configured": configured,
		}
		// resolved is what admission would pick RIGHT NOW, which differs from
		// target while the primary's only placements are unreachable hosts.
		if configured {
			entry["resolved"] = a.aliasTargetOrFallback(alias, target, models)
		}
		if chain := fallbacks[alias]; len(chain) > 0 {
			entry["fallbacks"] = chain
		}
		out[alias] = entry
	}
	writeJSON(w, 200, out)
}

type aliasUpdateRequest struct {
	Target string `json:"target"`
	// Fallbacks, when non-nil, replaces this alias's ordered fallback list. An
	// explicit empty array clears it; omitting the field leaves it untouched.
	Fallbacks []string `json:"fallbacks"`
}

// putAlias handles PUT /v1/llm/aliases/{alias}.
func (a *API) putAlias(w http.ResponseWriter, r *http.Request) {
	a.configMutationMu.Lock()
	defer a.configMutationMu.Unlock()

	alias := r.PathValue("alias")
	if alias == "" {
		writeError(w, 400, "alias name required")
		return
	}
	var req aliasUpdateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Target == "" {
		writeError(w, 400, "body must be {\"target\":\"llm:<model>\",\"fallbacks\":[\"llm:<model>\"]}")
		return
	}

	newAliases := a.aliasSnapshot()
	if newAliases == nil {
		newAliases = make(map[string]string)
	}
	oldTarget := newAliases[alias]
	newAliases[alias] = req.Target

	if err := validateLLMAliases(newAliases, a.config.CloneModels()); err != nil {
		writeError(w, 400, err.Error())
		return
	}

	newFallbacks := a.aliasFallbacksSnapshot()
	if req.Fallbacks != nil {
		if len(req.Fallbacks) == 0 {
			delete(newFallbacks, alias)
		} else {
			newFallbacks[alias] = slices.Clone(req.Fallbacks)
		}
	}
	// Retargeting can invalidate a previously-valid fallback list (e.g. the new
	// primary is already named as a fallback), so the whole map is revalidated.
	if err := validateLLMAliasFallbacks(newFallbacks, newAliases, a.config.CloneModels()); err != nil {
		writeError(w, 400, err.Error())
		return
	}

	if err := SaveLLMAliases(a.projectRoot, newAliases); err != nil {
		writeError(w, 500, fmt.Sprintf("persist alias: %s", err))
		return
	}
	if err := SaveLLMAliasFallbacks(a.projectRoot, newFallbacks); err != nil {
		writeError(w, 500, fmt.Sprintf("persist alias fallbacks: %s", err))
		return
	}
	a.replaceAliases(newAliases)
	a.replaceAliasFallbacks(newFallbacks)

	_, modelIDs := a.aliasStateSnapshot()
	resolved := a.aliasTargetOrFallback(alias, req.Target, modelIDs)
	a.logger.Log("llm.alias_updated", map[string]any{
		"alias":      alias,
		"old_target": oldTarget,
		"new_target": req.Target,
		"fallbacks":  newFallbacks[alias],
		"resolved":   resolved,
		"actor":      r.RemoteAddr,
	})
	writeJSON(w, 200, map[string]any{
		"alias":      alias,
		"old_target": oldTarget,
		"new_target": req.Target,
		"fallbacks":  newFallbacks[alias],
		"resolved":   resolved,
	})
}

// deleteAlias handles DELETE /v1/llm/aliases/{alias}.
func (a *API) deleteAlias(w http.ResponseWriter, r *http.Request) {
	a.configMutationMu.Lock()
	defer a.configMutationMu.Unlock()

	alias := r.PathValue("alias")
	if alias == "" {
		writeError(w, 400, "alias name required")
		return
	}
	aliases := a.aliasSnapshot()
	if _, ok := aliases[alias]; !ok {
		writeError(w, 404, fmt.Sprintf("alias not found: %s", alias))
		return
	}

	force := r.URL.Query().Get("force") == "1" || r.URL.Query().Get("force") == "true"
	if !force {
		cutoff := nowTS() - 24*3600
		n, err := a.store.CountRequestedModelSince(alias, cutoff)
		if err != nil {
			writeError(w, 500, fmt.Sprintf("check alias traffic: %s", err))
			return
		}
		if n > 0 {
			writeError(w, 409, fmt.Sprintf(
				"alias %q resolved %d job(s) in the last 24h; use ?force=1 to delete anyway",
				alias, n,
			))
			return
		}
	}

	newAliases := maps.Clone(aliases)
	delete(newAliases, alias)
	newFallbacks := a.aliasFallbacksSnapshot()
	delete(newFallbacks, alias)
	if err := SaveLLMAliases(a.projectRoot, newAliases); err != nil {
		writeError(w, 500, fmt.Sprintf("persist alias deletion: %s", err))
		return
	}
	if err := SaveLLMAliasFallbacks(a.projectRoot, newFallbacks); err != nil {
		writeError(w, 500, fmt.Sprintf("persist alias fallback deletion: %s", err))
		return
	}
	a.replaceAliases(newAliases)
	a.replaceAliasFallbacks(newFallbacks)

	a.logger.Log("llm.alias_deleted", map[string]any{"alias": alias, "force": force})
	writeJSON(w, 200, map[string]any{
		"alias":   alias,
		"deleted": true,
		"aliases": newAliases,
	})
}

func configuredModelIDs(models map[string]ModelConfig) map[string]struct{} {
	ids := make(map[string]struct{}, len(models))
	for modelID := range models {
		ids[modelID] = struct{}{}
	}
	return ids
}

func (a *API) aliasSnapshot() map[string]string {
	a.aliasMu.RLock()
	defer a.aliasMu.RUnlock()
	return maps.Clone(a.config.LLMAliases)
}

// aliasFallbackSnapshot returns the ordered fallback list configured for one
// alias, or nil when it has none.
func (a *API) aliasFallbackSnapshot(alias string) []string {
	a.aliasMu.RLock()
	defer a.aliasMu.RUnlock()
	return slices.Clone(a.config.LLMAliasFallbacks[alias])
}

// aliasFallbacksSnapshot returns every configured fallback list.
func (a *API) aliasFallbacksSnapshot() map[string][]string {
	a.aliasMu.RLock()
	defer a.aliasMu.RUnlock()
	out := make(map[string][]string, len(a.config.LLMAliasFallbacks))
	for alias, chain := range a.config.LLMAliasFallbacks {
		out[alias] = slices.Clone(chain)
	}
	return out
}

// replaceAliasFallbacks publishes a new fallback map under the alias lock.
func (a *API) replaceAliasFallbacks(fallbacks map[string][]string) {
	a.aliasMu.Lock()
	defer a.aliasMu.Unlock()
	next := make(map[string][]string, len(fallbacks))
	for alias, chain := range fallbacks {
		next[alias] = slices.Clone(chain)
	}
	a.config.LLMAliasFallbacks = next
}

func (a *API) aliasStateSnapshot() (map[string]string, map[string]struct{}) {
	a.aliasMu.RLock()
	defer a.aliasMu.RUnlock()
	return maps.Clone(a.config.LLMAliases), maps.Clone(a.aliasModels)
}

func (a *API) replaceAliases(aliases map[string]string) {
	a.aliasMu.Lock()
	defer a.aliasMu.Unlock()
	a.config.LLMAliases = maps.Clone(aliases)
	a.aliasModels = configuredModelIDs(a.config.CloneModels())
}

func (a *API) refreshAliasModels() {
	a.aliasMu.Lock()
	defer a.aliasMu.Unlock()
	a.aliasModels = configuredModelIDs(a.config.CloneModels())
}

func (a *API) modelAliasCollision(modelID string) (string, bool) {
	bare := strings.TrimPrefix(modelID, "llm:")
	aliases := a.aliasSnapshot()
	if _, exists := aliases[modelID]; exists {
		return modelID, true
	}
	if _, exists := aliases[bare]; exists {
		return bare, true
	}
	return "", false
}
