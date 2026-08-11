package main

import (
	"encoding/json"
	"fmt"
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

// resolveLLMModelID resolves a requested model string to a canonical model id,
// reporting whether an alias was used. Resolution order:
//  1. Exact configured model id (including llm:* ids).
//  2. Bare LLM name -> llm:<name> if configured.
//  3. Alias -> configured target.
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
			return target, requested, true
		}
	}
	return "", "", false
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
	for _, alias := range keys {
		target := aliases[alias]
		_, configured := models[target]
		out[alias] = map[string]any{
			"target":            target,
			"resolved":          target,
			"target_configured": configured,
		}
	}
	writeJSON(w, 200, out)
}

type aliasUpdateRequest struct {
	Target string `json:"target"`
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
		writeError(w, 400, "body must be {\"target\":\"llm:<model>\"}")
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

	if err := SaveLLMAliases(a.projectRoot, newAliases); err != nil {
		writeError(w, 500, fmt.Sprintf("persist alias: %s", err))
		return
	}
	a.replaceAliases(newAliases)

	a.logger.Log("llm.alias_updated", map[string]any{
		"alias":      alias,
		"old_target": oldTarget,
		"new_target": req.Target,
		"resolved":   req.Target,
		"actor":      r.RemoteAddr,
	})
	writeJSON(w, 200, map[string]any{
		"alias":      alias,
		"old_target": oldTarget,
		"new_target": req.Target,
		"resolved":   req.Target,
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
	if err := SaveLLMAliases(a.projectRoot, newAliases); err != nil {
		writeError(w, 500, fmt.Sprintf("persist alias deletion: %s", err))
		return
	}
	a.replaceAliases(newAliases)

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
