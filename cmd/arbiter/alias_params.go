package main

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
)

// aliasParamsReservedKeys are chat-body keys an alias may never set: they are
// routing (model), content (messages), or transport (stream) and belong to the
// caller and the alias layer itself, not to a role's generation policy.
var aliasParamsReservedKeys = []string{"model", "messages", "stream"}

// validateLLMAliasParams checks proposed per-alias enforced chat parameters.
// Each entry must name a configured alias, be a non-empty object, and avoid the
// reserved keys.
func validateLLMAliasParams(params map[string]map[string]any, aliases map[string]string) error {
	for alias, values := range params {
		if _, exists := aliases[alias]; !exists {
			return fmt.Errorf("params for %q name no configured alias", alias)
		}
		if len(values) == 0 {
			return fmt.Errorf("alias %q has empty params; omit the entry instead", alias)
		}
		for _, key := range aliasParamsReservedKeys {
			if _, set := values[key]; set {
				return fmt.Errorf("alias %q params may not set %q", alias, key)
			}
		}
	}
	return nil
}

// applyAliasParams merges a role's enforced parameters into a chat body. The
// role wins: an alias names *what the caller wants* (a summary, an extraction),
// so generation policy such as thinking and effort is a property of the role,
// not of whichever client happens to call it. Nested objects (for example
// chat_template_kwargs) are merged key-by-key so unrelated caller keys survive.
// It returns the rewritten body and the sorted top-level keys whose caller
// value was replaced by a different one.
func applyAliasParams(body []byte, params map[string]any) ([]byte, []string, error) {
	if len(params) == 0 {
		return body, nil, nil
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, nil, err
	}
	if m == nil {
		m = map[string]any{}
	}
	var overridden []string
	for key, value := range params {
		if previous, exists := m[key]; exists && !jsonEqual(previous, value) {
			overridden = append(overridden, key)
		}
		m[key] = mergeParamValue(m[key], value)
	}
	out, err := json.Marshal(m)
	if err != nil {
		return nil, nil, err
	}
	slices.Sort(overridden)
	return out, overridden, nil
}

// mergeParamValue returns the enforced value, recursively merging when both
// the caller's value and the enforced value are JSON objects.
func mergeParamValue(existing, enforced any) any {
	enforcedMap, enforcedIsMap := enforced.(map[string]any)
	existingMap, existingIsMap := existing.(map[string]any)
	if !enforcedIsMap || !existingIsMap {
		return enforced
	}
	merged := maps.Clone(existingMap)
	for key, value := range enforcedMap {
		merged[key] = mergeParamValue(merged[key], value)
	}
	return merged
}

func jsonEqual(a, b any) bool {
	left, errLeft := json.Marshal(a)
	right, errRight := json.Marshal(b)
	return errLeft == nil && errRight == nil && string(left) == string(right)
}

// cloneAliasParams deep-copies one alias's parameter object so callers can
// never mutate the live configuration.
func cloneAliasParams(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	data, err := json.Marshal(values)
	if err != nil {
		return nil
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil
	}
	return out
}

// aliasParamsSnapshot returns a deep copy of the enforced params for an alias,
// or nil when it has none.
func (a *API) aliasParamsSnapshot(alias string) map[string]any {
	if alias == "" {
		return nil
	}
	a.aliasMu.RLock()
	defer a.aliasMu.RUnlock()
	return cloneAliasParams(a.config.LLMAliasParams[alias])
}

// aliasParamsMapSnapshot returns a deep copy of every alias's enforced params.
func (a *API) aliasParamsMapSnapshot() map[string]map[string]any {
	a.aliasMu.RLock()
	defer a.aliasMu.RUnlock()
	out := make(map[string]map[string]any, len(a.config.LLMAliasParams))
	for alias, values := range a.config.LLMAliasParams {
		out[alias] = cloneAliasParams(values)
	}
	return out
}

// replaceAliasParams publishes a new params map under the alias lock.
func (a *API) replaceAliasParams(params map[string]map[string]any) {
	a.aliasMu.Lock()
	defer a.aliasMu.Unlock()
	next := make(map[string]map[string]any, len(params))
	for alias, values := range params {
		next[alias] = cloneAliasParams(values)
	}
	a.config.LLMAliasParams = next
}

// forgetAliases removes aliases and every per-alias setting (fallbacks and
// enforced params) from live state, used when a target model is deleted.
func (a *API) forgetAliases(names []string) {
	aliases := a.aliasSnapshot()
	fallbacks := a.aliasFallbacksSnapshot()
	params := a.aliasParamsMapSnapshot()
	for _, alias := range names {
		delete(aliases, alias)
		delete(fallbacks, alias)
		delete(params, alias)
	}
	a.replaceAliases(aliases)
	a.replaceAliasFallbacks(fallbacks)
	a.replaceAliasParams(params)
}

// enforceAliasChatParams applies the alias's enforced params to a canonical
// chat body and logs any caller value it replaced.
func (a *API) enforceAliasChatParams(body []byte, alias, modelID string) ([]byte, error) {
	params := a.aliasParamsSnapshot(alias)
	if len(params) == 0 {
		return body, nil
	}
	out, overridden, err := applyAliasParams(body, params)
	if err != nil {
		return nil, err
	}
	if len(overridden) > 0 {
		a.logger.Log("llm.alias_params_overrode", map[string]any{
			"alias":      alias,
			"model":      modelID,
			"overridden": overridden,
		})
	}
	return out, nil
}
