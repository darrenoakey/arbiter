package main

import (
	"net/http"
	"slices"
)

// APIMajorVersion is the served HTTP transport major version.
const APIMajorVersion = 1

const (
	versioningTransport    = "v1"
	versioningIndependent  = "independent"
	versioningAdditive     = "additive-minor-only"
	aliasRenameOverlapNote = "requires overlap window"
)

type capabilitiesResponse struct {
	APIVersion   int                    `json:"api_version"`
	JobTypes     []string               `json:"job_types"`
	ModelAliases map[string]aliasTarget `json:"model_aliases"`
	Versioning   capabilitiesVersioning `json:"versioning"`
}

type aliasTarget struct {
	Target string `json:"target"`
}

type capabilitiesVersioning struct {
	Transport    string `json:"transport"`
	JobTypes     string `json:"job_types"`
	ModelAliases string `json:"model_aliases"`
	Changes      string `json:"changes"`
	AliasRename  string `json:"alias_rename"`
}

func capabilitiesVersioningInfo() capabilitiesVersioning {
	return capabilitiesVersioning{
		Transport:    versioningTransport,
		JobTypes:     versioningIndependent,
		ModelAliases: versioningIndependent,
		Changes:      versioningAdditive,
		AliasRename:  aliasRenameOverlapNote,
	}
}

func liveJobTypes() []string {
	types := make([]string, 0, len(JobTypeToModel))
	for jobType := range JobTypeToModel {
		types = append(types, jobType)
	}
	slices.Sort(types)
	return types
}

func (a *API) liveModelAliases() map[string]aliasTarget {
	snapshot := a.aliasSnapshot()
	out := make(map[string]aliasTarget, len(snapshot))
	for alias, target := range snapshot {
		out[alias] = aliasTarget{Target: target}
	}
	return out
}

func (a *API) capabilitiesPayload() capabilitiesResponse {
	return capabilitiesResponse{
		APIVersion:   APIMajorVersion,
		JobTypes:     liveJobTypes(),
		ModelAliases: a.liveModelAliases(),
		Versioning:   capabilitiesVersioningInfo(),
	}
}

func (a *API) capabilities(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, a.capabilitiesPayload())
}
