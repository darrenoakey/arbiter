package main

import (
	"encoding/json"
	"net/http"
	"slices"
	"testing"
)

func TestCapabilitiesSchema(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	body := getCapabilities(t, api)
	if body.APIVersion != APIMajorVersion {
		t.Fatalf("api_version = %d, want %d", body.APIVersion, APIMajorVersion)
	}
	if len(body.JobTypes) == 0 {
		t.Fatal("job_types is empty")
	}
	if body.ModelAliases == nil {
		t.Fatal("model_aliases is null")
	}
	assertVersioning(t, body.Versioning)
	if !slices.Contains(body.JobTypes, "caption") {
		t.Fatalf("live job types missing caption: %v", body.JobTypes)
	}
}

func TestCapabilitiesIncludesNewlyRegisteredJobType(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	const probe = "capabilities-probe-type"
	JobTypeToModel[probe] = "capabilities-probe-model"
	t.Cleanup(func() { delete(JobTypeToModel, probe) })

	body := getCapabilities(t, api)
	if !slices.Contains(body.JobTypes, probe) {
		t.Fatalf("newly registered job type %q missing from %v", probe, body.JobTypes)
	}
}

func TestCapabilitiesIncludesLiveAliasTarget(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	api.replaceAliases(map[string]string{"local-capabilities": "llm:probe"})
	t.Cleanup(func() { api.replaceAliases(map[string]string{}) })

	body := getCapabilities(t, api)
	got, ok := body.ModelAliases["local-capabilities"]
	if !ok || got.Target != "llm:probe" {
		t.Fatalf("model_aliases = %#v, want local-capabilities -> llm:probe", body.ModelAliases)
	}
}

func getCapabilities(t *testing.T, api *API) capabilitiesResponse {
	t.Helper()
	rec := performRequest(api, http.MethodGet, "/v1/capabilities", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var body capabilitiesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return body
}

func assertVersioning(t *testing.T, info capabilitiesVersioning) {
	t.Helper()
	if info.Transport != versioningTransport {
		t.Fatalf("versioning.transport = %q", info.Transport)
	}
	if info.JobTypes != versioningIndependent || info.ModelAliases != versioningIndependent {
		t.Fatalf("job types and aliases must version independently: %+v", info)
	}
	if info.Changes != versioningAdditive {
		t.Fatalf("versioning.changes = %q", info.Changes)
	}
	if info.AliasRename != aliasRenameOverlapNote {
		t.Fatalf("alias rename overlap note missing: %+v", info)
	}
}
