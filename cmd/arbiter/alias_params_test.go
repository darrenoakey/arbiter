package main

import (
	"encoding/json"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

func thinkingOffParams() map[string]any {
	return map[string]any{
		"reasoning_effort":     "none",
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	}
}

func TestValidateLLMAliasParamsPolicy(t *testing.T) {
	aliases := map[string]string{"local-summariser": "llm:qwen"}
	invalid := map[string]map[string]map[string]any{
		"unknown alias": {"local-missing": {"reasoning_effort": "none"}},
		"empty object":  {"local-summariser": {}},
		"sets model":    {"local-summariser": {"model": "llm:gemma"}},
		"sets messages": {"local-summariser": {"messages": []any{}}},
		"sets stream":   {"local-summariser": {"stream": true}},
	}
	for name, params := range invalid {
		t.Run(name, func(t *testing.T) {
			if err := validateLLMAliasParams(params, aliases); err == nil {
				t.Fatal("invalid alias params accepted")
			}
		})
	}
	valid := map[string]map[string]any{"local-summariser": thinkingOffParams()}
	if err := validateLLMAliasParams(valid, aliases); err != nil {
		t.Fatalf("valid alias params rejected: %v", err)
	}
}

func TestApplyAliasParamsRoleWinsAndMergesNestedObjects(t *testing.T) {
	body := []byte(`{"model":"qwen","messages":[],"reasoning_effort":"high","max_tokens":77,` +
		`"chat_template_kwargs":{"enable_thinking":true,"keep":1}}`)
	out, overridden, err := applyAliasParams(body, thinkingOffParams())
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if got["reasoning_effort"] != "none" || got["max_tokens"] != float64(77) || got["model"] != "qwen" {
		t.Fatalf("merged body = %s", out)
	}
	wantKwargs := map[string]any{"enable_thinking": false, "keep": float64(1)}
	if !reflect.DeepEqual(got["chat_template_kwargs"], wantKwargs) {
		t.Fatalf("chat_template_kwargs = %v, want %v", got["chat_template_kwargs"], wantKwargs)
	}
	if !reflect.DeepEqual(overridden, []string{"chat_template_kwargs", "reasoning_effort"}) {
		t.Fatalf("overridden = %v", overridden)
	}

	same, overridden, err := applyAliasParams([]byte(`{"reasoning_effort":"none"}`),
		map[string]any{"reasoning_effort": "none"})
	if err != nil || len(overridden) != 0 || !strings.Contains(string(same), `"reasoning_effort":"none"`) {
		t.Fatalf("matching caller value: body=%s overridden=%v err=%v", same, overridden, err)
	}
}

func TestAliasParamsEnforcedOnJobsButNotConcreteModels(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	api.llmCache = nil
	api.replaceAliasParams(map[string]map[string]any{"local-chat": thinkingOffParams()})

	viaAlias := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","params":{"model":"local-chat","messages":[{"role":"user","content":"a"}]}}`)
	job := submittedJob(t, api, viaAlias)
	var payload map[string]any
	if err := json.Unmarshal(job.Payload, &payload); err != nil {
		t.Fatal(err)
	}
	if payload["reasoning_effort"] != "none" || payload["model"] != "qwen" {
		t.Fatalf("alias job payload missing enforced params: %s", job.Payload)
	}
	kwargs, _ := payload["chat_template_kwargs"].(map[string]any)
	if kwargs["enable_thinking"] != false {
		t.Fatalf("alias job payload chat_template_kwargs = %v", payload["chat_template_kwargs"])
	}

	viaConcrete := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","params":{"model":"qwen","messages":[{"role":"user","content":"b"}]}}`)
	concreteJob := submittedJob(t, api, viaConcrete)
	if strings.Contains(string(concreteJob.Payload), "reasoning_effort") {
		t.Fatalf("concrete-model job was rewritten: %s", concreteJob.Payload)
	}
}

func TestAliasParamsShapeSyncCacheKey(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	api.replaceAliasParams(map[string]map[string]any{"local-chat": {"reasoning_effort": "none"}})

	// The cache holds the result for the body the role actually runs: the
	// canonical model plus the enforced params. A bare alias call must hit it.
	enforced := []byte(`{"messages":[{"role":"user","content":"same"}],"model":"qwen","reasoning_effort":"none"}`)
	key, err := api.llmCache.Key(enforced)
	if err != nil {
		t.Fatal(err)
	}
	if err := api.llmCache.Put(key, chatResultWithModel("worker/raw-tag", "answer")); err != nil {
		t.Fatal(err)
	}
	hit := performRequest(api, http.MethodPost, "/v1/chat/completions",
		`{"model":"local-chat","messages":[{"role":"user","content":"same"}]}`)
	assertEchoAndHeaders(t, hit, "local-chat", "llm:qwen", "local-chat")
	if hit.Header().Get("X-Arbiter-Cache") != "hit" {
		t.Fatalf("bare alias call did not hit the enforced-body cache entry: cache=%q",
			hit.Header().Get("X-Arbiter-Cache"))
	}
}

func TestPutAliasPersistsParamsListsThemAndClears(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	writeConfigFixture(t, api.projectRoot, api.config.CloneModels(), map[string]string{"local-chat": "llm:qwen"})

	response := performRequest(api, http.MethodPut, "/v1/llm/aliases/local-chat",
		`{"target":"llm:qwen","params":{"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}}`)
	if response.Code != http.StatusOK {
		t.Fatalf("put alias status = %d: %s", response.Code, response.Body.String())
	}
	reloaded, err := LoadConfig(api.projectRoot)
	if err != nil {
		t.Fatalf("reload config: %v", err)
	}
	if !reflect.DeepEqual(reloaded.LLMAliasParams["local-chat"], thinkingOffParams()) {
		t.Fatalf("persisted params = %+v", reloaded.LLMAliasParams)
	}
	listed := decodeObject(t, performRequest(api, http.MethodGet, "/v1/llm/aliases", "").Body.Bytes())
	entry, _ := listed["local-chat"].(map[string]any)
	if !reflect.DeepEqual(entry["params"], thinkingOffParams()) {
		t.Fatalf("listed params = %v", entry["params"])
	}

	// Omitting params on retarget leaves them in place.
	if response := performRequest(api, http.MethodPut, "/v1/llm/aliases/local-chat", `{"target":"llm:gemma"}`); response.Code != http.StatusOK {
		t.Fatalf("retarget status = %d: %s", response.Code, response.Body.String())
	}
	if len(api.aliasParamsSnapshot("local-chat")) == 0 {
		t.Fatal("retarget without params dropped them")
	}

	reserved := performRequest(api, http.MethodPut, "/v1/llm/aliases/local-chat",
		`{"target":"llm:qwen","params":{"model":"llm:gemma"}}`)
	if reserved.Code != http.StatusBadRequest {
		t.Fatalf("reserved key status = %d: %s", reserved.Code, reserved.Body.String())
	}

	cleared := performRequest(api, http.MethodPut, "/v1/llm/aliases/local-chat", `{"target":"llm:qwen","params":{}}`)
	if cleared.Code != http.StatusOK {
		t.Fatalf("clear params status = %d: %s", cleared.Code, cleared.Body.String())
	}
	after, err := LoadConfig(api.projectRoot)
	if err != nil {
		t.Fatalf("reload after clear: %v", err)
	}
	if len(after.LLMAliasParams) != 0 || len(api.aliasParamsSnapshot("local-chat")) != 0 {
		t.Fatalf("params after explicit clear = %+v", after.LLMAliasParams)
	}
}

func TestDeleteAliasDropsParamsSoConfigReloads(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	writeConfigFixture(t, api.projectRoot, api.config.CloneModels(), map[string]string{"local-chat": "llm:qwen"})
	put := performRequest(api, http.MethodPut, "/v1/llm/aliases/local-chat",
		`{"target":"llm:qwen","params":{"reasoning_effort":"none"}}`)
	if put.Code != http.StatusOK {
		t.Fatalf("put alias status = %d: %s", put.Code, put.Body.String())
	}
	deleted := performRequest(api, http.MethodDelete, "/v1/llm/aliases/local-chat?force=1", "")
	if deleted.Code != http.StatusOK {
		t.Fatalf("delete alias status = %d: %s", deleted.Code, deleted.Body.String())
	}
	reloaded, err := LoadConfig(api.projectRoot)
	if err != nil {
		t.Fatalf("config with params for a deleted alias no longer loads: %v", err)
	}
	if len(reloaded.LLMAliasParams) != 0 {
		t.Fatalf("params survived alias deletion: %+v", reloaded.LLMAliasParams)
	}
}

func TestDeleteModelConfigDropsDependentAliasSettings(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	writeConfigFixture(t, api.projectRoot, api.config.CloneModels(), map[string]string{"local-chat": "llm:qwen"})
	if err := SaveLLMAliasFallbacks(api.projectRoot, map[string][]string{"local-chat": {"llm:gemma"}}); err != nil {
		t.Fatal(err)
	}
	if err := SaveLLMAliasParams(api.projectRoot, map[string]map[string]any{"local-chat": {"reasoning_effort": "none"}}); err != nil {
		t.Fatal(err)
	}
	if err := DeleteModelConfig(api.projectRoot, "llm:qwen", "local-chat"); err != nil {
		t.Fatal(err)
	}
	reloaded, err := LoadConfig(api.projectRoot)
	if err != nil {
		t.Fatalf("config no longer loads after deleting an aliased model: %v", err)
	}
	if len(reloaded.LLMAliases) != 0 || len(reloaded.LLMAliasFallbacks) != 0 || len(reloaded.LLMAliasParams) != 0 {
		t.Fatalf("dependent alias settings survived: aliases=%v fallbacks=%v params=%v",
			reloaded.LLMAliases, reloaded.LLMAliasFallbacks, reloaded.LLMAliasParams)
	}
}

func submittedJob(t *testing.T, api *API, response interface {
	Result() *http.Response
}) *Job {
	t.Helper()
	recorder := response.Result()
	defer recorder.Body.Close()
	var body map[string]any
	if err := json.NewDecoder(recorder.Body).Decode(&body); err != nil {
		t.Fatalf("decode submit response: %v", err)
	}
	if recorder.StatusCode != http.StatusOK {
		t.Fatalf("submit status = %d, body = %v", recorder.StatusCode, body)
	}
	jobID, _ := body["job_id"].(string)
	job, err := api.store.GetJob(jobID)
	if err != nil {
		t.Fatalf("get job %q: %v", jobID, err)
	}
	return job
}
