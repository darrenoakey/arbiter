package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func aliasTestModel() ModelConfig {
	return ModelConfig{
		MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(0),
		MaxRuntimeSec: 600, PressureIndex: floatPtr(1),
	}
}

func floatPtr(value float64) *float64 {
	return &value
}

func configureAliasTestAPI(api *API) {
	api.config.Models["llm:qwen"] = aliasTestModel()
	api.config.Models["llm:gemma"] = aliasTestModel()
	api.replaceAliases(map[string]string{"local-chat": "llm:qwen"})
}

func performRequest(api *API, method, target, body string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(method, target, strings.NewReader(body))
	response := httptest.NewRecorder()
	api.Handler().ServeHTTP(response, request)
	return response
}

func decodeObject(t *testing.T, data []byte) map[string]any {
	t.Helper()
	var object map[string]any
	if err := json.Unmarshal(data, &object); err != nil {
		t.Fatalf("decode response: %v: %s", err, data)
	}
	return object
}

func TestValidateLLMAliasesPolicy(t *testing.T) {
	models := map[string]ModelConfig{
		"llm:qwen":      aliasTestModel(),
		"llm:local-hit": aliasTestModel(),
	}
	tests := []struct {
		name    string
		aliases map[string]string
	}{
		{name: "unknown target", aliases: map[string]string{"local-chat": "llm:missing"}},
		{name: "bare target", aliases: map[string]string{"local-chat": "qwen"}},
		{name: "chain", aliases: map[string]string{"local-chat": "llm:local-other", "local-other": "llm:qwen"}},
		{name: "model shadow", aliases: map[string]string{"local-hit": "llm:qwen"}},
		{name: "malformed prefix", aliases: map[string]string{"cloud-chat": "llm:qwen"}},
		{name: "malformed case", aliases: map[string]string{"local-Chat": "llm:qwen"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := validateLLMAliases(test.aliases, models); err == nil {
				t.Fatal("invalid alias map accepted")
			}
		})
	}
	if err := validateLLMAliases(map[string]string{"local-chat": "llm:qwen"}, models); err != nil {
		t.Fatalf("valid alias rejected: %v", err)
	}
}

func TestAliasConfigPersistenceRoundTrip(t *testing.T) {
	root := t.TempDir()
	models := map[string]ModelConfig{"llm:qwen": aliasTestModel()}
	writeConfigFixture(t, root, models, nil)
	aliases := map[string]string{"local-chat": "llm:qwen"}
	if err := SaveLLMAliases(root, aliases); err != nil {
		t.Fatalf("save aliases: %v", err)
	}
	loaded, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("reload config: %v", err)
	}
	if got := loaded.LLMAliases["local-chat"]; got != "llm:qwen" {
		t.Fatalf("reloaded target = %q", got)
	}
}

func TestAliasConfigRejectsDuplicateNormalizedNames(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "local")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"models":{},"llm_aliases":{"local-chat":"llm:qwen","local-chat":"llm:gemma"}}`
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(root); err == nil || !strings.Contains(err.Error(), "duplicate normalized") {
		t.Fatalf("duplicate alias error = %v", err)
	}
}

func TestAliasResolutionAndCanonicalization(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)

	tests := []struct {
		requested string
		canonical string
		alias     string
		ok        bool
	}{
		{requested: "llm:qwen", canonical: "llm:qwen", ok: true},
		{requested: "qwen", canonical: "llm:qwen", ok: true},
		{requested: "local-chat", canonical: "llm:qwen", alias: "local-chat", ok: true},
		{requested: "missing", ok: false},
	}
	for _, test := range tests {
		canonical, alias, ok := api.resolveLLMModelID(test.requested)
		if canonical != test.canonical || alias != test.alias || ok != test.ok {
			t.Fatalf("resolve(%q) = %q, %q, %v", test.requested, canonical, alias, ok)
		}
	}

	concrete := []byte("{ \"model\": \"qwen\", \"messages\": [] }\n")
	rewritten, err := canonicalizeChatBody(concrete, "llm:qwen")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(concrete, rewritten) {
		t.Fatalf("concrete body changed bytes: %q", rewritten)
	}
	aliasBody, err := canonicalizeChatBody([]byte(`{"model":"local-chat"}`), "llm:qwen")
	if err != nil || !bytes.Contains(aliasBody, []byte(`"model":"qwen"`)) {
		t.Fatalf("alias body = %s, error = %v", aliasBody, err)
	}
}

func TestAliasAsyncTopLevelNestedAndUnknownResolution(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	api.llmCache = nil

	top := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","model":"local-chat","params":{"messages":[]}}`)
	assertSubmittedModel(t, api, top, "llm:qwen", "local-chat")

	nested := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","params":{"model":"local-chat","messages":[],"force":true}}`)
	assertSubmittedModel(t, api, nested, "llm:qwen", "local-chat")

	unknown := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","params":{"model":"local-missing","messages":[]}}`)
	if unknown.Code != http.StatusNotFound {
		t.Fatalf("nested unknown status = %d, body = %s", unknown.Code, unknown.Body.String())
	}
	topUnknown := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","model":"local-missing","params":{"messages":[]}}`)
	if topUnknown.Code != http.StatusNotFound {
		t.Fatalf("top-level unknown status = %d, body = %s", topUnknown.Code, topUnknown.Body.String())
	}

	previousDefault, hadDefault := JobTypeToModel["chat-completion"]
	JobTypeToModel["chat-completion"] = "llm:qwen"
	t.Cleanup(func() {
		if hadDefault {
			JobTypeToModel["chat-completion"] = previousDefault
		} else {
			delete(JobTypeToModel, "chat-completion")
		}
	})
	defaulted := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","params":{"messages":[],"force":true}}`)
	if defaulted.Code != http.StatusOK {
		t.Fatalf("absent nested model status = %d, body = %s", defaulted.Code, defaulted.Body.String())
	}
	defaultJob, err := api.store.GetJob(decodeObject(t, defaulted.Body.Bytes())["job_id"].(string))
	if err != nil || defaultJob.ModelID != "llm:qwen" || defaultJob.RequestedModel != "" {
		t.Fatalf("defaulted job = %+v, error = %v", defaultJob, err)
	}
}

func assertSubmittedModel(t *testing.T, api *API, response *httptest.ResponseRecorder, modelID, requested string) {
	t.Helper()
	if response.Code != http.StatusOK {
		t.Fatalf("submit status = %d, body = %s", response.Code, response.Body.String())
	}
	if response.Header().Get("X-Arbiter-Resolved-Model") != modelID {
		t.Fatalf("submit resolved header = %q", response.Header().Get("X-Arbiter-Resolved-Model"))
	}
	body := decodeObject(t, response.Body.Bytes())
	if body["status"] != "queued" {
		t.Fatalf("submission deduplicated unexpectedly: %s", response.Body.String())
	}
	job, err := api.store.GetJob(body["job_id"].(string))
	if err != nil {
		t.Fatal(err)
	}
	if job.ModelID != modelID || job.RequestedModel != requested {
		t.Fatalf("job model/requested = %q/%q", job.ModelID, job.RequestedModel)
	}
	expectedBare := strings.TrimPrefix(modelID, "llm:")
	if !bytes.Contains(job.Payload, []byte(fmt.Sprintf(`"model":"%s"`, expectedBare))) {
		t.Fatalf("worker payload is not canonical: %s", job.Payload)
	}
}

func TestAliasCacheEchoAndStreamingHeaders(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	body := []byte(`{"model":"qwen","messages":[{"role":"user","content":"same"}]}`)
	key, err := api.llmCache.Key(body)
	if err != nil {
		t.Fatal(err)
	}
	result := chatResultWithModel("worker/raw-tag", "answer")
	if err := api.llmCache.Put(key, result); err != nil {
		t.Fatal(err)
	}

	aliasResponse := performRequest(api, http.MethodPost, "/v1/chat/completions",
		`{"model":"local-chat","messages":[{"role":"user","content":"same"}]}`)
	assertEchoAndHeaders(t, aliasResponse, "local-chat", "llm:qwen", "local-chat")
	concreteResponse := performRequest(api, http.MethodPost, "/v1/chat/completions",
		`{"model":"qwen","messages":[{"role":"user","content":"same"}]}`)
	assertEchoAndHeaders(t, concreteResponse, "qwen", "llm:qwen", "")

	streamResponse := performRequest(api, http.MethodPost, "/v1/chat/completions",
		`{"model":"local-chat","stream":true,"messages":[{"role":"user","content":"same"}]}`)
	assertEchoAndHeaders(t, streamResponse, "worker/raw-tag", "llm:qwen", "local-chat")
	if !strings.Contains(streamResponse.Body.String(), `"model":"worker/raw-tag"`) {
		t.Fatalf("stream chunks were rewritten: %s", streamResponse.Body.String())
	}

	asyncResponse := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","params":{"model":"local-chat","messages":[{"role":"user","content":"same"}]}}`)
	asyncBody := decodeObject(t, asyncResponse.Body.Bytes())
	if asyncBody["status"] != "completed" {
		t.Fatalf("async cache response = %s", asyncResponse.Body.String())
	}
	retrieved := performRequest(api, http.MethodGet, "/v1/jobs/"+asyncBody["job_id"].(string), "")
	assertNestedJobEcho(t, retrieved, "local-chat")
}

func TestAliasFreshSyncResponseEcho(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.llmCache = nil
	workerPath := filepath.Join(api.projectRoot, "llm-worker")
	writeFreshChatWorker(t, workerPath)
	config := aliasTestModel()
	config.MaxInstances = intPtr(1)
	config.WorkerCmd = []string{workerPath}
	config.AdapterParams = map[string]string{"LLM_BACKEND": "llamacpp"}
	api.config.Models["llm:qwen"] = config
	api.replaceAliases(map[string]string{"local-chat": "llm:qwen"})
	api.mgr.ScaleModel("llm:qwen", 1, config)

	schedulerContext, cancel := context.WithCancel(context.Background())
	defer cancel()
	go api.scheduler.Run(schedulerContext)
	response := performRequest(api, http.MethodPost, "/v1/chat/completions",
		`{"model":"local-chat","messages":[{"role":"user","content":"fresh"}]}`)
	assertEchoAndHeaders(t, response, "local-chat", "llm:qwen", "local-chat")
	if response.Header().Get("X-Arbiter-Cache") != "miss" {
		t.Fatalf("fresh cache header = %q", response.Header().Get("X-Arbiter-Cache"))
	}
}

func assertEchoAndHeaders(t *testing.T, response *httptest.ResponseRecorder, model, resolved, alias string) {
	t.Helper()
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if response.Header().Get("X-Arbiter-Resolved-Model") != resolved {
		t.Fatalf("resolved header = %q", response.Header().Get("X-Arbiter-Resolved-Model"))
	}
	if response.Header().Get("X-Arbiter-Alias") != alias {
		t.Fatalf("alias header = %q", response.Header().Get("X-Arbiter-Alias"))
	}
	if !strings.Contains(response.Body.String(), fmt.Sprintf(`"model":"%s"`, model)) {
		t.Fatalf("response model does not echo %q: %s", model, response.Body.String())
	}
}

func TestAliasDedupSubscribersRetainOwnRequestedModel(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	api.llmCache = nil
	api.store.InitDedup()

	firstResponse := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","params":{"model":"local-chat","messages":[{"role":"user","content":"dedup"}]}}`)
	secondResponse := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","params":{"model":"qwen","messages":[{"role":"user","content":"dedup"}]}}`)
	firstID := decodeObject(t, firstResponse.Body.Bytes())["job_id"].(string)
	secondID := decodeObject(t, secondResponse.Body.Bytes())["job_id"].(string)
	result := chatResultWithModel("worker/raw-tag", "done")
	if err := api.store.UpdateState(firstID, "completed", WithResult(result), WithFinishedAt(nowTS())); err != nil {
		t.Fatal(err)
	}
	api.store.ResolveFollowers(firstID, "completed", &result, "", api.outputDir)

	first := performRequest(api, http.MethodGet, "/v1/jobs/"+firstID, "")
	second := performRequest(api, http.MethodGet, "/v1/jobs/"+secondID, "")
	assertNestedJobEcho(t, first, "local-chat")
	assertNestedJobEcho(t, second, "qwen")
}

func assertNestedJobEcho(t *testing.T, response *httptest.ResponseRecorder, requested string) {
	t.Helper()
	body := decodeObject(t, response.Body.Bytes())
	result := body["result"].(map[string]any)
	completion := result["response"].(map[string]any)
	if completion["model"] != requested || body["requested_model"] != requested {
		t.Fatalf("requested model not retained: %s", response.Body.String())
	}
}

func TestAliasRemapChangesAdmissionCacheAndDedupIdentity(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	api.llmCache = nil
	api.store.InitDedup()

	before := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","params":{"model":"local-chat","messages":[{"role":"user","content":"remap"}]}}`)
	assertSubmittedModel(t, api, before, "llm:qwen", "local-chat")
	update := performRequest(api, http.MethodPut, "/v1/llm/aliases/local-chat", `{"target":"llm:gemma"}`)
	if update.Code != http.StatusOK {
		t.Fatalf("remap status = %d, body = %s", update.Code, update.Body.String())
	}
	after := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","params":{"model":"local-chat","messages":[{"role":"user","content":"remap"}]}}`)
	assertSubmittedModel(t, api, after, "llm:gemma", "local-chat")

	qwenBody, _ := canonicalizeChatBody([]byte(`{"model":"local-chat","messages":[]}`), "llm:qwen")
	gemmaBody, _ := canonicalizeChatBody([]byte(`{"model":"local-chat","messages":[]}`), "llm:gemma")
	qwenKey, _ := NewLLMCache(t.TempDir(), 1).Key(qwenBody)
	gemmaKey, _ := NewLLMCache(t.TempDir(), 1).Key(gemmaBody)
	if qwenKey == gemmaKey {
		t.Fatal("cache identity survived target remap")
	}
	if computeJobHash("chat-completion", "llm:qwen", qwenBody) ==
		computeJobHash("chat-completion", "llm:gemma", gemmaBody) {
		t.Fatal("dedup identity survived target remap")
	}
}

func TestConcreteModelsDoNotDeduplicateAcrossTopLevelModel(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	api.llmCache = nil
	api.store.InitDedup()
	qwen := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","model":"qwen","params":{"messages":[{"role":"user","content":"same"}]}}`)
	gemma := performRequest(api, http.MethodPost, "/v1/jobs",
		`{"type":"chat-completion","model":"gemma","params":{"messages":[{"role":"user","content":"same"}]}}`)
	assertSubmittedModel(t, api, qwen, "llm:qwen", "qwen")
	assertSubmittedModel(t, api, gemma, "llm:gemma", "gemma")
}

func TestAliasManagementGuardsAndReverseDiscovery(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	writeConfigFixture(t, api.projectRoot, api.config.Models, api.aliasSnapshot())

	recent, err := api.store.CreateJobWithRequestedModel("llm:qwen", "chat-completion", json.RawMessage(`{}`), 0, "local-chat")
	if err != nil || recent == nil {
		t.Fatalf("create recent alias job: %v", err)
	}
	blockedDelete := performRequest(api, http.MethodDelete, "/v1/llm/aliases/local-chat", "")
	if blockedDelete.Code != http.StatusConflict {
		t.Fatalf("guarded alias delete = %d, body = %s", blockedDelete.Code, blockedDelete.Body.String())
	}

	collision := performRequest(api, http.MethodPost, "/v1/models", `{"model_id":"llm:local-chat"}`)
	if collision.Code != http.StatusConflict {
		t.Fatalf("reverse registration guard = %d, body = %s", collision.Code, collision.Body.String())
	}

	blockedModel := performRequest(api, http.MethodDelete, "/v1/models/qwen", "")
	if blockedModel.Code != http.StatusConflict || !strings.Contains(blockedModel.Body.String(), "local-chat") {
		t.Fatalf("dependent model delete = %d, body = %s", blockedModel.Code, blockedModel.Body.String())
	}
	forcedModel := performRequest(api, http.MethodDelete, "/v1/models/qwen?force=1", "")
	if forcedModel.Code != http.StatusOK {
		t.Fatalf("forced model delete = %d, body = %s", forcedModel.Code, forcedModel.Body.String())
	}
	if _, exists := api.aliasSnapshot()["local-chat"]; exists {
		t.Fatal("forced model delete left dependent alias live")
	}
	reloaded, loadErr := LoadConfig(api.projectRoot)
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	if _, exists := reloaded.Models["llm:qwen"]; exists || len(reloaded.LLMAliases) != 0 {
		t.Fatalf("forced deletion did not persist atomically: %+v", reloaded)
	}
}

func TestAliasPersistFailureAndConcurrentReads(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	writeConfigFixture(t, api.projectRoot, api.config.Models, api.aliasSnapshot())
	restore := makeConfigStorageUnwritable(t, api.projectRoot)
	failed := performRequest(api, http.MethodPut, "/v1/llm/aliases/local-chat", `{"target":"llm:gemma"}`)
	restore()
	if failed.Code < 500 || api.aliasSnapshot()["local-chat"] != "llm:qwen" {
		t.Fatalf("failed persist changed live state: %d %+v", failed.Code, api.aliasSnapshot())
	}

	var wait sync.WaitGroup
	errors := make(chan string, 100)
	for index := 0; index < 50; index++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			response := performRequest(api, http.MethodGet, "/v1/llm/aliases", "")
			var body map[string]map[string]any
			if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
				errors <- err.Error()
				return
			}
			target := body["local-chat"]["target"]
			if target != "llm:qwen" && target != "llm:gemma" {
				errors <- fmt.Sprint(target)
			}
		}()
	}
	update := performRequest(api, http.MethodPut, "/v1/llm/aliases/local-chat", `{"target":"llm:gemma"}`)
	wait.Wait()
	close(errors)
	if update.Code != http.StatusOK {
		t.Fatalf("concurrent update = %d, body = %s", update.Code, update.Body.String())
	}
	for invalid := range errors {
		t.Fatalf("reader observed partial target %q", invalid)
	}
}

func TestAliasListConcurrentWithModelDeletionIsAtomic(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)

	const readerCount = 16
	const iterations = 100
	start := make(chan struct{})
	errors := make(chan string, readerCount*iterations)
	var wait sync.WaitGroup
	for range readerCount {
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			for range iterations {
				response := performRequest(api, http.MethodGet, "/v1/llm/aliases", "")
				var body map[string]map[string]any
				if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
					errors <- err.Error()
					continue
				}
				if len(body) == 0 {
					continue
				}
				entry, ok := body["local-chat"]
				if !ok || len(body) != 1 || entry["target"] != "llm:qwen" || entry["target_configured"] != true {
					errors <- response.Body.String()
				}
			}
		}()
	}
	wait.Add(1)
	go func() {
		defer wait.Done()
		<-start
		for index := range iterations {
			api.configMutationMu.Lock()
			if index%2 == 0 {
				api.config.Models["llm:qwen"] = aliasTestModel()
				api.replaceAliases(map[string]string{"local-chat": "llm:qwen"})
			} else {
				delete(api.config.Models, "llm:qwen")
				api.replaceAliases(map[string]string{})
			}
			api.configMutationMu.Unlock()
		}
	}()
	close(start)
	wait.Wait()
	close(errors)
	for invalid := range errors {
		t.Fatalf("reader observed non-atomic alias/model state: %s", invalid)
	}
}

func TestAliasConcurrentUpdatesSerializeWithoutLostEntries(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	api.replaceAliases(map[string]string{
		"local-chat": "llm:qwen",
		"local-code": "llm:qwen",
	})
	writeConfigFixture(t, api.projectRoot, api.config.Models, api.aliasSnapshot())

	var wait sync.WaitGroup
	statuses := make(chan int, 2)
	for _, alias := range []string{"local-chat", "local-code"} {
		wait.Add(1)
		go func(alias string) {
			defer wait.Done()
			response := performRequest(api, http.MethodPut, "/v1/llm/aliases/"+alias, `{"target":"llm:gemma"}`)
			statuses <- response.Code
		}(alias)
	}
	wait.Wait()
	close(statuses)
	for status := range statuses {
		if status != http.StatusOK {
			t.Fatalf("concurrent PUT status = %d", status)
		}
	}
	aliases := api.aliasSnapshot()
	if aliases["local-chat"] != "llm:gemma" || aliases["local-code"] != "llm:gemma" {
		t.Fatalf("concurrent PUT lost an update: %+v", aliases)
	}
}

func TestAliasDeleteForceOverridesRecentTrafficGuard(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	writeConfigFixture(t, api.projectRoot, api.config.Models, api.aliasSnapshot())
	if _, err := api.store.CreateJobWithRequestedModel(
		"llm:qwen", "chat-completion", json.RawMessage(`{}`), 0, "local-chat",
	); err != nil {
		t.Fatal(err)
	}
	response := performRequest(api, http.MethodDelete, "/v1/llm/aliases/local-chat?force=1", "")
	if response.Code != http.StatusOK {
		t.Fatalf("forced alias delete = %d, body = %s", response.Code, response.Body.String())
	}
	if _, exists := api.aliasSnapshot()["local-chat"]; exists {
		t.Fatal("forced alias delete left live entry")
	}
	loaded, err := LoadConfig(api.projectRoot)
	if err != nil || len(loaded.LLMAliases) != 0 {
		t.Fatalf("forced alias delete did not persist: aliases=%v error=%v", loaded.LLMAliases, err)
	}
}

func TestAliasDiscoveryUsesSyntheticEntriesNotModels(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	configureAliasTestAPI(api)
	response := performRequest(api, http.MethodGet, "/v1/llm/models", "")
	var entries []map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &entries); err != nil {
		t.Fatal(err)
	}
	aliasEntries := 0
	for _, entry := range entries {
		if entry["name"] == "local-chat" {
			aliasEntries++
			if entry["alias_for"] != "llm:qwen" {
				t.Fatalf("alias_for = %v", entry["alias_for"])
			}
		}
	}
	if aliasEntries != 1 {
		t.Fatalf("synthetic alias entries = %d", aliasEntries)
	}
	if _, exists := api.config.Models["llm:local-chat"]; exists {
		t.Fatal("alias was represented as a pseudo-model")
	}
	modelResponse := performRequest(api, http.MethodGet, "/v1/models/qwen", "")
	modelBody := decodeObject(t, modelResponse.Body.Bytes())
	reverseAliases := modelBody["aliases"].([]any)
	if len(reverseAliases) != 1 || reverseAliases[0] != "local-chat" {
		t.Fatalf("model reverse aliases = %v", reverseAliases)
	}
	aliasManagement := performRequest(api, http.MethodGet, "/v1/models/local-chat", "")
	if aliasManagement.Code != http.StatusNotFound {
		t.Fatalf("management route accepted alias: %d", aliasManagement.Code)
	}

	config := api.config.Models["llm:qwen"]
	config.MaxInstances = intPtr(1)
	api.config.Models["llm:qwen"] = config
	api.mgr.ScaleModel("llm:qwen", 1, config)
	api.updatePSCache()
	status := performRequest(api, http.MethodGet, "/v1/ps", "")
	statusBody := decodeObject(t, status.Body.Bytes())
	foundReverse := false
	for _, rawModel := range statusBody["models"].([]any) {
		model := rawModel.(map[string]any)
		if model["id"] == "llm:qwen" {
			foundReverse = len(model["aliases"].([]any)) == 1
		}
	}
	if !foundReverse {
		t.Fatalf("/v1/ps omitted reverse aliases: %s", status.Body.String())
	}
}

func TestRequestedModelMigrationAndStreamingRowPersistence(t *testing.T) {
	root := t.TempDir()
	databasePath := filepath.Join(root, "old.db")
	database, err := sql.Open("sqlite", databasePath)
	if err != nil {
		t.Fatal(err)
	}
	_, err = database.Exec(`CREATE TABLE jobs (
		id TEXT PRIMARY KEY, model_id TEXT NOT NULL, job_type TEXT NOT NULL,
		state TEXT NOT NULL DEFAULT 'queued', priority REAL NOT NULL DEFAULT 0,
		payload TEXT NOT NULL DEFAULT '{}', result TEXT, error TEXT,
		created_at REAL NOT NULL, started_at REAL, finished_at REAL,
		canonical_job_id TEXT, excluded_hosts TEXT
	)`)
	if closeErr := database.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if err != nil {
		t.Fatal(err)
	}
	store, err := NewStore(databasePath)
	if err != nil {
		t.Fatalf("migrate store: %v", err)
	}
	defer store.Close()
	job, err := store.CreateJobWithRequestedModel(
		"llm:qwen", "chat-completion-stream", json.RawMessage(`{"model":"qwen"}`), 0, "local-chat",
	)
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := store.GetJob(job.ID)
	if err != nil || reloaded.RequestedModel != "local-chat" {
		t.Fatalf("stream requested model = %q, error = %v", reloaded.RequestedModel, err)
	}
}

func chatResultWithModel(model, content string) json.RawMessage {
	response := map[string]any{
		"id": "chatcmpl-alias", "model": model,
		"choices": []map[string]any{{
			"message": map[string]any{"role": "assistant", "content": content},
		}},
	}
	responseBytes, _ := json.Marshal(response)
	result, _ := json.Marshal(map[string]any{"format": "json", "response": json.RawMessage(responseBytes)})
	return result
}

func writeFreshChatWorker(t *testing.T, path string) {
	t.Helper()
	script := `#!/usr/bin/env python3
import json
import sys
for line in sys.stdin:
    message = json.loads(line)
    command = message.get("cmd")
    request_id = message.get("req_id", "")
    if command == "load":
        print(json.dumps({"status":"ok","req_id":request_id}), flush=True)
    elif command == "infer":
        result = {
            "format":"json",
            "response":{
                "id":"fresh",
                "model":"worker/raw-tag",
                "choices":[{"message":{"role":"assistant","content":"fresh-answer"}}]
            }
        }
        print(json.dumps({"status":"ok","req_id":request_id,"result":result}), flush=True)
    elif command in ("unload", "shutdown"):
        print(json.dumps({"status":"ok","req_id":request_id}), flush=True)
        break
`
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
}

func writeConfigFixture(t *testing.T, root string, models map[string]ModelConfig, aliases map[string]string) {
	t.Helper()
	directory := filepath.Join(root, "local")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	workerPath := filepath.Join(root, "llm-worker")
	if err := os.WriteFile(workerPath, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	persistedModels := make(map[string]ModelConfig, len(models))
	for modelID, config := range models {
		if strings.HasPrefix(modelID, "llm:") {
			config.WorkerCmd = []string{workerPath}
			config.AdapterParams = map[string]string{"LLM_BACKEND": "llamacpp"}
		}
		persistedModels[modelID] = config
	}
	data, err := json.Marshal(map[string]any{
		"vram_budget_gb": 100,
		"models":         persistedModels,
		"llm_aliases":    aliases,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "config.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}
