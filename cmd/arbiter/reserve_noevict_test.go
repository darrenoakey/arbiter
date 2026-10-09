package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

func reserveSnapshot(m *InstanceManager) (reserved float64, used float64, reservations, instances int) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.reservedGB, m.usedGB, len(m.reservations), len(m.instances)
}

func TestCreateReservationNoEvictConcurrentStaysUnderBudget(t *testing.T) {
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 70}, "python3", t.TempDir())
	var wg sync.WaitGroup
	var mu sync.Mutex
	ok, insufficient := 0, 0
	for i := 0; i < 40; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := mgr.CreateReservationNoEvict(10, fmt.Sprintf("c%d", i))
			mu.Lock()
			defer mu.Unlock()
			switch {
			case err == nil:
				ok++
			case errors.Is(err, ErrReservationInsufficient):
				insufficient++
			default:
				t.Errorf("unexpected error: %v", err)
			}
		}(i)
	}
	wg.Wait()
	reserved, _, count, _ := reserveSnapshot(mgr)
	if ok != 7 || insufficient != 33 || count != 7 || reserved != 70 {
		t.Fatalf("ok=%d insufficient=%d count=%d reserved=%v", ok, insufficient, count, reserved)
	}
}

func TestCreateReservationNoEvictFailureLeavesStateUntouched(t *testing.T) {
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 70}, "python3", t.TempDir())
	mgr.mu.Lock()
	mgr.usedGB = 60
	mgr.mu.Unlock()
	r0, u0, c0, i0 := reserveSnapshot(mgr)
	if _, err := mgr.CreateReservationNoEvict(11, "too-big"); !errors.Is(err, ErrReservationInsufficient) {
		t.Fatalf("want insufficient, got %v", err)
	}
	if _, err := mgr.CreateReservationNoEvict(-1, "bad"); err == nil {
		t.Fatal("negative accepted")
	}
	r1, u1, c1, i1 := reserveSnapshot(mgr)
	if r0 != r1 || u0 != u1 || c0 != c1 || i0 != i1 {
		t.Fatalf("state changed: %v/%v/%v/%v -> %v/%v/%v/%v", r0, u0, c0, i0, r1, u1, c1, i1)
	}
	if _, err := mgr.CreateReservationNoEvict(10, "fits-exactly"); err != nil {
		t.Fatalf("exact fit rejected: %v", err)
	}
}

func reserveReq(t *testing.T, h http.Handler, method, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		_ = json.NewEncoder(&buf).Encode(body)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(method, path, &buf))
	return rec
}

func TestReserveHTTPCapabilityNoEvictAndDefaultCompat(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	srv := httptest.NewServer(api.Handler())
	defer srv.Close()
	h := api.Handler()

	rec := reserveReq(t, h, http.MethodGet, "/v1/reserve/capabilities", nil)
	var caps map[string]any
	if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &caps) != nil || caps["atomic_non_evicting"] != true {
		t.Fatalf("capabilities: %d %s", rec.Code, rec.Body.String())
	}

	rec = reserveReq(t, h, http.MethodGet, "/v1/capabilities", nil)
	var full struct {
		Reservations struct {
			NoEvict bool `json:"no_evict"`
			Atomic  bool `json:"atomic"`
		} `json:"reservations"`
	}
	if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &full) != nil || !full.Reservations.NoEvict || !full.Reservations.Atomic {
		t.Fatalf("/v1/capabilities reservations: %d %s", rec.Code, rec.Body.String())
	}

	// Real TCP request too.
	resp, err := http.Get(srv.URL + "/v1/reserve/capabilities")
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("tcp capabilities: %v %v", err, resp)
	}
	_ = resp.Body.Close()

	// no_evict success.
	rec = reserveReq(t, h, http.MethodPost, "/v1/reserve", map[string]any{"memory_gb": 60, "label": "a", "no_evict": true})
	var ok map[string]any
	if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &ok) != nil || ok["reservation_id"] == "" || ok["no_evict"] != true || fmt.Sprint(ok["evicted"]) != "[]" {
		t.Fatalf("no_evict reserve: %d %s", rec.Code, rec.Body.String())
	}
	// no_evict conflict (budget 100, 60 reserved).
	rec = reserveReq(t, h, http.MethodPost, "/v1/reserve", map[string]any{"memory_gb": 50, "no_evict": true})
	var bad map[string]any
	if rec.Code != 409 || json.Unmarshal(rec.Body.Bytes(), &bad) != nil || bad["code"] != "insufficient" {
		t.Fatalf("no_evict conflict: %d %s", rec.Code, rec.Body.String())
	}
	if r, _, c, _ := reserveSnapshot(api.mgr); r != 60 || c != 1 {
		t.Fatalf("conflict changed state: reserved=%v count=%d", r, c)
	}
	// Default semantics unchanged: omitted no_evict still succeeds when it fits,
	// and the response carries no no_evict field.
	rec = reserveReq(t, h, http.MethodPost, "/v1/reserve", map[string]any{"memory_gb": 30, "label": "d"})
	var def map[string]any
	if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &def) != nil || def["reservation_id"] == "" {
		t.Fatalf("default reserve: %d %s", rec.Code, rec.Body.String())
	}
	if _, has := def["no_evict"]; has {
		t.Fatalf("default response gained no_evict: %v", def)
	}
	// Default path with nothing evictable still fails with the old 409 shape.
	rec = reserveReq(t, h, http.MethodPost, "/v1/reserve", map[string]any{"memory_gb": 50})
	if rec.Code != 409 {
		t.Fatalf("default overflow: %d %s", rec.Code, rec.Body.String())
	}
}
