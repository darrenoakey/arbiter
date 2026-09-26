package main

import (
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestMiniMaxH3FrameAdmissionUsesCanonicalStagingPaths(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.config.Models["minimax-h3"] = ModelConfig{}
	api.refreshAliasModels()
	share := filepath.Join(t.TempDir(), "share")
	inbox := filepath.Join(share, "inbox")
	if err := os.MkdirAll(filepath.Join(inbox, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}
	api.config.ShareMount = share
	frame := filepath.Join(inbox, "frame.png")
	if err := os.WriteFile(frame, []byte("frame"), 0o644); err != nil {
		t.Fatal(err)
	}
	lastFrame := filepath.Join(inbox, "last.png")
	if err := os.WriteFile(lastFrame, []byte("last"), 0o644); err != nil {
		t.Fatal(err)
	}
	acceptedBody := fmt.Sprintf(
		`{"type":"video-generate","model":"minimax-h3","params":{"prompt":"shot","duration":15,"resolution":"2K","ratio":"9:16","first_frame_path":%q,"last_frame_path":%q}}`,
		frame, lastFrame,
	)
	accepted := performRequest(api, "POST", "/v1/jobs", acceptedBody)
	if accepted.Code != 400 {
		t.Fatalf("cloud MiniMax job status=%d body=%s", accepted.Code, accepted.Body.String())
	}

	outside := filepath.Join(t.TempDir(), "outside.png")
	if err := os.WriteFile(outside, []byte("outside"), 0o644); err != nil {
		t.Fatal(err)
	}
	symlink := filepath.Join(inbox, "linked.png")
	if err := os.Symlink(frame, symlink); err != nil {
		t.Fatal(err)
	}
	traversal := filepath.Join(inbox, "nested") + string(filepath.Separator) + ".." + string(filepath.Separator) + "frame.png"
	for name, path := range map[string]string{"outside": outside, "symlink": symlink, "traversal": traversal} {
		t.Run(name, func(t *testing.T) {
			response := submitMiniMaxFrame(t, api, path)
			if response.Code != 400 {
				t.Fatalf("unsafe frame status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func submitMiniMaxFrame(t *testing.T, api *API, path string) *httptest.ResponseRecorder {
	t.Helper()
	body := fmt.Sprintf(
		`{"type":"video-generate","model":"minimax-h3","params":{"prompt":"shot","duration":4,"resolution":"768P","first_frame_path":%q}}`,
		path,
	)
	return performRequest(api, "POST", "/v1/jobs", body)
}
