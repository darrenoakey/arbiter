"""Caller-provenance (who/why) helpers in arbiter.client.

The server-side persistence and API surface are covered by the Go integration
tests (cmd/arbiter/job_source_test.go); these tests pin the client-side
ambient/explicit merge rules that every Python caller inherits.
"""

from arbiter.client import MAX_WHY_BYTES, MAX_WHO_BYTES, _effective_source, ambient_source


class TestAmbientSource:
    def test_absent_env_means_no_source(self, monkeypatch):
        monkeypatch.delenv("ARBITER_WHO", raising=False)
        monkeypatch.delenv("ARBITER_WHY", raising=False)
        assert ambient_source() is None

    def test_env_who_and_why_are_inherited(self, monkeypatch):
        monkeypatch.setenv("ARBITER_WHO", "waggler")
        monkeypatch.setenv("ARBITER_WHY", "hang williams video")
        assert ambient_source() == {"who": "waggler", "why": "hang williams video"}

    def test_whitespace_only_env_is_ignored(self, monkeypatch):
        monkeypatch.setenv("ARBITER_WHO", "   ")
        monkeypatch.setenv("ARBITER_WHY", "")
        assert ambient_source() is None

    def test_env_values_are_capped(self, monkeypatch):
        monkeypatch.setenv("ARBITER_WHO", "x" * 500)
        monkeypatch.setenv("ARBITER_WHY", "y" * 500)
        source = ambient_source()
        assert len(source["who"]) == MAX_WHO_BYTES
        assert len(source["why"]) == MAX_WHY_BYTES


class TestEffectiveSource:
    def test_explicit_overrides_ambient(self, monkeypatch):
        monkeypatch.setenv("ARBITER_WHO", "ambient-owner")
        monkeypatch.setenv("ARBITER_WHY", "ambient reason")
        source = _effective_source("waggler", "spring song")
        assert source == {"who": "waggler", "why": "spring song"}

    def test_explicit_partial_keeps_other_ambient_field(self, monkeypatch):
        monkeypatch.setenv("ARBITER_WHY", "hang williams video")
        source = _effective_source("beezle3", None)
        assert source == {"who": "beezle3", "why": "hang williams video"}

    def test_explicit_empty_string_clears_ambient_field(self, monkeypatch):
        monkeypatch.setenv("ARBITER_WHY", "ambient reason")
        source = _effective_source(None, "")
        assert source is None

    def test_no_explicit_no_ambient_is_none(self, monkeypatch):
        monkeypatch.delenv("ARBITER_WHO", raising=False)
        monkeypatch.delenv("ARBITER_WHY", raising=False)
        assert _effective_source(None, None) is None

    def test_who_only(self):
        assert _effective_source("photo-namer", None) == {"who": "photo-namer"}
