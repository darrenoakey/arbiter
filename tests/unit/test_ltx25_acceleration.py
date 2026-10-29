"""CPU contract tests for the opt-in LTX 2.5 acceleration benchmark.

These check parsing, caps, fixed weight identities, content attestation,
timing bookkeeping and the pinned upstream constants. They do not and cannot
show GPU correctness; that is verified by real Arbiter benchmark jobs.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import subprocess
import tarfile
import threading
import time
import uuid
from pathlib import Path

import pytest
import torch

from arbiter.adapters import ltx25_acceleration as accel
from arbiter.adapters import ltx25_denoise1
from arbiter.adapters.base import InferenceError

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "runtime" / "ltx25" / "manifest.json"


class TestProfileParsing:
    def test_absent_means_production_path(self):
        assert accel.parse_acceleration_benchmark({"encoded_file": "/e.pt"}) is None

    @pytest.mark.parametrize("profile", ["stock", "distilled", "distilled-conv"])
    def test_accepts_each_profile(self, profile):
        assert accel.parse_acceleration_benchmark({"acceleration_benchmark": profile}) == profile

    @pytest.mark.parametrize(
        "value", ["Stock", "distilled_conv", "dev-lora", "", None, True, 1, ["stock"]]
    )
    def test_rejects_anything_else(self, value):
        with pytest.raises(InferenceError, match="acceleration_benchmark"):
            accel.parse_acceleration_benchmark({"acceleration_benchmark": value})

    def test_denoise_adapter_rejects_bad_profile_before_load_check(self, tmp_path):
        adapter = ltx25_denoise1.LTX25Denoise1Adapter()
        import threading

        with pytest.raises(InferenceError, match="acceleration_benchmark"):
            adapter.infer({"acceleration_benchmark": "turbo"}, tmp_path, threading.Event())

    def test_denoise_adapter_rejects_stock_with_historical_a2v_5(self, tmp_path):
        adapter = ltx25_denoise1.LTX25Denoise1Adapter()
        import threading

        with pytest.raises(InferenceError, match="a2v_guidance_scale"):
            adapter.infer(
                {"acceleration_benchmark": "stock", "a2v_guidance_scale": 5.0},
                tmp_path,
                threading.Event(),
            )


class TestRequestContract:
    def test_stock_accepts_absent_or_fixed_defaults(self):
        accel.require_benchmark_request("stock", {})
        accel.require_benchmark_request(
            "stock", {"num_inference_steps": 30, "a2v_guidance_scale": 3.0}
        )
        accel.require_benchmark_request("stock", {"a2v_guidance_scale": 3})

    @pytest.mark.parametrize(
        "params",
        [
            {"num_inference_steps": 8},
            {"num_inference_steps": 30.0},
            {"num_inference_steps": True},
            {"a2v_guidance_scale": 5.0},
            {"a2v_guidance_scale": "3.0"},
        ],
    )
    def test_stock_rejects_changed_steps_or_guidance(self, params):
        with pytest.raises(InferenceError, match="stock fixes"):
            accel.require_benchmark_request("stock", params)

    @pytest.mark.parametrize("profile", ["distilled", "distilled-conv"])
    @pytest.mark.parametrize("key,value", [("num_inference_steps", 8), ("a2v_guidance_scale", 1.0)])
    def test_distilled_rejects_any_schedule_or_guidance_param(self, profile, key, value):
        with pytest.raises(InferenceError, match="must be absent"):
            accel.require_benchmark_request(profile, {key: value})

    @pytest.mark.parametrize(
        "key,value",
        [
            ("stage1_guiding_keyframes", True),
            ("generated_keyframes", 2),
            ("generated_keyframes", True),
            ("generated_keyframe_positions", [10]),
            ("stage2_drop_end_image", True),
            ("end_image_strength", 0.5),
        ],
    )
    @pytest.mark.parametrize("profile", ["stock", "distilled", "distilled-conv"])
    def test_rejects_enabled_conflicting_opt_ins(self, profile, key, value):
        with pytest.raises(InferenceError, match=key):
            accel.require_benchmark_request(profile, {key: value})

    def test_allows_conflicting_opt_ins_at_their_noop_default(self):
        accel.require_benchmark_request(
            "distilled",
            {
                "stage1_guiding_keyframes": False,
                "generated_keyframes": 0,
                "generated_keyframe_positions": [],
                "stage2_drop_end_image": False,
                "end_image_strength": None,
                "end_image_attention_strengths": [0.0, 1.0],
            },
        )


class TestBundleCaps:
    @pytest.mark.parametrize(
        "frames,height,width",
        [(49, 512, 768), (49, 320, 512), (9, 64, 64), (25, 384, 640)],
    )
    def test_accepts_capped_valid_shapes(self, frames, height, width):
        data = {"num_frames": frames, "height": height, "width": width}
        assert accel.require_benchmark_bundle(data) == (frames, height, width)

    @pytest.mark.parametrize(
        "frames,height,width,field",
        [
            (57, 320, 512, "num_frames"),
            (48, 320, 512, "num_frames"),
            (121, 1088, 1920, "num_frames"),
            (49, 576, 512, "height"),
            (49, 352, 512, "height"),
            (49, 320, 832, "width"),
            (49, 320, 544, "width"),
            (49, 1088, 1920, "height"),
        ],
    )
    def test_rejects_over_cap_or_misaligned(self, frames, height, width, field):
        data = {"num_frames": frames, "height": height, "width": width}
        with pytest.raises(InferenceError, match=field):
            accel.require_benchmark_bundle(data)

    @pytest.mark.parametrize("value", [None, 49.0, True, "49", 0])
    def test_rejects_non_int_dims(self, value):
        with pytest.raises(InferenceError, match="num_frames"):
            accel.require_benchmark_bundle({"num_frames": value, "height": 320, "width": 512})


class TestFixedWeights:
    def test_profiles_select_only_fixed_paths(self):
        stock = accel.profile_spec("stock")
        distilled = accel.profile_spec("distilled")
        conv = accel.profile_spec("distilled-conv")
        assert stock.transformer is accel.DEV_TRANSFORMER
        assert distilled.transformer is accel.DISTILLED_TRANSFORMER
        assert conv.transformer is accel.DISTILLED_TRANSFORMER
        assert stock.decoder is accel.DIFF_VAE
        assert distilled.decoder is accel.DIFF_VAE
        assert conv.decoder is accel.CONV_VAE
        assert not stock.distilled and distilled.distilled and conv.distilled

    def test_distilled_is_a_separate_checkpoint_not_dev(self):
        assert accel.DISTILLED_TRANSFORMER.path != accel.DEV_TRANSFORMER.path
        assert accel.DISTILLED_TRANSFORMER.sha256 != accel.DEV_TRANSFORMER.sha256
        assert accel.DISTILLED_TRANSFORMER.sha256 == (
            "31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4"
        )
        assert accel.DISTILLED_TRANSFORMER.path.startswith(
            "/mnt/arbiter-store/output/benchmarks/ltx25-acceleration/weights/"
        )

    def test_every_weight_is_absolute_with_hex_sha(self):
        for spec in (accel.profile_spec(p) for p in accel.PROFILES):
            for weight in spec.weights():
                assert Path(weight.path).is_absolute()
                assert len(weight.sha256) == 64
                int(weight.sha256, 16)
                assert weight.size > 0

    def test_unknown_profile_has_no_spec(self):
        with pytest.raises(InferenceError):
            accel.profile_spec("dev-lora")

    def test_runner_pin_matches_repo_manifest(self):
        assert accel.require_pinned_runner(MANIFEST) == accel.PINNED_RUNNER_SHA256

    def test_runner_pin_rejects_other_runner(self, tmp_path):
        manifest = json.loads(MANIFEST.read_text())
        manifest["patched_files"]["video_fast_gpu.py"] = "0" * 64
        changed = tmp_path / f"manifest-{uuid.uuid4().hex}.json"
        changed.write_text(json.dumps(manifest))
        with pytest.raises(InferenceError, match="re-audit"):
            accel.require_pinned_runner(changed)


class TestAttestation:
    def _weight(self, tmp_path, payload: bytes, sha: str | None = None, size: int | None = None):
        path = tmp_path / f"weight-{uuid.uuid4().hex}.safetensors"
        path.write_bytes(payload)
        return path, accel.WeightFile(
            "test_weight",
            str(path),
            len(payload) if size is None else size,
            sha or hashlib.sha256(payload).hexdigest(),
        )

    def test_hashes_then_caches_unchanged_file(self, tmp_path):
        _, weight = self._weight(tmp_path, os.urandom(4096))
        first = accel.attest_weight(weight)
        second = accel.attest_weight(weight)
        assert first["cached"] is False and second["cached"] is True
        assert first["sha256"] == weight.sha256

    def test_rehashes_after_modification_and_fails_closed(self, tmp_path):
        path, weight = self._weight(tmp_path, b"a" * 1024)
        accel.attest_weight(weight)
        path.write_bytes(b"b" * 1024)
        later = time.time_ns() + 5_000_000_000
        os.utime(path, ns=(later, later))
        with pytest.raises(InferenceError, match="sha256"):
            accel.attest_weight(weight)

    def test_wrong_sha_fails(self, tmp_path):
        _, weight = self._weight(tmp_path, b"payload", sha="0" * 64)
        with pytest.raises(InferenceError, match="sha256"):
            accel.attest_weight(weight)

    def test_wrong_size_fails_before_hashing(self, tmp_path):
        _, weight = self._weight(tmp_path, b"payload", size=99)
        with pytest.raises(InferenceError, match="size"):
            accel.attest_weight(weight)

    def test_missing_file_fails(self, tmp_path):
        weight = accel.WeightFile("test_weight", str(tmp_path / "absent"), 1, "0" * 64)
        with pytest.raises(InferenceError, match="missing"):
            accel.attest_weight(weight)

    def test_symlink_attests_target_content(self, tmp_path):
        target, weight = self._weight(tmp_path, os.urandom(2048))
        link = tmp_path / f"link-{uuid.uuid4().hex}"
        link.symlink_to(target)
        linked = accel.WeightFile(weight.role, str(link), weight.size, weight.sha256)
        result = accel.attest_weight(linked)
        assert result["resolved_path"] == str(target.resolve())


class TestHashProgress:
    def test_reports_real_bytes_at_bounded_intervals(self, tmp_path):
        payload = os.urandom(5 * 1024 * 1024 + 123)
        path = tmp_path / f"hash-{uuid.uuid4().hex}.bin"
        path.write_bytes(payload)
        events = []
        digest = accel.sha256_file(
            path,
            lambda done, total, elapsed: events.append((done, total, elapsed)),
            progress_bytes=1024 * 1024,
            progress_seconds=3600.0,
            block_bytes=256 * 1024,
        )
        assert digest == hashlib.sha256(payload).hexdigest()
        done = [e[0] for e in events]
        assert done == sorted(done) and done[-1] == len(payload)
        assert all(e[1] == len(payload) for e in events)
        assert len(events) == 6
        assert all(b - a >= 1024 * 1024 for a, b in zip([0] + done[:-2], done[:-1]))

    def test_defaults_are_one_gib_or_thirty_seconds(self):
        assert accel.HASH_PROGRESS_BYTES == 1024**3
        assert accel.HASH_PROGRESS_SECONDS == 30.0


class TestWorkerLoadMetadata:
    def _timing(self):
        return {
            "runtime_import_s": 1.5,
            "pipeline_init_and_load_denoise_models_s": 20.25,
            "load_total_s": 21.75,
            "loaded_at_unix": 1793305136.0,
        }

    def test_reports_measured_load_and_first_job_marker(self):
        meta = accel.worker_load_metadata(
            {"timing": self._timing(), "benchmark_infers_since_load": 1}
        )
        assert meta["timing"] == self._timing()
        assert meta["benchmark_infers_since_load"] == 1
        assert "Transformers are not part of worker load" in meta["note"]
        assert "counts benchmark jobs only" in meta["note"]
        assert "not proof of a cold process" in meta["note"]
        json.dumps(meta)

    def test_absent_timing_is_reported_as_none_not_invented(self):
        meta = accel.worker_load_metadata(None)
        assert meta["timing"] is None and meta["benchmark_infers_since_load"] is None

    @pytest.mark.parametrize(
        "change",
        [
            {"load_total_s": -1.0},
            {"load_total_s": float("nan")},
            {"runtime_import_s": 2},
            {"extra_s": 1.0},
        ],
    )
    def test_rejects_malformed_timing(self, change):
        timing = {**self._timing(), **change}
        with pytest.raises(InferenceError):
            accel.worker_load_metadata({"timing": timing, "benchmark_infers_since_load": 1})

    @pytest.mark.parametrize("count", [0, True, 1.0])
    def test_rejects_bad_infer_count(self, count):
        with pytest.raises(InferenceError):
            accel.worker_load_metadata({"timing": self._timing(), "benchmark_infers_since_load": count})

    def test_fresh_adapter_has_no_load_timing(self):
        adapter = ltx25_denoise1.LTX25Denoise1Adapter()
        assert adapter._load_timing is None and adapter._infers_since_load == 0


class TestStageClock:
    def test_segments_record_once_and_without_cuda_have_no_peaks(self):
        clock = accel.StageClock(torch if not torch.cuda.is_available() else None)
        with clock.segment("work"):
            torch.ones(256, 256) @ torch.ones(256, 256)
        assert clock.timings["work"] >= 0
        assert clock.overall_peak() is None
        with pytest.raises(InferenceError, match="twice"):
            clock.record("work", 1.0)

    def test_failed_segment_still_records(self):
        clock = accel.StageClock(None)
        with pytest.raises(ValueError):
            with clock.segment("broken"):
                raise ValueError("boom")
        assert "broken" in clock.timings

    def test_rejects_invalid_duration(self):
        clock = accel.StageClock(None)
        with pytest.raises(InferenceError):
            clock.record("negative", -1.0)

    def test_long_transformer_build_emits_output_until_denoise_starts(self):
        clock = accel.StageClock(None)
        build_output_seen = threading.Event()
        messages = []

        class OutputHandler(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                if "transformer build still running" in message:
                    messages.append(message)
                    build_output_seen.set()

        handler = OutputHandler()
        previous_level = accel.log.level
        accel.log.setLevel(logging.INFO)
        accel.log.addHandler(handler)
        try:
            def stage(loop, **kwargs):
                assert build_output_seen.wait(5.0)
                return loop(answer=42)

            result = clock.run_stage(
                "stage1",
                stage,
                lambda answer: answer,
                lambda: None,
                liveness_seconds=0.02,
            )
        finally:
            accel.log.removeHandler(handler)
            accel.log.setLevel(previous_level)

        assert result == 42
        assert messages
        assert all("stage1" in message for message in messages)
        assert clock.timings["stage1_build"] >= 0
        message_count = len(messages)
        threading.Event().wait(0.06)
        assert len(messages) == message_count

    def test_fast_transformer_build_does_not_emit_liveness_output(self, caplog):
        caplog.set_level("INFO", logger=accel.log.name)
        clock = accel.StageClock(None)
        result = clock.run_stage(
            "stage2",
            lambda loop: loop(answer="done"),
            lambda answer: answer,
            lambda: None,
            liveness_seconds=30.0,
        )
        assert result == "done"
        assert not [r for r in caplog.records if "transformer build still running" in r.getMessage()]


class TestContractMetadata:
    @pytest.mark.parametrize("profile", ["stock", "distilled", "distilled-conv"])
    def test_contract_is_json_and_labels_frozen_audio(self, profile):
        contract = accel.profile_contract(profile)
        json.dumps(contract)
        assert contract["audio"]["mode"] == "frozen_original_audio_latent"
        assert "not stock joint T2AV" in contract["audio"]["note"]
        assert "pixel copy" in contract["anchors"]

    def test_stock_matches_runner_literals(self):
        contract = accel.profile_contract("stock")
        assert contract["stage1"]["steps"] == 30
        assert contract["stage1"]["video_guider"] == {
            "cfg_scale": 3.0,
            "stg_scale": 1.0,
            "rescale_scale": 0.7,
            "modality_scale": 3.0,
            "skip_step": 0,
            "stg_blocks": [28],
        }
        assert contract["stage2"]["sigmas"] == [0.909, 0.725, 0.422, 0.0]

    def test_distilled_is_eight_step_guidance_free_without_lora(self):
        for profile in ("distilled", "distilled-conv"):
            contract = accel.profile_contract(profile)
            assert contract["stage1"]["steps"] == 8
            assert contract["stage1"]["loras"] == [] and contract["stage2"]["loras"] == []
            assert "EulerAncestral" in contract["stage1"]["sampler"]
            assert "deterministic" in contract["stage2"]["sampler"]
            assert contract["stage2"]["noise_scale"] == 0.909375


def _pinned_archive() -> Path:
    manifest = json.loads(MANIFEST.read_text())
    common = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    archive = Path(common).parent / "local" / "ltx25-runtime-sources" / manifest["source"]["archive"]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == manifest["source"]["archive_sha256"], archive
    return archive


def _archive_source(archive: Path, name: str) -> str:
    with tarfile.open(archive, "r:gz") as tar:
        return tar.extractfile(tar.getmember(name)).read().decode()


def _assigned(source: str) -> dict:
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue
    return values


class TestPinnedUpstreamSource:
    """The benchmark's constants equal the checksum-pinned installed source."""

    def test_distilled_sigmas_match_installed_constants(self):
        source = _archive_source(
            _pinned_archive(), "packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py"
        )
        values = _assigned(source)
        assert tuple(values["DISTILLED_SIGMA_VALUES"]) == accel.DISTILLED_STAGE1_SIGMAS
        assert tuple(values["STAGE_2_DISTILLED_SIGMA_VALUES"]) == accel.DISTILLED_STAGE2_SIGMAS

    def test_ancestral_contract_matches_installed_distilled_pipeline(self):
        source = _archive_source(
            _pinned_archive(), "packages/ltx-pipelines/src/ltx_pipelines/distilled.py"
        )
        values = _assigned(source)
        assert values["ANCESTRAL_ETA"] == accel.ANCESTRAL_ETA
        assert values["ANCESTRAL_S_NOISE"] == accel.ANCESTRAL_S_NOISE
        assert values["ANCESTRAL_NOISE_SEED_OFFSET"] == accel.ANCESTRAL_NOISE_SEED_OFFSET
        assert values["ANCESTRAL_SAMPLER_SINCE_VERSION"] == (2, 5)
        # Both official stages use SimpleDenoiser and stage 2 the default stepper.
        assert source.count("denoiser=SimpleDenoiser(video_context, audio_context)") == 2
        assert "noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET" in source
        assert "noise_scale=stage_2_sigmas[0].item()" in source

    def test_stock_runner_literals_match_pinned_runner_source(self):
        source = _archive_source(_pinned_archive(), "video_fast_gpu.py")
        for literal in (
            "cfg_scale=3.0",
            "stg_scale=1.0",
            "rescale_scale=0.7",
            "modality_scale=a2v_guidance_scale",
            "stg_blocks=[28]",
            "frozen=True",
            "diffvae_optimization=DiffVAEMode.CHUNKED_EAGER",
            "self._make_denoise_stage(self.distilled_lora)",
        ):
            assert literal in source, literal

    def test_runner_constants_pin_paths_and_stage2_sigmas(self):
        source = _archive_source(_pinned_archive(), "constants.py")
        assert "STAGE_2_DISTILLED_SIGMAS = torch.tensor([0.909, 0.725, 0.422, 0.0])" in source
        assert 'MODEL_DIR = Path("/home/darren/models/ltx25")' in source
        dev = accel.DEV_TRANSFORMER.path.removeprefix("/home/darren/models/ltx25/")
        vae = accel.DIFF_VAE.path.removeprefix("/home/darren/models/ltx25/")
        assert f'CHECKPOINT = MODEL_DIR / "{dev}"' in source
        assert f'VIDEO_VAE = MODEL_DIR / "{vae}"' in source


def test_pinned_runtime_oracle_has_precise_audited_ownership():
    """Real repository oracle I/O must not force every unrelated release full."""
    from scripts.python_test_impact import AUDITED, select_tests

    test_path = "tests/unit/test_ltx25_acceleration.py"
    assert AUDITED[test_path] == frozenset({"runtime/ltx25/"})
    assert test_path in select_tests(["src/arbiter/adapters/ltx25_acceleration.py"])
    assert test_path in select_tests(["src/arbiter/adapters/ltx25_denoise1.py"])
    assert select_tests(["deploy-to-spark.sh", "docs/notes.md"]) == [
        "tests/unit/test_drain_wait.py",
        "tests/unit/test_no_cloud_video.py",
    ]
    assert test_path not in select_tests(["src/arbiter/adapters/aesthetic_scorer.py"])
