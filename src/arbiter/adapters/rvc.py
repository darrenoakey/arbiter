"""RVC v2 voice-conversion adapters (train + convert), backed by Applio.

Applio (IAHispano/Applio) is a fairseq-free RVC v2 implementation with a
headless ``core.py`` CLI (preprocess -> extract -> train -> index -> infer).
It loads the ContentVec content encoder via HuggingFace transformers and
rmvpe as a plain torch checkpoint, so nothing needs the fairseq-format
``hubert_base.pt`` that fails to build on aarch64 / Python 3.12.

Both adapters run in venvs/rvc (torch cu130 + Applio deps, NOT Applio's pinned
cu128 torch) via a worker_cmd in local/config.json. sys.executable is that
venv's python, so the Applio subprocesses inherit the working CUDA build.

- ``rvc-train``: a dir/zip of wav samples -> a trained RVC model persisted
  under RVC_MODELS_DIR/<model_id>/ (model.pth + model.index).
- ``rvc-convert``: model (id or .pth path) + input audio (+ transpose) ->
  converted wav.
"""

from __future__ import annotations

import base64
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

APPLIO_DIR = Path("/home/darren/src/Applio")
APPLIO_CORE = APPLIO_DIR / "core.py"
APPLIO_LOGS = APPLIO_DIR / "logs"
# Stable, deploy-independent home for trained voice models.
RVC_MODELS_DIR = Path("/home/darren/rvc-models")

# Cap inline base64 (raw bytes). rvc-convert returns ONE stem; keep its b64
# under the worker stdout pipe ceiling (96MB, see proc.go). 60MB raw -> ~80MB
# b64. Larger output gets an omission note and must be fetched as a file.
_B64_CAP_BYTES = 60 * 1024 * 1024


def _ensure_applio_config() -> None:
    """Applio's extract_model reads assets/config.json (for model_author); a
    headless clone lacks it, which makes the deployable-weight extraction fail
    silently (caught+printed, non-fatal) so training produces only G_/D_
    checkpoints and no usable model. The documented headless setup copies the
    template — do it self-healingly here."""
    cfg = APPLIO_DIR / "assets" / "config.json"
    if cfg.is_file():
        return
    tmpl = APPLIO_DIR / "assets" / "config_template.json"
    if tmpl.is_file():
        shutil.copy2(tmpl, cfg)


def _sanitize_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    if not clean:
        raise InferenceError(
            "rvc: 'name'/'model' must contain at least one alphanumeric char"
        )
    return clean


def _run(cmd: list[str], timeout: int, cancel_flag: threading.Event, label: str) -> str:
    """Run an Applio CLI step from APPLIO_DIR. Raise InferenceError on failure."""
    if cancel_flag.is_set():
        from arbiter.adapters.base import CancelledException

        raise CancelledException(f"cancelled before {label}")
    log.info("rvc: %s -> %s", label, " ".join(cmd[1:]))
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(APPLIO_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise InferenceError(f"rvc {label} timed out after {timeout}s") from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1200:]
        raise InferenceError(f"rvc {label} failed (exit {proc.returncode}): {tail}")
    log.info("rvc: %s done in %.1fs", label, time.time() - t0)
    return proc.stdout or ""


@register
class RvcTrainAdapter(ModelAdapter):
    model_id = "rvc-train"

    def __init__(self):
        self._ready = False

    def load(self, device: str = "cuda") -> None:
        if not APPLIO_CORE.is_file():
            raise LoadError(f"Applio not found at {APPLIO_CORE}")
        _ensure_applio_config()
        RVC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._ready = True
        log.info("rvc-train ready (Applio at %s).", APPLIO_DIR)

    def unload(self) -> None:
        self._ready = False
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        if not self._ready:
            raise InferenceError("rvc-train not loaded")
        self._check_cancel(cancel_flag)

        name = _sanitize_name(params.get("name") or params.get("model_name") or "")
        epochs = int(params.get("epochs", 300))
        sample_rate = int(params.get("sample_rate", 40000))
        batch_size = int(params.get("batch_size", 4))
        f0_method = params.get("f0_method", "rmvpe")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        n_wavs = self._materialize_dataset(params, dataset_dir)
        if n_wavs == 0:
            raise InferenceError("rvc-train: dataset contained no .wav files")
        log.info(
            "rvc-train '%s': %d wav(s), sr=%d, epochs=%d",
            name,
            n_wavs,
            sample_rate,
            epochs,
        )

        # Clean any stale Applio run dir for this name so globbing picks fresh outputs.
        run_dir = APPLIO_LOGS / name
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        py = sys.executable
        _run(
            [
                py,
                "core.py",
                "preprocess",
                "--model_name",
                name,
                "--dataset_path",
                str(dataset_dir),
                "--sample_rate",
                str(sample_rate),
                "--cpu_cores",
                "8",
                "--cut_preprocess",
                "Automatic",
            ],
            timeout=3600,
            cancel_flag=cancel_flag,
            label="preprocess",
        )
        self._check_cancel(cancel_flag)
        _run(
            [
                py,
                "core.py",
                "extract",
                "--model_name",
                name,
                "--sample_rate",
                str(sample_rate),
                "--f0_method",
                f0_method,
                "--embedder_model",
                "contentvec",
                "--gpu",
                "0",
                "--cpu_cores",
                "8",
                "--include_mutes",
                "2",
            ],
            timeout=3600,
            cancel_flag=cancel_flag,
            label="extract",
        )
        self._check_cancel(cancel_flag)
        # save_every_epoch must be <= total so the final epoch is a save point;
        # this Applio build does not force a save at completion, so a run whose
        # epoch count never hits the interval produces no deployable weight.
        save_every = max(1, min(25, epochs))
        t_train = time.time()
        _run(
            [
                py,
                "core.py",
                "train",
                "--model_name",
                name,
                "--sample_rate",
                str(sample_rate),
                "--total_epoch",
                str(epochs),
                "--batch_size",
                str(batch_size),
                "--save_every_epoch",
                str(save_every),
                "--save_only_latest",
                "False",
                "--save_every_weights",
                "True",
                "--gpu",
                "0",
                "--pretrained",
                "True",
                "--vocoder",
                "HiFi-GAN",
            ],
            timeout=6 * 3600,
            cancel_flag=cancel_flag,
            label="train",
        )
        train_seconds = time.time() - t_train
        self._check_cancel(cancel_flag)
        _run(
            [py, "core.py", "index", "--model_name", name],
            timeout=1800,
            cancel_flag=cancel_flag,
            label="index",
        )

        pth = self._pick_weight(run_dir)
        index = self._pick_index(run_dir)
        if pth is None:
            raise InferenceError(f"rvc-train: no deployable .pth produced in {run_dir}")

        model_dir = RVC_MODELS_DIR / name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pth"
        shutil.copy2(pth, model_path)
        index_path = None
        if index is not None:
            index_path = model_dir / "model.index"
            shutil.copy2(index, index_path)
        (model_dir / "info.json").write_text(
            f'{{"model_id":"{name}","sample_rate":{sample_rate},"epochs":{epochs},'
            f'"f0_method":"{f0_method}","source_weight":"{pth.name}",'
            f'"trained_at":{time.time():.0f}}}\n'
        )

        return {
            "format": "rvc-model",
            "model_id": name,
            "model_path": str(model_path),
            "index_path": str(index_path) if index_path else None,
            "sample_rate": sample_rate,
            "epochs": epochs,
            "train_wavs": n_wavs,
            "train_runtime_seconds": round(train_seconds, 1),
            "source_weight": pth.name,
        }

    def _materialize_dataset(self, params: dict, dataset_dir: Path) -> int:
        """Populate dataset_dir with .wav files from dataset_file/dataset_b64."""
        raw = None
        src_file = params.get("dataset_file")
        if src_file:
            p = Path(src_file)
            if p.is_dir():
                count = 0
                for w in p.rglob("*.wav"):
                    shutil.copy2(w, dataset_dir / f"{count:04d}_{w.name}")
                    count += 1
                return count
            raw = p.read_bytes()
        elif params.get("dataset_b64"):
            raw = base64.b64decode(params["dataset_b64"])
        if raw is None:
            raise InferenceError(
                "rvc-train: provide dataset_file (dir/zip/wav) or dataset_b64"
            )

        if raw[:2] == b"PK":  # zip archive
            zpath = dataset_dir.parent / "_dataset.zip"
            zpath.write_bytes(raw)
            count = 0
            with zipfile.ZipFile(zpath) as zf:
                for nm in zf.namelist():
                    if nm.lower().endswith(".wav") and not nm.startswith("__MACOSX"):
                        data = zf.read(nm)
                        (dataset_dir / f"{count:04d}_{Path(nm).name}").write_bytes(data)
                        count += 1
            zpath.unlink(missing_ok=True)
            return count
        # Single wav payload.
        (dataset_dir / "0000_sample.wav").write_bytes(raw)
        return 1

    @staticmethod
    def _pick_weight(run_dir: Path) -> Path | None:
        if not run_dir.is_dir():
            return None
        cands = [
            p for p in run_dir.glob("*.pth") if not p.name.startswith(("G_", "D_"))
        ]
        if not cands:
            return None
        return max(cands, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _pick_index(run_dir: Path) -> Path | None:
        if not run_dir.is_dir():
            return None
        idx = list(run_dir.glob("*.index"))
        if not idx:
            return None
        added = [p for p in idx if "added" in p.name.lower()]
        pool = added or idx
        return max(pool, key=lambda p: p.stat().st_size)

    def estimate_time(self, params: dict) -> float:
        epochs = int(params.get("epochs", 300))
        return epochs * 6000.0 + 120000.0  # rough: ~6s/epoch small set + overhead


@register
class RvcConvertAdapter(ModelAdapter):
    model_id = "rvc-convert"

    def __init__(self):
        self._ready = False

    def load(self, device: str = "cuda") -> None:
        if not APPLIO_CORE.is_file():
            raise LoadError(f"Applio not found at {APPLIO_CORE}")
        self._ready = True
        log.info("rvc-convert ready.")

    def unload(self) -> None:
        self._ready = False
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        if not self._ready:
            raise InferenceError("rvc-convert not loaded")
        self._check_cancel(cancel_flag)

        pth, index = self._resolve_model(params)
        transpose = int(params.get("transpose", params.get("pitch", 0)))
        index_rate = float(params.get("index_rate", 0.5))
        f0_method = params.get("f0_method", "rmvpe")
        protect = float(params.get("protect", 0.33))
        return_b64 = bool(params.get("return_b64", False))

        try:
            audio_bytes = self._resolve_media(params, "audio")
        except Exception as e:  # noqa: BLE001
            raise InferenceError(f"rvc-convert: failed to resolve audio: {e}") from e

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        in_path = output_dir / "_input.wav"
        in_path.write_bytes(audio_bytes)
        out_path = output_dir / "result.wav"

        cmd = [
            sys.executable,
            "core.py",
            "infer",
            "--pth_path",
            str(pth),
            "--index_path",
            str(index) if index is not None else "",
            "--input_path",
            str(in_path),
            "--output_path",
            str(out_path),
            "--f0_method",
            f0_method,
            "--pitch",
            str(transpose),
            "--index_rate",
            str(index_rate),
            "--protect",
            str(protect),
            "--embedder_model",
            "contentvec",
            "--export_format",
            "WAV",
            "--clean_audio",
            "False",
            "--split_audio",
            "False",
        ]
        _run(cmd, timeout=1800, cancel_flag=cancel_flag, label="infer")
        in_path.unlink(missing_ok=True)

        if not out_path.is_file():
            # Applio may append a suffix depending on export_format; find it.
            alt = sorted(output_dir.glob("result*.wav"))
            if alt:
                out_path = alt[0]
            else:
                raise InferenceError("rvc-convert produced no output wav")

        result = {
            "format": "wav",
            "file": out_path.name,
            "audio": out_path.name,
            "converted": out_path.name,
        }
        if return_b64:
            raw = out_path.read_bytes()
            if len(raw) <= _B64_CAP_BYTES:
                result["audio_b64"] = base64.b64encode(raw).decode("ascii")
            else:
                result["audio_b64_skipped"] = f"{len(raw)} bytes exceeds cap"
        return result

    @staticmethod
    def _resolve_model(params: dict) -> tuple[Path, Path | None]:
        model = (
            params.get("model") or params.get("model_path") or params.get("model_id")
        )
        if not model:
            raise InferenceError("rvc-convert: 'model' (id or .pth path) is required")
        explicit_index = params.get("index_path")
        p = Path(model)
        if p.suffix == ".pth" and p.is_file():
            pth = p
            index = Path(explicit_index) if explicit_index else None
            if index is None:
                sib = list(p.parent.glob("*.index"))
                index = sib[0] if sib else None
        else:
            model_dir = RVC_MODELS_DIR / _sanitize_name(str(model))
            pth = model_dir / "model.pth"
            if not pth.is_file():
                raise InferenceError(f"rvc-convert: model '{model}' not found at {pth}")
            idx = model_dir / "model.index"
            index = (
                idx
                if idx.is_file()
                else (Path(explicit_index) if explicit_index else None)
            )
        return pth, index

    def estimate_time(self, params: dict) -> float:
        return 60000.0
