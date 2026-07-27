"""voice-fit adapter — fit custom Kokoro voice packs (.pt style tensors) from
reference audio, backed by the voxsmith package.

voxsmith implements direct gradient descent on Kokoro's 256-dim style vector
through Kokoro's differentiable decoder, scored by a differentiable ECAPA-TDNN
speaker encoder (SpeechBrain spkrec-ecapa-voxceleb), with a multi-target
round-robin objective so the fitted voice generalises across phonetic content.
It is the successor to the KVoiceWalk-style random walk (still available in
voxsmith as --method walk, ~100x slower).

The adapter runs in a dedicated venv (venvs/voxsmith, torch cu130 + kokoro +
speechbrain + the voxsmith package itself) via worker_cmd in local/config.json.
Fitted packs persist at /home/darren/voice-models/<name>.pt (deploy-independent,
same convention as rvc-models).

voxsmith is imported lazily inside load()/infer() — the deploy-time adapter
smoke test imports this module with the main .venv, which does NOT have
voxsmith installed.

Job type:
    {"type": "voice-fit", "params": {
        "name": "leo-laporte",            # required -> /home/darren/voice-models/<name>.pt
        "targets_dir": "/path/on/spark",  # dir of *.wav + sibling *.txt transcripts
        # or "targets_file": "<staged zip of the same layout>",
        "seed_voice": "auto",             # stock voice, or "auto" (embedding search)
        "exclude": "am_adam,am_michael",  # optional: exclude from auto seed pool
        "init_pack": "",                  # optional: name under voice-models/, or abs .pt path
        "steps": 300, "lr": 0.05, "w_self": 0.5, "w_reg": 1.0, "eval_every": 10,
        # mode="finetune": FULL fine-tune of decoder+predictor+style vector
        # (voxsmith.finetune.finetune_full). Requires init_pack. Extra params:
        # "lr_model": 2e-5, "lr_vec": 0.01. Artifacts: <name>.pt pack,
        # <name>-model.pth (full Kokoro state_dict), <name>.json sidecar.
        # Run with memory_gb ~20 (backward through trainable decoder).
        "mode": "fit"
    }}
    # result: {model_id, pack_path, seed, baseline_sim, final_sim, per_target,
    #          self_sim, steps, seconds, history_tail} + the pack itself as
    #          result.pt (inlined base64 by the API for remote download)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

# Stable, deploy-independent home for fitted Kokoro voice packs (and the stock
# voice cache used for seed search), mirroring the rvc-models convention.
VOICE_MODELS_DIR = Path("/home/darren/voice-models")


def _sanitize_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    if not clean:
        raise InferenceError("voice-fit: 'name' must contain at least one alphanumeric char")
    return clean


@register
class VoiceFitAdapter(ModelAdapter):
    model_id = "voice-fit"

    def __init__(self):
        self._synth = None
        self._enc = None
        self._device = "cpu"

    # ------------------------------------------------------------------ load
    def load(self, device: str = "cuda") -> None:
        import torch  # noqa: PLC0415

        dev = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        try:
            from voxsmith.encoder import SpeakerEncoder  # noqa: PLC0415
            from voxsmith.synth import KokoroSynth  # noqa: PLC0415
        except ImportError as e:
            raise LoadError(f"voxsmith package not importable in this worker venv: {e}") from e

        t0 = time.time()
        VOICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._synth = KokoroSynth(device=dev)
        self._enc = SpeakerEncoder(device=dev)
        self._device = dev
        log.info("voice-fit ready in %.1fs (device=%s)", time.time() - t0, dev)

    def unload(self) -> None:
        self._synth = None
        self._enc = None
        self._cleanup_gpu()

    # ----------------------------------------------------------------- infer
    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        if self._synth is None or self._enc is None:
            raise InferenceError("voice-fit not loaded")
        self._check_cancel(cancel_flag)

        from voxsmith.fit import FitCancelled, fit_gradient, load_targets_dir, select_seed  # noqa: PLC0415
        from voxsmith.voices import STOCK_VOICES, load_pack, pack_from_vector, save_pack  # noqa: PLC0415

        name = _sanitize_name(params.get("name") or params.get("model_id") or "")
        steps = int(params.get("steps", 300))
        lr = float(params.get("lr", 0.05))
        w_self = float(params.get("w_self", 0.5))
        w_reg = float(params.get("w_reg", 1.0))
        eval_every = int(params.get("eval_every", 10))
        mode = (params.get("mode") or "fit").strip()

        output_dir = Path(output_dir)
        work_dir = output_dir / "work"
        targets_dir = self._materialize_targets(params, output_dir, work_dir)
        targets = load_targets_dir(targets_dir, work_dir, log=lambda m: log.info("%s", m))
        if not targets:
            raise InferenceError("voice-fit: no targets resolved")
        log.info("voice-fit '%s': %d target(s), steps=%d lr=%.3f", name, len(targets), steps, lr)

        if mode == "finetune":
            return self._infer_finetune(params, output_dir, targets, name, cancel_flag)
        if mode != "fit":
            raise InferenceError(f"voice-fit: unknown mode {mode!r} (expected fit|finetune)")

        # --- seed: explicit init pack > stock voice > auto embedding search ---
        init_pack = (params.get("init_pack") or "").strip()
        seed_voice = (params.get("seed_voice") or "auto").strip()
        if init_pack:
            seed_pack, seed_name = self._resolve_init_pack(init_pack)
        elif seed_voice == "auto":
            exclude = {s.strip() for s in (params.get("exclude") or "").split(",") if s.strip()}
            candidates = [v for v in STOCK_VOICES if v not in exclude]
            seed_name, seed_pack, _ = select_seed(
                self._synth, self._enc, targets[0].wav, targets[0].text,
                VOICE_MODELS_DIR, candidates, log=lambda m: log.info("%s", m),
            )
        else:
            seed_name = seed_voice
            seed_pack = load_pack(seed_voice, VOICE_MODELS_DIR)
        self._check_cancel(cancel_flag)
        log.info("voice-fit '%s': seed=%s", name, seed_name)

        pack_path = VOICE_MODELS_DIR / f"{name}.pt"
        t0 = time.time()
        try:
            res = fit_gradient(
                self._synth, self._enc, targets, seed_pack, seed_name,
                steps=steps, lr=lr, w_self=w_self, w_reg=w_reg,
                eval_every=eval_every,
                save_best_path=str(pack_path),
                cancel_check=cancel_flag.is_set,
                log=lambda m: log.info("%s", m),
            )
        except FitCancelled:
            from arbiter.adapters.base import CancelledException  # noqa: PLC0415

            raise CancelledException(f"voice-fit '{name}' cancelled") from None

        pack = pack_from_vector(res.vector)
        save_pack(pack, pack_path)
        # Also write the pack as the job's result file: the arbiter API inlines
        # output/jobs/<id>/result.<format> as base64 in result.data, which lets
        # remote clients (e.g. `voxsmith arbiter-fit`) download the pack with no
        # filesystem access to spark.
        save_pack(pack, output_dir / "result.pt")
        fit_seconds = time.time() - t0
        (VOICE_MODELS_DIR / f"{name}.json").write_text(json.dumps({
            "model_id": name,
            "seed": res.seed_name,
            "targets": [t.name for t in targets],
            "steps": res.steps_run,
            "lr": lr, "w_self": w_self, "w_reg": w_reg,
            "baseline_sim": round(res.seed_sim, 4),
            "final_sim": round(res.final_sim, 4),
            "per_target": {k: round(v, 4) for k, v in res.per_target_sim.items()},
            "self_sim": round(res.self_sim, 4),
            "seconds": round(fit_seconds, 1),
            "fitted_at": round(time.time()),
        }, indent=1) + "\n")
        log.info("voice-fit '%s': done in %.0fs, sim %.4f -> %.4f",
                 name, fit_seconds, res.seed_sim, res.final_sim)

        return {
            "format": "pt",
            "model_id": name,
            "pack_path": str(pack_path),
            "seed": res.seed_name,
            "n_targets": len(targets),
            "baseline_sim": round(res.seed_sim, 4),
            "final_sim": round(res.final_sim, 4),
            "delta": round(res.final_sim - res.seed_sim, 4),
            "per_target": {k: round(v, 4) for k, v in res.per_target_sim.items()},
            "self_sim": round(res.self_sim, 4),
            "steps": res.steps_run,
            "seconds": round(fit_seconds, 1),
            "history_tail": res.history[-10:],
        }

    # ------------------------------------------------------------- finetune
    def _infer_finetune(self, params: dict, output_dir: Path, targets, name: str,
                        cancel_flag: threading.Event) -> dict:
        """mode=finetune: full fine-tune of decoder+predictor+style vector."""
        import torch  # noqa: PLC0415
        from voxsmith.fit import FitCancelled  # noqa: PLC0415
        from voxsmith.finetune import finetune_full  # noqa: PLC0415
        from voxsmith.voices import pack_from_vector, save_pack  # noqa: PLC0415

        steps = int(params.get("steps", 300))
        lr_model = float(params.get("lr_model", 2e-5))
        lr_vec = float(params.get("lr_vec", 0.01))
        eval_every = int(params.get("eval_every", 10))
        w_silence = float(params.get("w_silence", 0.2))
        gate_db = float(params.get("gate_db", -35.0))
        floor_db = float(params.get("floor_db", -50.0))
        w_asym = float(params.get("w_asym", 0.0))
        asym_lo_weight = float(params.get("asym_lo_weight", 1.0))
        init_pack = (params.get("init_pack") or "").strip()
        if not init_pack:
            raise InferenceError("voice-fit finetune: init_pack is required "
                                 "(style-vector init, e.g. paris-ensemble)")
        pack, pack_name = self._resolve_init_pack(init_pack)

        model_path = VOICE_MODELS_DIR / f"{name}-model.pth"
        pack_path = VOICE_MODELS_DIR / f"{name}.pt"
        t0 = time.time()
        log.info("voice-fit finetune '%s': init=%s steps=%d lr_model=%g lr_vec=%g",
                 name, pack_name, steps, lr_model, lr_vec)
        try:
            res = finetune_full(
                self._synth, self._enc, targets, pack,
                steps=steps, lr_model=lr_model, lr_vec=lr_vec,
                eval_every=eval_every, w_silence=w_silence,
                gate_db=gate_db, floor_db=floor_db, w_asym=w_asym,
                asym_lo_weight=asym_lo_weight,
                save_best_model_path=str(model_path),
                save_best_pack_path=str(pack_path),
                cancel_check=cancel_flag.is_set,
                log=lambda m: log.info("%s", m),
            )
        except FitCancelled:
            from arbiter.adapters.base import CancelledException  # noqa: PLC0415

            raise CancelledException(f"voice-fit finetune '{name}' cancelled") from None

        # Final artifacts (best weights were already streamed to these paths,
        # but write them explicitly so a failed save_best can't lose the run).
        torch.save(res.state_dict, model_path)
        final_pack = pack_from_vector(res.vector)
        save_pack(final_pack, pack_path)
        save_pack(final_pack, output_dir / "result.pt")
        fit_seconds = time.time() - t0
        (VOICE_MODELS_DIR / f"{name}.json").write_text(json.dumps({
            "model_id": name,
            "mode": "finetune",
            "init_pack": pack_name,
            "targets": [t.name for t in targets],
            "steps": res.steps_run,
            "lr_model": lr_model, "lr_vec": lr_vec,
            "train_params_m": res.extras.get("train_params_m"),
            "baseline_sim": round(res.init_sim, 4),
            "final_sim": round(res.final_sim, 4),
            "per_target": {k: round(v, 4) for k, v in res.per_target_sim.items()},
            "self_sim": round(res.self_sim, 4),
            "recommended_speed": res.extras.get("recommended_speed"),
            "seconds": round(fit_seconds, 1),
            "fitted_at": round(time.time()),
        }, indent=1) + "\n")
        log.info("voice-fit finetune '%s': done in %.0fs, sim %.4f -> %.4f",
                 name, fit_seconds, res.init_sim, res.final_sim)

        return {
            "format": "pt",
            "model_id": name,
            "mode": "finetune",
            "pack_path": str(pack_path),
            "model_weights_path": str(model_path),
            "init_pack": pack_name,
            "n_targets": len(targets),
            "baseline_sim": round(res.init_sim, 4),
            "final_sim": round(res.final_sim, 4),
            "delta": round(res.final_sim - res.init_sim, 4),
            "per_target": {k: round(v, 4) for k, v in res.per_target_sim.items()},
            "self_sim": round(res.self_sim, 4),
            "steps": res.steps_run,
            "seconds": round(fit_seconds, 1),
            "history_tail": res.history[-10:],
        }

    # ------------------------------------------------------------- targets
    def _materialize_targets(self, params: dict, output_dir: Path, work_dir: Path) -> Path:
        """Return a directory of *.wav + sibling *.txt pairs."""
        out = output_dir / "targets"
        out.mkdir(parents=True, exist_ok=True)

        targets_dir = (params.get("targets_dir") or "").strip()
        targets_file = (params.get("targets_file") or "").strip()
        if targets_dir:
            src = Path(targets_dir)
            if not src.is_dir():
                raise InferenceError(f"voice-fit: targets_dir not a directory: {src}")
            for f in src.iterdir():
                if f.suffix.lower() in (".wav", ".txt"):
                    shutil.copy2(f, out / f.name)
        elif targets_file:
            src = Path(targets_file)
            if src.is_dir():
                for f in src.iterdir():
                    if f.suffix.lower() in (".wav", ".txt"):
                        shutil.copy2(f, out / f.name)
            elif src.is_file() and src.suffix.lower() == ".zip":
                with zipfile.ZipFile(src) as zf:
                    for nm in zf.namelist():
                        p = Path(nm)
                        if p.suffix.lower() in (".wav", ".txt") and not nm.startswith("__MACOSX"):
                            (out / p.name).write_bytes(zf.read(nm))
            else:
                raise InferenceError(f"voice-fit: targets_file must be a dir or .zip: {src}")
        else:
            raise InferenceError("voice-fit: provide targets_dir (spark path) or targets_file (staged zip)")

        n_wavs = len(list(out.glob("*.wav")))
        if n_wavs == 0:
            raise InferenceError("voice-fit: targets contained no .wav files")
        missing = [w.name for w in out.glob("*.wav") if not w.with_suffix(".txt").exists()]
        if missing:
            raise InferenceError(f"voice-fit: .wav files missing sibling .txt transcripts: {missing}")
        work_dir.mkdir(parents=True, exist_ok=True)
        return out

    # ------------------------------------------------------------ init pack
    @staticmethod
    def _resolve_init_pack(init_pack: str):
        from voxsmith.voices import load_pack  # noqa: PLC0415

        p = Path(init_pack)
        if p.suffix != ".pt":
            p = VOICE_MODELS_DIR / f"{_sanitize_name(init_pack)}.pt"
        if not p.is_file():
            raise InferenceError(f"voice-fit: init_pack not found: {p}")
        return load_pack(p, VOICE_MODELS_DIR), f"init:{p.stem}"

    # ----------------------------------------------------------- estimate
    def estimate_time(self, params: dict) -> float:
        steps = int(params.get("steps", 300))
        seed_overhead = 120000.0 if (params.get("seed_voice") or "auto") == "auto" else 10000.0
        return seed_overhead + steps * 1500.0  # ~1.5s/step on GB10 + eval passes
