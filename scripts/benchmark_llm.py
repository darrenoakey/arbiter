#!/usr/bin/env python3
"""LLM backend benchmark runner.

Compares two registered arbiter LLM models (e.g. one llama.cpp + one vLLM,
both serving the same checkpoint at the same quant) across:

  - cold + warm load time
  - single-request token/s at several prompt sizes
  - throughput vs concurrency sweep (finds ideal max_concurrent for each)
  - validity check (planted-token echo)
  - VRAM + RSS snapshots from /v1/ps

Procedure:
  1. POST /v1/admin/benchmark/enable      → pause external dispatch
  2. POST /v1/admin/models/unload_all     → clean baseline
  3. For each model under test:
       - POST /v1/admin/models/preload    → cold load time + memory
       - POST /v1/admin/models/preload    → warm load time
       - For each prompt size: 3 runs single-request, record tokens/sec
       - For each N in concurrency sweep: time aggregate, record throughput + p50/p95
       - POST /v1/admin/models/unload_all
  4. POST /v1/admin/benchmark/disable
  5. Write JSON + self-contained HTML report; open HTML in browser.

The report's headline output per backend is the recommended max_concurrent —
the largest N where throughput gain over N/2 is >= 5% AND p95 latency stays
within 2x of N=1.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import subprocess
import sys
import time
import webbrowser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

DEFAULT_ARBITER = os.environ.get("ARBITER_URL", "http://10.0.0.254:8400")

# Concurrency sweep — geometric so we can spot the knee.
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]

# Prompt sizes (approx tokens) → (label, prompt_text_builder, completion_max_tokens)
# We seed each prompt with a unique planted token the model is asked to echo
# back, and validate the echo to catch silent failures (truncations, refusals).
def _filler(n_words: int) -> str:
    base = ("the quick brown fox jumps over the lazy dog and then continues "
            "walking through the forest while reciting prime numbers. ")
    out = []
    while sum(len(s) for s in out) < n_words * 6:
        out.append(base)
    return "".join(out)

PROMPT_SIZES = [
    ("small",  256,    128),  # ~256 prompt tokens, 128 completion
    ("medium", 4096,   256),  # ~4k prompt tokens
]

PLANT_TOKEN = "ZX9PLANT42"


def build_prompt(approx_tokens: int) -> str:
    # 1 token ≈ 4 chars for English; budget filler accordingly.
    target_chars = max(0, approx_tokens * 4 - 200)
    filler = _filler(target_chars // 6)[:target_chars]
    return (
        f"{filler}\n\n"
        f"You will receive context above. Reply with EXACTLY the token "
        f"{PLANT_TOKEN} followed by a one-sentence summary of the topic. "
        f"Begin your reply with {PLANT_TOKEN}."
    )


@dataclass
class RunResult:
    ok: bool
    elapsed_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float = 0.0  # time-to-first-token (non-streaming: same as elapsed)
    error: str = ""
    valid: bool = False  # planted-token echo present


@dataclass
class CellResult:
    """One (model, prompt_size, N) cell of the sweep."""
    model: str
    prompt_label: str
    n_concurrent: int
    runs: list[RunResult] = field(default_factory=list)

    @property
    def aggregate_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.runs if r.ok)

    @property
    def wall_clock_s(self) -> float:
        # set externally
        return getattr(self, "_wall_clock_s", 0.0)

    def set_wall_clock(self, s: float) -> None:
        self._wall_clock_s = s

    @property
    def aggregate_throughput_tps(self) -> float:
        if self.wall_clock_s <= 0:
            return 0.0
        return self.aggregate_completion_tokens / self.wall_clock_s

    @property
    def per_request_latencies(self) -> list[float]:
        return [r.elapsed_s for r in self.runs if r.ok]

    @property
    def p50(self) -> float:
        xs = self.per_request_latencies
        return statistics.median(xs) if xs else 0.0

    @property
    def p95(self) -> float:
        xs = sorted(self.per_request_latencies)
        if not xs:
            return 0.0
        idx = max(0, int(round(0.95 * len(xs))) - 1)
        return xs[idx]

    @property
    def success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.ok and r.valid) / len(self.runs)


def _http_post(url: str, payload: dict | None = None, timeout: float = 1800) -> dict:
    body = b""
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=body, method="POST",
                             headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return {}
        return json.loads(raw)


def _http_get(url: str, timeout: float = 30) -> dict:
    with urlrequest.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def admin_enable_benchmark(base: str) -> None:
    _http_post(f"{base}/v1/admin/benchmark/enable")


def admin_disable_benchmark(base: str) -> None:
    _http_post(f"{base}/v1/admin/benchmark/disable")


def admin_unload_all(base: str) -> dict:
    return _http_post(f"{base}/v1/admin/models/unload_all")


def admin_preload(base: str, model_id: str) -> dict:
    return _http_post(f"{base}/v1/admin/models/preload", {"model_id": model_id})


def get_ps(base: str) -> dict:
    try:
        return _http_get(f"{base}/v1/ps")
    except (HTTPError, URLError, TimeoutError):
        return {}


def memory_for_model(ps: dict, model_id: str) -> dict[str, float]:
    """Extract memory for a model from the /v1/ps response. The arbiter exposes
    it as a single memory_gb field per model (process-tree aggregate); we
    surface it as vram_gb since on the GB10 unified-memory box VRAM == RSS."""
    out = {"vram_gb": 0.0, "rss_gb": 0.0}
    if not isinstance(ps, dict):
        return out
    for m in ps.get("models", []) or []:
        if not isinstance(m, dict):
            continue
        if m.get("id") != model_id:
            continue
        v = m.get("memory_gb")
        if isinstance(v, (int, float)):
            out["vram_gb"] = float(v)
            out["rss_gb"] = float(v)
    return out


def chat_once(base: str, model_name: str, prompt: str, max_tokens: int) -> RunResult:
    """Send one chat completion. model_name is 'gemma4-26b' (no llm: prefix)."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    try:
        resp = _http_post(f"{base}/v1/chat/completions", payload, timeout=1800)
    except Exception as e:
        return RunResult(ok=False, elapsed_s=time.perf_counter() - t0, error=str(e))
    elapsed = time.perf_counter() - t0
    text = ""
    usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
    choices = resp.get("choices", []) if isinstance(resp, dict) else []
    if choices and isinstance(choices, list):
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        text = msg.get("content") or msg.get("reasoning_content") or ""
    valid = PLANT_TOKEN in text
    return RunResult(
        ok=True,
        elapsed_s=elapsed,
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        ttft_s=elapsed,
        valid=valid,
    )


def sweep_concurrency(base: str, model_name: str, prompt: str, max_tokens: int,
                      n: int, runs_per_n: int = 1) -> CellResult:
    cell = CellResult(model=model_name, prompt_label="", n_concurrent=n)
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futures = []
        for _ in range(n * runs_per_n):
            futures.append(ex.submit(chat_once, base, model_name, prompt, max_tokens))
        for f in cf.as_completed(futures):
            cell.runs.append(f.result())
    cell.set_wall_clock(time.perf_counter() - t0)
    return cell


@dataclass
class ModelReport:
    model_id: str
    name: str
    backend: str
    cold_load_s: float = 0.0
    warm_load_s: float = 0.0
    memory_after_load: dict = field(default_factory=dict)
    cells: list[CellResult] = field(default_factory=list)
    recommended_max_concurrent: int = 1
    notes: list[str] = field(default_factory=list)


def recommend_max_concurrent(cells_by_n: dict[int, CellResult],
                             min_throughput_gain: float = 0.05,
                             max_p95_blowup: float = 2.0) -> int:
    """Largest N where throughput gain >= min_throughput_gain over the next-lower
    sample AND p95 latency stays within max_p95_blowup x of N=1."""
    levels = sorted(cells_by_n.keys())
    if not levels:
        return 1
    base_p95 = cells_by_n[levels[0]].p95 or 1e-9
    best = levels[0]
    for i, n in enumerate(levels):
        if i == 0:
            continue
        prev_n = levels[i - 1]
        prev = cells_by_n[prev_n].aggregate_throughput_tps or 1e-9
        cur = cells_by_n[n].aggregate_throughput_tps
        gain = (cur - prev) / prev
        p95 = cells_by_n[n].p95
        if gain >= min_throughput_gain and p95 <= base_p95 * max_p95_blowup:
            best = n
        else:
            break
    return best


def run_for_model(base: str, model_id: str, name: str, backend: str) -> ModelReport:
    rep = ModelReport(model_id=model_id, name=name, backend=backend)

    # Cold load
    admin_unload_all(base)
    t0 = time.perf_counter()
    res = admin_preload(base, model_id)
    rep.cold_load_s = time.perf_counter() - t0
    if "load_ms" in res:
        rep.notes.append(f"server-reported cold load_ms={res['load_ms']}")

    # Warm load (should be near-zero — already loaded)
    t0 = time.perf_counter()
    admin_preload(base, model_id)
    rep.warm_load_s = time.perf_counter() - t0

    rep.memory_after_load = memory_for_model(get_ps(base), model_id)

    # For each prompt size, sweep N
    for label, prompt_tokens, max_tokens in PROMPT_SIZES:
        prompt = build_prompt(prompt_tokens)
        for n in CONCURRENCY_LEVELS:
            cell = sweep_concurrency(base, name, prompt, max_tokens, n)
            cell.prompt_label = label
            rep.cells.append(cell)

    # Recommend max_concurrent based on the medium-prompt sweep.
    cells_med: dict[int, CellResult] = {
        c.n_concurrent: c for c in rep.cells if c.prompt_label == "medium"
    }
    rep.recommended_max_concurrent = recommend_max_concurrent(cells_med)

    admin_unload_all(base)
    return rep


def render_html(reports: list[ModelReport], out_path: Path) -> None:
    def cell_to_dict(c: CellResult) -> dict[str, Any]:
        return {
            "model": c.model, "prompt": c.prompt_label, "n": c.n_concurrent,
            "throughput_tps": round(c.aggregate_throughput_tps, 2),
            "p50_s": round(c.p50, 3), "p95_s": round(c.p95, 3),
            "wall_s": round(c.wall_clock_s, 3),
            "ok": sum(1 for r in c.runs if r.ok),
            "valid": sum(1 for r in c.runs if r.valid),
            "total": len(c.runs),
            "completion_tokens": c.aggregate_completion_tokens,
        }

    data = []
    for rep in reports:
        data.append({
            "model_id": rep.model_id, "name": rep.name, "backend": rep.backend,
            "cold_load_s": round(rep.cold_load_s, 2),
            "warm_load_s": round(rep.warm_load_s, 2),
            "memory": rep.memory_after_load,
            "cells": [cell_to_dict(c) for c in rep.cells],
            "recommended_max_concurrent": rep.recommended_max_concurrent,
            "notes": rep.notes,
        })

    vllm_note = ""
    if not any(r.backend == "vllm" for r in reports):
        vllm_note = """
<div class="headline" style="background:#fde7e7;border-left-color:#b00">
<b>vLLM was excluded from this run — Gemma-4 4-bit quants are not yet loadable.</b><br>
We tried <code>unsloth/gemma-4-26B-A4B-it-GGUF</code> (Q4_K_M): vLLM 0.19.2 rejects with
<code>GGUF model with architecture gemma4 is not supported yet</code> (the
<code>load_gguf_checkpoint</code> path in transformers has no Gemma-4 entry).<br>
We tried <code>cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit</code> (compressed-tensors):
vLLM's <code>gemma4.py</code> weight loader hits
<code>KeyError: 'layers.0.moe.experts.0.down_proj_packed'</code> — the packed MoE down-projection
naming used by this checkpoint is not in vLLM's Gemma-4 expected-keys map.<br>
vLLM's model registry <i>does</i> include <code>Gemma4ForCausalLM</code> /
<code>Gemma4ForConditionalGeneration</code>, so support is partial — only full BF16
(<code>unsloth/gemma-4-26B-A4B-it</code>, ~52 GB) is likely to load today, but that
would not be a fair quant comparison vs llama.cpp Q4_K_M.<br>
<b>Re-run this benchmark</b> against <code>llm:gemma4-26b-vllm</code> when vLLM ships
a Gemma-4 4-bit-supporting release, or against any other model where both backends
load the same quant cleanly. The harness handles N models generically — pass them
comma-separated to <code>--models</code>.
</div>"""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LLM Backend Benchmark</title>
<style>
body {{ font: 14px -apple-system, sans-serif; margin: 24px; color: #222; }}
h1 {{ font-size: 22px; }}
h2 {{ margin-top: 28px; font-size: 17px; }}
table {{ border-collapse: collapse; margin: 8px 0 16px; }}
th, td {{ border: 1px solid #ddd; padding: 5px 9px; text-align: right; }}
th {{ background: #f4f4f4; }}
td.label, th.label {{ text-align: left; }}
.headline {{ background: #fff7d6; padding: 10px 14px; border-left: 4px solid #d4a017; margin: 14px 0; }}
.bad {{ color: #b00; }}
.muted {{ color: #888; font-size: 12px; }}
</style></head><body>
<h1>LLM Backend Benchmark</h1>
<p class="muted">Generated {time.strftime("%Y-%m-%d %H:%M:%S")}. Concurrency sweep: {CONCURRENCY_LEVELS}. Prompt sizes: {[p[0] for p in PROMPT_SIZES]}.</p>
{vllm_note}
<div id="root"></div>
<script>
const data = {json.dumps(data)};
const root = document.getElementById("root");

function tbl(headers, rows) {{
  let h = "<table><tr>" + headers.map(x => `<th class="label">${{x}}</th>`).join("") + "</tr>";
  for (const r of rows) {{
    h += "<tr>" + r.map((v,i) => `<td class="${{i===0?'label':''}}">${{v}}</td>`).join("") + "</tr>";
  }}
  return h + "</table>";
}}

// Headline comparison
let html = "<div class='headline'><b>Headline</b><br>";
for (const m of data) {{
  html += `<b>${{m.name}}</b> [${{m.backend}}] — cold load ${{m.cold_load_s}}s, recommended max_concurrent = <b>${{m.recommended_max_concurrent}}</b><br>`;
}}
html += "</div>";

// Per-model details
for (const m of data) {{
  html += `<h2>${{m.name}} — ${{m.backend}}</h2>`;
  html += `<p>Cold load: <b>${{m.cold_load_s}}s</b>, warm load: ${{m.warm_load_s}}s. VRAM after load: ${{(m.memory && m.memory.vram_gb) || '?'}} GB, RSS: ${{(m.memory && m.memory.rss_gb) || '?'}} GB.</p>`;
  html += `<p>Recommended max_concurrent: <b>${{m.recommended_max_concurrent}}</b></p>`;
  // throughput / latency table per prompt size
  const sizes = [...new Set(m.cells.map(c => c.prompt))];
  for (const s of sizes) {{
    html += `<h3>Prompt: ${{s}}</h3>`;
    const rows = m.cells.filter(c => c.prompt === s).sort((a,b)=>a.n-b.n).map(c => [
      c.n,
      c.throughput_tps,
      c.p50_s,
      c.p95_s,
      c.wall_s,
      `${{c.valid}}/${{c.total}}`,
      c.completion_tokens,
    ]);
    html += tbl(["N","throughput tok/s","p50 s","p95 s","wall s","valid/total","tokens"], rows);
  }}
}}

// Side-by-side
if (data.length >= 2) {{
  html += "<h2>Side-by-side throughput (medium prompt)</h2>";
  const a = data[0], b = data[1];
  const ns = [...new Set(a.cells.filter(c=>c.prompt==='medium').map(c=>c.n))].sort((x,y)=>x-y);
  const rows = ns.map(n => {{
    const ca = a.cells.find(c => c.prompt==='medium' && c.n===n) || {{throughput_tps:'-', p95_s:'-'}};
    const cb = b.cells.find(c => c.prompt==='medium' && c.n===n) || {{throughput_tps:'-', p95_s:'-'}};
    return [n, ca.throughput_tps, cb.throughput_tps, ca.p95_s, cb.p95_s];
  }});
  html += tbl(["N", a.name+" tok/s", b.name+" tok/s", a.name+" p95 s", b.name+" p95 s"], rows);
}}

root.innerHTML = html;
</script></body></html>"""
    out_path.write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arbiter", default=DEFAULT_ARBITER)
    ap.add_argument("--models", required=True,
                    help="comma-separated model_ids, e.g. 'llm:gemma4-26b,llm:gemma4-26b-vllm'")
    ap.add_argument("--out", default=str(Path.home() / "src/arbiter/benchmark_report.html"))
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    base = args.arbiter.rstrip("/")
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"Arbiter: {base}", flush=True)
    print(f"Enabling benchmark mode (external dispatch will pause)...", flush=True)
    admin_enable_benchmark(base)
    try:
        reports: list[ModelReport] = []
        for mid in model_ids:
            print(f"\n=== Running {mid} ===", flush=True)
            # Look up backend + name from the live config (URL-quote the colon).
            from urllib.parse import quote
            try:
                cfg = _http_get(f"{base}/v1/models/{quote(mid, safe='')}")
            except Exception:
                cfg = {}
            name = cfg.get("llm_name") or mid.split(":", 1)[-1]
            backend = (cfg.get("adapter_params", {}) or {}).get("LLM_BACKEND", "llamacpp")
            rep = run_for_model(base, mid, name, backend)
            reports.append(rep)
            print(f"  cold load: {rep.cold_load_s:.1f}s | recommended max_concurrent: {rep.recommended_max_concurrent}",
                  flush=True)
    finally:
        print("\nDisabling benchmark mode (queue will drain)...", flush=True)
        try:
            admin_disable_benchmark(base)
        except Exception as e:
            print(f"  WARN: failed to disable benchmark mode: {e}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps([asdict(r) for r in reports], default=str, indent=2))
    render_html(reports, out)
    print(f"\nReport: {out}\nJSON: {json_out}", flush=True)
    if not args.no_open:
        if sys.platform == "darwin":
            subprocess.run(["open", str(out)], check=False)
        else:
            webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
