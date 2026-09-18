"""Climb conversation over an arbiter-served multimodal LLM.

Replaces photo-enhance's Ollama VisionChat with OpenAI-compatible chat
completions against the arbiter's own vision model (qwen3-vl on spark).
The conversation protocol is unchanged: user turns carry text plus images
(base64 data URLs); earlier turns drop their images to bound context; the
model's proposals and our verdicts stay in the thread.

The transcript is ALSO bounded in length. A climb runs dozens of turns and
each verdict adds ~1k characters, so an unbounded thread walks into the
served context window: on 2026-09-18 job 0ba205f57358 reached 13885 input
tokens at turn ~20 and every later turn died with vllm's
"maximum context length is 16384 tokens" validation error, which the 5xx
retry path then sat on for 60s at a time before failing a photo that had
already cost 25 minutes of GPU. Old turns are dropped (oldest first, the
opening rules turn and the newest turns always kept) to fit a budget, and a
context-overflow answer from the backend re-trims and retries IMMEDIATELY
instead of backing off — it is a permanent error, not a restarting worker.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import urllib.request

from PIL import Image

from .settings import Settings

log = logging.getLogger(__name__)

# qwen3-vl on spark can sit behind a long queue of photo-namer drip jobs or
# a multi-minute vllm (re)load; a climb turn is not worth killing over that.
VISION_TIMEOUT = 900
MAX_ATTEMPTS = 2
JPEG_QUALITY = 88
MAX_TOKENS = 2500
TEMPERATURE = 0.5

# Context budgeting. The served qwen3-vl runs with max_model_len 16384; the
# prompt must leave room for the completion plus tokenizer/template overhead.
# Image cost is deliberately over-estimated: trimming one extra verdict costs
# a little context quality, while under-estimating costs the whole photo.
MODEL_CONTEXT_TOKENS = 16384
CONTEXT_SAFETY_TOKENS = 768
IMAGE_TOKEN_COST = 1600
CHARS_PER_TOKEN = 3.5
# newest turns that are never dropped (the latest verdict + the proposal it
# answers) and the note that replaces whatever was dropped.
MIN_KEPT_TURNS = 2
DROPPED_TURNS_NOTE = "(earlier turns of this climb were dropped to fit the model's context window)"
CONTEXT_ERROR_MARKERS = ("maximum context length", "context length", "input_tokens")
CONTEXT_RETRIES = 3


def encode_image(image: Image.Image, max_side: int = 1024) -> str:
    """JPEG base64 at a bounded size for the vision model's context."""
    copy = image.convert("RGB").copy()
    copy.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class VisionChat:
    """One multi-turn conversation with an arbiter-served multimodal model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.messages: list[dict] = []

    # ##################################################################
    # user
    def user(self, text: str, images: list[Image.Image] | None = None) -> None:
        for message in self.messages:
            message.pop("images", None)
        turn: dict = {"role": "user", "content": text}
        if images:
            turn["images"] = [encode_image(image) for image in images]
        self.messages.append(turn)

    # ##################################################################
    # assistant
    def assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    # ##################################################################
    # ask
    def ask(self) -> str:
        last_err: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self._ask_once()
            except (TimeoutError, OSError) as err:
                last_err = err
                log.warning("vision chat attempt %d failed: %s", attempt, err)
        raise RuntimeError(f"vision chat failed after {MAX_ATTEMPTS} attempts: {last_err}")

    # ##################################################################
    # ask once
    # the vision backend (vllm-served qwen3-vl) restarts several times a
    # day under fleet load and answers 5xx while down; a climb turn must
    # ride that out instead of failing the whole photo after the outer
    # retry. 5xx and connection errors retry with a backoff (3 tries);
    # a 4xx is a real protocol bug and fails immediately.
    retry_backoff_seconds = 60.0

    def _ask_once(self) -> str:
        import time
        import urllib.error

        extra_trim = 0
        trims = 0
        failures = 0
        last_error: Exception | None = None
        while failures < 3:
            messages = self.fit_to_context(self.messages, self.prompt_budget() - extra_trim)
            body = {
                "model": self.settings.vision_model,
                "stream": False,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "messages": [self._openai_turn(m) for m in messages],
            }
            request = urllib.request.Request(
                f"{self.settings.arbiter_url}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=VISION_TIMEOUT) as response:
                    payload = json.load(response)
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as err:
                detail = read_error_body(err)
                overflow = context_overflow_tokens(detail)
                if overflow is not None and trims < CONTEXT_RETRIES:
                    # permanent for this prompt size: trim by the reported
                    # overflow (plus a margin) and retry straight away. A trim
                    # is not a backend failure, so it does not burn a retry.
                    extra_trim += overflow + CONTEXT_SAFETY_TOKENS
                    trims += 1
                    log.warning("vision chat over context by ~%d tokens; trimming history and retrying", overflow)
                    last_error = err
                    continue
                if err.code < 500:
                    raise
                last_error = err
            except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                last_error = err
            failures += 1
            log.warning("vision chat attempt %d failed (%s); backing off", failures, last_error)
            time.sleep(self.retry_backoff_seconds)
        raise RuntimeError(f"vision chat failed after 3 attempts: {last_error}")

    # ##################################################################
    # prompt budget
    # tokens the prompt may occupy: the served window minus the completion we
    # ask for and a safety margin for the chat template's own tokens.
    @staticmethod
    def prompt_budget() -> int:
        return MODEL_CONTEXT_TOKENS - MAX_TOKENS - CONTEXT_SAFETY_TOKENS

    # ##################################################################
    # fit to context
    # keep the opening turn (it carries the rules the model must follow) and
    # the newest turns, dropping the oldest middle turns until the estimate
    # fits. A note replaces the dropped block so the model knows the thread
    # is abridged rather than contradicting itself.
    @staticmethod
    def fit_to_context(messages: list[dict], budget: int) -> list[dict]:
        if not messages or estimate_tokens(messages) <= budget:
            return list(messages)
        opening, rest = messages[0], messages[1:]
        note = {"role": "user", "content": DROPPED_TURNS_NOTE}
        for drop in range(len(rest)):
            kept = rest[drop:]
            candidate = [opening, note, *kept]
            if estimate_tokens(candidate) <= budget or len(kept) <= MIN_KEPT_TURNS:
                return candidate
        return [opening, note, *rest[-MIN_KEPT_TURNS:]]


    # ##################################################################
    # openai turn
    # our internal turns carry a plain `content` string plus an optional
    # `images` list of base64 JPEGs; OpenAI wants content as typed parts.
    @staticmethod
    def _openai_turn(turn: dict) -> dict:
        content: list[dict] = [{"type": "text", "text": turn["content"]}]
        for encoded in turn.get("images", []):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        return {"role": turn["role"], "content": content}

    # ##################################################################
    # transcript
    def transcript(self) -> list[dict]:
        return [{"role": m["role"], "text": m["content"]} for m in self.messages]


# ##################################################################
# estimate tokens
# characters/CHARS_PER_TOKEN for text plus a fixed cost per attached image.
# Only an estimate: the reactive trim in _ask_once corrects any shortfall the
# backend actually reports.
def estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        total += int(len(message.get("content", "")) / CHARS_PER_TOKEN) + 1
        total += IMAGE_TOKEN_COST * len(message.get("images", []))
    return total


# ##################################################################
# read error body
# an HTTPError is a readable response; the body carries the backend's reason
# (arbiter forwards the worker's error text) and is safe to consume once.
def read_error_body(err: object) -> str:
    try:
        raw = err.read()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - body already consumed or absent
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw)


# ##################################################################
# context overflow tokens
# how many tokens over the window the backend says we are, or None when the
# error is not a context-length rejection. vllm reports both the total and the
# prompt size: "maximum context length is 16384 tokens. However, you requested
# 2500 output tokens and your prompt contains at least 13885 input tokens, for
# a total of at least 16385 tokens".
def context_overflow_tokens(detail: str) -> int | None:
    text = detail.lower()
    if not any(marker in text for marker in CONTEXT_ERROR_MARKERS):
        return None
    limit = re.search(r"maximum context length is (\d+)", text)
    total = re.search(r"total of at least (\d+)", text)
    if limit and total:
        return max(int(total.group(1)) - int(limit.group(1)), 1)
    prompt = re.search(r"(\d+) input tokens", text)
    if limit and prompt:
        return max(int(prompt.group(1)) + MAX_TOKENS - int(limit.group(1)), 1)
    # recognised the error but not the numbers: trim a fixed slice and retry.
    return MAX_TOKENS


# ##################################################################
# parse json object
# pull the first JSON object out of a reply that may be wrapped in fences
# or prose (or preceded by chain-of-thought); raises ValueError with the
# offending text.
def parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in reply: {text[:300]!r}")
    return json.loads(match.group(0))
