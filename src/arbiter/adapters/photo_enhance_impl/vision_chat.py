"""Climb conversation over an arbiter-served multimodal LLM.

Replaces photo-enhance's Ollama VisionChat with OpenAI-compatible chat
completions against the arbiter's own vision model (qwen3-vl on spark).
The conversation protocol is unchanged: user turns carry text plus images
(base64 data URLs); earlier turns drop their images to bound context; the
model's proposals and our verdicts stay in the thread.
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

        body = {
            "model": self.settings.vision_model,
            "stream": False,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "messages": [self._openai_turn(m) for m in self.messages],
        }
        request = urllib.request.Request(
            f"{self.settings.arbiter_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=VISION_TIMEOUT) as response:
                    payload = json.load(response)
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as err:
                if err.code < 500:
                    raise
                last_error = err
            except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                last_error = err
            log.warning("vision chat attempt %d failed (%s); backing off", attempt + 1, last_error)
            time.sleep(self.retry_backoff_seconds)
        raise RuntimeError(f"vision chat failed after 3 attempts: {last_error}")

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
