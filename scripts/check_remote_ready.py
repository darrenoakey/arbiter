"""Fail before CPU-heavy validation if the real Go integration fixtures are absent."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ENDPOINTS = ("http://127.0.0.1:11434", "http://127.0.0.1:11435")
REQUIRED = {"llama3.2:3b", "nomic-embed-text:latest"}


def validate_inventory(data: object) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError("Ollama inventory must contain a models list")
    models = data["models"]
    if any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in models):
        raise ValueError("Ollama inventory contains an invalid model identity")
    missing = REQUIRED - {item["name"] for item in models}
    if missing:
        raise ValueError("required integration models absent: " + ", ".join(sorted(missing)))


def check_endpoint(endpoint: str) -> None:
    with urlopen(endpoint + "/api/tags", timeout=5) as response:
        validate_inventory(json.load(response))


def main() -> int:
    with ThreadPoolExecutor(max_workers=len(ENDPOINTS)) as executor:
        results = [(endpoint, executor.submit(check_endpoint, endpoint)) for endpoint in ENDPOINTS]
        for endpoint, result in results:
            try:
                result.result()
            except (HTTPError, URLError, ValueError, OSError) as error:
                print(f"Go integration preflight failed at {endpoint}: {error}", file=sys.stderr)
                return 1
    print("Go integration fixtures ready on both real Ollama endpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
