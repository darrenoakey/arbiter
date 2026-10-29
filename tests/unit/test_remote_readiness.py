"""Readiness is checked against the same real services as the Go suite."""
from __future__ import annotations

import socket

import pytest

from scripts.check_remote_ready import ENDPOINTS, check_endpoint, main, validate_inventory


def test_real_required_ollama_fixtures_are_present():
    for endpoint in ENDPOINTS:
        check_endpoint(endpoint)
    assert main() == 0


@pytest.mark.parametrize("inventory", [None, {}, {"models": []}, {"models": [{}]}, {"models": [{"name": "llama3.2:3b"}]}])
def test_missing_or_ambiguous_inventory_fails_closed(inventory):
    with pytest.raises(ValueError):
        validate_inventory(inventory)


def test_unreachable_service_fails_closed():
    # Reserve a real OS-assigned port without listening: no fake HTTP server.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        with pytest.raises(OSError):
            check_endpoint(f"http://127.0.0.1:{reserved.getsockname()[1]}")
