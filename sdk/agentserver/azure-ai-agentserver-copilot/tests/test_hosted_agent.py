# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Integration tests for the Copilot hosted agent adapter.

These tests validate the full HTTP surface area described in the
hosted agents tutorial:
https://learn.microsoft.com/azure/ai-foundry/agents/concepts/hosted-agents
"""

import json

import pytest
import uvicorn
from copilot import SessionConfig

from azure.ai.agentserver.copilot.copilot_adapter import CopilotAdapter


@pytest.fixture(scope="module")
def _server():
    """Start the adapter HTTP server for the duration of the test module."""
    import threading

    import httpx

    adapter = CopilotAdapter(SessionConfig(model="gpt-5"))
    adapter.init_tracing()
    config = uvicorn.Config(adapter.app, host="127.0.0.1", port=18099, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # wait for server to be ready
    import time
    for _ in range(30):
        try:
            with httpx.Client() as client:
                r = client.get("http://127.0.0.1:18099/liveness", timeout=1)
                if r.status_code == 200:
                    break
        except Exception:
            pass
        time.sleep(0.5)
    yield server
    server.should_exit = True
    thread.join(timeout=5)


@pytest.mark.integration
def test_liveness_endpoint(_server):
    """GET /liveness returns 200."""
    import httpx
    with httpx.Client() as client:
        r = client.get("http://127.0.0.1:18099/liveness")
    assert r.status_code == 200


@pytest.mark.integration
def test_readiness_endpoint(_server):
    """GET /readiness returns 200 with status ready."""
    import httpx
    with httpx.Client() as client:
        r = client.get("http://127.0.0.1:18099/readiness")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"


@pytest.mark.integration
def test_responses_endpoint_non_streaming(_server):
    """POST /responses with stream=false returns a JSON response."""
    import httpx
    with httpx.Client(timeout=120) as client:
        r = client.post(
            "http://127.0.0.1:18099/responses",
            json={"input": "Reply with one word: hello", "stream": False},
        )
    assert r.status_code == 200
    body = r.json()
    # Should have the 'output' key with at least one output item
    assert "output" in body
    assert isinstance(body["output"], list)
    assert len(body["output"]) >= 1


@pytest.mark.integration
def test_responses_endpoint_streaming(_server):
    """POST /responses with stream=true returns SSE events."""
    import httpx
    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            "http://127.0.0.1:18099/responses",
            json={"input": "Reply with one word: hello", "stream": True},
        ) as r:
            assert r.status_code == 200
            events = []
            for line in r.iter_lines():
                if line.startswith("data: "):
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        break
                    events.append(json.loads(payload))

    # Should have at least created + completed events
    assert len(events) >= 2
    event_types = [e.get("type") for e in events]
    assert "response.created" in event_types
