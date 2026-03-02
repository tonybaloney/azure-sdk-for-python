"""OTel span export verification test.

Spins up the CopilotAdapter server with an InMemorySpanExporter,
sends a real streaming request, then asserts the expected span tree.

Run with:
    uv run pytest tests/test_otel_spans.py -v -s
"""
import threading
import time
from typing import Sequence

import httpx
import pytest
import uvicorn
from copilot import SessionConfig
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult

from azure.ai.agentserver.copilot.copilot_adapter import CopilotAdapter


# ── Helpers ────────────────────────────────────────────────────────────────────

def _start_server_with_exporter(port: int) -> tuple[InMemorySpanExporter, uvicorn.Server, threading.Thread]:
    """Create an adapter whose tracer points at an InMemorySpanExporter."""
    exporter = InMemorySpanExporter()
    resource = Resource.create({"service.name": "copilot-adapter-test"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    adapter = CopilotAdapter(SessionConfig(model="gpt-5"))
    # init_tracing() in the base class will call trace.get_tracer() which now
    # uses our in-memory provider.
    adapter.init_tracing()

    config = uvicorn.Config(adapter.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(30):
        try:
            with httpx.Client() as c:
                if c.get(f"http://127.0.0.1:{port}/liveness", timeout=1).status_code == 200:
                    break
        except Exception:
            pass
        time.sleep(0.3)

    return exporter, server, thread


def _spans_by_name(spans, name: str):
    return [s for s in spans if s.name == name or s.name.startswith(name)]


def _print_span_tree(spans):
    """Print a readable tree of captured spans."""
    id_to_span = {s.context.span_id: s for s in spans}
    roots = [s for s in spans if s.parent is None or s.parent.span_id not in id_to_span]

    def _print_node(span, indent=0):
        status = span.status.status_code.name
        attrs = dict(span.attributes or {})
        attr_str = "  ".join(f"{k}={v!r}" for k, v in attrs.items())
        print(f"{'  ' * indent}▶ [{span.name}]  status={status}  {attr_str}")
        for child in spans:
            if child.parent and child.parent.span_id == span.context.span_id:
                _print_node(child, indent + 1)

    print("\n── Captured OTel Spans ──────────────────────────────────────")
    for root in roots:
        _print_node(root)
    print("─────────────────────────────────────────────────────────────\n")


# ── Fixtures ────────────────────────────────────────────────────────────────────

PORT = 18199

@pytest.fixture(scope="module")
def _otel_server():
    exporter, server, thread = _start_server_with_exporter(PORT)
    yield exporter, server
    server.should_exit = True
    thread.join(timeout=5)
    otel_trace.set_tracer_provider(otel_trace.ProxyTracerProvider())


# ── Tests ───────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_streaming_request_produces_invoke_agent_span(_otel_server):
    """A streaming /responses request must produce an invoke_agent span."""
    exporter, _ = _otel_server
    exporter.clear()

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"http://127.0.0.1:{PORT}/responses",
            json={"input": "Reply with one word: hello", "stream": True},
        ) as resp:
            resp.raise_for_status()
            # consume the full stream
            for _ in resp.iter_lines():
                pass

    # Force flush — SimpleSpanProcessor is synchronous so spans are already recorded
    spans = exporter.get_finished_spans()
    _print_span_tree(spans)

    agent_spans = _spans_by_name(spans, "invoke_agent")
    assert agent_spans, f"No invoke_agent span found. Got: {[s.name for s in spans]}"

    span = agent_spans[0]
    assert span.status.status_code.name != "ERROR", f"invoke_agent span has error status: {span.status}"


@pytest.mark.integration
def test_invoke_agent_span_has_gen_ai_attributes(_otel_server):
    """invoke_agent span must carry required gen_ai.* attributes."""
    exporter, _ = _otel_server
    exporter.clear()

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"http://127.0.0.1:{PORT}/responses",
            json={"input": "Reply with one word: hello", "stream": True},
        ) as resp:
            resp.raise_for_status()
            for _ in resp.iter_lines():
                pass

    spans = exporter.get_finished_spans()
    agent_spans = _spans_by_name(spans, "invoke_agent")
    assert agent_spans, "No invoke_agent span found"

    attrs = dict(agent_spans[0].attributes or {})
    assert attrs.get("gen_ai.operation.name") == "invoke_agent", f"Wrong gen_ai.operation.name: {attrs}"
    assert attrs.get("gen_ai.provider.name") == "github.copilot", f"Missing gen_ai.provider.name: {attrs}"
    assert "gen_ai.agent.name" in attrs, f"Missing gen_ai.agent.name: {attrs}"


@pytest.mark.integration
def test_invoke_agent_span_records_finish_reason(_otel_server):
    """invoke_agent span must have gen_ai.response.finish_reasons=['stop']."""
    exporter, _ = _otel_server
    exporter.clear()

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"http://127.0.0.1:{PORT}/responses",
            json={"input": "Reply with one word: hello", "stream": True},
        ) as resp:
            resp.raise_for_status()
            for _ in resp.iter_lines():
                pass

    spans = exporter.get_finished_spans()
    agent_spans = _spans_by_name(spans, "invoke_agent")
    assert agent_spans

    attrs = dict(agent_spans[0].attributes or {})
    finish_reasons = attrs.get("gen_ai.response.finish_reasons")
    assert finish_reasons is not None, f"gen_ai.response.finish_reasons not set: {attrs}"
    assert "stop" in finish_reasons, f"Expected 'stop' in finish_reasons: {finish_reasons}"


@pytest.mark.integration
def test_span_is_closed_not_leaked(_otel_server):
    """All spans must have an end_time — no leaked/unclosed spans."""
    exporter, _ = _otel_server
    exporter.clear()

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"http://127.0.0.1:{PORT}/responses",
            json={"input": "Reply with one word: hi", "stream": True},
        ) as resp:
            resp.raise_for_status()
            for _ in resp.iter_lines():
                pass

    spans = exporter.get_finished_spans()
    assert spans, "No spans captured at all"

    leaked = [s for s in spans if s.end_time is None]
    assert not leaked, f"Unclosed (leaked) spans: {[s.name for s in leaked]}"


@pytest.mark.integration
def test_non_streaming_request_produces_span(_otel_server):
    """A non-streaming /responses request must also produce an invoke_agent span."""
    exporter, _ = _otel_server
    exporter.clear()

    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"http://127.0.0.1:{PORT}/responses",
            json={"input": "Reply with one word: hello", "stream": False},
        )
        resp.raise_for_status()

    spans = exporter.get_finished_spans()
    _print_span_tree(spans)

    # Non-streaming now creates its own invoke_agent span (like streaming does).
    agent_spans = _spans_by_name(spans, "invoke_agent")
    assert agent_spans, f"No invoke_agent span found for non-streaming request. Got: {[s.name for s in spans]}"

    attrs = dict(agent_spans[0].attributes or {})
    assert attrs.get("gen_ai.operation.name") == "invoke_agent", f"Wrong operation name: {attrs}"
