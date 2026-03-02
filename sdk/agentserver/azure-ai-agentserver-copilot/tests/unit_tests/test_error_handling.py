# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for OTel error recording compliance.

Validates that error handling follows the OTel recording-errors guidance:
https://opentelemetry.io/docs/specs/semconv/general/recording-errors/

Key requirements:
    - Span status code MUST be left unset for successful operations
    - On error: SHOULD set status to Error, SHOULD set error.type,
      SHOULD set status description (exception message)
    - Exceptions SHOULD be recorded on the span (record_exception)
    - error.type should use fully-qualified exception type names
    - Copilot SDK SESSION_ERROR events SHOULD be treated as errors
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict
from unittest.mock import MagicMock, patch

import pytest

from copilot.generated.session_events import SessionEventType
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from azure.ai.agentserver.copilot.copilot_adapter import (
    CopilotAdapter,
    _error_type_name,
)
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext

# Applied per-class below (not module-level) to avoid warnings on sync tests.

_PATCH_ITER = "azure.ai.agentserver.copilot.copilot_adapter._iter_copilot_events"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type: SessionEventType,
    *,
    content: str | None = None,
    message: str | None = None,
    model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    mcp_tool_name: str | None = None,
    mcp_server_name: str | None = None,
    arguments: Any = None,
    result: Any = None,
    success: bool | None = None,
) -> MagicMock:
    """Build a minimal Copilot SessionEvent mock."""
    event = MagicMock()
    event.type = event_type

    if any(v is not None for v in (
        content, message, model, input_tokens, output_tokens,
        tool_call_id, tool_name, mcp_tool_name, mcp_server_name,
        arguments, result, success,
    )):
        data = MagicMock()
        data.content = content
        data.message = message
        data.model = model
        data.input_tokens = input_tokens
        data.output_tokens = output_tokens
        data.tool_call_id = tool_call_id
        data.tool_name = tool_name
        data.mcp_tool_name = mcp_tool_name
        data.mcp_server_name = mcp_server_name
        data.arguments = arguments
        data.result = result
        data.success = success
        event.data = data
    else:
        event.data = None
    return event


class FakeSession:
    """Minimal session stub that delivers pre-loaded events to listeners."""

    def __init__(self) -> None:
        self._callbacks: list[Callable] = []
        self._auto_events: list[MagicMock] = []

    def on(self, cb: Callable) -> Callable:
        self._callbacks.append(cb)
        def unsubscribe() -> None:
            self._callbacks.remove(cb)
        return unsubscribe

    async def send(self, options: Any) -> None:
        loop = asyncio.get_running_loop()
        events = list(self._auto_events)
        loop.call_soon(self._push_many, events)

    def _push_many(self, events: list) -> None:
        for e in events:
            for cb in list(self._callbacks):
                cb(e)


def _make_context(*, stream: bool = True) -> AgentRunContext:
    """Build a minimal AgentRunContext."""
    payload: Dict[str, Any] = {
        "input": "hello",
        "stream": stream,
        "agent": {"type": "agent_id", "name": "test-agent", "version": "1.0"},
        "conversation": {"id": "conv_123"},
    }
    return AgentRunContext(payload)


def _make_config() -> Dict[str, Any]:
    return {"model": "gpt-4.1"}


def _make_attachments() -> MagicMock:
    """Build a mock ConvertedAttachments."""
    attachments = MagicMock()
    attachments.attachments = None
    attachments.cleanup = MagicMock()
    return attachments


@pytest.fixture
def exporter_and_tracer():
    """Create an InMemorySpanExporter + TracerProvider for capturing spans."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-error-handling")
    return exporter, tracer, provider


def _make_adapter(tracer) -> CopilotAdapter:
    """Create a minimal CopilotAdapter with a real tracer for span tests."""
    adapter = CopilotAdapter.__new__(CopilotAdapter)
    adapter.tracer = tracer
    adapter._session_config = _make_config()
    return adapter


# Fake async generators that raise immediately — avoids real 120s timeouts.

async def _raising_conn_error(*_args: Any, **_kwargs: Any):
    """Async generator that raises ConnectionError immediately."""
    raise ConnectionError("simulated network failure")
    yield  # noqa: unreachable — makes this a generator


async def _raising_timeout(*_args: Any, **_kwargs: Any):
    """Async generator that raises asyncio.TimeoutError."""
    raise asyncio.TimeoutError("session timed out")
    yield  # noqa: unreachable


async def _tool_start_then_raise(*_args: Any, **_kwargs: Any):
    """Yield a tool-start event, then raise to simulate stream abort."""
    yield _make_event(
        SessionEventType.TOOL_EXECUTION_START,
        tool_call_id="call_orphan",
        tool_name="slow_tool",
        mcp_tool_name="ns:slow_tool",
        mcp_server_name=None,
        arguments=None,
    )
    raise ConnectionError("connection lost")


# ===========================================================================
# _error_type_name helper
# ===========================================================================


@pytest.mark.unit
class TestErrorTypeName:
    """The ``_error_type_name`` helper should return fully-qualified type names."""

    def test_builtin_exception(self) -> None:
        """Builtins should return just the qualname (no module prefix)."""
        assert _error_type_name(ValueError("bad")) == "ValueError"

    def test_standard_library_exception(self) -> None:
        """Standard library exceptions should include the module."""
        exc = asyncio.TimeoutError("timed out")
        result = _error_type_name(exc)
        # asyncio.TimeoutError is actually builtins.TimeoutError in newer Python
        assert "TimeoutError" in result

    def test_custom_exception(self) -> None:
        """Custom exception classes include their module path."""

        class MyCustomError(Exception):
            pass

        exc = MyCustomError("oops")
        result = _error_type_name(exc)
        assert "MyCustomError" in result
        assert "." in result  # Should have module prefix

    def test_exception_group(self) -> None:
        """ExceptionGroup from SDK 0.1.29 should be handled."""
        eg = ExceptionGroup("stop errors", [RuntimeError("a")])
        result = _error_type_name(eg)
        assert "ExceptionGroup" in result


# ===========================================================================
# Streaming path: Python exception → span error
# ===========================================================================


@pytest.mark.unit
@pytest.mark.asyncio(loop_scope="function")
class TestStreamingExceptionError:
    """When _run_streaming raises a Python exception, the invoke_agent span
    SHOULD have: error.type (FQ name), StatusCode.ERROR, recorded exception."""

    async def test_exception_sets_error_type_and_status(self, exporter_and_tracer) -> None:
        """ConnectionError → error.type is FQ name, status is ERROR, description set."""
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        with patch(_PATCH_ITER, _raising_conn_error):
            with pytest.raises(ConnectionError):
                async for _ in adapter._run_streaming(
                    FakeSession(), "hi", _make_attachments(), context, _make_config()
                ):
                    pass

        spans = exporter.get_finished_spans()
        invoke_spans = [s for s in spans if s.name.startswith("invoke_agent")]
        assert len(invoke_spans) == 1
        span = invoke_spans[0]

        # error.type SHOULD be the fully-qualified exception type name
        assert "error.type" in dict(span.attributes)
        assert "ConnectionError" in span.attributes["error.type"]

        # Span status SHOULD be ERROR
        assert span.status.status_code == StatusCode.ERROR

        # Status description SHOULD contain the exception message
        # (record_exception may format it as "Type: message")
        assert "simulated network failure" in span.status.description

    async def test_exception_records_exception_event(self, exporter_and_tracer) -> None:
        """record_exception creates a span event with exception details."""
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        with patch(_PATCH_ITER, _raising_conn_error):
            with pytest.raises(ConnectionError):
                async for _ in adapter._run_streaming(
                    FakeSession(), "hi", _make_attachments(), context, _make_config()
                ):
                    pass

        spans = exporter.get_finished_spans()
        invoke_span = [s for s in spans if s.name.startswith("invoke_agent")][0]

        # SHOULD have an exception event recorded
        exception_events = [e for e in invoke_span.events if e.name == "exception"]
        assert len(exception_events) >= 1
        exc_event = exception_events[0]
        assert "exception.type" in dict(exc_event.attributes)
        assert "exception.message" in dict(exc_event.attributes)
        assert exc_event.attributes["exception.message"] == "simulated network failure"

    async def test_timeout_error_sets_error_type(self, exporter_and_tracer) -> None:
        """asyncio.TimeoutError → error.type contains 'TimeoutError'."""
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        with patch(_PATCH_ITER, _raising_timeout):
            with pytest.raises(asyncio.TimeoutError):
                async for _ in adapter._run_streaming(
                    FakeSession(), "hi", _make_attachments(), context, _make_config()
                ):
                    pass

        spans = exporter.get_finished_spans()
        invoke_span = [s for s in spans if s.name.startswith("invoke_agent")][0]

        assert "TimeoutError" in invoke_span.attributes["error.type"]
        assert invoke_span.status.status_code == StatusCode.ERROR

    async def test_exception_does_not_set_stop_finish_reason(self, exporter_and_tracer) -> None:
        """On exception, finish_reasons should NOT be 'stop'."""
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        with patch(_PATCH_ITER, _raising_conn_error):
            with pytest.raises(ConnectionError):
                async for _ in adapter._run_streaming(
                    FakeSession(), "hi", _make_attachments(), context, _make_config()
                ):
                    pass

        spans = exporter.get_finished_spans()
        invoke_span = [s for s in spans if s.name.startswith("invoke_agent")][0]
        finish_reasons = invoke_span.attributes.get("gen_ai.response.finish_reasons")
        # On exception the except block runs before the finish_reasons line,
        # so finish_reasons may not be set at all — but it MUST NOT be ("stop",).
        if finish_reasons is not None:
            assert finish_reasons != ("stop",)


# ===========================================================================
# Streaming path: SESSION_ERROR event → span error
# ===========================================================================


@pytest.mark.unit
@pytest.mark.asyncio(loop_scope="function")
class TestStreamingSessionError:
    """When the Copilot SDK delivers a SESSION_ERROR event (not a Python
    exception), the invoke_agent span SHOULD still be marked as errored."""

    async def test_session_error_sets_error_type(self, exporter_and_tracer) -> None:
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        session = FakeSession()
        session._auto_events = [
            _make_event(SessionEventType.ASSISTANT_TURN_START),
            _make_event(SessionEventType.SESSION_ERROR, message="Model overloaded"),
            _make_event(SessionEventType.SESSION_IDLE),
        ]

        async for _ in adapter._run_streaming(
            session, "hi", _make_attachments(), context, _make_config()
        ):
            pass

        spans = exporter.get_finished_spans()
        invoke_span = [s for s in spans if s.name.startswith("invoke_agent")][0]

        # error.type SHOULD be "session_error"
        assert invoke_span.attributes["error.type"] == "session_error"

    async def test_session_error_sets_status_error(self, exporter_and_tracer) -> None:
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        session = FakeSession()
        session._auto_events = [
            _make_event(SessionEventType.SESSION_ERROR, message="Auth failure"),
            _make_event(SessionEventType.SESSION_IDLE),
        ]

        async for _ in adapter._run_streaming(
            session, "hi", _make_attachments(), context, _make_config()
        ):
            pass

        spans = exporter.get_finished_spans()
        invoke_span = [s for s in spans if s.name.startswith("invoke_agent")][0]

        assert invoke_span.status.status_code == StatusCode.ERROR
        assert invoke_span.status.description == "Auth failure"

    async def test_session_error_finish_reason_is_error(self, exporter_and_tracer) -> None:
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        session = FakeSession()
        session._auto_events = [
            _make_event(SessionEventType.SESSION_ERROR, message="rate limited"),
            _make_event(SessionEventType.SESSION_IDLE),
        ]

        async for _ in adapter._run_streaming(
            session, "hi", _make_attachments(), context, _make_config()
        ):
            pass

        spans = exporter.get_finished_spans()
        invoke_span = [s for s in spans if s.name.startswith("invoke_agent")][0]

        finish_reasons = invoke_span.attributes.get("gen_ai.response.finish_reasons")
        assert finish_reasons == ("error",)

    async def test_session_error_with_content_attr(self, exporter_and_tracer) -> None:
        """SESSION_ERROR with data.content (no data.message) should still be captured."""
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        session = FakeSession()
        error_event = _make_event(SessionEventType.SESSION_ERROR, content="server overloaded")
        error_event.data.message = None
        session._auto_events = [
            error_event,
            _make_event(SessionEventType.SESSION_IDLE),
        ]

        async for _ in adapter._run_streaming(
            session, "hi", _make_attachments(), context, _make_config()
        ):
            pass

        spans = exporter.get_finished_spans()
        invoke_span = [s for s in spans if s.name.startswith("invoke_agent")][0]

        assert invoke_span.status.status_code == StatusCode.ERROR
        assert invoke_span.status.description == "server overloaded"


# ===========================================================================
# Successful operation: no error state
# ===========================================================================


@pytest.mark.unit
@pytest.mark.asyncio(loop_scope="function")
class TestSuccessfulOperation:
    """A successful streaming operation MUST NOT have error.type or ERROR status."""

    async def test_success_has_no_error_type(self, exporter_and_tracer) -> None:
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        session = FakeSession()
        session._auto_events = [
            _make_event(SessionEventType.ASSISTANT_TURN_START),
            _make_event(SessionEventType.ASSISTANT_MESSAGE, content="Hello!"),
            _make_event(SessionEventType.ASSISTANT_TURN_END),
            _make_event(SessionEventType.SESSION_IDLE),
        ]

        async for _ in adapter._run_streaming(
            session, "hi", _make_attachments(), context, _make_config()
        ):
            pass

        spans = exporter.get_finished_spans()
        invoke_span = [s for s in spans if s.name.startswith("invoke_agent")][0]

        assert "error.type" not in dict(invoke_span.attributes)
        assert invoke_span.status.status_code == StatusCode.UNSET
        assert invoke_span.attributes["gen_ai.response.finish_reasons"] == ("stop",)

    async def test_success_has_no_exception_events(self, exporter_and_tracer) -> None:
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        session = FakeSession()
        session._auto_events = [
            _make_event(SessionEventType.ASSISTANT_MESSAGE, content="OK"),
            _make_event(SessionEventType.SESSION_IDLE),
        ]

        async for _ in adapter._run_streaming(
            session, "hi", _make_attachments(), context, _make_config()
        ):
            pass

        spans = exporter.get_finished_spans()
        invoke_span = [s for s in spans if s.name.startswith("invoke_agent")][0]

        exception_events = [e for e in invoke_span.events if e.name == "exception"]
        assert len(exception_events) == 0


# ===========================================================================
# Tool span error handling
# ===========================================================================


@pytest.mark.unit
@pytest.mark.asyncio(loop_scope="function")
class TestToolSpanErrors:
    """Tool spans should follow recording-errors guidance for failures."""

    async def test_tool_error_sets_status_error(self, exporter_and_tracer) -> None:
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        session = FakeSession()
        session._auto_events = [
            _make_event(
                SessionEventType.TOOL_EXECUTION_START,
                tool_call_id="call_1",
                tool_name="my_tool",
                mcp_tool_name="ns:my_tool",
                mcp_server_name="test-server",
                arguments={"x": 1},
            ),
            _make_event(
                SessionEventType.TOOL_EXECUTION_COMPLETE,
                tool_call_id="call_1",
                result="error: not found",
                success=False,
            ),
            _make_event(SessionEventType.SESSION_IDLE),
        ]

        async for _ in adapter._run_streaming(
            session, "hi", _make_attachments(), context, _make_config()
        ):
            pass

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name.startswith("tools/call")]
        assert len(tool_spans) == 1
        tool_span = tool_spans[0]

        assert tool_span.attributes["error.type"] == "tool_error"
        assert tool_span.status.status_code == StatusCode.ERROR
        assert tool_span.status.description == "tool execution failed"

    async def test_tool_success_no_error(self, exporter_and_tracer) -> None:
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        session = FakeSession()
        session._auto_events = [
            _make_event(
                SessionEventType.TOOL_EXECUTION_START,
                tool_call_id="call_2",
                tool_name="good_tool",
                mcp_tool_name="ns:good_tool",
                mcp_server_name=None,
                arguments=None,
            ),
            _make_event(
                SessionEventType.TOOL_EXECUTION_COMPLETE,
                tool_call_id="call_2",
                result='{"ok": true}',
                success=True,
            ),
            _make_event(SessionEventType.SESSION_IDLE),
        ]

        async for _ in adapter._run_streaming(
            session, "hi", _make_attachments(), context, _make_config()
        ):
            pass

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name.startswith("tools/call")]
        assert len(tool_spans) == 1
        tool_span = tool_spans[0]

        assert "error.type" not in dict(tool_span.attributes)
        assert tool_span.status.status_code == StatusCode.UNSET

    async def test_stream_abort_marks_open_tool_spans(self, exporter_and_tracer) -> None:
        """Tool spans still open when the stream aborts SHOULD be closed
        with error.type='stream_aborted' and ERROR status."""
        exporter, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)

        with patch(_PATCH_ITER, _tool_start_then_raise):
            with pytest.raises(ConnectionError):
                async for _ in adapter._run_streaming(
                    FakeSession(), "hi", _make_attachments(), context, _make_config()
                ):
                    pass

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name.startswith("tools/call")]
        assert len(tool_spans) == 1
        tool_span = tool_spans[0]

        assert tool_span.attributes["error.type"] == "stream_aborted"
        assert tool_span.status.status_code == StatusCode.ERROR
        assert tool_span.status.description == "stream aborted"


# ===========================================================================
# Attachments cleanup on error
# ===========================================================================


@pytest.mark.unit
@pytest.mark.asyncio(loop_scope="function")
class TestCleanupOnError:
    """Converted attachments are always cleaned up, even on error."""

    async def test_cleanup_called_on_exception(self, exporter_and_tracer) -> None:
        _, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)
        attachments = _make_attachments()

        with patch(_PATCH_ITER, _raising_conn_error):
            with pytest.raises(ConnectionError):
                async for _ in adapter._run_streaming(
                    FakeSession(), "hi", attachments, context, _make_config()
                ):
                    pass

        attachments.cleanup.assert_called_once()

    async def test_cleanup_called_on_success(self, exporter_and_tracer) -> None:
        _, tracer, _ = exporter_and_tracer
        adapter = _make_adapter(tracer)
        context = _make_context(stream=True)
        attachments = _make_attachments()

        session = FakeSession()
        session._auto_events = [
            _make_event(SessionEventType.ASSISTANT_MESSAGE, content="done"),
            _make_event(SessionEventType.SESSION_IDLE),
        ]

        async for _ in adapter._run_streaming(
            session, "hi", attachments, context, _make_config()
        ):
            pass

        attachments.cleanup.assert_called_once()
