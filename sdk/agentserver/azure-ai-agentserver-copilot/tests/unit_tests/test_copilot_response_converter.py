# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for CopilotStreamingResponseConverter."""
import datetime
import uuid
from unittest.mock import MagicMock

import pytest

from copilot.generated.session_events import Data, SessionEvent, SessionEventType

from azure.ai.agentserver.core.models.projects import (
    ResponseCompletedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext
from azure.ai.agentserver.copilot.models.copilot_response_converter import CopilotStreamingResponseConverter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(event_type: SessionEventType, content: str | None = None, **kwargs) -> SessionEvent:
    """Create a real SessionEvent dataclass instance."""
    data = Data(content=content, **kwargs)
    return SessionEvent(
        data=data,
        id=uuid.uuid4(),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        type=event_type,
    )


def _make_context(*, with_agent: bool = True, with_conversation: bool = True) -> AgentRunContext:
    payload: dict = {"input": "hello", "stream": True}
    if with_agent:
        payload["agent"] = {"type": "agent_id", "name": "test-agent", "version": "1"}
    if with_conversation:
        payload["_conversation_id"] = "conv_test123"
    return AgentRunContext(payload)


def _run_simple_turn(context: AgentRunContext, text: str = "Hello!"):
    """Simulate the guaranteed runtime event order for a simple text response:
    TURN_START → USAGE → MESSAGE → TURN_END → SESSION_IDLE
    """
    converter = CopilotStreamingResponseConverter(context)
    events = []
    events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
    events.extend(converter._convert_event(
        _make_event(SessionEventType.ASSISTANT_USAGE, input_tokens=10, output_tokens=5), context))
    events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_MESSAGE, text), context))
    events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
    events.extend(converter._convert_event(_make_event(SessionEventType.SESSION_IDLE), context))
    return events, converter


def _run_streaming_turn(context: AgentRunContext, deltas: list[str], full_text: str):
    """Simulate streaming: TURN_START → DELTA×N → USAGE → MESSAGE → TURN_END → IDLE"""
    converter = CopilotStreamingResponseConverter(context)
    events = []
    events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
    for delta in deltas:
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, delta), context))
    events.extend(converter._convert_event(
        _make_event(SessionEventType.ASSISTANT_USAGE, input_tokens=10, output_tokens=5), context))
    events.extend(converter._convert_event(
        _make_event(SessionEventType.ASSISTANT_MESSAGE, full_text), context))
    events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
    events.extend(converter._convert_event(_make_event(SessionEventType.SESSION_IDLE), context))
    return events, converter


# ---------------------------------------------------------------------------
# _build_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildResponse:
    """Tests for CopilotStreamingResponseConverter._build_response()."""

    def test_includes_object_field(self):
        conv = CopilotStreamingResponseConverter(_make_context())
        resp = conv._build_response("in_progress")
        assert resp.get("object") == "response"

    def test_includes_id(self):
        ctx = _make_context()
        conv = CopilotStreamingResponseConverter(ctx)
        resp = conv._build_response("in_progress")
        assert resp.get("id") == ctx.response_id

    def test_includes_created_at(self):
        conv = CopilotStreamingResponseConverter(_make_context())
        resp = conv._build_response("in_progress")
        created_at = resp.get("created_at")
        assert isinstance(created_at, int) and created_at > 0

    def test_created_at_is_consistent(self):
        conv = CopilotStreamingResponseConverter(_make_context())
        r1 = conv._build_response("in_progress")
        r2 = conv._build_response("completed")
        assert r1.get("created_at") == r2.get("created_at")

    def test_includes_agent_id_when_present(self):
        conv = CopilotStreamingResponseConverter(_make_context(with_agent=True))
        resp = conv._build_response("in_progress")
        assert resp.get("agent_id") is not None
        assert resp.get("agent_id").get("name") == "test-agent"

    def test_omits_agent_id_when_absent(self):
        conv = CopilotStreamingResponseConverter(_make_context(with_agent=False))
        resp = conv._build_response("in_progress")
        assert resp.get("agent_id") is None

    def test_includes_conversation_when_present(self):
        ctx = _make_context(with_conversation=True)
        conv = CopilotStreamingResponseConverter(ctx)
        resp = conv._build_response("in_progress")
        assert resp.get("conversation") is not None

    def test_includes_output_when_provided(self):
        conv = CopilotStreamingResponseConverter(_make_context())
        fake_item = MagicMock()
        resp = conv._build_response("completed", output=[fake_item])
        assert resp.get("output") == [fake_item]

    def test_output_absent_when_not_provided(self):
        conv = CopilotStreamingResponseConverter(_make_context())
        resp = conv._build_response("in_progress")
        assert resp.get("output") is None


# ---------------------------------------------------------------------------
# ASSISTANT_TURN_START
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTurnStartEvents:

    def test_emits_response_created_first(self):
        events, _ = _run_simple_turn(_make_context())
        assert isinstance(events[0], ResponseCreatedEvent)

    def test_emits_response_in_progress_second(self):
        events, _ = _run_simple_turn(_make_context())
        assert isinstance(events[1], ResponseInProgressEvent)

    def test_created_event_response_has_required_fields(self):
        events, _ = _run_simple_turn(_make_context())
        resp = events[0].get("response")
        assert resp.get("object") == "response"
        assert resp.get("id") is not None
        assert resp.get("status") == "in_progress"
        assert resp.get("created_at") is not None

    def test_in_progress_event_response_has_required_fields(self):
        events, _ = _run_simple_turn(_make_context())
        resp = events[1].get("response")
        assert resp.get("object") == "response"
        assert resp.get("id") is not None
        assert resp.get("status") == "in_progress"
        assert resp.get("created_at") is not None

    def test_sequence_numbers_are_sequential(self):
        events, _ = _run_simple_turn(_make_context())
        seq_numbers = [e.get("sequence_number") for e in events]
        for i in range(1, len(seq_numbers)):
            assert seq_numbers[i] == seq_numbers[i - 1] + 1


# ---------------------------------------------------------------------------
# ASSISTANT_MESSAGE emits done events + response.completed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMessageEvents:

    def test_last_event_is_response_completed(self):
        events, _ = _run_simple_turn(_make_context())
        assert isinstance(events[-1], ResponseCompletedEvent)

    def test_completed_event_response_has_required_fields(self):
        events, _ = _run_simple_turn(_make_context())
        resp = events[-1].get("response")
        assert resp.get("object") == "response"
        assert resp.get("id") is not None
        assert resp.get("status") == "completed"
        assert resp.get("created_at") is not None

    def test_completed_response_includes_output_item(self):
        events, _ = _run_simple_turn(_make_context())
        resp = events[-1].get("response")
        output = resp.get("output")
        assert output is not None and len(output) == 1

    def test_created_at_consistent_across_turn(self):
        events, _ = _run_simple_turn(_make_context())
        created_at_start = events[0].get("response").get("created_at")
        created_at_end = events[-1].get("response").get("created_at")
        assert created_at_start == created_at_end

    def test_emits_synthetic_delta_when_no_streaming(self):
        """When no ASSISTANT_MESSAGE_DELTA events arrive, ASSISTANT_MESSAGE
        should emit a synthetic delta so clients always see at least one."""
        events, _ = _run_simple_turn(_make_context(), text="Hi there")
        deltas = [e for e in events if isinstance(e, ResponseTextDeltaEvent)]
        assert len(deltas) == 1
        assert deltas[0].get("delta") == "Hi there"


# ---------------------------------------------------------------------------
# Multi-turn (tool-using) behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiTurn:

    def _run_two_turns(self, context: AgentRunContext):
        """Simulate: turn1 (tool-calling, no text content in message),
        turn2 (actual answer).  Runtime order per turn:
        TURN_START → USAGE → MESSAGE → TURN_END
        """
        converter = CopilotStreamingResponseConverter(context)
        events = []
        # Turn 1: tool-calling turn — MESSAGE has no text content
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_USAGE, input_tokens=5, output_tokens=2), context))
        # Tool-calling turns emit ASSISTANT_MESSAGE with tool_requests but no text content.
        # Our converter ignores messages without content, so we send TURN_END directly.
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
        # Turn 2: actual answer
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_USAGE, input_tokens=10, output_tokens=5), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_MESSAGE, "Done!"), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.SESSION_IDLE), context))
        return events, converter

    def test_response_created_emitted_only_once(self):
        events, _ = self._run_two_turns(_make_context())
        created = [e for e in events if isinstance(e, ResponseCreatedEvent)]
        assert len(created) == 1

    def test_response_in_progress_emitted_only_once(self):
        events, _ = self._run_two_turns(_make_context())
        in_prog = [e for e in events if isinstance(e, ResponseInProgressEvent)]
        assert len(in_prog) == 1

    def test_second_turn_gets_different_item_id(self):
        events, _ = self._run_two_turns(_make_context())
        added = [e for e in events if isinstance(e, ResponseOutputItemAddedEvent)]
        assert len(added) == 2
        id1 = added[0].get("item", {}).get("id")
        id2 = added[1].get("item", {}).get("id")
        assert id1 != id2

    def test_no_response_completed_on_first_turn_when_no_content(self):
        """Tool-calling turns with no text content must NOT emit response.completed."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = []
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 0

    def test_response_completed_emitted_once_at_end(self):
        events, _ = self._run_two_turns(_make_context())
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1


# ---------------------------------------------------------------------------
# Token usage from ASSISTANT_USAGE
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUsage:

    def test_usage_stored_after_assistant_usage_event(self):
        context = _make_context()
        _, converter = _run_simple_turn(context)
        assert converter._usage is not None
        assert converter._usage.get("input_tokens") == 10
        assert converter._usage.get("output_tokens") == 5

    def test_usage_total_tokens_computed(self):
        context = _make_context()
        _, converter = _run_simple_turn(context)
        assert converter._usage.get("total_tokens") == 15

    def test_usage_included_in_response_completed(self):
        context = _make_context()
        events, _ = _run_simple_turn(context)
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        usage = completed[0].get("response").get("usage")
        assert usage is not None
        assert usage.get("input_tokens") == 10
        assert usage.get("output_tokens") == 5

    def test_no_usage_when_assistant_usage_not_received(self):
        """If ASSISTANT_USAGE is missing (error case), response.completed still works."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = []
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
        # Skip ASSISTANT_USAGE — go straight to message
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_MESSAGE, "Hi"), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        assert completed[0].get("response").get("usage") is None


# ---------------------------------------------------------------------------
# Streaming with deltas
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamingDeltas:

    def test_delta_events_emitted(self):
        events, _ = _run_streaming_turn(_make_context(), ["Hello, ", "world!"], "Hello, world!")
        deltas = [e for e in events if isinstance(e, ResponseTextDeltaEvent)]
        assert len(deltas) == 2
        assert deltas[0].get("delta") == "Hello, "
        assert deltas[1].get("delta") == "world!"

    def test_no_synthetic_delta_when_streaming(self):
        """When deltas already arrived, ASSISTANT_MESSAGE should NOT emit another delta."""
        events, _ = _run_streaming_turn(_make_context(), ["Hi"], "Hi")
        deltas = [e for e in events if isinstance(e, ResponseTextDeltaEvent)]
        assert len(deltas) == 1

    def test_emits_output_text_done(self):
        events, _ = _run_streaming_turn(_make_context(), ["A", "B"], "AB")
        done = [e for e in events if isinstance(e, ResponseTextDoneEvent)]
        assert len(done) == 1
        assert done[0].get("text") == "AB"

    def test_emits_content_part_done(self):
        events, _ = _run_streaming_turn(_make_context(), ["Hi"], "Hi")
        assert any(isinstance(e, ResponseContentPartDoneEvent) for e in events)

    def test_emits_output_item_done(self):
        events, _ = _run_streaming_turn(_make_context(), ["Hi"], "Hi")
        assert any(isinstance(e, ResponseOutputItemDoneEvent) for e in events)

    def test_emits_response_completed(self):
        events, _ = _run_streaming_turn(_make_context(), ["Hi"], "Hi")
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1

    def test_response_completed_contains_output_item(self):
        events, _ = _run_streaming_turn(_make_context(), ["Hello, ", "world!"], "Hello, world!")
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)][0]
        output = completed.get("response").get("output")
        assert output is not None and len(output) == 1
        content = output[0].get("content", [])
        assert any(c.get("text") == "Hello, world!" for c in content)

    def test_delta_event_ordering(self):
        events, _ = _run_streaming_turn(_make_context(), ["A", "B"], "AB")
        last_delta = max(
            (i for i, e in enumerate(events) if isinstance(e, ResponseTextDeltaEvent)), default=-1)
        first_done = next(
            (i for i, e in enumerate(events) if isinstance(e, ResponseTextDoneEvent)), 9999)
        assert last_delta < first_done


# ---------------------------------------------------------------------------
# SESSION_IDLE safety net
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSessionIdleSafetyNet:

    def test_session_idle_forces_completion_when_no_message(self):
        """If SESSION_IDLE arrives without ASSISTANT_MESSAGE (error path),
        response.completed is still emitted."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = []
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.SESSION_IDLE), context))
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1

    def test_session_idle_does_not_double_complete(self):
        """SESSION_IDLE must not emit response.completed if already emitted by ASSISTANT_MESSAGE."""
        events, _ = _run_simple_turn(_make_context())
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1

    def test_session_idle_noop_without_turns(self):
        """SESSION_IDLE before any TURN_START should emit nothing."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = list(converter._convert_event(_make_event(SessionEventType.SESSION_IDLE), context))
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Full event sequence validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFullEventSequence:
    """Verify the complete event sequence matches the working reference stream."""

    def test_simple_turn_event_types(self):
        """The event type sequence for a simple turn must match the expected pattern."""
        events, _ = _run_simple_turn(_make_context(), text="Hi!")
        types = [e.get("type") or type(e).__name__ for e in events]
        expected = [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            # USAGE produces no RAPI events
            "response.output_text.delta",       # synthetic delta from MESSAGE
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
            # TURN_END + SESSION_IDLE produce nothing
        ]
        assert types == expected

    def test_streaming_turn_event_types(self):
        """Streaming turn: deltas before done events, completed at end."""
        events, _ = _run_streaming_turn(_make_context(), ["A", "B"], "AB")
        types = [e.get("type") or type(e).__name__ for e in events]
        expected = [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",       # delta "A"
            "response.output_text.delta",       # delta "B"
            # USAGE produces no RAPI events
            # MESSAGE: no synthetic delta (already have deltas)
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert types == expected
