# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for CopilotStreamingResponseConverter response event attributes.

These tests verify that every Response object embedded in stream events
has the required fields (object, id, status, created_at, agent_id, conversation)
that the portal depends on. The absence of these fields was causing portal crashes.
"""
import datetime
import uuid
from unittest.mock import MagicMock

import pytest

from copilot.generated.session_events import Data, SessionEvent, SessionEventType

from azure.ai.agentserver.core.models.projects import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
)
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext
from azure.ai.agentserver.copilot.models.copilot_response_converter import CopilotStreamingResponseConverter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(event_type: SessionEventType, content: str | None = None, **kwargs) -> SessionEvent:
    """Create a real SessionEvent dataclass instance (MagicMock won't match dataclass patterns)."""
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


def _events_for_turn(context: AgentRunContext, *, include_message: bool = True, trigger: str = "idle"):
    """Run ASSISTANT_TURN_START (and optionally a message + turn end) through the converter.

    ``trigger`` controls how response.completed is fired after ASSISTANT_TURN_END:
    - ``"usage"``  — send ASSISTANT_USAGE (with dummy token counts)
    - ``"idle"``   — send SESSION_IDLE  (default; simulates no usage event)
    """
    converter = CopilotStreamingResponseConverter(context)
    events = []
    events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
    if include_message:
        text_event = _make_event(SessionEventType.ASSISTANT_MESSAGE, "Hello!")
        events.extend(converter._convert_event(text_event, context))
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
        if trigger == "usage":
            events.extend(
                converter._convert_event(
                    _make_event(SessionEventType.ASSISTANT_USAGE, input_tokens=10, output_tokens=5),
                    context,
                )
            )
        else:  # "idle"
            events.extend(converter._convert_event(_make_event(SessionEventType.SESSION_IDLE), context))
    return events, converter


# ---------------------------------------------------------------------------
# _build_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildResponse:
    """Tests for CopilotStreamingResponseConverter._build_response()."""

    def test_includes_object_field(self):
        """Response must have object='response'."""
        conv = CopilotStreamingResponseConverter(_make_context())
        resp = conv._build_response("in_progress")
        assert resp.get("object") == "response"

    def test_includes_id(self):
        """Response must carry the response_id from context."""
        ctx = _make_context()
        conv = CopilotStreamingResponseConverter(ctx)
        resp = conv._build_response("in_progress")
        assert resp.get("id") == ctx.response_id

    def test_includes_created_at(self):
        """Response must carry a Unix-timestamp created_at."""
        conv = CopilotStreamingResponseConverter(_make_context())
        resp = conv._build_response("in_progress")
        created_at = resp.get("created_at")
        assert created_at is not None
        assert isinstance(created_at, int)
        assert created_at > 0

    def test_created_at_is_consistent(self):
        """All Response objects from one converter share the same created_at."""
        conv = CopilotStreamingResponseConverter(_make_context())
        r1 = conv._build_response("in_progress")
        r2 = conv._build_response("completed")
        assert r1.get("created_at") == r2.get("created_at")

    def test_includes_agent_id_when_present(self):
        """agent_id is included when context has an agent."""
        conv = CopilotStreamingResponseConverter(_make_context(with_agent=True))
        resp = conv._build_response("in_progress")
        agent_id = resp.get("agent_id")
        assert agent_id is not None
        assert agent_id.get("name") == "test-agent"

    def test_omits_agent_id_when_absent(self):
        """agent_id is omitted when context has no agent (no KeyError)."""
        conv = CopilotStreamingResponseConverter(_make_context(with_agent=False))
        resp = conv._build_response("in_progress")
        assert resp.get("agent_id") is None

    def test_includes_conversation_when_present(self):
        """conversation is included when context has a conversation_id."""
        ctx = _make_context(with_conversation=True)
        conv = CopilotStreamingResponseConverter(ctx)
        resp = conv._build_response("in_progress")
        # conversation_id is auto-generated by AgentRunContext if not overridden
        assert resp.get("conversation") is not None

    def test_includes_output_when_provided(self):
        """output list is serialised into the Response when supplied."""
        conv = CopilotStreamingResponseConverter(_make_context())
        fake_item = MagicMock()
        resp = conv._build_response("completed", output=[fake_item])
        assert resp.get("output") == [fake_item]

    def test_output_absent_when_not_provided(self):
        """output key is absent/None when not supplied."""
        conv = CopilotStreamingResponseConverter(_make_context())
        resp = conv._build_response("in_progress")
        assert resp.get("output") is None


# ---------------------------------------------------------------------------
# ResponseCreatedEvent / ResponseInProgressEvent at ASSISTANT_TURN_START
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTurnStartEvents:
    """Verify the first two events emitted at ASSISTANT_TURN_START."""

    def test_emits_response_created_first(self):
        """First event from ASSISTANT_TURN_START must be ResponseCreatedEvent."""
        events, _ = _events_for_turn(_make_context(), include_message=False)
        assert isinstance(events[0], ResponseCreatedEvent)

    def test_emits_response_in_progress_second(self):
        """Second event from ASSISTANT_TURN_START must be ResponseInProgressEvent."""
        events, _ = _events_for_turn(_make_context(), include_message=False)
        assert isinstance(events[1], ResponseInProgressEvent)

    def test_created_event_response_has_required_fields(self):
        """ResponseCreatedEvent.response has object, id, status, created_at."""
        events, _ = _events_for_turn(_make_context(), include_message=False)
        resp = events[0].get("response")
        assert resp.get("object") == "response"
        assert resp.get("id") is not None
        assert resp.get("status") == "in_progress"
        assert resp.get("created_at") is not None

    def test_in_progress_event_response_has_required_fields(self):
        """ResponseInProgressEvent.response has object, id, status, created_at."""
        events, _ = _events_for_turn(_make_context(), include_message=False)
        resp = events[1].get("response")
        assert resp.get("object") == "response"
        assert resp.get("id") is not None
        assert resp.get("status") == "in_progress"
        assert resp.get("created_at") is not None

    def test_sequence_numbers_are_sequential(self):
        """Sequence numbers must be strictly increasing."""
        events, _ = _events_for_turn(_make_context(), include_message=False)
        seq_numbers = [e.get("sequence_number") for e in events]
        for i in range(1, len(seq_numbers)):
            assert seq_numbers[i] == seq_numbers[i - 1] + 1


# ---------------------------------------------------------------------------
# ResponseCompletedEvent at ASSISTANT_TURN_END
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTurnEndEvents:
    """Verify ResponseCompletedEvent emitted at ASSISTANT_TURN_END."""

    def test_last_event_is_response_completed(self):
        """Final event of a full turn must be ResponseCompletedEvent."""
        events, _ = _events_for_turn(_make_context())
        assert isinstance(events[-1], ResponseCompletedEvent)

    def test_completed_event_response_has_required_fields(self):
        """ResponseCompletedEvent.response has object, id, status=completed, created_at."""
        events, _ = _events_for_turn(_make_context())
        resp = events[-1].get("response")
        assert resp.get("object") == "response"
        assert resp.get("id") is not None
        assert resp.get("status") == "completed"
        assert resp.get("created_at") is not None

    def test_completed_response_includes_output_item(self):
        """ResponseCompletedEvent.response.output contains the assistant message item."""
        events, _ = _events_for_turn(_make_context())
        resp = events[-1].get("response")
        output = resp.get("output")
        assert output is not None
        assert len(output) == 1

    def test_created_at_consistent_across_turn(self):
        """created_at is the same in ResponseCreatedEvent and ResponseCompletedEvent."""
        events, _ = _events_for_turn(_make_context())
        created_at_start = events[0].get("response").get("created_at")
        created_at_end = events[-1].get("response").get("created_at")
        assert created_at_start == created_at_end


# ---------------------------------------------------------------------------
# Multi-turn (tool-using) behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiTurn:
    """Verify correct event emission across two ASSISTANT_TURN_START cycles."""

    def _run_two_turns(self, context: AgentRunContext):
        """Simulate: turn1 (no text), tool, turn2 (with text).

        Returns the full list of events from both turns.
        """
        converter = CopilotStreamingResponseConverter(context)
        events = []
        # Turn 1: tool-calling turn — no text content
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
        # Turn 2: actual answer turn
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_MESSAGE, "Done!"), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
        # ASSISTANT_USAGE arrives after ASSISTANT_TURN_END — triggers response.completed
        events.extend(
            converter._convert_event(
                _make_event(SessionEventType.ASSISTANT_USAGE, input_tokens=10, output_tokens=5),
                context,
            )
        )
        return events, converter

    def test_response_created_emitted_only_once(self):
        """response.created must appear exactly once across a two-turn tool exchange."""
        events, _ = self._run_two_turns(_make_context())
        created = [e for e in events if isinstance(e, ResponseCreatedEvent)]
        assert len(created) == 1

    def test_response_in_progress_emitted_only_once(self):
        """response.in_progress must appear exactly once across a two-turn tool exchange."""
        events, _ = self._run_two_turns(_make_context())
        in_prog = [e for e in events if isinstance(e, ResponseInProgressEvent)]
        assert len(in_prog) == 1

    def test_second_turn_gets_different_item_id(self):
        """The output_item.added events for turn 1 and turn 2 must use different item IDs."""
        events, _ = self._run_two_turns(_make_context())
        added = [e for e in events if isinstance(e, ResponseOutputItemAddedEvent)]
        assert len(added) == 2
        id1 = added[0].get("item", {}).get("id")
        id2 = added[1].get("item", {}).get("id")
        assert id1 is not None
        assert id2 is not None
        assert id1 != id2

    def test_no_response_completed_on_first_turn_when_no_content(self):
        """Tool-calling turns (no text output) must NOT emit response.completed."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = []
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 0

    def test_response_completed_emitted_once_at_end(self):
        """response.completed must appear exactly once — only for the final content turn."""
        events, _ = self._run_two_turns(_make_context())
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1


# ---------------------------------------------------------------------------
# Token usage from ASSISTANT_USAGE
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUsage:
    """Verify ASSISTANT_USAGE is captured and embedded in response.completed."""

    def _run_turn_with_usage(self, context: AgentRunContext, input_tokens: int, output_tokens: int):
        converter = CopilotStreamingResponseConverter(context)
        events = []
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_START), context))
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_MESSAGE, "Hi"), context))
        # Real SDK order: ASSISTANT_TURN_END comes before ASSISTANT_USAGE
        events.extend(converter._convert_event(_make_event(SessionEventType.ASSISTANT_TURN_END), context))
        usage_event = _make_event(
            SessionEventType.ASSISTANT_USAGE,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        events.extend(converter._convert_event(usage_event, context))
        return events, converter

    def test_usage_stored_after_assistant_usage_event(self):
        """After ASSISTANT_USAGE, converter._usage is populated."""
        context = _make_context()
        _, converter = self._run_turn_with_usage(context, input_tokens=10, output_tokens=5)
        assert converter._usage is not None
        assert converter._usage.get("input_tokens") == 10
        assert converter._usage.get("output_tokens") == 5

    def test_usage_total_tokens_computed(self):
        """total_tokens is the sum of input + output."""
        context = _make_context()
        _, converter = self._run_turn_with_usage(context, input_tokens=10, output_tokens=5)
        assert converter._usage.get("total_tokens") == 15

    def test_usage_included_in_response_completed(self):
        """response.completed.response must contain usage when ASSISTANT_USAGE was received."""
        context = _make_context()
        events, _ = self._run_turn_with_usage(context, input_tokens=10, output_tokens=5)
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        usage = completed[0].get("response").get("usage")
        assert usage is not None
        assert usage.get("input_tokens") == 10
        assert usage.get("output_tokens") == 5

    def test_no_usage_when_assistant_usage_not_received(self):
        """If no ASSISTANT_USAGE event arrives, response.completed has no usage field."""
        events, _ = _events_for_turn(_make_context())
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        # usage key should be absent / None
        assert completed[0].get("response").get("usage") is None
