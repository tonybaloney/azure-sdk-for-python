# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for ASSISTANT_REASONING → RAPI reasoning summary event conversion.

Validates that the CopilotStreamingResponseConverter correctly maps Copilot
SDK reasoning events (ASSISTANT_REASONING, ASSISTANT_REASONING_DELTA) to the
RAPI reasoning summary streaming events:
    - response.output_item.added  (type: reasoning)
    - response.reasoning_summary_part.added
    - response.reasoning_summary_text.delta
    - response.reasoning_summary_text.done
    - response.reasoning_summary_part.done
    - response.output_item.done   (type: reasoning)
"""
import datetime
import uuid
from typing import Any

import pytest
from copilot.generated.session_events import Data, SessionEvent, SessionEventType

from azure.ai.agentserver.core.models.projects import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext
from azure.ai.agentserver.copilot.models.copilot_response_converter import (
    CopilotStreamingResponseConverter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(event_type: SessionEventType, **kwargs: Any) -> SessionEvent:
    data = Data(**kwargs)
    return SessionEvent(
        data=data,
        id=uuid.uuid4(),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        type=event_type,
    )


def _make_context() -> AgentRunContext:
    payload = {
        "input": "hello",
        "stream": True,
        "agent": {"type": "agent_id", "name": "test-agent", "version": "1"},
        "conversation": {"id": "conv_test"},
    }
    return AgentRunContext(payload)


def _run_reasoning_then_message(
    context: AgentRunContext,
    *,
    reasoning_deltas: list[str] | None = None,
    reasoning_full: str = "I should help the user.",
    message_deltas: list[str] | None = None,
    message_full: str = "Hello!",
):
    """Simulate: TURN_START → REASONING_DELTA×N → REASONING → MESSAGE_DELTA×N → USAGE → MESSAGE → TURN_END → IDLE"""
    converter = CopilotStreamingResponseConverter(context)
    events = []

    events.extend(converter._convert_event(
        _make_event(SessionEventType.ASSISTANT_TURN_START), context
    ))

    if reasoning_deltas:
        for chunk in reasoning_deltas:
            events.extend(converter._convert_event(
                _make_event(SessionEventType.ASSISTANT_REASONING_DELTA, delta_content=chunk), context
            ))

    events.extend(converter._convert_event(
        _make_event(SessionEventType.ASSISTANT_REASONING, content=reasoning_full), context
    ))

    if message_deltas:
        for chunk in message_deltas:
            events.extend(converter._convert_event(
                _make_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, content=chunk), context
            ))

    events.extend(converter._convert_event(
        _make_event(SessionEventType.ASSISTANT_USAGE, input_tokens=100, output_tokens=50), context
    ))
    events.extend(converter._convert_event(
        _make_event(SessionEventType.ASSISTANT_MESSAGE, content=message_full), context
    ))
    events.extend(converter._convert_event(
        _make_event(SessionEventType.ASSISTANT_TURN_END), context
    ))
    events.extend(converter._convert_event(
        _make_event(SessionEventType.SESSION_IDLE), context
    ))

    return events, converter


# ===========================================================================
# ASSISTANT_REASONING_DELTA → reasoning summary text deltas
# ===========================================================================


@pytest.mark.unit
class TestReasoningDelta:
    """ASSISTANT_REASONING_DELTA produces RAPI reasoning summary events."""

    def test_first_delta_emits_added_events(self):
        """First reasoning delta should emit output_item.added + summary_part.added + text.delta."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING_DELTA, delta_content="Let me think"),
            context,
        ))
        assert len(events) == 3
        assert isinstance(events[0], ResponseOutputItemAddedEvent)
        assert events[0]["item"]["type"] == "reasoning"
        assert isinstance(events[1], ResponseReasoningSummaryPartAddedEvent)
        assert isinstance(events[2], ResponseReasoningSummaryTextDeltaEvent)
        assert events[2]["delta"] == "Let me think"

    def test_subsequent_deltas_only_emit_text_delta(self):
        """Second and later deltas should only produce text.delta events."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        # First delta triggers added events
        list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING_DELTA, delta_content="A"), context
        ))
        # Second delta — no added events
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING_DELTA, delta_content="B"), context
        ))
        assert len(events) == 1
        assert isinstance(events[0], ResponseReasoningSummaryTextDeltaEvent)
        assert events[0]["delta"] == "B"

    def test_delta_item_id_is_consistent(self):
        """All deltas should reference the same reasoning item_id."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        all_events = []
        for chunk in ["X", "Y", "Z"]:
            all_events.extend(converter._convert_event(
                _make_event(SessionEventType.ASSISTANT_REASONING_DELTA, delta_content=chunk), context
            ))
        deltas = [e for e in all_events if isinstance(e, ResponseReasoningSummaryTextDeltaEvent)]
        item_ids = {e["item_id"] for e in deltas}
        assert len(item_ids) == 1

    def test_delta_output_index_is_zero(self):
        """Reasoning should claim output_index=0 (first item in turn)."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING_DELTA, delta_content="A"), context
        ))
        added = events[0]
        assert added["output_index"] == 0

    def test_summary_index_is_zero(self):
        """summary_index should be 0 for the single summary part."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING_DELTA, delta_content="A"), context
        ))
        part_added = [e for e in events if isinstance(e, ResponseReasoningSummaryPartAddedEvent)]
        assert len(part_added) == 1
        assert part_added[0]["summary_index"] == 0


# ===========================================================================
# ASSISTANT_REASONING → done events
# ===========================================================================


@pytest.mark.unit
class TestReasoningDone:
    """ASSISTANT_REASONING emits done events and finalises the reasoning item."""

    def test_reasoning_without_deltas_emits_full_lifecycle(self):
        """When no deltas preceded, REASONING emits added + delta + done events."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING, content="I need to greet them."),
            context,
        ))
        types = [e.get("type") for e in events]
        assert types == [
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
        ]

    def test_reasoning_with_deltas_emits_only_done_events(self):
        """When deltas preceded, REASONING only emits done events (no added/delta)."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        # Simulate prior deltas
        list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING_DELTA, delta_content="I need to "), context
        ))
        list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING_DELTA, delta_content="greet them."), context
        ))
        # REASONING (done)
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING, content="I need to greet them."),
            context,
        ))
        types = [e.get("type") for e in events]
        assert types == [
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
        ]

    def test_reasoning_done_text_matches_full_text(self):
        """The text in the done event should be the authoritative full reasoning text."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING, content="Full reasoning"),
            context,
        ))
        text_done = [e for e in events if isinstance(e, ResponseReasoningSummaryTextDoneEvent)]
        assert len(text_done) == 1
        assert text_done[0]["text"] == "Full reasoning"

    def test_reasoning_done_part_has_summary_text(self):
        """The summary part done event should contain the full text."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING, content="Thinking..."),
            context,
        ))
        part_done = [e for e in events if isinstance(e, ResponseReasoningSummaryPartDoneEvent)]
        assert len(part_done) == 1
        assert part_done[0]["part"]["text"] == "Thinking..."
        assert part_done[0]["part"]["type"] == "summary_text"

    def test_reasoning_output_item_done_has_summary(self):
        """output_item.done for reasoning should include the summary array."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING, content="Deep thought"),
            context,
        ))
        item_done = [e for e in events if isinstance(e, ResponseOutputItemDoneEvent)]
        assert len(item_done) == 1
        item = item_done[0]["item"]
        assert item["type"] == "reasoning"
        assert len(item["summary"]) == 1
        assert item["summary"][0]["text"] == "Deep thought"


# ===========================================================================
# Full turn: reasoning + message interleaved
# ===========================================================================


@pytest.mark.unit
class TestReasoningWithMessage:
    """Reasoning events should interleave correctly with message events in a full turn."""

    def test_event_type_ordering_with_streaming_deltas(self):
        """Full turn with streaming reasoning and message deltas."""
        events, _ = _run_reasoning_then_message(
            _make_context(),
            reasoning_deltas=["I ", "should ", "help."],
            reasoning_full="I should help.",
            message_deltas=["He", "llo!"],
            message_full="Hello!",
        )
        types = [e.get("type") for e in events]
        expected = [
            # TURN_START — created + in_progress
            "response.created",
            "response.in_progress",
            # REASONING_DELTA "I " — first delta: added + part_added + text.delta
            "response.output_item.added",       # reasoning item
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            # REASONING_DELTA " should " — text.delta only
            "response.reasoning_summary_text.delta",
            # REASONING_DELTA " help." — text.delta only
            "response.reasoning_summary_text.delta",
            # REASONING — done events
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",        # reasoning done
            # MESSAGE_DELTA "He" — deferred message added + delta
            "response.output_item.added",       # message item
            "response.content_part.added",
            "response.output_text.delta",
            # MESSAGE_DELTA "llo!" — delta only
            "response.output_text.delta",
            # ASSISTANT_MESSAGE — no synthetic delta (already have deltas)
            # TURN_END — deferred done-events
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",        # message done
            "response.completed",
        ]
        assert types == expected

    def test_event_type_ordering_without_streaming(self):
        """Full turn: no streaming deltas for either reasoning or message."""
        events, _ = _run_reasoning_then_message(
            _make_context(),
            reasoning_deltas=None,
            reasoning_full="Thinking...",
            message_deltas=None,
            message_full="Hi!",
        )
        types = [e.get("type") for e in events]
        expected = [
            # TURN_START
            "response.created",
            "response.in_progress",
            # REASONING (no prior deltas) — full lifecycle
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
            # MESSAGE (no prior deltas) — deferred message added + synthetic delta
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",       # synthetic delta from MESSAGE
            # TURN_END
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert types == expected

    def test_reasoning_output_index_before_message(self):
        """Reasoning item gets output_index=0, message gets output_index=1."""
        events, _ = _run_reasoning_then_message(
            _make_context(),
            reasoning_full="Think",
            message_full="Hello",
        )
        added = [e for e in events if isinstance(e, ResponseOutputItemAddedEvent)]
        assert len(added) == 2
        # Reasoning first (index 0), message second (index 1)
        assert added[0]["item"]["type"] == "reasoning"
        assert added[0]["output_index"] == 0
        assert added[1]["item"]["type"] == "message"
        assert added[1]["output_index"] == 1

    def test_message_deltas_use_correct_output_index(self):
        """Message text deltas should use the message's output_index, not 0."""
        events, _ = _run_reasoning_then_message(
            _make_context(),
            reasoning_deltas=["think"],
            reasoning_full="think",
            message_deltas=["Hi"],
            message_full="Hi",
        )
        text_deltas = [e for e in events if isinstance(e, ResponseTextDeltaEvent)]
        assert len(text_deltas) == 1
        assert text_deltas[0]["output_index"] == 1  # After reasoning at index 0

    def test_completed_response_includes_both_items(self):
        """response.completed output array should include reasoning + message."""
        events, _ = _run_reasoning_then_message(
            _make_context(),
            reasoning_full="I should say hi.",
            message_full="Hi!",
        )
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        output = completed[0]["response"]["output"]
        assert len(output) == 2
        assert output[0]["type"] == "reasoning"
        assert output[0]["summary"][0]["text"] == "I should say hi."
        assert output[1]["type"] == "message"

    def test_completed_response_text_content(self):
        """The message text in the completed response should be correct."""
        events, _ = _run_reasoning_then_message(
            _make_context(),
            reasoning_full="Hmm...",
            message_full="The answer is 42.",
        )
        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        msg_item = completed[0]["response"]["output"][1]
        assert msg_item["content"][0]["text"] == "The answer is 42."


# ===========================================================================
# No reasoning: backward compatibility
# ===========================================================================


@pytest.mark.unit
class TestNoReasoning:
    """Turns without reasoning should work exactly as before."""

    def test_message_gets_output_index_zero_without_reasoning(self):
        """Without reasoning, message claims output_index=0 as before."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = []
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_TURN_START), context
        ))
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, content="Hi"), context
        ))
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_MESSAGE, content="Hi"), context
        ))
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_TURN_END), context
        ))
        events.extend(converter._convert_event(
            _make_event(SessionEventType.SESSION_IDLE), context
        ))

        added = [e for e in events if isinstance(e, ResponseOutputItemAddedEvent)]
        assert len(added) == 1
        assert added[0]["output_index"] == 0
        assert added[0]["item"]["type"] == "message"

    def test_completed_output_has_only_message_without_reasoning(self):
        """Without reasoning, completed output has only the message item."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = []
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_TURN_START), context
        ))
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_MESSAGE, content="Hello!"), context
        ))
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_TURN_END), context
        ))
        events.extend(converter._convert_event(
            _make_event(SessionEventType.SESSION_IDLE), context
        ))

        completed = [e for e in events if isinstance(e, ResponseCompletedEvent)]
        assert len(completed) == 1
        output = completed[0]["response"]["output"]
        assert len(output) == 1
        assert output[0]["type"] == "message"

    def test_sequence_numbers_are_sequential(self):
        """All events should have monotonically increasing sequence numbers."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = []
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_TURN_START), context
        ))
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_MESSAGE, content="OK"), context
        ))
        events.extend(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_TURN_END), context
        ))
        seq_nums = [e["sequence_number"] for e in events]
        for i in range(1, len(seq_nums)):
            assert seq_nums[i] == seq_nums[i - 1] + 1


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.unit
class TestReasoningEdgeCases:
    """Edge cases for reasoning event handling."""

    def test_reasoning_without_content_is_ignored(self):
        """ASSISTANT_REASONING with no data.content should produce no events."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        # Event with empty content
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING), context
        ))
        assert len(events) == 0

    def test_reasoning_delta_without_delta_content_is_ignored(self):
        """ASSISTANT_REASONING_DELTA with no delta_content should produce no events."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING_DELTA), context
        ))
        assert len(events) == 0

    def test_reasoning_reset_between_turns(self):
        """Reasoning state should reset between turns (multi-turn scenario)."""
        context = _make_context()
        converter = CopilotStreamingResponseConverter(context)

        # Turn 1: tool-only (no reasoning or message content)
        list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_TURN_START), context
        ))
        list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_TURN_END), context
        ))

        # Turn 2: with reasoning
        list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_TURN_START), context
        ))
        events = list(converter._convert_event(
            _make_event(SessionEventType.ASSISTANT_REASONING_DELTA, delta_content="thinking..."), context
        ))
        # Should emit added + part_added + delta (fresh reasoning state)
        assert len(events) == 3
        assert isinstance(events[0], ResponseOutputItemAddedEvent)
