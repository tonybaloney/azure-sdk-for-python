# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import datetime
import time
from typing import Any, Dict, Generator, Optional

from copilot.generated.session_events import SessionEvent, SessionEventType

from azure.ai.agentserver.core.models import Response as OpenAIResponse
from azure.ai.agentserver.core.models.projects import (
    ItemContentOutputText,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponsesAssistantMessageItemResource,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext

import logging

logger = logging.getLogger(__name__)


class CopilotResponseConverter:
    @staticmethod
    def to_response(text: str, context: AgentRunContext) -> OpenAIResponse:
        """Build a non-streaming OpenAI Response from the final assistant text.

        If *text* is empty, a fallback message is used so the response is
        never blank.
        """
        item_id = context.id_generator.generate_message_id()
        if not text.strip():
            text = "(No response text was produced by the agent.)"
        return OpenAIResponse(
            id=context.response_id,
            created_at=datetime.datetime.now(),
            output=[
                ResponsesAssistantMessageItemResource(
                    id=item_id,
                    status="completed",
                    content=[
                        ItemContentOutputText(text=text, annotations=[]),
                    ],
                )
            ],
        )


class CopilotStreamingResponseConverter:
    """Converts Copilot SDK session events into RAPI streaming response events.

    The runtime emits events in a strict, guaranteed order per turn:

        ASSISTANT_TURN_START
        ASSISTANT_MESSAGE_DELTA ×N   (streaming text chunks)
        ASSISTANT_USAGE              (token counts — arrives BEFORE message)
        ASSISTANT_MESSAGE            (authoritative full text — always emitted)
        ASSISTANT_TURN_END           (always emitted, even on error)
        SESSION_IDLE                 (session finished processing)

    In multi-turn (tool-calling) flows the turn sequence repeats.  Only the
    final turn that produces text content gets ``response.completed``.

    The converter is intentionally simple: each Copilot event maps to a fixed
    set of RAPI events with no deferred state or pending flags.
    """

    def __init__(self, context: AgentRunContext):
        self.context = context
        self._sequence = -1
        self._created_at: int = int(time.time())
        self._accumulated_text: str = ""
        self._turn_count: int = 0
        self._item_id: str = context.id_generator.generate_message_id()
        self._usage: Optional[Dict[str, Any]] = None
        self._completed: bool = False

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _build_response(
        self,
        status: str,
        output: Optional[list] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> OpenAIResponse:
        """Build a Response dict with all required fields, matching the langgraph/agentframework pattern."""
        response_data: Dict[str, Any] = {
            "object": "response",
            "id": self.context.response_id,
            "status": status,
            "created_at": self._created_at,
        }
        agent_id = self.context.get_agent_id_object()
        if agent_id is not None:
            response_data["agent_id"] = agent_id
        conversation = self.context.get_conversation_object()
        if conversation is not None:
            response_data["conversation"] = conversation
        if output is not None:
            response_data["output"] = output
        if usage is not None:
            response_data["usage"] = usage
        return OpenAIResponse(response_data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_stream_events(
        self, events: list[SessionEvent], context: AgentRunContext,
    ) -> Generator[ResponseStreamEvent, None, None]:
        """Convert a collected batch of Copilot SessionEvents into RAPI stream events."""
        for event in events:
            yield from self._convert_event(event, context)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convert_event(
        self, event: SessionEvent, context: AgentRunContext,
    ) -> Generator[ResponseStreamEvent, None, None]:
        """Yield zero or more RAPI ResponseStreamEvents for a single Copilot session event."""
        item_id = self._item_id

        match event:

            # ── Turn start ────────────────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_TURN_START):
                self._item_id = context.id_generator.generate_message_id()
                item_id = self._item_id
                self._accumulated_text = ""
                is_first_turn = self._turn_count == 0
                self._turn_count += 1

                if is_first_turn:
                    yield ResponseCreatedEvent(
                        sequence_number=self.next_sequence(),
                        response=self._build_response("in_progress"),
                    )
                    yield ResponseInProgressEvent(
                        sequence_number=self.next_sequence(),
                        response=self._build_response("in_progress"),
                    )
                yield ResponseOutputItemAddedEvent(
                    sequence_number=self.next_sequence(),
                    output_index=0,
                    item=ResponsesAssistantMessageItemResource(
                        id=item_id,
                        status="in_progress",
                        content=[],
                    ),
                )
                yield ResponseContentPartAddedEvent(
                    sequence_number=self.next_sequence(),
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    part=ItemContentOutputText(text="", annotations=[]),
                )

            # ── Streaming text delta ──────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_MESSAGE_DELTA, data=data) if data and data.content:
                self._accumulated_text += data.content
                yield ResponseTextDeltaEvent(
                    sequence_number=self.next_sequence(),
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    delta=data.content,
                )

            # ── Token / model usage (arrives BEFORE ASSISTANT_MESSAGE) ────────
            case SessionEvent(type=SessionEventType.ASSISTANT_USAGE, data=data) if data:
                usage: Dict[str, Any] = {}
                if data.input_tokens is not None:
                    usage["input_tokens"] = int(data.input_tokens)
                if data.output_tokens is not None:
                    usage["output_tokens"] = int(data.output_tokens)
                total = (int(data.input_tokens) if data.input_tokens is not None else 0) + \
                        (int(data.output_tokens) if data.output_tokens is not None else 0)
                if total:
                    usage["total_tokens"] = total
                if usage:
                    self._usage = usage

            # ── Full assistant message (authoritative, always emitted) ────────
            case SessionEvent(type=SessionEventType.ASSISTANT_MESSAGE, data=data) if data and data.content:
                text = data.content

                if not self._accumulated_text:
                    self._accumulated_text = text
                    yield ResponseTextDeltaEvent(
                        sequence_number=self.next_sequence(),
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        delta=text,
                    )

                completed_item = ResponsesAssistantMessageItemResource(
                    id=item_id,
                    status="completed",
                    content=[ItemContentOutputText(text=text, annotations=[])],
                )

                yield ResponseTextDoneEvent(
                    sequence_number=self.next_sequence(),
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    text=text,
                )
                yield ResponseContentPartDoneEvent(
                    sequence_number=self.next_sequence(),
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    part=ItemContentOutputText(text=text, annotations=[]),
                )
                yield ResponseOutputItemDoneEvent(
                    sequence_number=self.next_sequence(),
                    output_index=0,
                    item=completed_item,
                )
                yield ResponseCompletedEvent(
                    sequence_number=self.next_sequence(),
                    response=self._build_response(
                        "completed",
                        output=[completed_item],
                        usage=self._usage,
                    ),
                )
                self._completed = True

            # ── Turn end (no-op; completion already emitted from ASSISTANT_MESSAGE)
            case SessionEvent(type=SessionEventType.ASSISTANT_TURN_END):
                pass

            # ── Session idle (safety net) ─────────────────────────────────────
            case SessionEvent(type=SessionEventType.SESSION_IDLE):
                if not self._completed and self._turn_count > 0:
                    logger.warning("SESSION_IDLE without response.completed — forcing completion")
                    output: list = []
                    if self._accumulated_text:
                        completed_item = ResponsesAssistantMessageItemResource(
                            id=item_id,
                            status="completed",
                            content=[ItemContentOutputText(text=self._accumulated_text, annotations=[])],
                        )
                        output = [completed_item]
                    yield ResponseCompletedEvent(
                        sequence_number=self.next_sequence(),
                        response=self._build_response("completed", output=output, usage=self._usage),
                    )
                    self._completed = True

            # ── Reasoning ─────────────────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_REASONING, data=data):
                if data and data.content:
                    logger.debug(f"Copilot reasoning: {data.content[:120]!r}")

            # ── All other events ──────────────────────────────────────────────
            case _:
                ename = event.type.name if event.type else "UNKNOWN"
                logger.debug(f"Unhandled Copilot event: {ename}")

    # ------------------------------------------------------------------
    # Legacy shim kept for any existing call sites
    # ------------------------------------------------------------------

    def as_response_stream_event(
        self, event: SessionEvent, context: AgentRunContext,
    ) -> ResponseStreamEvent | None:
        """Convert a single event.  Returns the first RAPI event produced, or None.

        .. deprecated::
            Prefer ``_convert_event`` (a generator) which may yield multiple events.
        """
        for e in self._convert_event(event, context):
            return e
        return None
