# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import datetime
import json
import time
from typing import Any, Dict, Generator, List, Optional

from copilot.generated.session_events import SessionEvent, SessionEventType

from azure.ai.agentserver.core.models import Response as OpenAIResponse
from azure.ai.agentserver.core.models.projects import (
    ItemContentOutputText,
    MCPApprovalRequestItemResource,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseIncompleteEvent,
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
    """Converts Copilot SDK session events into OpenAI-format streaming response events.

    Event mapping (based on observed GitHub Copilot SDK event flow):

    Copilot event                  RAPI events emitted
    ─────────────────────────────  ──────────────────────────────────────────────────────
    ASSISTANT_TURN_START           response.created
                                   response.output_item.added  (assistant message item)
                                   response.content_part.added (output_text part)
    ASSISTANT_MESSAGE_DELTA        response.output_text.delta
                                   (may not fire for all models; content may arrive in
                                    one shot via ASSISTANT_MESSAGE instead)
    ASSISTANT_MESSAGE              response.output_text.delta  (if no prior deltas)
                                   response.output_text.done
                                   response.content_part.done
                                   response.output_item.done
    ASSISTANT_TURN_END             response.completed
    ASSISTANT_REASONING            (logged only; no RAPI event — not in spec yet)
    everything else                (ignored)
    """

    def __init__(self, context: AgentRunContext):
        self.context = context
        # Sequence numbers start at -1; next_sequence() pre-increments before use.
        self._sequence = -1
        # Capture creation timestamp once so all Response objects share it.
        self._created_at: int = int(time.time())
        # Text accumulated from deltas so we can detect when ASSISTANT_MESSAGE
        # arrives without prior deltas and synthesise a single delta for it.
        self._accumulated_text: str = ""
        # The completed assistant message item built during ASSISTANT_MESSAGE;
        # stored so ASSISTANT_TURN_END can embed it in ResponseCompletedEvent.
        self._completed_item: Optional[ResponsesAssistantMessageItemResource] = None

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _build_response(self, status: str, output: Optional[list] = None) -> OpenAIResponse:
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
        return OpenAIResponse(response_data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_stream_events(
        self, events: list[SessionEvent], context: AgentRunContext,
    ) -> Generator[ResponseStreamEvent, None, None]:
        """Convert a collected batch of Copilot SessionEvents into RAPI stream events."""
        item_id = context.id_generator.generate_message_id()
        for event in events:
            yield from self._convert_event(event, context, item_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convert_event(
        self, event: SessionEvent, context: AgentRunContext, item_id: str,
    ) -> Generator[ResponseStreamEvent, None, None]:
        """Yield zero or more RAPI ResponseStreamEvents for a single Copilot session event."""

        match event:

            # ── Turn start ────────────────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_TURN_START):
                # response.created
                yield ResponseCreatedEvent(
                    sequence_number=self.next_sequence(),
                    response=self._build_response("in_progress"),
                )
                # response.in_progress
                yield ResponseInProgressEvent(
                    sequence_number=self.next_sequence(),
                    response=self._build_response("in_progress"),
                )
                # response.output_item.added  (empty assistant item, will be filled later)
                yield ResponseOutputItemAddedEvent(
                    sequence_number=self.next_sequence(),
                    output_index=0,
                    item=ResponsesAssistantMessageItemResource(
                        id=item_id,
                        status="in_progress",
                        content=[],
                    ),
                )
                # response.content_part.added  (empty text part)
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

            # ── Full assistant message ────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_MESSAGE, data=data) if data and data.content:
                text = data.content

                # If no deltas arrived (e.g. Claude returns all at once), emit
                # a synthetic delta now so clients get at least one delta event.
                if not self._accumulated_text:
                    self._accumulated_text = text
                    yield ResponseTextDeltaEvent(
                        sequence_number=self.next_sequence(),
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        delta=text,
                    )

                # response.output_text.done
                yield ResponseTextDoneEvent(
                    sequence_number=self.next_sequence(),
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    text=text,
                )

                # Build the completed item now so ASSISTANT_TURN_END can embed it.
                self._completed_item = ResponsesAssistantMessageItemResource(
                    id=item_id,
                    status="completed",
                    content=[ItemContentOutputText(text=text, annotations=[])],
                )

                # response.content_part.done
                yield ResponseContentPartDoneEvent(
                    sequence_number=self.next_sequence(),
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    part=ItemContentOutputText(text=text, annotations=[]),
                )

                # response.output_item.done
                yield ResponseOutputItemDoneEvent(
                    sequence_number=self.next_sequence(),
                    output_index=0,
                    item=self._completed_item,
                )

            # ── Turn end ──────────────────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_TURN_END):
                output = [self._completed_item] if self._completed_item else []
                yield ResponseCompletedEvent(
                    sequence_number=self.next_sequence(),
                    response=self._build_response("completed", output=output),
                )

            # ── Reasoning (extended thinking) ─────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_REASONING, data=data):
                # Not yet part of the standard RAPI spec; log and skip.
                if data and data.content:
                    logger.debug(f"Copilot reasoning (skipped): {data.content[:120]!r}")

            # ── All other events ──────────────────────────────────────────────
            case _:
                ename = event.type.name if event.type else "UNKNOWN"
                logger.debug(f"Unhandled Copilot event (skipped): {ename}")

    # ------------------------------------------------------------------
    # Legacy shim kept for any existing call sites
    # ------------------------------------------------------------------

    def as_response_stream_event(
        self, event: SessionEvent, context: AgentRunContext, item_id: str,
    ) -> ResponseStreamEvent | None:
        """Convert a single event.  Returns the first RAPI event produced, or None.

        .. deprecated::
            Prefer ``_convert_event`` (a generator) which may yield multiple events.
        """
        for e in self._convert_event(event, context, item_id):
            return e
        return None
