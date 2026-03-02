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
    ResponseError,
    ResponseFailedEvent,
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

    To avoid a burst-then-close race at the Container App ingress layer,
    done-events (text_done → content_part.done → output_item.done →
    response.completed) are **deferred** from ``ASSISTANT_MESSAGE`` to
    ``ASSISTANT_TURN_END``.  This ensures they are yielded in a separate
    async iteration of the event loop, giving uvicorn time to flush the
    earlier chunks before the connection closes.
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
        self._failed: bool = False
        self._session_error: Optional[str] = None
        self._pending_message_text: Optional[str] = None

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _build_response(
        self,
        status: str,
        output: Optional[list] = None,
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[ResponseError] = None,
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
        if error is not None:
            response_data["error"] = error
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
            # Only emit a synthetic delta here if no streaming deltas arrived.
            # Done-events are deferred to ASSISTANT_TURN_END so they cross an
            # async boundary, preventing a burst-then-close race at the proxy.
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

                # Store text — done-events emitted by ASSISTANT_TURN_END.
                self._pending_message_text = text

            # ── Session error ───────────────────────────────────────────
            case SessionEvent(type=SessionEventType.SESSION_ERROR, data=data):
                error_msg = ""
                if data:
                    error_msg = getattr(data, 'message', None) or getattr(data, 'content', None) or repr(data)
                self._session_error = error_msg
                logger.error("Copilot session error: %s", error_msg)

                if not self._completed and not self._failed:
                    error_obj = ResponseError(code="server_error", message=error_msg)
                    yield ResponseFailedEvent(
                        sequence_number=self.next_sequence(),
                        response=self._build_response("failed", error=error_obj),
                    )
                    self._failed = True

            # ── Turn end — emit deferred done-events ─────────────────────────
            # Done-events are deferred here (instead of ASSISTANT_MESSAGE) so
            # they are yielded in a separate async iteration, giving the proxy
            # layer time to flush earlier SSE chunks.
            case SessionEvent(type=SessionEventType.ASSISTANT_TURN_END):
                if self._pending_message_text is not None:
                    text = self._pending_message_text
                    self._pending_message_text = None

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

            # ── Session idle (safety net) ─────────────────────────────────────
            case SessionEvent(type=SessionEventType.SESSION_IDLE):
                if not self._completed and not self._failed and self._turn_count > 0:
                    logger.warning("SESSION_IDLE without response.completed — forcing completion")
                    # Use accumulated text, or error message, or fallback.
                    text = self._accumulated_text
                    if not text.strip():
                        if self._session_error:
                            text = f"(Agent error: {self._session_error})"
                        else:
                            text = "(No response text was produced by the agent.)"
                    completed_item = ResponsesAssistantMessageItemResource(
                        id=item_id,
                        status="completed",
                        content=[ItemContentOutputText(text=text, annotations=[])],
                    )
                    output = [completed_item]
                    # Emit the full done-event chain so clients see well-formed output.
                    yield ResponseTextDeltaEvent(
                        sequence_number=self.next_sequence(),
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        delta=text,
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
                        response=self._build_response("completed", output=output, usage=self._usage),
                    )
                    self._completed = True

            # ── Session warning ────────────────────────────────────────────────
            case SessionEvent(type=SessionEventType.SESSION_WARNING, data=data):
                warning_type = getattr(data, 'warning_type', None) or "unknown" if data else "unknown"
                warning_msg = getattr(data, 'message', None) or "" if data else ""
                logger.warning("Copilot session warning: type=%s message=%s", warning_type, warning_msg)

            # ── Reasoning ─────────────────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_REASONING, data=data):
                if data and data.content:
                    logger.debug("Copilot reasoning: %r", data.content[:120])

            # ── Reasoning delta (streaming chunks) ────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_REASONING_DELTA, data=data):
                if data and data.delta_content:
                    logger.debug("Copilot reasoning delta: %r", data.delta_content[:120])

            # ── Intent classification ─────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_INTENT, data=data):
                if data and data.intent:
                    logger.debug("Copilot intent: %s", data.intent)

            # ── All other events ──────────────────────────────────────────────
            case _:
                ename = event.type.name if event.type else "UNKNOWN"
                logger.debug("Unhandled Copilot event: %s", ename)
