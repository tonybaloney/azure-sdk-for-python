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
    ASSISTANT_TURN_START (1st)     response.created
                                   response.in_progress
                                   response.output_item.added  (assistant message item)
                                   response.content_part.added (output_text part)
    ASSISTANT_TURN_START (2nd+)    response.output_item.added  (new item, new item_id)
                                   response.content_part.added
    ASSISTANT_MESSAGE_DELTA        response.output_text.delta
                                   (may not fire for all models; content may arrive in
                                    one shot via ASSISTANT_MESSAGE instead)
    ASSISTANT_MESSAGE              response.output_text.delta  (if no prior deltas)
                                   response.output_text.done
                                   response.content_part.done
                                   response.output_item.done
    ASSISTANT_TURN_END             response.completed  (only if there is text content;
                                   tool-calling turns with no text are silently skipped)
    ASSISTANT_USAGE                (stored internally; included in response.completed)
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
        # Turn counter: tracks how many ASSISTANT_TURN_START events have been seen.
        # Only the first turn gets response.created / response.in_progress.
        self._turn_count: int = 0
        # Per-turn item ID — regenerated at each ASSISTANT_TURN_START so that
        # the second (actual-answer) turn in a tool-using exchange gets a unique ID.
        self._item_id: str = context.id_generator.generate_message_id()
        # Token usage received from ASSISTANT_USAGE; included in response.completed.
        self._usage: Optional[Dict[str, Any]] = None
        # Pending response.completed awaiting ASSISTANT_USAGE (set True on ASSISTANT_TURN_END
        # when there is text content; emitted when ASSISTANT_USAGE or SESSION_IDLE arrives).
        self._pending_completed: bool = False

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
                # Generate a fresh item ID for this turn so the second turn in a
                # tool-using exchange doesn't reuse the first turn's item ID.
                self._item_id = context.id_generator.generate_message_id()
                item_id = self._item_id
                # Reset per-turn state so the second turn starts clean.
                self._accumulated_text = ""
                self._completed_item = None
                is_first_turn = self._turn_count == 0
                self._turn_count += 1

                if is_first_turn:
                    # response.created — only on initial turn
                    yield ResponseCreatedEvent(
                        sequence_number=self.next_sequence(),
                        response=self._build_response("in_progress"),
                    )
                    # response.in_progress — only on initial turn
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
                # Some models stream text entirely via ASSISTANT_MESSAGE_DELTA
                # events without ever emitting a final ASSISTANT_MESSAGE.  In
                # that case _completed_item is None but _accumulated_text holds
                # the full text.  Synthesise the missing "done" events here so
                # the client always gets a well-formed sequence.
                logger.info(
                    f"ASSISTANT_TURN_END: _completed_item={'SET' if self._completed_item is not None else 'NONE'}, "
                    f"_accumulated_text={len(self._accumulated_text)} chars"
                )
                if self._completed_item is None and self._accumulated_text:
                    text = self._accumulated_text
                    self._completed_item = ResponsesAssistantMessageItemResource(
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
                        item=self._completed_item,
                    )

                if self._completed_item is not None:
                    # Defer response.completed until ASSISTANT_USAGE arrives so
                    # token counts can be embedded.  SESSION_IDLE is the fallback.
                    self._pending_completed = True
                else:
                    logger.debug("ASSISTANT_TURN_END with no text content (tool-calling turn) — skipping response.completed")

            # ── Token / model usage ───────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_USAGE, data=data) if data:
                # Store usage so it can be embedded in the final response.completed.
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
                    logger.debug(f"Usage recorded: {usage}")
                # Emit response.completed if ASSISTANT_TURN_END was already seen.
                if self._pending_completed and self._completed_item is not None:
                    self._pending_completed = False
                    yield ResponseCompletedEvent(
                        sequence_number=self.next_sequence(),
                        response=self._build_response(
                            "completed",
                            output=[self._completed_item],
                            usage=self._usage,
                        ),
                    )

            # ── Session idle: emit any deferred response.completed ─────────────
            case SessionEvent(type=SessionEventType.SESSION_IDLE):
                # Fallback: ASSISTANT_USAGE did not arrive before SESSION_IDLE.
                if self._pending_completed and self._completed_item is not None:
                    self._pending_completed = False
                    yield ResponseCompletedEvent(
                        sequence_number=self.next_sequence(),
                        response=self._build_response(
                            "completed",
                            output=[self._completed_item],
                            usage=self._usage,
                        ),
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
        self, event: SessionEvent, context: AgentRunContext,
    ) -> ResponseStreamEvent | None:
        """Convert a single event.  Returns the first RAPI event produced, or None.

        .. deprecated::
            Prefer ``_convert_event`` (a generator) which may yield multiple events.
        """
        for e in self._convert_event(event, context):
            return e
        return None
