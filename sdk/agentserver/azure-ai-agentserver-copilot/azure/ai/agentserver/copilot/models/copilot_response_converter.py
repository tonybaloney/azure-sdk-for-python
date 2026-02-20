# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import datetime
import json
from typing import Any, Dict, Generator, List, Optional

from copilot.generated.session_events import SessionEvent, SessionEventType

from azure.ai.agentserver.core.models import Response as OpenAIResponse
from azure.ai.agentserver.core.models.projects import (
    ItemContentOutputText,
    MCPApprovalRequestItemResource,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponsesAssistantMessageItemResource,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext


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


class CopilotResponseStreamingConverter:
    """Converts Copilot SDK session events into OpenAI-format responses."""

    def __init__(self, context: AgentRunContext):
        self.context = context
        # sequence numbers must start at 0 for first emitted event. Counter is incremented in next_sequence() before being assigned to event, so it starts at -1 here.
        self._sequence = -1
        self._response_id = None
        self._response_created_at = None
        self._next_output_index = 0

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def as_response_stream_event(self, event: SessionEvent, context: AgentRunContext) -> ResponseStreamEvent | None:
        """Convert a single SessionEvent into a ResponseStreamEvent, if possible."""
        match event:
            case SessionEvent(type=SessionEventType.SESSION_START, data=data):
                return ResponseCreatedEvent(
                    sequence_number=self.next_sequence(),
                    response=OpenAIResponse(
                        id=context.response_id, 
                        status="in_progress",
                        ),
                )
            case SessionEvent(type=SessionEventType.SESSION_SHUTDOWN, data=data):
                return ResponseCompletedEvent(
                    sequence_number=self.next_sequence(),
                    response=OpenAIResponse(
                        id=context.response_id, 
                        status="completed",
                        ),
                )
            case SessionEvent(type=SessionEventType.ASSISTANT_MESSAGE_DELTA, data=data) if data and hasattr(data, "content"):
                return ResponseTextDeltaEvent(
                    sequence_number=self.next_sequence(),
                    delta=data.content,
                )
            case SessionEvent(type=SessionEventType.ASSISTANT_MESSAGE, data=data) if data and hasattr(data, "content"):
                return ResponseTextDoneEvent(
                    sequence_number=self.next_sequence(),
                    text=data.content,
                )
            
        return None

    def to_stream_events(
            self, events: list[SessionEvent], context: AgentRunContext,
    ) -> Generator[ResponseStreamEvent, None, None]:
        """Convert a batch of Copilot SessionEvents into streaming response events."""
        item_id = context.id_generator.generate_message_id()

        for event in events:
            stream_event = self.as_response_stream_event(event, context)
            if stream_event is not None:
                yield stream_event
