# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import datetime
from typing import Generator, Optional

from copilot.generated.session_events import SessionEvent, SessionEventType

from azure.ai.agentserver.core.models import Response as OpenAIResponse
from azure.ai.agentserver.core.models.projects import (
    ItemContentOutputText,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseOutputItemAddedEvent,
    ResponsesAssistantMessageItemResource,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext


class CopilotResponseConverter:
    """Converts Copilot SDK session events into OpenAI-format responses."""

    @staticmethod
    def to_response(text: str, context: AgentRunContext) -> OpenAIResponse:
        """Build a non-streaming OpenAI Response from the final assistant text.

        :param text: The assembled assistant text.
        :type text: str
        :param context: The agent run context.
        :type context: AgentRunContext
        :return: An OpenAI-compatible Response.
        :rtype: OpenAIResponse
        """
        item_id = context.id_generator.generate_message_id()
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

    @staticmethod
    def to_stream_events(
        events: list[SessionEvent], context: AgentRunContext
    ) -> Generator[ResponseStreamEvent, None, None]:
        """Convert a batch of Copilot SessionEvents into streaming response events.

        :param events: Copilot session events collected during the run.
        :type events: list[SessionEvent]
        :param context: The agent run context.
        :type context: AgentRunContext
        :return: A generator of ResponseStreamEvent objects.
        :rtype: Generator[ResponseStreamEvent, None, None]
        """
        item_id = context.id_generator.generate_message_id()

        yield ResponseCreatedEvent(response=OpenAIResponse(output=[]))
        yield ResponseOutputItemAddedEvent(
            output_index=0,
            item=ResponsesAssistantMessageItemResource(
                id=item_id,
                status="in_progress",
                content=[ItemContentOutputText(text="", annotations=[])],
            ),
        )

        assembled = ""
        for event in events:
            text = _extract_text(event)
            if text:
                assembled += text
                yield ResponseTextDeltaEvent(
                    output_index=0,
                    content_index=0,
                    delta=text,
                )

        yield ResponseTextDoneEvent(output_index=0, content_index=0, text=assembled)
        yield ResponseCompletedEvent(
            response=OpenAIResponse(
                id=context.response_id,
                created_at=datetime.datetime.now(),
                output=[
                    ResponsesAssistantMessageItemResource(
                        id=item_id,
                        status="completed",
                        content=[ItemContentOutputText(text=assembled, annotations=[])],
                    )
                ],
            )
        )


def _extract_text(event: SessionEvent) -> Optional[str]:
    """Extract text content from a Copilot SessionEvent.

    :param event: A Copilot session event.
    :type event: SessionEvent
    :return: The text content or None.
    :rtype: Optional[str]
    """
    if event.type == SessionEventType.ASSISTANT_MESSAGE:
        data = event.data
        if data and hasattr(data, "content") and data.content:
            return data.content
    if event.type == SessionEventType.ASSISTANT_MESSAGE_DELTA:
        data = event.data
        if data and hasattr(data, "content") and data.content:
            return data.content
    return None
