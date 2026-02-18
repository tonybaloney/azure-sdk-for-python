# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import datetime
import json
from typing import Any, Dict, Generator, Optional

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
    def to_approval_request_response(
        approval_info: Dict[str, Any], context: AgentRunContext,
    ) -> OpenAIResponse:
        """Build a non-streaming incomplete response containing an approval request.

        :param approval_info: Dict with ``request_id`` and ``permission_request``.
        :param context: The agent run context.
        :return: An OpenAI-compatible Response with status ``incomplete``.
        """
        perm = approval_info["permission_request"]
        return OpenAIResponse(
            id=context.response_id,
            created_at=datetime.datetime.now(),
            status="incomplete",
            output=[
                MCPApprovalRequestItemResource(
                    id=approval_info["request_id"],
                    server_label="copilot-cli",
                    name=perm.get("kind", "unknown"),
                    arguments=json.dumps(
                        {k: v for k, v in perm.items() if k != "kind"},
                        default=str,
                    ),
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

        yield ResponseCreatedEvent(response=OpenAIResponse(id=context.response_id, output=[]))
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

    @staticmethod
    def to_stream_events_with_approval(
        events: list[SessionEvent],
        approval_info: Dict[str, Any],
        context: AgentRunContext,
    ) -> Generator[ResponseStreamEvent, None, None]:
        """Yield text events collected so far, then an approval request, then ``response.incomplete``.

        :param events: Copilot session events collected before the approval was requested.
        :param approval_info: Dict with ``request_id`` and ``permission_request``.
        :param context: The agent run context.
        """
        msg_id = context.id_generator.generate_message_id()

        yield ResponseCreatedEvent(response=OpenAIResponse(id=context.response_id, output=[]))

        # Emit any text that was produced before the tool approval
        assembled = ""
        text_output_index = 0
        has_text = False
        for event in events:
            text = _extract_text(event)
            if text:
                if not has_text:
                    yield ResponseOutputItemAddedEvent(
                        output_index=text_output_index,
                        item=ResponsesAssistantMessageItemResource(
                            id=msg_id,
                            status="in_progress",
                            content=[ItemContentOutputText(text="", annotations=[])],
                        ),
                    )
                    has_text = True
                assembled += text
                yield ResponseTextDeltaEvent(
                    output_index=text_output_index,
                    content_index=0,
                    delta=text,
                )

        if has_text:
            yield ResponseTextDoneEvent(
                output_index=text_output_index, content_index=0, text=assembled
            )
            yield ResponseOutputItemDoneEvent(
                output_index=text_output_index,
                item=ResponsesAssistantMessageItemResource(
                    id=msg_id,
                    status="completed",
                    content=[ItemContentOutputText(text=assembled, annotations=[])],
                ),
            )

        # Emit the approval request as an output item
        perm = approval_info["permission_request"]
        approval_item = MCPApprovalRequestItemResource(
            id=approval_info["request_id"],
            server_label="copilot-cli",
            name=perm.get("kind", "unknown"),
            arguments=json.dumps(
                {k: v for k, v in perm.items() if k != "kind"},
                default=str,
            ),
        )
        approval_output_index = 1 if has_text else 0
        yield ResponseOutputItemAddedEvent(
            output_index=approval_output_index,
            item=approval_item,
        )
        yield ResponseOutputItemDoneEvent(
            output_index=approval_output_index,
            item=approval_item,
        )

        # End the response as incomplete (awaiting approval)
        output_items = []
        if has_text:
            output_items.append(
                ResponsesAssistantMessageItemResource(
                    id=msg_id,
                    status="completed",
                    content=[ItemContentOutputText(text=assembled, annotations=[])],
                )
            )
        output_items.append(approval_item)

        yield ResponseIncompleteEvent(
            response=OpenAIResponse(
                id=context.response_id,
                created_at=datetime.datetime.now(),
                status="incomplete",
                output=output_items,
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
