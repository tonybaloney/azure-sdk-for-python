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
    """Converts Copilot SDK session events into OpenAI-format responses."""

    @staticmethod
    def to_response(text: str, context: AgentRunContext) -> OpenAIResponse:
        """Build a non-streaming OpenAI Response from the final assistant text."""
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
        text: str,
        denied_requests: List[Dict[str, Any]],
        original_prompt: str,
        context: AgentRunContext,
    ) -> OpenAIResponse:
        """Build an incomplete response with text + approval request items.

        The response includes any text the model produced before the tool
        call, plus ``mcp_approval_request`` items for each denied tool.
        Each approval item stashes the original prompt and permission
        request so the caller can re-submit them.
        """
        output: list = []

        # Include text if any
        if text:
            msg_id = context.id_generator.generate_message_id()
            output.append(
                ResponsesAssistantMessageItemResource(
                    id=msg_id,
                    status="completed",
                    content=[ItemContentOutputText(text=text, annotations=[])],
                )
            )

        # Add approval request items
        for req in denied_requests:
            perm = req["permission_request"]
            output.append(
                MCPApprovalRequestItemResource(
                    id=req["request_id"],
                    server_label="copilot-cli",
                    name=perm.get("kind", "unknown"),
                    arguments=json.dumps({
                        **{k: v for k, v in perm.items() if k != "kind"},
                        "_original_prompt": original_prompt,
                    }, default=str),
                )
            )

        return OpenAIResponse(
            id=context.response_id,
            created_at=datetime.datetime.now(),
            status="incomplete",
            output=output,
        )

    @staticmethod
    def to_stream_events(
        events: list[SessionEvent], context: AgentRunContext,
    ) -> Generator[ResponseStreamEvent, None, None]:
        """Convert a batch of Copilot SessionEvents into streaming response events."""
        item_id = context.id_generator.generate_message_id()

        yield ResponseCreatedEvent(
            response=OpenAIResponse(id=context.response_id, output=[]),
        )
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
                    output_index=0, content_index=0, delta=text,
                )

        yield ResponseTextDoneEvent(
            output_index=0, content_index=0, text=assembled,
        )
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
        denied_requests: List[Dict[str, Any]],
        original_prompt: str,
        context: AgentRunContext,
    ) -> Generator[ResponseStreamEvent, None, None]:
        """Yield text events, then approval request items, then ``response.incomplete``."""
        msg_id = context.id_generator.generate_message_id()

        yield ResponseCreatedEvent(
            response=OpenAIResponse(id=context.response_id, output=[]),
        )

        # Emit text
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
                    output_index=text_output_index, content_index=0, delta=text,
                )

        if has_text:
            yield ResponseTextDoneEvent(
                output_index=text_output_index, content_index=0, text=assembled,
            )
            yield ResponseOutputItemDoneEvent(
                output_index=text_output_index,
                item=ResponsesAssistantMessageItemResource(
                    id=msg_id,
                    status="completed",
                    content=[ItemContentOutputText(text=assembled, annotations=[])],
                ),
            )

        # Emit approval request items
        next_index = 1 if has_text else 0
        for req in denied_requests:
            perm = req["permission_request"]
            approval_item = MCPApprovalRequestItemResource(
                id=req["request_id"],
                server_label="copilot-cli",
                name=perm.get("kind", "unknown"),
                arguments=json.dumps({
                    **{k: v for k, v in perm.items() if k != "kind"},
                    "_original_prompt": original_prompt,
                }, default=str),
            )
            yield ResponseOutputItemAddedEvent(
                output_index=next_index, item=approval_item,
            )
            yield ResponseOutputItemDoneEvent(
                output_index=next_index, item=approval_item,
            )
            next_index += 1

        # Build final output list
        output_items: list = []
        if has_text:
            output_items.append(
                ResponsesAssistantMessageItemResource(
                    id=msg_id,
                    status="completed",
                    content=[ItemContentOutputText(text=assembled, annotations=[])],
                )
            )
        for req in denied_requests:
            perm = req["permission_request"]
            output_items.append(
                MCPApprovalRequestItemResource(
                    id=req["request_id"],
                    server_label="copilot-cli",
                    name=perm.get("kind", "unknown"),
                    arguments=json.dumps({
                        **{k: v for k, v in perm.items() if k != "kind"},
                        "_original_prompt": original_prompt,
                    }, default=str),
                )
            )

        yield ResponseIncompleteEvent(
            response=OpenAIResponse(
                id=context.response_id,
                created_at=datetime.datetime.now(),
                status="incomplete",
                output=output_items,
            )
        )


def _extract_text(event: SessionEvent) -> Optional[str]:
    """Extract text content from a Copilot SessionEvent."""
    if event.type == SessionEventType.ASSISTANT_MESSAGE:
        data = event.data
        if data and hasattr(data, "content") and data.content:
            return data.content
    if event.type == SessionEventType.ASSISTANT_MESSAGE_DELTA:
        data = event.data
        if data and hasattr(data, "content") and data.content:
            return data.content
    return None
