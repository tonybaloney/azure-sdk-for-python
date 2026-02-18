# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for the tool approval flow in CopilotAdapter."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from azure.ai.agentserver.copilot.copilot_adapter import (
    CopilotAdapter,
)
from azure.ai.agentserver.copilot.models.copilot_response_converter import CopilotResponseConverter


def _make_context(stream=False, input_items=None):
    """Build a minimal AgentRunContext mock."""
    ctx = MagicMock()
    ctx.stream = stream
    ctx.response_id = "resp_test123"
    ctx.request = {"input": input_items or [{"role": "user", "content": "Hello"}]}
    ctx.id_generator.generate_message_id.return_value = "msg_test456"
    return ctx


def _make_event(event_type, content=None):
    """Build a minimal SessionEvent mock."""
    event = MagicMock()
    event.type = event_type
    if content is not None:
        event.data = MagicMock()
        event.data.content = content
    else:
        event.data = None
    return event


@pytest.mark.unit
class TestFindApprovalResponse:
    """Tests for CopilotAdapter._find_approval_response."""

    def test_no_approval_in_input(self):
        ctx = _make_context(input_items=[{"role": "user", "content": "hi"}])
        result = CopilotAdapter._find_approval_response(ctx)
        assert result is None

    def test_approval_found(self):
        items = [
            {"type": "message", "role": "user", "content": "hi"},
            {
                "type": "mcp_approval_response",
                "approval_request_id": "abc-123",
                "approve": True,
            },
        ]
        ctx = _make_context(input_items=items)
        result = CopilotAdapter._find_approval_response(ctx)
        assert result is not None
        assert result["approval_request_id"] == "abc-123"
        assert result["approve"] is True

    def test_no_list_input(self):
        ctx = _make_context()
        ctx.request = {"input": "just a string"}
        result = CopilotAdapter._find_approval_response(ctx)
        assert result is None


@pytest.mark.unit
class TestApprovalRequestResponse:
    """Tests for CopilotResponseConverter.to_approval_request_response."""

    def test_builds_incomplete_response_with_text(self):
        ctx = _make_context()
        denied = [
            {
                "request_id": "req-abc",
                "permission_request": {"kind": "write", "toolCallId": "tc-1"},
            }
        ]
        resp = CopilotResponseConverter.to_approval_request_response(
            "I'd like to write a file.", denied, "create a file", ctx,
        )
        d = resp.as_dict()
        assert d["status"] == "incomplete"
        assert len(d["output"]) == 2
        # First item is the text
        assert d["output"][0]["type"] == "message"
        assert "write a file" in d["output"][0]["content"][0]["text"]
        # Second item is the approval request
        item = d["output"][1]
        assert item["type"] == "mcp_approval_request"
        assert item["server_label"] == "copilot-cli"
        assert item["name"] == "write"
        args = json.loads(item["arguments"])
        assert args["toolCallId"] == "tc-1"
        assert args["_original_prompt"] == "create a file"

    def test_no_text_before_approval(self):
        ctx = _make_context()
        denied = [
            {
                "request_id": "req-shell",
                "permission_request": {"kind": "shell", "toolCallId": "tc-2"},
            }
        ]
        resp = CopilotResponseConverter.to_approval_request_response(
            "", denied, "run a command", ctx,
        )
        d = resp.as_dict()
        assert d["status"] == "incomplete"
        # Only the approval item, no text item
        assert len(d["output"]) == 1
        assert d["output"][0]["type"] == "mcp_approval_request"
        assert d["output"][0]["name"] == "shell"

    def test_multiple_denied_tools(self):
        ctx = _make_context()
        denied = [
            {"request_id": "req-1", "permission_request": {"kind": "write", "toolCallId": "tc-1"}},
            {"request_id": "req-2", "permission_request": {"kind": "shell", "toolCallId": "tc-2"}},
        ]
        resp = CopilotResponseConverter.to_approval_request_response(
            "Need to do things.", denied, "prompt", ctx,
        )
        d = resp.as_dict()
        approval_items = [o for o in d["output"] if o.get("type") == "mcp_approval_request"]
        assert len(approval_items) == 2


@pytest.mark.unit
class TestStreamEventsWithApproval:
    """Tests for CopilotResponseConverter.to_stream_events_with_approval."""

    def test_yields_approval_and_incomplete(self):
        from copilot.generated.session_events import SessionEventType

        events = [_make_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, "thinking...")]
        denied = [
            {"request_id": "req-xyz", "permission_request": {"kind": "mcp", "toolCallId": "tc-3"}},
        ]
        ctx = _make_context(stream=True)
        stream = list(
            CopilotResponseConverter.to_stream_events_with_approval(
                events, denied, "do something", ctx,
            )
        )
        types = [e.type for e in stream]
        assert "response.created" in types
        assert "response.incomplete" in types
        assert "response.output_item.added" in types

        last = stream[-1]
        assert last.type == "response.incomplete"
        resp_dict = last.as_dict()
        assert resp_dict["response"]["status"] == "incomplete"

        output = resp_dict["response"]["output"]
        approval_items = [o for o in output if o.get("type") == "mcp_approval_request"]
        assert len(approval_items) == 1
        assert approval_items[0]["id"] == "req-xyz"

    def test_no_text_before_approval(self):
        denied = [
            {"request_id": "req-empty", "permission_request": {"kind": "write", "toolCallId": "tc-4"}},
        ]
        ctx = _make_context(stream=True)
        stream = list(
            CopilotResponseConverter.to_stream_events_with_approval(
                [], denied, "write file", ctx,
            )
        )
        types = [e.type for e in stream]
        assert "response.text.delta" not in types
        assert "response.output_item.added" in types
        assert "response.incomplete" in types


@pytest.mark.unit
class TestHandleApprovalResponse:
    """Tests for CopilotAdapter._handle_approval_response."""

    def test_denied_by_caller_returns_text(self):
        async def _run():
            adapter = CopilotAdapter.__new__(CopilotAdapter)
            adapter._session_config = {"model": "gpt-5"}
            adapter._credential = None
            adapter._client = None

            ctx = _make_context()
            approval = {"type": "mcp_approval_response", "approve": False}
            return await adapter._handle_approval_response(approval, ctx)

        result = asyncio.run(_run())
        d = result.as_dict()
        assert "denied" in d["output"][0]["content"][0]["text"].lower()

    @patch.object(CopilotAdapter, "_run_session")
    def test_approved_reruns_with_tools(self, mock_run_session):
        """Approving re-runs the session with the tool pre-approved."""
        mock_response = MagicMock()
        mock_run_session.return_value = mock_response

        async def _run():
            adapter = CopilotAdapter.__new__(CopilotAdapter)
            adapter._session_config = {"model": "gpt-5"}
            adapter._credential = None
            adapter._client = None

            ctx = _make_context()
            approval = {
                "type": "mcp_approval_response",
                "approve": True,
                "_original_prompt": "create a python script",
                "_permission_request": {"kind": "write", "toolCallId": "tc-99"},
            }
            return await adapter._handle_approval_response(approval, ctx)

        asyncio.run(_run())
        # Should have called _run_session with approved_tools
        mock_run_session.assert_called_once()
        call_args = mock_run_session.call_args
        assert call_args[0][0] == "create a python script"
        assert call_args[1]["approved_tools"] == [{"kind": "write", "toolCallId": "tc-99"}]
