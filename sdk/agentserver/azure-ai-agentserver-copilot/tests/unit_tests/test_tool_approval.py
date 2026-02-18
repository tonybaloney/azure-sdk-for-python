# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for the tool approval flow in CopilotAdapter."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.ai.agentserver.copilot.copilot_adapter import (
    CopilotAdapter,
    _drain_queue,
    _PendingSession,
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
class TestDrainQueue:
    """Tests for _drain_queue helper."""

    def test_drain_empty(self):
        q = asyncio.Queue()
        assert _drain_queue(q) == []

    def test_drain_items(self):
        q = asyncio.Queue()
        q.put_nowait("a")
        q.put_nowait("b")
        q.put_nowait("c")
        assert _drain_queue(q) == ["a", "b", "c"]
        assert q.empty()


@pytest.mark.unit
class TestApprovalRequestResponse:
    """Tests for CopilotResponseConverter.to_approval_request_response."""

    def test_builds_incomplete_response(self):
        ctx = _make_context()
        approval_info = {
            "request_id": "req-abc",
            "permission_request": {"kind": "write", "toolCallId": "tc-1"},
        }
        resp = CopilotResponseConverter.to_approval_request_response(approval_info, ctx)
        d = resp.as_dict()
        assert d["status"] == "incomplete"
        assert len(d["output"]) == 1
        item = d["output"][0]
        assert item["type"] == "mcp_approval_request"
        assert item["server_label"] == "copilot-cli"
        assert item["name"] == "write"
        args = json.loads(item["arguments"])
        assert args["toolCallId"] == "tc-1"

    def test_shell_kind(self):
        ctx = _make_context()
        approval_info = {
            "request_id": "req-shell",
            "permission_request": {"kind": "shell", "toolCallId": "tc-2"},
        }
        resp = CopilotResponseConverter.to_approval_request_response(approval_info, ctx)
        d = resp.as_dict()
        assert d["output"][0]["name"] == "shell"


@pytest.mark.unit
class TestStreamEventsWithApproval:
    """Tests for CopilotResponseConverter.to_stream_events_with_approval."""

    def test_yields_approval_and_incomplete(self):
        from copilot.generated.session_events import SessionEventType

        events = [_make_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, "thinking...")]
        approval_info = {
            "request_id": "req-xyz",
            "permission_request": {"kind": "mcp", "toolCallId": "tc-3"},
        }
        ctx = _make_context(stream=True)
        stream = list(
            CopilotResponseConverter.to_stream_events_with_approval(events, approval_info, ctx)
        )
        types = [e.type for e in stream]
        # Should have: created, output_item.added (text), text.delta, text.done,
        # output_item.done (text), output_item.added (approval), output_item.done (approval),
        # response.incomplete
        assert "response.created" in types
        assert "response.incomplete" in types
        assert "response.output_item.added" in types

        # Last event should be incomplete
        last = stream[-1]
        assert last.type == "response.incomplete"
        resp_dict = last.as_dict()
        assert resp_dict["response"]["status"] == "incomplete"

        # Should contain the approval item
        output = resp_dict["response"]["output"]
        approval_items = [o for o in output if o.get("type") == "mcp_approval_request"]
        assert len(approval_items) == 1
        assert approval_items[0]["id"] == "req-xyz"
        assert approval_items[0]["name"] == "mcp"

    def test_no_text_before_approval(self):
        """When no text events precede the approval, only approval items are emitted."""
        approval_info = {
            "request_id": "req-empty",
            "permission_request": {"kind": "write", "toolCallId": "tc-4"},
        }
        ctx = _make_context(stream=True)
        stream = list(
            CopilotResponseConverter.to_stream_events_with_approval([], approval_info, ctx)
        )
        types = [e.type for e in stream]
        # No text.delta events
        assert "response.text.delta" not in types
        # Still has approval
        assert "response.output_item.added" in types
        assert "response.incomplete" in types


@pytest.mark.unit
class TestHandleApprovalResponse:
    """Tests for CopilotAdapter._handle_approval_response."""

    def test_unknown_approval_id_returns_error(self):
        """When no pending session matches, return an error message."""
        async def _run():
            adapter = CopilotAdapter.__new__(CopilotAdapter)
            adapter._pending_sessions = {}
            adapter._session_config = {"model": "gpt-5"}
            adapter._credential = None
            adapter._client = None

            ctx = _make_context()
            approval = {"approval_request_id": "nonexistent", "approve": True}
            return await adapter._handle_approval_response(approval, ctx)

        result = asyncio.run(_run())
        d = result.as_dict()
        assert "no pending tool approval" in d["output"][0]["content"][0]["text"].lower()

    def test_approved_resumes_session(self):
        """Approving a pending session resolves the future and collects events."""
        from copilot.generated.session_events import SessionEventType

        async def _run():
            adapter = CopilotAdapter.__new__(CopilotAdapter)
            adapter._session_config = {"model": "gpt-5"}
            adapter._credential = None
            adapter._client = None

            loop = asyncio.get_running_loop()
            future = loop.create_future()
            done_event = asyncio.Event()
            event_queue = asyncio.Queue()

            msg_event = _make_event(SessionEventType.ASSISTANT_MESSAGE, "File created!")
            idle_event = _make_event(SessionEventType.SESSION_IDLE)
            event_queue.put_nowait(msg_event)
            event_queue.put_nowait(idle_event)
            done_event.set()

            mock_session = AsyncMock()
            pending = _PendingSession(
                session=mock_session,
                approval_future=future,
                event_queue=event_queue,
                done_event=done_event,
                approval_request_id="req-resume",
                permission_request={"kind": "write"},
            )
            adapter._pending_sessions = {"req-resume": pending}

            ctx = _make_context()
            approval = {"approval_request_id": "req-resume", "approve": True}
            result = await adapter._handle_approval_response(approval, ctx)
            return result, future, mock_session

        result, future, mock_session = asyncio.run(_run())
        assert future.result()["kind"] == "approved"
        mock_session.destroy.assert_awaited_once()
        d = result.as_dict()
        assert "File created!" in d["output"][0]["content"][0]["text"]

    def test_denied_sends_denied_result(self):
        """Denying a pending session sends denied result to the future."""
        from copilot.generated.session_events import SessionEventType

        async def _run():
            adapter = CopilotAdapter.__new__(CopilotAdapter)
            adapter._session_config = {"model": "gpt-5"}
            adapter._credential = None
            adapter._client = None

            loop = asyncio.get_running_loop()
            future = loop.create_future()
            done_event = asyncio.Event()
            event_queue = asyncio.Queue()

            msg = _make_event(SessionEventType.ASSISTANT_MESSAGE, "Denied action.")
            idle = _make_event(SessionEventType.SESSION_IDLE)
            event_queue.put_nowait(msg)
            event_queue.put_nowait(idle)
            done_event.set()

            mock_session = AsyncMock()
            pending = _PendingSession(
                session=mock_session,
                approval_future=future,
                event_queue=event_queue,
                done_event=done_event,
                approval_request_id="req-deny",
                permission_request={"kind": "shell"},
            )
            adapter._pending_sessions = {"req-deny": pending}

            ctx = _make_context()
            approval = {
                "approval_request_id": "req-deny",
                "approve": False,
                "reason": "Not safe",
            }
            result = await adapter._handle_approval_response(approval, ctx)
            return result, future

        result, future = asyncio.run(_run())
        assert future.result()["kind"] == "denied-interactively-by-user"
