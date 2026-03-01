# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for the async streaming path.

Tests ``_iter_copilot_events`` (the async generator that replaced
``_send_and_collect``) and verifies that ``_run_streaming`` converts Copilot
events to RAPI events on-the-fly without batching.
"""
import asyncio
import uuid
from contextlib import aclosing
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

from copilot.generated.session_events import SessionEventType

from azure.ai.agentserver.copilot.copilot_adapter import _iter_copilot_events

pytestmark = pytest.mark.asyncio(loop_scope="function")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(event_type: SessionEventType, content: str | None = None) -> MagicMock:
    event = MagicMock()
    event.type = event_type
    if content is not None:
        event.data = MagicMock()
        event.data.content = content
    else:
        event.data = None
    return event


class FakeSession:
    """Minimal Copilot session stub.

    Call ``push(*events)`` to deliver events to registered listeners, then
    ``idle()`` to push SESSION_IDLE and close the iterator.
    """

    def __init__(self):
        self._callbacks: list[Callable] = []
        self._sent_prompts: list[str] = []
        # Events to deliver automatically from ``send()``
        self._auto_events: list[MagicMock] = []

    def on(self, cb: Callable) -> Callable:
        self._callbacks.append(cb)
        def unsubscribe():
            self._callbacks.remove(cb)
        return unsubscribe

    async def send(self, options) -> None:
        self._sent_prompts.append(getattr(options, "prompt", None))
        # Deliver pre-loaded events after a tiny yield so the queue is read
        loop = asyncio.get_running_loop()
        events = list(self._auto_events)
        loop.call_soon(self._push_many, events)

    def _push_many(self, events: list) -> None:
        for e in events:
            for cb in list(self._callbacks):
                cb(e)

    def push(self, *events) -> None:
        for e in events:
            for cb in list(self._callbacks):
                cb(e)

    def idle(self) -> None:
        self.push(_make_event(SessionEventType.SESSION_IDLE))


# ---------------------------------------------------------------------------
# _iter_copilot_events
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIterCopilotEvents:

    async def test_yields_events_in_order(self):
        """Events arrive in the same order they are pushed by the session."""
        session = FakeSession()
        session._auto_events = [
            _make_event(SessionEventType.ASSISTANT_TURN_START),
            _make_event(SessionEventType.ASSISTANT_MESSAGE, "hello"),
            _make_event(SessionEventType.ASSISTANT_TURN_END),
            _make_event(SessionEventType.SESSION_IDLE),
        ]
        received = []
        async for event in _iter_copilot_events(session, "hi"):
            received.append(event.type)
        assert received == [
            SessionEventType.ASSISTANT_TURN_START,
            SessionEventType.ASSISTANT_MESSAGE,
            SessionEventType.ASSISTANT_TURN_END,
            SessionEventType.SESSION_IDLE,
        ]

    async def test_stops_after_session_idle(self):
        """Generator closes cleanly after SESSION_IDLE; subsequent events are not yielded."""
        session = FakeSession()
        session._auto_events = [
            _make_event(SessionEventType.ASSISTANT_TURN_START),
            _make_event(SessionEventType.SESSION_IDLE),
            # This arrives after idle — should NOT be yielded
            _make_event(SessionEventType.ASSISTANT_TURN_END),
        ]
        received = []
        async for event in _iter_copilot_events(session, "hi"):
            received.append(event.type)
        # SESSION_IDLE is the last yielded event; ASSISTANT_TURN_END is dropped
        assert SessionEventType.SESSION_IDLE in received
        assert SessionEventType.ASSISTANT_TURN_END not in received

    async def test_deduplicates_consecutive_identical_events(self):
        """Consecutive events with the same (type, content) are dropped (resume_session artefact)."""
        session = FakeSession()
        e_start = _make_event(SessionEventType.ASSISTANT_TURN_START)
        session._auto_events = [
            e_start,
            e_start,  # duplicate — should be skipped
            _make_event(SessionEventType.SESSION_IDLE),
        ]
        received = []
        async for event in _iter_copilot_events(session, "hi"):
            received.append(event.type)
        assert received.count(SessionEventType.ASSISTANT_TURN_START) == 1

    async def test_unsubscribes_listener_on_completion(self):
        """The session listener is unsubscribed once the generator is exhausted."""
        session = FakeSession()
        session._auto_events = [_make_event(SessionEventType.SESSION_IDLE)]
        async for _ in _iter_copilot_events(session, "hi"):
            pass
        assert len(session._callbacks) == 0, "Listener should have been unsubscribed after iteration"

    async def test_unsubscribes_on_early_break(self):
        """The listener is unsubscribed when the generator is explicitly closed (aclosing pattern)."""
        session = FakeSession()
        session._auto_events = [
            _make_event(SessionEventType.ASSISTANT_TURN_START),
            _make_event(SessionEventType.ASSISTANT_MESSAGE, "x"),
            _make_event(SessionEventType.SESSION_IDLE),
        ]
        # aclosing() calls aclose() on exit, which triggers the finally block
        # immediately — the correct pattern when abandoning an async generator early.
        async with aclosing(_iter_copilot_events(session, "hi")) as gen:
            async for event in gen:
                if event.type == SessionEventType.ASSISTANT_TURN_START:
                    break  # abandon iteration early
        assert len(session._callbacks) == 0

    async def test_records_sent_prompt(self):
        """The prompt is forwarded to ``session.send``."""
        session = FakeSession()
        session._auto_events = [_make_event(SessionEventType.SESSION_IDLE)]
        async for _ in _iter_copilot_events(session, "What is 2+2?"):
            pass
        # The prompt string is inside a MessageOptions object; we check send was called with something
        assert len(session._sent_prompts) == 1

    async def test_timeout_raises(self):
        """If no SESSION_IDLE arrives within the timeout, TimeoutError is raised."""
        session = FakeSession()
        # Never push SESSION_IDLE — session just hangs
        session._auto_events = []
        received = []
        with pytest.raises(asyncio.TimeoutError):
            async for event in _iter_copilot_events(session, "hi", timeout=0.05):
                received.append(event.type)
