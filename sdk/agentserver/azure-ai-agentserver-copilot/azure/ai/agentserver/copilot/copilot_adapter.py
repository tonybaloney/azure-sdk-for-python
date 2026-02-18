# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation,broad-exception-caught
import os
from typing import Optional

from copilot import CopilotClient, MessageOptions, SessionConfig
from copilot.generated.session_events import SessionEventType

from azure.ai.agentserver.core.constants import Constants
from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.server.base import FoundryCBAgent
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext

from .models.copilot_request_converter import CopilotRequestConverter
from .models.copilot_response_converter import CopilotResponseConverter

logger = get_logger()


class CopilotAdapter(FoundryCBAgent):
    """Adapter that bridges a GitHub Copilot SDK session to an Azure AI Agent Server.

    :param session_config: Configuration for creating Copilot sessions (model selection, etc.).
    :type session_config: Optional[SessionConfig]
    """

    def __init__(self, session_config: Optional[SessionConfig] = None):
        super().__init__()
        self._session_config = session_config or SessionConfig(model="gpt-5")
        self._client: Optional[CopilotClient] = None

    async def _ensure_client(self) -> CopilotClient:
        """Lazily start the CopilotClient."""
        if self._client is None:
            self._client = CopilotClient()
            await self._client.start()
            logger.info("CopilotClient started")
        return self._client

    async def agent_run(self, context: AgentRunContext):
        prompt = CopilotRequestConverter(context.request).convert()
        logger.debug(f"Copilot prompt: {prompt!r}")

        client = await self._ensure_client()
        session = await client.create_session(self._session_config)
        try:
            if not context.stream:
                return await self._run_non_stream(session, prompt, context)
            return self._run_stream(session, prompt, context)
        except Exception as e:
            logger.error(f"Error during Copilot agent run: {e}")
            raise

    async def _run_non_stream(self, session, prompt: str, context: AgentRunContext):
        """Execute a non-streaming request and return a single Response."""
        import asyncio

        collected_events = []
        done = asyncio.Event()

        def on_event(event):
            collected_events.append(event)
            if event.type == SessionEventType.SESSION_IDLE:
                done.set()

        session.on(on_event)
        await session.send(MessageOptions(prompt=prompt))
        await asyncio.wait_for(done.wait(), timeout=300)
        await session.destroy()

        # Extract final text from collected events
        text_parts = []
        for event in collected_events:
            if event.type == SessionEventType.ASSISTANT_MESSAGE:
                if event.data and hasattr(event.data, "content") and event.data.content:
                    text_parts.append(event.data.content)
        full_text = "\n".join(text_parts) if text_parts else ""

        return CopilotResponseConverter.to_response(full_text, context)

    async def _run_stream(self, session, prompt: str, context: AgentRunContext):
        """Execute a streaming request, yielding ResponseStreamEvent objects."""
        import asyncio

        collected_events = []
        done = asyncio.Event()

        def on_event(event):
            collected_events.append(event)
            if event.type == SessionEventType.SESSION_IDLE:
                done.set()

        session.on(on_event)
        await session.send(MessageOptions(prompt=prompt))
        await asyncio.wait_for(done.wait(), timeout=300)
        await session.destroy()

        # Convert collected events to streaming events
        return CopilotResponseConverter.to_stream_events(collected_events, context)

    def get_trace_attributes(self):
        attrs = super().get_trace_attributes()
        attrs["service.namespace"] = "azure.ai.agentserver.copilot"
        return attrs

    def get_agent_identifier(self) -> str:
        agent_name = os.getenv(Constants.AGENT_NAME)
        if agent_name:
            return agent_name
        agent_id = os.getenv(Constants.AGENT_ID)
        if agent_id:
            return agent_id
        return "HostedAgent-Copilot"
