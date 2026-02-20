# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation,broad-exception-caught
import asyncio
import dataclasses
import os
import uuid
from typing import Any, Dict, List, Optional

from copilot import CopilotClient, MessageOptions, ProviderConfig, SessionConfig
from copilot.generated.session_events import SessionEventType
from copilot.types import PermissionRequest, PermissionRequestResult, ResumeSessionConfig

from azure.ai.agentserver.core.constants import Constants
from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.server.base import FoundryCBAgent
from azure.ai.agentserver.core.server.common.agent_run_context import AgentRunContext

from .models.copilot_request_converter import CopilotRequestConverter
from .models.copilot_response_converter import CopilotResponseConverter, CopilotStreamingResponseConverter

logger = get_logger()

_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


def _build_session_config() -> SessionConfig:
    """Build a SessionConfig from environment variables.

    When ``AZURE_AI_FOUNDRY_RESOURCE_URL`` is set the adapter runs in
    **BYOK mode** against Azure AI Foundry.

    Authentication priority:
    1. ``AZURE_AI_FOUNDRY_API_KEY`` — static API key (easiest for local dev)
    2. ``DefaultAzureCredential`` — Managed Identity / Azure CLI (for prod)

    When using an API key the token is set once at startup.  When using
    ``DefaultAzureCredential`` the bearer token starts as a placeholder and
    is refreshed lazily by ``_refresh_token_if_needed()`` before each
    session.

    Environment variables
    ---------------------
    AZURE_AI_FOUNDRY_RESOURCE_URL
        The Azure AI Foundry resource URL, e.g.
        ``https://<resource>.cognitiveservices.azure.com``.
    AZURE_AI_FOUNDRY_API_KEY
        Static API key for the Foundry resource.  If set, takes priority
        over ``DefaultAzureCredential``.
    COPILOT_MODEL
        Model deployment name (default ``gpt-4.1``).
    """
    foundry_url = os.getenv("AZURE_AI_FOUNDRY_RESOURCE_URL")
    model = os.getenv("COPILOT_MODEL", "gpt-4.1")

    if foundry_url:
        base_url = foundry_url.rstrip("/") + "/openai/v1/"
        api_key = os.getenv("AZURE_AI_FOUNDRY_API_KEY")

        if api_key:
            logger.info(f"BYOK mode (API key): {base_url}")
            return SessionConfig(
                model=model,
                provider=ProviderConfig(
                    type="openai",
                    base_url=base_url,
                    bearer_token=api_key,
                    wire_api="responses",
                ),
            )

        logger.info(f"BYOK mode (Managed Identity): {base_url}")
        return SessionConfig(
            model=model,
            provider=ProviderConfig(
                type="openai",
                base_url=base_url,
                bearer_token="placeholder",  # refreshed before first use
                wire_api="responses",
            ),
        )

    # Fallback: default GitHub Copilot models
    return SessionConfig(model=os.getenv("COPILOT_MODEL", "gpt-5"))


class CopilotAdapter(FoundryCBAgent):
    """Adapter that bridges a GitHub Copilot SDK session to an Azure AI Agent Server.

    When ``AZURE_AI_FOUNDRY_RESOURCE_URL`` is set the adapter uses Azure AI
    Foundry models via BYOK with Managed Identity authentication.  Otherwise
    it falls back to the default GitHub Copilot models.

    :param session_config: Override for the Copilot session config.  When
        *None* the config is built automatically from environment variables.
    :type session_config: Optional[SessionConfig]
    """

    def __init__(self, session_config: Optional[SessionConfig] = None):
        super().__init__()
        self._session_config = session_config or _build_session_config()
        self._client: Optional[CopilotClient] = None
        self._credential = None

        # Map approval_request_id → {session_id, prompt, denied_requests}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}

        # Keep credential for token refresh when using Foundry with Managed Identity
        if os.getenv("AZURE_AI_FOUNDRY_RESOURCE_URL") and not os.getenv("AZURE_AI_FOUNDRY_API_KEY"):
            from azure.identity import DefaultAzureCredential

            self._credential = DefaultAzureCredential()

    def _refresh_token_if_needed(self) -> SessionConfig:
        """Return the session config, refreshing the bearer token if using Foundry."""
        if self._credential is None or "provider" not in self._session_config:
            return self._session_config

        token = self._credential.get_token(_COGNITIVE_SERVICES_SCOPE).token
        self._session_config["provider"]["bearer_token"] = token
        return self._session_config

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
        config = self._refresh_token_if_needed()

        # First run: deny all tool requests and capture them
        denied_requests: List[Dict[str, Any]] = []

        def _on_permission(req: PermissionRequest, _ctx: dict) -> PermissionRequestResult:
            request_id = str(uuid.uuid4())
            kind = req.get("kind", "unknown")
            logger.info(f"Denied tool (needs approval): kind={kind} id={request_id}")
            denied_requests.append({
                "request_id": request_id,
                "permission_request": dict(req),
            })
            return PermissionRequestResult(
                kind="denied-no-approval-rule-and-could-not-request-from-user"
            )

        session_config = SessionConfig(**config, on_permission_request=_on_permission)
        session = await client.create_session(session_config)

        try:
            collected_events, _ = await _send_and_collect(session, prompt)


            # TODO: decide when to destroy session.
            # await session.destroy()
            if not context.stream:
                # Non-streaming support is a dumb idea. IMO this should yield a not-supported-exception.
                return CopilotResponseConverter.to_response("", context)
            response_converter = CopilotStreamingResponseConverter(context)
            return response_converter.to_stream_events(
                collected_events, context,
            )

        except Exception as e:
            logger.error(f"Error during Copilot agent run: {e}")
            raise

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


async def _send_and_collect(session, prompt: str):
    """Send a message and collect all events until SESSION_IDLE.

    The Copilot SDK emits every event twice after ``resume_session``.
    These duplicates are always **consecutive**, so we skip an event
    only when it is identical to the immediately preceding one.
    """
    collected = []
    last_key = None
    done = asyncio.Event()

    def on_event(event):
        nonlocal last_key
        # Build a dedup key from type + content
        text = ""
        if event.data and hasattr(event.data, "content") and event.data.content:
            text = event.data.content
        key = (event.type, text)

        # Skip only consecutive duplicates (resume_session replay artefact)
        if key == last_key:
            logger.debug(f"Skipping consecutive duplicate: {event.type}")
            return
        last_key = key

        collected.append(event)
        if logger.isEnabledFor(10):  # DEBUG
            data_fields: Dict[str, Any] = {}
            if event.data is not None:
                try:
                    raw = dataclasses.asdict(event.data)
                    data_fields = {k: v for k, v in raw.items() if v is not None}
                except Exception:
                    data_fields = {"repr": repr(event.data)}
            logger.debug(
                f"Event #{len(collected):03d} {event.type.name if event.type else '?'} "
                f"data={data_fields}"
            )
        if event.type == SessionEventType.SESSION_IDLE:
            done.set()

    session.on(on_event)
    await session.send(MessageOptions(prompt=prompt))
    await asyncio.wait_for(done.wait(), timeout=120)
    return collected, done

