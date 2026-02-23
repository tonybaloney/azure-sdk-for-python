# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""A2A response converter for Copilot SDK events.

This module converts Copilot SDK SessionEvents into A2A protocol events,
enabling richer visibility into agent operations than RAPI provides.

Key benefits over RAPI:
- Tool execution events become visible as task status updates
- Tool outputs become typed artifacts
- Progress messages are streamed to clients
- Sub-agent delegation can be represented
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from copilot.generated.session_events import SessionEvent, SessionEventType

from .a2a_types import (
    A2AStreamEvent,
    Artifact,
    DataPart,
    Message,
    Part,
    Role,
    Skill,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
    artifact_event,
    message_event,
    task_status_event,
)

logger = logging.getLogger(__name__)


class A2AResponseConverter:
    """Converts Copilot SDK session events into A2A streaming events.

    Unlike the RAPI converter which collapses tool execution into invisible
    operations, the A2A converter exposes:
    - Tool execution start/progress/complete as task status updates
    - Tool results as typed artifacts
    - Reasoning (opt-in) as artifacts
    - Sub-agent delegation as status messages
    """

    def __init__(
        self,
        task_id: str,
        context_id: str,
        expose_reasoning: bool = False,
    ):
        self.task_id = task_id
        self.context_id = context_id
        self.expose_reasoning = expose_reasoning

        self._task = Task(
            task_id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.SUBMITTED),
        )
        self._accumulated_text: str = ""
        self._artifact_index: int = 0
        self._tool_artifacts: Dict[str, Artifact] = {}  # call_id → artifact
        self._completed: bool = False

    @property
    def task(self) -> Task:
        """Return the current task state."""
        return self._task

    def convert_event(
        self, event: SessionEvent
    ) -> Generator[A2AStreamEvent, None, None]:
        """Convert a single Copilot event to zero or more A2A stream events."""

        match event:

            # ── Turn start ────────────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_TURN_START):
                self._task.status = TaskStatus(
                    state=TaskState.WORKING,
                    message="Processing request...",
                )
                yield task_status_event(self._task)

            # ── Streaming text delta ──────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_MESSAGE_DELTA, data=data) if data and data.content:
                self._accumulated_text += data.content
                # Don't yield individual deltas — aggregate into final artifact

            # ── Full assistant message ────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_MESSAGE, data=data) if data and data.content:
                text = data.content
                if not self._accumulated_text:
                    self._accumulated_text = text

                # Create response artifact
                artifact = Artifact(
                    name="Response",
                    parts=[TextPart(text=text)],
                    index=self._next_artifact_index(),
                    description="Agent response text",
                )
                self._task.artifacts.append(artifact)
                yield artifact_event(self.task_id, artifact)

            # ── Tool execution start ──────────────────────────────────────
            case SessionEvent(type=SessionEventType.TOOL_EXECUTION_START, data=data) if data:
                tool_name = data.mcp_tool_name or data.tool_name or "unknown"
                call_id = data.tool_call_id or str(uuid.uuid4())

                # Update task status with tool info
                self._task.status = TaskStatus(
                    state=TaskState.WORKING,
                    message=f"Executing tool: {tool_name}",
                )
                yield task_status_event(self._task)

                # Create placeholder artifact for tool output
                artifact = Artifact(
                    artifact_id=f"tool-{call_id}",
                    name=f"Tool: {tool_name}",
                    parts=[],
                    index=self._next_artifact_index(),
                    description=f"Output from {tool_name}",
                    last_chunk=False,  # Will be updated on complete
                )
                self._tool_artifacts[call_id] = artifact

                # Include arguments if available
                if data.arguments is not None:
                    try:
                        import json
                        args_str = json.dumps(data.arguments, indent=2)
                        artifact.parts.append(DataPart(
                            data={"arguments": data.arguments},
                            media_type="application/json",
                        ))
                    except Exception:
                        pass

            # ── Tool execution progress ───────────────────────────────────
            case SessionEvent(type=SessionEventType.TOOL_EXECUTION_PROGRESS, data=data) if data:
                call_id = data.tool_call_id
                progress_msg = getattr(data, "progress_message", None) or "Working..."

                self._task.status = TaskStatus(
                    state=TaskState.WORKING,
                    message=progress_msg,
                )
                yield task_status_event(self._task)

            # ── Tool execution complete ───────────────────────────────────
            case SessionEvent(type=SessionEventType.TOOL_EXECUTION_COMPLETE, data=data) if data:
                call_id = data.tool_call_id
                artifact = self._tool_artifacts.get(call_id) if call_id else None

                if artifact:
                    # Add result to artifact
                    result = data.result
                    if result is not None:
                        if isinstance(result, str):
                            artifact.parts.append(TextPart(text=result))
                        else:
                            try:
                                artifact.parts.append(DataPart(
                                    data=result if isinstance(result, dict) else {"result": result},
                                ))
                            except Exception:
                                artifact.parts.append(TextPart(text=str(result)))
                    artifact.last_chunk = True
                    self._task.artifacts.append(artifact)
                    yield artifact_event(self.task_id, artifact)

                # Update status
                self._task.status = TaskStatus(
                    state=TaskState.WORKING,
                    message="Tool completed, continuing...",
                )
                yield task_status_event(self._task)

            # ── Sub-agent events ──────────────────────────────────────────
            case SessionEvent(type=SessionEventType.SUBAGENT_STARTED, data=data) if data:
                agent_name = getattr(data, "agent_display_name", None) or getattr(data, "agent_name", "sub-agent")
                self._task.status = TaskStatus(
                    state=TaskState.WORKING,
                    message=f"Delegating to: {agent_name}",
                )
                yield task_status_event(self._task)

            case SessionEvent(type=SessionEventType.SUBAGENT_COMPLETED, data=data):
                self._task.status = TaskStatus(
                    state=TaskState.WORKING,
                    message="Sub-agent completed",
                )
                yield task_status_event(self._task)

            case SessionEvent(type=SessionEventType.SUBAGENT_FAILED, data=data):
                self._task.status = TaskStatus(
                    state=TaskState.WORKING,
                    message="Sub-agent failed, continuing...",
                )
                yield task_status_event(self._task)

            # ── Reasoning (opt-in) ────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_REASONING, data=data) if data and data.content and self.expose_reasoning:
                artifact = Artifact(
                    name="Reasoning",
                    parts=[TextPart(text=data.content)],
                    index=self._next_artifact_index(),
                    description="Agent reasoning trace",
                )
                self._task.artifacts.append(artifact)
                yield artifact_event(self.task_id, artifact)

            # ── Session error ─────────────────────────────────────────────
            case SessionEvent(type=SessionEventType.SESSION_ERROR, data=data):
                error_msg = ""
                if data:
                    error_msg = getattr(data, "message", None) or getattr(data, "content", None) or str(data)

                self._task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message=error_msg or "Session error",
                )
                self._completed = True
                yield task_status_event(self._task)

            # ── Turn end ──────────────────────────────────────────────────
            case SessionEvent(type=SessionEventType.ASSISTANT_TURN_END):
                # Only update status if we have content
                if self._accumulated_text and not self._completed:
                    self._task.status = TaskStatus(
                        state=TaskState.WORKING,
                        message="Turn completed",
                    )
                    yield task_status_event(self._task)

            # ── Session idle (completion) ─────────────────────────────────
            case SessionEvent(type=SessionEventType.SESSION_IDLE):
                if not self._completed:
                    self._task.status = TaskStatus(
                        state=TaskState.COMPLETED,
                        message="Task completed successfully",
                    )
                    self._completed = True
                    yield task_status_event(self._task)

            # ── Model change ──────────────────────────────────────────────
            case SessionEvent(type=SessionEventType.SESSION_MODEL_CHANGE, data=data) if data:
                new_model = getattr(data, "new_model", None) or "unknown"
                self._task.status = TaskStatus(
                    state=TaskState.WORKING,
                    message=f"Model changed to: {new_model}",
                )
                yield task_status_event(self._task)

            # ── Other events ──────────────────────────────────────────────
            case _:
                # Log but don't emit for unhandled events
                ename = event.type.name if event.type else "UNKNOWN"
                logger.debug(f"A2A: Unhandled Copilot event: {ename}")

    def _next_artifact_index(self) -> int:
        """Get next artifact index."""
        idx = self._artifact_index
        self._artifact_index += 1
        return idx

    def force_completion(self, error: Optional[str] = None) -> A2AStreamEvent:
        """Force task completion (used for error recovery)."""
        if error:
            self._task.status = TaskStatus(
                state=TaskState.FAILED,
                message=error,
            )
        else:
            self._task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message="Task completed",
            )
        self._completed = True
        return task_status_event(self._task)


# ---------------------------------------------------------------------------
# Agent Card Loading
# ---------------------------------------------------------------------------

_AGENT_CARD_CACHE: Optional[Dict[str, Any]] = None


def load_agent_card_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    """Load agent card configuration from a YAML file.

    The YAML file should contain agent identity, capabilities, and skills.
    See samples/hosted_agent/agent_card.yaml for the schema reference.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the agent_card.yaml file.

    Returns
    -------
    Dict[str, Any]
        Parsed agent card data ready for AgentCard construction.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML is invalid or missing required fields.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Agent card YAML not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Agent card YAML must be a mapping, got {type(data).__name__}")

    # Validate required fields
    if "name" not in data:
        raise ValueError("Agent card YAML missing required field: name")

    return data


def build_agent_card(
    name: Optional[str] = None,
    description: Optional[str] = None,
    url: Optional[str] = None,
    version: Optional[str] = None,
    yaml_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an A2A Agent Card for discovery.

    The agent card is built from three sources (in priority order):
    1. Explicit parameters passed to this function
    2. YAML file (from yaml_path or A2A_AGENT_CARD_PATH env var)
    3. Default fallback values

    Parameters
    ----------
    name : str, optional
        Agent name (overrides YAML).
    description : str, optional
        Agent description (overrides YAML).
    url : str, optional
        Agent URL (typically set dynamically from request).
    version : str, optional
        Agent version (overrides YAML).
    yaml_path : str, optional
        Path to agent_card.yaml (defaults to A2A_AGENT_CARD_PATH env var,
        then ./agent_card.yaml, then built-in defaults).

    Returns
    -------
    Dict[str, Any]
        Agent card as a dictionary ready for JSON serialization.
    """
    global _AGENT_CARD_CACHE
    from .a2a_types import AgentCard, Skill

    # Try to load from YAML (with caching)
    yaml_data: Dict[str, Any] = {}
    if _AGENT_CARD_CACHE is not None:
        yaml_data = _AGENT_CARD_CACHE
    else:
        yaml_file = yaml_path or os.getenv("A2A_AGENT_CARD_PATH")
        if yaml_file is None:
            # Check common locations
            for candidate in ["agent_card.yaml", "/app/agent_card.yaml"]:
                if Path(candidate).exists():
                    yaml_file = candidate
                    break

        if yaml_file and Path(yaml_file).exists():
            try:
                yaml_data = load_agent_card_yaml(yaml_file)
                _AGENT_CARD_CACHE = yaml_data
                logger.info(f"Loaded A2A agent card from: {yaml_file}")
            except Exception as e:
                logger.warning(f"Failed to load agent card YAML ({yaml_file}): {e}")

    # Build skills from YAML or defaults
    skills_data = yaml_data.get("skills", [])
    if skills_data:
        skills = [
            Skill(
                id=s.get("id", f"skill-{i}"),
                name=s.get("name", "Unknown Skill"),
                description=s.get("description", ""),
                tags=s.get("tags", []),
            )
            for i, s in enumerate(skills_data)
        ]
    else:
        # Default skills
        skills = [
            Skill(
                id="code-assistant",
                name="Code Assistant",
                description="Help with coding tasks, explanations, and debugging",
                tags=["coding", "programming", "ai"],
            ),
            Skill(
                id="shell-execution",
                name="Shell Execution",
                description="Execute shell commands (subject to ACL)",
                tags=["tools", "shell", "automation"],
            ),
            Skill(
                id="file-operations",
                name="File Operations",
                description="Read and write files (subject to ACL)",
                tags=["tools", "files"],
            ),
            Skill(
                id="web-fetch",
                name="Web Fetch",
                description="Fetch content from URLs (subject to ACL)",
                tags=["tools", "web"],
            ),
        ]

    # Build capabilities
    caps = yaml_data.get("capabilities", {})
    capabilities = {
        "streaming": caps.get("streaming", True),
        "pushNotifications": caps.get("pushNotifications", False),
    }

    # Resolve final values: explicit > env > yaml > default
    final_name = (
        name
        or os.getenv("A2A_AGENT_NAME")
        or yaml_data.get("name")
        or "copilot-hosted-agent"
    )
    final_description = (
        description
        or os.getenv("A2A_AGENT_DESCRIPTION")
        or yaml_data.get("description")
        or "Azure AI Foundry hosted Copilot agent"
    )
    final_version = (
        version
        or yaml_data.get("version")
        or "1.0.0"
    )
    final_url = url or os.getenv("A2A_AGENT_URL") or ""

    card = AgentCard(
        name=final_name,
        description=final_description,
        version=final_version,
        url=final_url,
        skills=skills,
        default_input_modes=yaml_data.get("defaultInputModes", ["text/plain"]),
        default_output_modes=yaml_data.get("defaultOutputModes", ["text/plain"]),
        capabilities=capabilities,
    )
    return card.to_dict()
