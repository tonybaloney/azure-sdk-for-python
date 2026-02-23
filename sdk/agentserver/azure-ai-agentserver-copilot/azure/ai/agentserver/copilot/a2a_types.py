# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""A2A (Agent-to-Agent) protocol types for the Copilot adapter.

This module implements the core A2A data structures following the
A2A Protocol Specification v0.3:
https://a2a-protocol.org/specification/

The A2A protocol enables richer agent communication than RAPI, including:
- Tool execution visibility
- Task lifecycle with state transitions
- Artifact-based outputs
- Progress updates during long operations
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskState(str, Enum):
    """A2A Task states."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class Role(str, Enum):
    """Message role in A2A."""
    USER = "user"
    AGENT = "agent"


@dataclass
class TextPart:
    """Text content part."""
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text}


@dataclass
class DataPart:
    """Structured data part."""
    data: Dict[str, Any]
    media_type: str = "application/json"

    def to_dict(self) -> Dict[str, Any]:
        return {"data": self.data, "mediaType": self.media_type}


Part = TextPart | DataPart


@dataclass
class Message:
    """A2A Message."""
    role: Role
    parts: List[Part]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_id: Optional[str] = None
    task_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: _iso_now())

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "messageId": self.message_id,
            "role": self.role.value,
            "parts": [p.to_dict() for p in self.parts],
            "timestamp": self.timestamp,
        }
        if self.context_id:
            d["contextId"] = self.context_id
        if self.task_id:
            d["taskId"] = self.task_id
        return d


@dataclass
class Artifact:
    """A2A Artifact — output from task execution."""
    name: str
    parts: List[Part]
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None
    index: int = 0
    append: bool = False
    last_chunk: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "artifactId": self.artifact_id,
            "name": self.name,
            "parts": [p.to_dict() for p in self.parts],
            "index": self.index,
            "append": self.append,
            "lastChunk": self.last_chunk,
        }
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class TaskStatus:
    """A2A Task status."""
    state: TaskState
    message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: _iso_now())

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "state": self.state.value,
            "timestamp": self.timestamp,
        }
        if self.message:
            d["message"] = self.message
        return d


@dataclass
class Task:
    """A2A Task — unit of work."""
    task_id: str
    context_id: str
    status: TaskStatus
    artifacts: List[Artifact] = field(default_factory=list)
    history: List[Message] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.task_id,
            "contextId": self.context_id,
            "status": self.status.to_dict(),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "history": [m.to_dict() for m in self.history],
        }


@dataclass
class Skill:
    """Agent skill descriptor."""
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
        }


@dataclass
class AgentCard:
    """A2A Agent Card — discovery manifest."""
    name: str
    description: str
    version: str
    url: str
    skills: List[Skill] = field(default_factory=list)
    default_input_modes: List[str] = field(default_factory=lambda: ["text/plain"])
    default_output_modes: List[str] = field(default_factory=lambda: ["text/plain"])
    capabilities: Dict[str, bool] = field(default_factory=lambda: {
        "streaming": True,
        "pushNotifications": False,
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url,
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "capabilities": self.capabilities,
            "skills": [s.to_dict() for s in self.skills],
        }


# ---------------------------------------------------------------------------
# SSE Event Helpers
# ---------------------------------------------------------------------------


@dataclass
class A2AStreamEvent:
    """A2A SSE stream event."""
    event_type: str
    data: Dict[str, Any]

    def to_sse(self) -> str:
        import json
        return f"event: {self.event_type}\ndata: {json.dumps(self.data)}\n\n"


def task_status_event(task: Task) -> A2AStreamEvent:
    """Create a task status update event."""
    return A2AStreamEvent(
        event_type="task.status",
        data={"task": task.to_dict()},
    )


def artifact_event(task_id: str, artifact: Artifact) -> A2AStreamEvent:
    """Create an artifact update event."""
    return A2AStreamEvent(
        event_type="task.artifact",
        data={"taskId": task_id, "artifact": artifact.to_dict()},
    )


def message_event(message: Message) -> A2AStreamEvent:
    """Create a message event."""
    return A2AStreamEvent(
        event_type="message",
        data={"message": message.to_dict()},
    )


def _iso_now() -> str:
    """Return current time in ISO 8601 format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
