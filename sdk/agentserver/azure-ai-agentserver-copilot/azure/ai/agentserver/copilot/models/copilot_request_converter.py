# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from typing import Any, Dict, List, Union

from azure.ai.agentserver.core.models import CreateResponse


class CopilotRequestConverter:
    """Converts an AgentRunContext request into a prompt string for the Copilot SDK."""

    def __init__(self, request: CreateResponse):
        self._request = request

    def convert(self) -> str:
        """Extract a prompt string from the incoming CreateResponse request.

        Handles several input shapes:
        - ``str``: returned as-is
        - ``list[dict]``: messages are concatenated with role prefixes
        - ``dict`` with ``content`` key: treated as a single implicit user message

        :return: The extracted prompt string.
        :rtype: str
        """
        raw_input = self._request.get("input")
        if raw_input is None:
            return ""
        if isinstance(raw_input, str):
            return raw_input
        if isinstance(raw_input, list):
            return self._convert_message_list(raw_input)
        if isinstance(raw_input, dict):
            return self._extract_content(raw_input)
        raise ValueError(f"Unsupported input type: {type(raw_input)}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convert_message_list(self, messages: List[Dict[str, Any]]) -> str:
        """Flatten a list of message dicts into a single prompt string."""
        parts: List[str] = []
        for msg in messages:
            content = self._extract_content(msg)
            if content:
                parts.append(content)
        return "\n".join(parts)

    @staticmethod
    def _extract_content(msg: Union[Dict[str, Any], str]) -> str:
        """Pull the text content out of a single message dict or string."""
        if isinstance(msg, str):
            return msg
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        # content may be a list of content parts (e.g. [{type: input_text, text: ...}])
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        text_parts.append(text)
                elif isinstance(part, str):
                    text_parts.append(part)
            return " ".join(text_parts)
        return str(content) if content else ""
