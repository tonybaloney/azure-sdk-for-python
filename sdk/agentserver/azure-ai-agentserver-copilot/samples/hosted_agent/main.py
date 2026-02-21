# Copyright (c) Microsoft. All rights reserved.

"""Hosted agent backed by the GitHub Copilot SDK.

This sample shows how to wrap the Copilot SDK as a hosted agent that
exposes the Foundry Responses API.  It follows the pattern described in
the Microsoft Learn tutorial:

    https://learn.microsoft.com/azure/ai-foundry/agents/concepts/hosted-agents

Usage
-----
1. Start the agent server locally (defaults to 0.0.0.0:8088)::

       python main.py

2. Test with a non-streaming request::

       curl -sS -H "Content-Type: application/json" \\
         -X POST http://localhost:8088/responses \\
         -d '{"input":"What is the capital of France?","stream":false}'

3. Test with a streaming request (SSE)::

       curl -N -H "Content-Type: application/json" \\
         -X POST http://localhost:8088/responses \\
         -d '{"input":"Explain Python decorators briefly.","stream":true}'

Tool Access Control List (ACL)
--------------------------------
The ``tools_acl.yaml`` file in this directory controls which tools the agent
is allowed to call at runtime.  It is loaded automatically when the server
starts.  You can:

* Edit ``tools_acl.yaml`` to customise the allow/deny rules.
* Set the ``TOOL_ACL_PATH`` environment variable to point to a different ACL
  file (overrides the side-by-side file).
* Remove ``tools_acl.yaml`` **and** unset ``TOOL_ACL_PATH`` to fall back to
  approve-all mode (useful for debugging, not recommended for production).

See the README for the full YAML schema reference.

Environment variables
---------------------
AZURE_AI_FOUNDRY_RESOURCE_URL
    Azure AI Foundry resource URL (e.g. https://<name>.cognitiveservices.azure.com).
    When set, the adapter uses Foundry models via BYOK.
AZURE_AI_FOUNDRY_API_KEY
    Static API key for the Foundry resource (for local dev). If not set,
    falls back to DefaultAzureCredential (Managed Identity / Azure CLI).
COPILOT_MODEL
    Model deployment name (default: gpt-4.1 for Foundry, gpt-5 for GitHub).
TOOL_ACL_PATH
    Path to a YAML ACL file.  Overrides the side-by-side tools_acl.yaml.
"""

import asyncio
import os
from pathlib import Path

from azure.ai.agentserver.copilot import from_copilot

# Resolve the ACL file: TOOL_ACL_PATH env var takes priority; otherwise fall
# back to tools_acl.yaml sitting next to this script.
_HERE = Path(__file__).parent
_DEFAULT_ACL = _HERE / "tools_acl.yaml"

_acl_path = os.getenv("TOOL_ACL_PATH") or (str(_DEFAULT_ACL) if _DEFAULT_ACL.exists() else None)


async def main() -> None:
    agent = from_copilot(acl_path=_acl_path)
    await agent.run_async()


if __name__ == "__main__":
    asyncio.run(main())

