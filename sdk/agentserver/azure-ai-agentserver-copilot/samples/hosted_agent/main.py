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

Environment variables
---------------------
AZURE_AI_FOUNDRY_RESOURCE_URL
    Azure AI Foundry resource URL (e.g. https://<name>.openai.azure.com).
    When set, the adapter uses Foundry models via BYOK with Managed Identity.
COPILOT_MODEL
    Model deployment name (default: gpt-4.1 for Foundry, gpt-5 for GitHub).
"""

import asyncio

from azure.ai.agentserver.copilot import from_copilot


async def main() -> None:
    agent = from_copilot()
    await agent.run_async()


if __name__ == "__main__":
    asyncio.run(main())
