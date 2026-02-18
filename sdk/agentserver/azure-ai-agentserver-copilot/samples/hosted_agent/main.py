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
COPILOT_MODEL   Model to use for Copilot sessions (default: gpt-5).
"""

import asyncio
import os

from copilot import SessionConfig

from azure.ai.agentserver.copilot import from_copilot

MODEL = os.getenv("COPILOT_MODEL", "gpt-5")


async def main() -> None:
    agent = from_copilot(SessionConfig(model=MODEL))
    await agent.run_async()


if __name__ == "__main__":
    asyncio.run(main())
