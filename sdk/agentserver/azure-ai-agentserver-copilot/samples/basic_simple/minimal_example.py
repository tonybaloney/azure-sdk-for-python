# Copyright (c) Microsoft. All rights reserved.

import asyncio

from azure.ai.agentserver.copilot import from_copilot


async def main() -> None:
    """Start an agent server backed by GitHub Copilot.

    The server exposes ``POST /responses`` which forwards prompts to the
    Copilot SDK, supporting both streaming and non-streaming modes.
    """
    agent = from_copilot({"model": "gpt-5"})
    await agent.run_async()


if __name__ == "__main__":
    asyncio.run(main())
