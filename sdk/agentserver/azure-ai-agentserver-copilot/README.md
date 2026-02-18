# Azure AI Agent Server – GitHub Copilot SDK Adapter

This package provides an adapter that lets you run a
[GitHub Copilot SDK](https://pypi.org/project/github-copilot-sdk/) session as
an Azure AI Agent Server endpoint, in the same way that
`azure-ai-agentserver-langgraph` and `azure-ai-agentserver-agentframework`
wrap their respective frameworks.

## Getting Started

### Install

```bash
pip install azure-ai-agentserver-copilot
```

### Prerequisites

* Python 3.10+
* [GitHub Copilot CLI](https://github.com/github/copilot-sdk) installed and in `$PATH`
* Valid GitHub Copilot authentication (`gh auth login`)

### Usage

```python
import asyncio
from azure.ai.agentserver.copilot import from_copilot

async def main():
    agent = from_copilot({"model": "gpt-5"})
    await agent.run_async()

asyncio.run(main())
```

Once the server is running, send requests:

```bash
# Non-streaming
curl -sS -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d '{"input":"What is the capital of France?","stream":false}'

# Streaming
curl -N -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d '{"input":"Explain Python decorators briefly.","stream":true}'
```

## Key Concepts

The adapter converts incoming OpenAI-format requests into Copilot SDK
`MessageOptions`, runs them through a Copilot session, and converts the
resulting `SessionEvent` objects back into OpenAI-format responses.

Both streaming (Server-Sent Events) and non-streaming (JSON) modes are
supported.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CopilotClient` fails to start | Ensure the Copilot CLI is installed and `gh auth login` has been run. |
| Timeout errors | The default session timeout is 300 s. For long-running prompts, set a higher timeout. |

## Next Steps

See the [samples](./samples) directory for runnable examples.
