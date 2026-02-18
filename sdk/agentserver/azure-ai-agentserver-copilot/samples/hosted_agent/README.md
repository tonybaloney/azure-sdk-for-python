# Hosted Agent – GitHub Copilot SDK

This sample shows how to deploy a GitHub Copilot-backed agent as a
**hosted agent** on Microsoft Foundry Agent Service, following the
[hosted agents tutorial](https://learn.microsoft.com/azure/ai-foundry/agents/concepts/hosted-agents).

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                  Foundry Agent Service                 │
│                                                       │
│  ┌─────────────┐   HTTP    ┌───────────────────────┐  │
│  │  Responses   │ ──────►  │  CopilotAdapter       │  │
│  │  API         │ ◄──────  │  (FoundryCBAgent)     │  │
│  └─────────────┘           │                       │  │
│                            │  ┌─────────────────┐  │  │
│                            │  │ CopilotClient   │  │  │
│                            │  │ (SDK session)   │  │  │
│                            │  └─────────────────┘  │  │
│                            └───────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

The adapter converts Foundry Responses API requests into Copilot SDK
`MessageOptions`, runs them through a Copilot session, and converts the
results back into OpenAI-format responses (streaming or non-streaming).

## Prerequisites

* Python 3.10+
* [GitHub Copilot CLI](https://github.com/github/copilot-sdk) installed and in `$PATH`
* Valid GitHub authentication (`gh auth login`)

## Run locally

```bash
pip install -r requirements.txt
python main.py
```

The server starts on `http://0.0.0.0:8088`.

### Test with REST calls

**Non-streaming:**

```bash
curl -sS \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d '{"input":"What is the capital of France?","stream":false}'
```

**Streaming (SSE):**

```bash
curl -N \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d '{"input":"Explain Python decorators briefly.","stream":true}'
```

**Liveness / readiness probes:**

```bash
curl http://localhost:8088/liveness
curl http://localhost:8088/readiness
```

## Build Docker image

```bash
docker build --platform linux/amd64 -t copilot-hosted-agent:latest .
```

## Deploy to Foundry Agent Service

Follow the [hosted agents tutorial](https://learn.microsoft.com/azure/ai-foundry/agents/concepts/hosted-agents)
to push the image to Azure Container Registry and register it as a hosted agent:

```python
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    ImageBasedHostedAgentDefinition,
    ProtocolVersionRecord,
    AgentProtocol,
)
from azure.identity import DefaultAzureCredential

client = AIProjectClient(
    endpoint="https://<project>.services.ai.azure.com/api/projects/<name>",
    credential=DefaultAzureCredential(),
)

agent = client.agents.create_version(
    agent_name="copilot-hosted-agent",
    definition=ImageBasedHostedAgentDefinition(
        container_protocol_versions=[
            ProtocolVersionRecord(protocol=AgentProtocol.RESPONSES, version="v1")
        ],
        cpu="1",
        memory="2Gi",
        image="<acr>.azurecr.io/copilot-hosted-agent:latest",
        environment_variables={
            "COPILOT_MODEL": "gpt-5",
        },
    ),
)
print(f"Agent created: {agent.name} v{agent.version}")
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COPILOT_MODEL` | `gpt-5` | Model for Copilot sessions |
| `COPILOT_CLI_PATH` | *(auto)* | Override path to `copilot` CLI |
