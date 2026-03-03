# Skill: Deploy a GitHub Copilot Hosted Agent on Microsoft Foundry

## What This Project Is

This repository contains `azure-ai-agentserver-copilot`, a Python adapter that lets you run a [GitHub Copilot SDK](https://pypi.org/project/github-copilot-sdk/) session as a **hosted agent** on Microsoft Foundry Agent Service. It exposes the standard OpenAI Responses API (RAPI) over HTTP, translating between the Copilot SDK's event protocol and RAPI streaming/non-streaming formats.

The adapter handles:

- Converting OpenAI-format requests into Copilot SDK `MessageOptions`
- Running multi-turn Copilot sessions with persistent conversation state
- Evaluating tool permission requests against a YAML-based Access Control List (ACL)
- Converting Copilot SDK events back into RAPI Server-Sent Events (streaming) or JSON (non-streaming)
- OpenTelemetry tracing for observability

The package is **not published on PyPI**. It must be installed from source by copying the package directory into the Docker image.

## Tutorials and References

- [Hosted Agents concept overview](https://learn.microsoft.com/azure/ai-foundry/agents/concepts/hosted-agents) — explains what hosted agents are and how Foundry runs them
- [Azure AI Foundry Agent Service documentation](https://learn.microsoft.com/azure/ai-foundry/agents/) — the broader Foundry agents docs
- [GitHub Copilot SDK (PyPI)](https://pypi.org/project/github-copilot-sdk/) — the upstream Copilot SDK package
- [ARCHITECTURE.md](../ARCHITECTURE.md) — detailed architecture of this adapter
- [DESIGN.md](../DESIGN.md) — streaming protocol design and event mapping

## Prerequisites

- Python 3.11+
- Docker (for building the container image)
- An Azure subscription with:
  - An Azure AI Foundry project
  - An Azure Container Registry (ACR)
  - A deployed model (e.g. `gpt-4.1`, `gpt-5`) accessible from the Foundry resource
- Azure CLI (`az`) installed and authenticated

## Setting Up Azure Resources (if you don't have them yet)

If you don't already have a Microsoft Foundry project and a deployed model, follow these steps using the Azure CLI. This uses the **new Foundry** model (not Foundry Classic) — there is no hub to create.

### 1. Create a resource group

Pick a region that supports hosted agents and the model you want. For coding agents, **`gpt-5.3-codex`** is recommended — it is only available in a few regions. **`eastus2`** (US East 2) is a reliable choice.

```bash
az group create --name <resource-group> --location eastus2
```

### 2. Create a Foundry resource (Cognitive Services account)

In the new Foundry, the **Cognitive Services account** _is_ the Foundry resource. A default project is created automatically with it.

```bash
az cognitiveservices account create \
  --name <account-name> \
  --resource-group <resource-group> \
  --location eastus2 \
  --kind AIServices \
  --sku S0
```

### 3. Create a project (if you need a non-default one)

The account above automatically has a default project. If you need additional projects, the easiest way is through the [Microsoft Foundry portal](https://ai.azure.com) or the Azure AI Projects SDK. For most hosted agent scenarios, the default project is sufficient.

To find your default project name:

```bash
az cognitiveservices account show \
  --name <account-name> \
  --resource-group <resource-group> \
  --query "properties.endpoints" -o json
```

Your project endpoint will look like: `https://<account-name>.services.ai.azure.com/api/projects/<project-name>`

### 4. Deploy a model

For coding-focused agents, deploy `gpt-5.3-codex`. This model is only available in a few regions — `eastus2` is a safe pick.

```bash
az cognitiveservices account deployment create \
  --name <account-name> \
  --resource-group <resource-group> \
  --deployment-name gpt-5.3-codex \
  --model-name gpt-5.3-codex \
  --model-version "2026-02-14" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name GlobalStandard
```

> **Tip:** You can list available models in your region with:
> ```bash
> az cognitiveservices model list \
>   --resource-group <resource-group> \
>   --name <account-name> \
>   -o table
> ```

### 5. Create an Azure Container Registry

```bash
az acr create \
  --name <acr-name> \
  --resource-group <resource-group> \
  --location eastus2 \
  --sku Basic \
  --admin-enabled true
```

### 6. Create the account-level capability host

Hosted agents require a **capability host** with public hosting enabled on the Foundry account. This is a one-time setup step per account.

You will need your subscription ID, resource group name, and account name. Use `az rest` to create it:

```bash
az rest --method put \
  --url "https://management.azure.com/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<account-name>/capabilityHosts/accountcaphost?api-version=2025-10-01-preview" \
  --headers "content-type=application/json" \
  --body '{
    "properties": {
      "capabilityHostKind": "Agents",
      "enablePublicHostingEnvironment": true
    }
  }'
```

> **Note:** Updating capability hosts isn't supported. If you already have one and need to change it, delete it and recreate it with `enablePublicHostingEnvironment` set to `true`.

### 7. Get the Foundry resource URL

This is the value you'll set as `AZURE_AI_FOUNDRY_RESOURCE_URL`:

```bash
az cognitiveservices account show \
  --name <account-name> \
  --resource-group <resource-group> \
  --query properties.endpoint -o tsv
```

The output will look like `https://<account-name>.cognitiveservices.azure.com/`. Use this as the `AZURE_AI_FOUNDRY_RESOURCE_URL` and set `COPILOT_MODEL` to `gpt-5.3-codex` (or whatever deployment name you chose).

## GitHub PAT Is NOT Required

This project does **not** require a GitHub Personal Access Token (PAT). When deployed as a hosted agent with `AZURE_AI_FOUNDRY_RESOURCE_URL` set, the adapter uses **Azure AI Foundry models via BYOK (Bring Your Own Key)** — it talks directly to the Foundry model endpoint, not to GitHub's API. Authentication is handled via Managed Identity (production) or API key (development). See the [Authentication](#authentication-managed-identity) section below.

The `azure.yaml` file in the repo root contains a `get-github-token.mjs` hook, but that is only used for local development with `azd app run` and is not needed when deploying as a hosted agent container.

## Creating a Hosted Agent Image

A hosted agent is a Docker container that exposes an HTTP server on **port 8088** implementing the Responses API. The Foundry Agent Service runs this container and routes requests to it.

### Project Structure for a Hosted Agent

Your hosted agent needs three files:

1. **`main.py`** — the entry point that starts the agent server
2. **`requirements.txt`** — Python dependencies (for local development)
3. **`Dockerfile`** — container image definition
4. **`tools_acl.yaml`** (recommended) — tool access control rules

You can optionally include any additional Python modules, data files, or configuration alongside these.

### Step 1: Create `main.py`

This is the minimal entry point. It uses `from_copilot()` to create the adapter and starts the HTTP server:

```python
import asyncio
import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from azure.ai.agentserver.copilot import from_copilot

# Resolve ACL file: TOOL_ACL_PATH env var takes priority,
# otherwise fall back to tools_acl.yaml next to this script.
_HERE = Path(__file__).parent
_DEFAULT_ACL = _HERE / "tools_acl.yaml"
_acl_path = os.getenv("TOOL_ACL_PATH") or (
    str(_DEFAULT_ACL) if _DEFAULT_ACL.exists() else None
)


async def main() -> None:
    agent = from_copilot(acl_path=_acl_path)
    await agent.run_async()


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Create `tools_acl.yaml`

The ACL controls which tools the Copilot agent can invoke. **Always use `default_action: deny` in production.** See the full schema in the [hosted_agent sample README](hosted_agent/README.md).

Minimal example:

```yaml
version: "1"
default_action: deny

rules:
  # Deny dangerous shell commands
  - kind: shell
    action: deny
    when:
      command: "\\brm\\b|\\bsudo\\b|\\bchmod\\b"

  # Allow safe shell commands
  - kind: shell
    action: allow
    when:
      command: "^(ls|cat|head|tail|grep|find|echo|pwd|python)"

  # Allow reading safe directories
  - kind: read
    action: allow
    when:
      path: "^(/tmp/|/home/app/)"

  # Allow only trusted URL domains
  - kind: url
    action: allow
    when:
      url: "^https://(.*\\.)?(microsoft|github)\\.com/"
```

### Step 3: Create the `Dockerfile`

Since `azure-ai-agentserver-copilot` is not published on PyPI, the Dockerfile copies the **entire package source** into the image and installs it with `pip install ./package/`:

```dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy the azure-ai-agentserver-copilot package source and install it.
# The package is not on PyPI, so we install from the local source tree.
COPY . package/
RUN pip install ./package/ azure-identity "github-copilot-sdk==0.1.29"

# Copy your agent entry point and ACL config
COPY samples/hosted_agent/main.py main.py
COPY samples/hosted_agent/tools_acl.yaml tools_acl.yaml

# Run as non-root for security
RUN useradd -m appuser
USER appuser

EXPOSE 8088
CMD ["python", "main.py"]
```

> **Note:** The `COPY . package/` line copies the entire repository root (which contains the `azure/ai/agentserver/copilot/` Python package and `pyproject.toml`) into the image. The `pip install ./package/` then installs it along with its dependencies (`azure-ai-agentserver-core`, `azure-identity`, `github-copilot-sdk`, `PyYAML`).

If you are creating a **new** agent project outside this repo, adjust the `COPY` paths so that the package source is available inside the image. For example:

```dockerfile
# If you have the package source at /path/to/azure-ai-agentserver-copilot/
COPY /path/to/azure-ai-agentserver-copilot/ package/
RUN pip install ./package/

# Copy your own agent files
COPY main.py main.py
COPY tools_acl.yaml tools_acl.yaml
```

## Bundling Custom Code Into the Image

You can include **any custom Python code** in the Docker image. This is useful for:

- Custom tool implementations
- Data files or configuration
- Helper libraries or utilities
- Pre-processing or post-processing logic

Simply `COPY` your code into the image and import it from `main.py`:

```dockerfile
# Copy custom modules
COPY my_custom_tools/ /app/my_custom_tools/
COPY config/ /app/config/
```

```python
# In main.py
from my_custom_tools import my_tool
```

### Loading Skills via the Copilot SDK

The Copilot SDK supports **skills** — specialized capabilities that can be loaded into a session. If you have skills defined as Python modules or YAML configurations, you can bundle them into the Docker image and they will be available to the Copilot SDK at runtime.

Skills are loaded by the Copilot SDK session when it starts. Any skill files or modules present in the container's filesystem and referenced by the session configuration will be picked up automatically. Include them in your Dockerfile:

```dockerfile
COPY my_skills/ /app/my_skills/
```

The Copilot SDK's session manager discovers and invokes skills as part of its agentic loop. Events like `SKILL_INVOKED` are emitted and logged by the adapter for observability.

## Building and Pushing the Docker Image

### Build for AMD64 (important!)

Foundry Agent Service runs containers on **linux/amd64** infrastructure. If you are building on an ARM64 machine (e.g. Apple Silicon Mac, ARM-based Linux), you **must** specify the target platform:

```bash
docker build --platform linux/amd64 -t <acr-name>.azurecr.io/<image-name>:v1 .
```

If you are already on an amd64 machine, the `--platform` flag is optional but harmless.

### Push to Azure Container Registry

1. **Log in to ACR:**

```bash
az acr login --name <acr-name>
```

2. **Push the image:**

```bash
docker push <acr-name>.azurecr.io/<image-name>:v1
```

### Full build-and-push example

```bash
# From the repository root directory
docker build --platform linux/amd64 \
  -t <acr-name>.azurecr.io/<image-name>:v1 \
  -f samples/hosted_agent/Dockerfile .

az acr login --name <acr-name>
docker push <acr-name>.azurecr.io/<image-name>:v1
```

### Building Remotely with ACR (No Local Docker Required)

If Docker Desktop is not running or you are on a machine without Docker, you can build and push the image entirely in the cloud using Azure Container Registry:

```bash
az acr build \
  --registry <acr-name> \
  --image <image-name>:v1 \
  --platform linux/amd64 \
  --file samples/hosted_agent/Dockerfile \
  .
```

This uploads the build context to ACR, builds the image remotely, and stores it directly in the registry — no local Docker daemon needed.

## Deploying the Hosted Agent to Foundry

> **IMPORTANT:** You **must** set the `AZURE_AI_FOUNDRY_RESOURCE_URL` environment variable when deploying to Foundry. Without it, the adapter cannot reach any model backend and requests will hang indefinitely. The container will start and pass health checks, but every inference request will time out because the Copilot SDK has no model endpoint to connect to.

### Register the Agent

Use the Azure AI Projects SDK to register the container image as a hosted agent and link it to a Foundry model.

The `environment_variables` dict **must** include `AZURE_AI_FOUNDRY_RESOURCE_URL` — this tells the adapter to use the Foundry model endpoint via Managed Identity instead of trying to reach GitHub's API (which is not accessible from hosted containers):

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
    agent_name="<agent-name>",
    definition=ImageBasedHostedAgentDefinition(
        container_protocol_versions=[
            ProtocolVersionRecord(
                protocol=AgentProtocol.RESPONSES, version="v1"
            )
        ],
        cpu="1",
        memory="2Gi",
        image="<acr-name>.azurecr.io/<image-name>:v1",
        environment_variables={
            "COPILOT_MODEL": "gpt-4.1",
            "TOOL_ACL_PATH": "/app/tools_acl.yaml",
            "AZURE_AI_FOUNDRY_RESOURCE_URL": "https://<resource>.cognitiveservices.azure.com",
        },
    ),
)
print(f"Agent created: {agent.name} v{agent.version}")
```

### Or Register via Azure CLI

```bash
az cognitiveservices agent create \
  --account-name <resource> \
  --project-name <project> \
  --name <agent-name> \
  --image <acr-name>.azurecr.io/<image-name>:v1 \
  --env "COPILOT_MODEL=gpt-4.1" \
       "AZURE_AI_FOUNDRY_RESOURCE_URL=https://<resource>.cognitiveservices.azure.com"
```

### Start the Agent Deployment

After registering the agent with `create_version()` or `az cognitiveservices agent create`, the container does **not** start automatically. You must explicitly start it:

```bash
az cognitiveservices agent start \
  --account-name <account-name> \
  --project-name <project-name> \
  --name <agent-name> \
  --agent-version <version-number>
```

The command returns immediately with a status of `Starting`. It may take a few minutes for the container to become fully ready. Check the status with:

```bash
az cognitiveservices agent show \
  --account-name <account-name> \
  --project-name <project-name> \
  --name <agent-name>
```

### Stop the Agent Deployment

To stop a running agent (e.g. to save costs or before redeploying):

```bash
az cognitiveservices agent stop \
  --account-name <account-name> \
  --project-name <project-name> \
  --name <agent-name>
```

### Update a Running Agent (Without Versioning)

To change properties like replica count or description without creating a new version:

```bash
az cognitiveservices agent update \
  --account-name <account-name> \
  --project-name <project-name> \
  --name <agent-name> \
  --min-replicas 1 \
  --max-replicas 3
```

### View Container Logs

To stream console logs from a running agent container for debugging:

```
GET https://<account-name>.services.ai.azure.com/api/projects/<project-name>/agents/<agent-name>/versions/<version>/containers/default:logstream?kind=console&api-version=2025-11-15-preview
```

You can also use `kind=system` for system-level logs.

### Delete an Agent

To delete a specific version:

```bash
az cognitiveservices agent delete \
  --account-name <account-name> \
  --project-name <project-name> \
  --name <agent-name> \
  --agent-version <version-number>
```

To delete the agent entirely (all versions):

```bash
az cognitiveservices agent delete \
  --account-name <account-name> \
  --project-name <project-name> \
  --name <agent-name>
```

### Linking to a Foundry Model

The agent connects to a Foundry model via the `AZURE_AI_FOUNDRY_RESOURCE_URL` and `COPILOT_MODEL` environment variables:

- **`AZURE_AI_FOUNDRY_RESOURCE_URL`** — the Foundry resource endpoint (e.g. `https://<resource>.cognitiveservices.azure.com`). The adapter normalizes this to append `/openai/v1/` if needed.
- **`COPILOT_MODEL`** — the model deployment name (e.g. `gpt-4.1`, `gpt-5`, `gpt-5.1-chat`). Defaults to `gpt-4.1` if not set.

## Authentication: Managed Identity

In production, the hosted agent authenticates to the backend Foundry model using **Azure Managed Identity**, not an API key.

When `AZURE_AI_FOUNDRY_RESOURCE_URL` is set **without** `AZURE_AI_FOUNDRY_API_KEY`, the adapter uses `DefaultAzureCredential` which resolves to the container's Managed Identity. This is the recommended approach for production deployments.

### Setup Requirements

1. The container's Managed Identity must have the **`Cognitive Services OpenAI User`** role on the Foundry resource.
2. Do **not** set `AZURE_AI_FOUNDRY_API_KEY` in production — let the adapter use Managed Identity.

### Auth Flow

```
Container → Managed Identity → Azure AD → Bearer Token → Foundry Model Endpoint
```

### Local Development (API Key)

For local development and testing, you can use a static API key instead:

```bash
export AZURE_AI_FOUNDRY_RESOURCE_URL="https://<resource>.cognitiveservices.azure.com"
export AZURE_AI_FOUNDRY_API_KEY="your-key-here"
export COPILOT_MODEL="gpt-4.1"
python main.py
```

This is convenient for debugging but should **never** be used in production.

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_AI_FOUNDRY_RESOURCE_URL` | **Yes** (hosted agents) | Foundry resource endpoint URL. **Required** — without this, the adapter has no model backend and requests will hang. |
| `AZURE_AI_FOUNDRY_API_KEY` | No | Static API key (local dev only; omit for Managed Identity) |
| `COPILOT_MODEL` | No | Model deployment name (default: `gpt-4.1`) |
| `TOOL_ACL_PATH` | Recommended | Path to YAML ACL file inside the container |

## Summary

To deploy a Copilot-backed hosted agent on Foundry:

1. Create `main.py`, `tools_acl.yaml`, and `Dockerfile` (see above or copy from `samples/hosted_agent/`)
2. Bundle any custom code, skills, or data files into the image
3. Build the Docker image for **linux/amd64**: `docker build --platform linux/amd64 -t <acr>.azurecr.io/<image-name>:v1 .`
4. Push to ACR: `docker push <acr>.azurecr.io/<image-name>:v1`
5. Register the agent in Foundry with `create_version()` or `az cognitiveservices agent create` — **include `AZURE_AI_FOUNDRY_RESOURCE_URL` in the environment variables**
6. **Start the agent** with `az cognitiveservices agent start` — the container does not start automatically after registration
7. Ensure the container's Managed Identity has `Cognitive Services OpenAI User` role on the Foundry resource
8. No GitHub PAT is needed — the agent authenticates to Foundry models via Managed Identity
```