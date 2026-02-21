# Hosted Agent – GitHub Copilot SDK

This sample shows how to deploy a GitHub Copilot-backed agent as a
**hosted agent** on Microsoft Foundry Agent Service, following the
[hosted agents tutorial](https://learn.microsoft.com/azure/ai-foundry/agents/concepts/hosted-agents).

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    Foundry Agent Service                   │
│                                                           │
│  ┌─────────────┐   HTTP    ┌───────────────────────────┐  │
│  │  Responses   │ ──────►  │  CopilotAdapter           │  │
│  │  API         │ ◄──────  │  (FoundryCBAgent)         │  │
│  └─────────────┘           │                           │  │
│                            │  ┌─────────────────────┐  │  │
│                            │  │  CopilotClient      │  │  │
│                            │  │  (SDK session)      │  │  │
│                            │  └─────────────────────┘  │  │
│                            │                           │  │
│                            │  ┌─────────────────────┐  │  │
│                            │  │  ToolAcl            │  │  │
│                            │  │  (tools_acl.yaml)   │  │  │
│                            │  └─────────────────────┘  │  │
│                            └───────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

The adapter converts Foundry Responses API requests into Copilot SDK
`MessageOptions`, runs them through a Copilot session, and converts the
results back into OpenAI-format responses (streaming or non-streaming).

Every tool permission request emitted by the Copilot SDK is evaluated
against the `ToolAcl` rules before the tool executes.

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
            "TOOL_ACL_PATH": "/app/tools_acl.yaml",
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
| `TOOL_ACL_PATH` | `tools_acl.yaml` (same dir) | Path to YAML tool ACL file; omit to approve all tools |

---

## Tool Access Control List (ACL)

The `ToolAcl` system lets you allow or deny tool invocations before they
execute, using a YAML rule file.  The file bundled in this sample
(`tools_acl.yaml`) is a good starting point — edit it to suit your
security requirements.

### How it works

1. The Copilot SDK emits a **permission request** for each tool call.
2. `CopilotAdapter` passes the request to `ToolAcl.evaluate()`.
3. Rules are checked **in declaration order**; the **first matching rule
   wins**.
4. If no rule matches, `default_action` is applied (`"deny"` by default).

### YAML schema

```yaml
version: "1"                     # required; only "1" is supported
default_action: deny             # optional; "allow" or "deny"  (default: "deny")

rules:
  - kind:   shell                # one of: shell, read, write, url, mcp
    action: deny                 # "allow" or "deny"
    when:                        # optional; omit to match ANY request of this kind
      command: "regex"           # shell  — full command text
      path:    "regex"           # read / write — file or directory path
      url:     "regex"           # url   — the full URL
      tool:    "regex"           # mcp   — MCP tool name
      server:  "regex"           # mcp   — MCP server name
```

All `when` conditions within a rule are **ANDed**: every specified pattern
must match for the rule to fire.  Patterns are Python `re.search` patterns
(not anchored; add `^`/`$` when needed).

### Permission-request fields by kind

| `kind` | Fields available | `when` key |
|--------|-----------------|------------|
| `shell` | `fullCommandText` (str), `commands[]`, `possiblePaths`, `possibleUrls`, `hasWriteFileRedirection` | `command` |
| `read` | `path` (str) | `path` |
| `write` | `fileName` (str), `diff`, `newFileContents` | `path` |
| `url` | `url` (str) | `url` |
| `mcp` | `toolName` (str), `serverName` (str) | `tool`, `server` |

### Example rules

**Deny `rm` and `sudo`:**
```yaml
- kind: shell
  action: deny
  when:
    command: "\\brm\\b|\\bsudo\\b"
```

**Allow reading only under `/tmp` and the project directory:**
```yaml
- kind: read
  action: allow
  when:
    path: "^(/tmp/|/home/app/myproject/)"

- kind: read
  action: deny          # catch-all deny for anything else
```

**Allow only safe domains for URL fetches:**
```yaml
- kind: url
  action: allow
  when:
    url: "^https://(.*\\.)?microsoft\\.com/"

- kind: url
  action: deny
```

**Block dangerous MCP tools:**
```yaml
- kind: mcp
  action: deny
  when:
    tool: "^(write_|create_|delete_)"
```

**Permissive development posture (allow everything by default):**
```yaml
version: "1"
default_action: allow
rules: []
```

### Programmatic usage

```python
from azure.ai.agentserver.copilot import from_copilot, ToolAcl

# Load from file
agent = from_copilot(acl_path="/path/to/tools_acl.yaml")

# Or build an ACL inline and pass it directly
from azure.ai.agentserver.copilot.tool_acl import ToolAcl

acl = ToolAcl.from_file("tools_acl.yaml")
acl.is_allowed({"kind": "shell", "fullCommandText": "echo hello"})  # True or False

from azure.ai.agentserver.copilot import CopilotAdapter
adapter = CopilotAdapter(acl=acl)
```

The `TOOL_ACL_PATH` environment variable is checked automatically at
startup; you do not need to pass `acl_path` explicitly when deploying via
Docker / Foundry Agent Service.
