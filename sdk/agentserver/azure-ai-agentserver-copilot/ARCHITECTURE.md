# Copilot SDK Adapter — Architecture Overview

This document provides a high-level technical overview of the GitHub Copilot
SDK adapter for Microsoft Foundry Agent Server.  For detailed streaming protocol
implementation, see [DESIGN.md](DESIGN.md).

## Executive Summary

The Copilot adapter enables GitHub Copilot's AI capabilities — including tool
execution, multi-turn conversations, and code understanding — to be deployed
as a **hosted agent** on Microsoft Foundry.  Clients interact using the
standard OpenAI Responses API; the adapter translates between the two
protocols transparently.

```mermaid
flowchart TB
    subgraph foundry["Microsoft Foundry"]
        subgraph container["Hosted Agent Container"]
            adapter["CopilotAdapter<br/>(FoundryCBAgent)"]
            copilot["CopilotClient<br/>(SDK session)"]
            acl["ToolAcl<br/>(permission gate)"]
            converter["ResponseConverter<br/>(→ RAPI)"]
            adapter --> copilot
            adapter --> acl
            adapter --> converter
        end
        clients["Clients<br/>(OpenAI SDK)"] <-->|"HTTP / SSE"| adapter
        models[("Foundry Models /<br/>GitHub API")]
        copilot --> models
    end
```

---

## Key Components

### 1. CopilotAdapter

The central orchestrator that bridges the OpenAI Responses API to the Copilot
SDK.  It:

- Receives RAPI requests (JSON or streaming)
- Converts them to Copilot `MessageOptions`
- Manages Copilot sessions per conversation
- Evaluates tool permission requests against the ACL
- Converts Copilot events back to RAPI streaming events
- Emits OpenTelemetry traces for observability

```python
from azure.ai.agentserver.copilot import from_copilot

agent = from_copilot(acl_path="/app/tools_acl.yaml")
await agent.run_async()  # starts HTTP server on :8088
```

### 2. Tool Access Control List (ToolAcl)

A YAML-based security layer that gates every tool invocation before execution.
When Copilot wants to run a shell command, read a file, fetch a URL, or invoke
an MCP tool, the adapter consults the ACL rules **before** allowing it.

**Rule evaluation:**
1. Rules are checked in declaration order
2. First matching rule's action (`allow` / `deny`) is applied
3. If no rule matches, `default_action` applies (default: `deny`)

**Example ACL (tools_acl.yaml):**

```yaml
version: "1"
default_action: deny

rules:
  # Block dangerous shell commands
  - kind: shell
    action: deny
    when:
      command: "\\brm\\b|\\bsudo\\b|\\bchmod\\b"

  # Allow safe shell commands
  - kind: shell
    action: allow
    when:
      command: "^(ls|cat|head|tail|grep|find|echo|pwd|python)"

  # Allow reading only safe directories
  - kind: read
    action: allow
    when:
      path: "^(/tmp/|/home/app/|/workspace/)"

  # Block URL fetches to untrusted domains
  - kind: url
    action: allow
    when:
      url: "^https://(.*\\.)?(microsoft|github|python)\\.com/"
```

**Permission request kinds:**

| Kind | Description | Key fields |
|------|-------------|------------|
| `shell` | Execute shell commands | `fullCommandText`, `commands[]` |
| `read` | Read files or directories | `path` |
| `write` | Write or modify files | `fileName`, `diff` |
| `url` | Fetch HTTP/HTTPS URLs | `url` |
| `mcp` | Invoke MCP server tools | `toolName`, `serverName` |
| `custom-tool` | Custom tool invocation | `toolName` |

### 3. Session Management

The adapter maintains **persistent Copilot sessions** for multi-turn
conversations.  This preserves context (memory, tool state, conversation
history) across multiple requests.

```mermaid
sequenceDiagram
    participant Client
    participant Adapter
    participant SessionCache
    participant CopilotSDK

    Client->>Adapter: Request 1 (conversation_id: "abc123")
    Adapter->>SessionCache: lookup("abc123")
    SessionCache-->>Adapter: not found
    Adapter->>CopilotSDK: create_session()
    Adapter->>SessionCache: cache("abc123", session)
    Adapter-->>Client: Response

    Client->>Adapter: Request 2 (conversation_id: "abc123")
    Adapter->>SessionCache: lookup("abc123")
    SessionCache-->>Adapter: cached session
    Note over Adapter: Context preserved
    Adapter-->>Client: Response

    Client->>Adapter: Request 3 (conversation_id: "xyz789")
    Adapter->>SessionCache: lookup("xyz789")
    SessionCache-->>Adapter: not found
    Adapter->>CopilotSDK: create_session()
    Adapter->>SessionCache: cache("xyz789", session)
    Adapter-->>Client: Response
```

Session mapping is keyed by the `conversation_id` from the RAPI request.
Sessions without a conversation ID are ephemeral (not cached).

### 4. Response Conversion

The `CopilotStreamingResponseConverter` translates Copilot SDK events into
RAPI Server-Sent Events.  The mapping follows strict ordering guarantees
from the Copilot SDK:

| Copilot Event | Handling | RAPI Events |
|---------------|----------|-------------|
| `ASSISTANT_TURN_START` | Converted | `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added` |
| `ASSISTANT_MESSAGE_DELTA` | Converted | `response.output_text.delta` |
| `ASSISTANT_USAGE` | Stored | *(enriches `response.completed` and OTel span)* |
| `ASSISTANT_MESSAGE` | Deferred | *(triggers done-event chain at `TURN_END`)* |
| `ASSISTANT_TURN_END` | Converted | `response.output_text.done`, `response.content_part.done`, `response.output_item.done`, `response.completed` |
| `SESSION_ERROR` | Converted | `response.failed` |
| `SESSION_IDLE` | Safety net | Forces `response.completed` if not already emitted |
| `SESSION_WARNING` | Logged | *(WARNING-level log with `warning_type` and message)* |
| `ASSISTANT_REASONING` | Logged | *(DEBUG-level log of full reasoning text)* |
| `ASSISTANT_REASONING_DELTA` | Logged | *(DEBUG-level log of streaming reasoning chunks)* |
| `ASSISTANT_INTENT` | Logged | *(DEBUG-level log of classified intent)* |
| `TOOL_EXECUTION_START` | OTel span | Opens `tools/call` child span |
| `TOOL_EXECUTION_COMPLETE` | OTel span | Closes `tools/call` child span |
| `TOOL_EXECUTION_PROGRESS` | Logged | *(DEBUG-level — no RAPI equivalent)* |
| `TOOL_EXECUTION_PARTIAL_RESULT` | Logged | *(DEBUG-level — no RAPI equivalent)* |
| `TOOL_USER_REQUESTED` | Logged | *(DEBUG-level)* |
| `MCP_APPROVAL_REQUEST` | SDK | *(handled by permission system / ToolAcl)* |
| `SESSION_START` | Logged | *(DEBUG-level — session lifecycle)* |
| `SESSION_RESUME` | Logged | *(DEBUG-level)* |
| `SESSION_SHUTDOWN` | Logged | *(DEBUG-level — includes shutdown metrics)* |
| `SESSION_CONTEXT_CHANGED` | Logged | *(DEBUG-level)* |
| `SESSION_MODEL_CHANGE` | Logged | *(DEBUG-level)* |
| `SESSION_MODE_CHANGED` | Logged | *(DEBUG-level)* |
| `SESSION_TITLE_CHANGED` | Logged | *(DEBUG-level)* |
| `SESSION_PLAN_CHANGED` | Logged | *(DEBUG-level)* |
| `SESSION_TRUNCATION` | Logged | *(DEBUG-level — token/message stats)* |
| `SESSION_COMPACTION_START` | Logged | *(DEBUG-level)* |
| `SESSION_COMPACTION_COMPLETE` | Logged | *(DEBUG-level — compaction token stats)* |
| `SESSION_SNAPSHOT_REWIND` | Logged | *(DEBUG-level)* |
| `SESSION_HANDOFF` | Logged | *(DEBUG-level)* |
| `SESSION_WORKSPACE_FILE_CHANGED` | Logged | *(DEBUG-level)* |
| `SESSION_TASK_COMPLETE` | Logged | *(DEBUG-level)* |
| `SESSION_USAGE_INFO` | Logged | *(DEBUG-level)* |
| `SUBAGENT_SELECTED` | Logged | *(DEBUG-level — no RAPI equivalent)* |
| `SUBAGENT_STARTED` | Logged | *(DEBUG-level)* |
| `SUBAGENT_COMPLETED` | Logged | *(DEBUG-level)* |
| `SUBAGENT_FAILED` | Logged | *(DEBUG-level)* |
| `SUBAGENT_DESELECTED` | Logged | *(DEBUG-level)* |
| `HOOK_START` / `HOOK_END` | Logged | *(DEBUG-level — no RAPI equivalent)* |
| `SKILL_INVOKED` | Logged | *(DEBUG-level — no RAPI equivalent)* |
| `SYSTEM_MESSAGE` / `USER_MESSAGE` | Logged | *(DEBUG-level)* |
| `PENDING_MESSAGES_MODIFIED` | Logged | *(DEBUG-level)* |
| `ABORT` / `UNKNOWN` | Logged | *(DEBUG-level)* |

---

## Deployment Architecture

### Container Build

The adapter is packaged as a Docker container for deployment to Azure
Container Apps via the Foundry Agent Service:

```dockerfile
FROM python:3.12-slim

# Requires Python >=3.11 (SDK 0.1.29+ dependency)
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY . package/
RUN pip install ./package/ azure-identity "github-copilot-sdk==0.1.29"

COPY samples/hosted_agent/main.py main.py
COPY samples/hosted_agent/tools_acl.yaml tools_acl.yaml

EXPOSE 8088
CMD ["python", "main.py"]
```

**Build and push:**

```bash
docker build --platform linux/amd64 -t <acr>.azurecr.io/copilot-agent:v1 .
docker push <acr>.azurecr.io/copilot-agent:v1
```

### Hosted Agent Registration

```bash
az cognitiveservices agent create \
  --account-name <resource> \
  --project-name <project> \
  --name copilot-hosted-agent \
  --image <acr>.azurecr.io/copilot-agent:v1 \
  --env "COPILOT_MODEL=gpt-5.1-chat" \
       "AZURE_AI_FOUNDRY_RESOURCE_URL=https://<resource>.cognitiveservices.azure.com"
```

### Authentication

The adapter supports two authentication modes for accessing models:

**1. Managed Identity (production)**

When `AZURE_AI_FOUNDRY_RESOURCE_URL` is set without an API key, the adapter
uses `DefaultAzureCredential` to obtain tokens.  The container's managed
identity must have `Cognitive Services OpenAI User` role on the resource.

```mermaid
flowchart LR
    container["Container"] --> mi["Managed Identity"]
    mi --> aad["Azure AD"]
    aad --> token["Bearer Token"]
    token --> aoai["Azure OpenAI Endpoint"]
```

**2. API Key (development)**

For local development, set `AZURE_AI_FOUNDRY_API_KEY` with a static key:

```bash
export AZURE_AI_FOUNDRY_RESOURCE_URL="https://myresource.cognitiveservices.azure.com"
export AZURE_AI_FOUNDRY_API_KEY="your-key-here"
python main.py
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_AI_FOUNDRY_RESOURCE_URL` | For BYOK | Microsoft Foundry endpoint |
| `AZURE_AI_FOUNDRY_API_KEY` | No | Static API key (dev only) |
| `COPILOT_MODEL` | No | Model deployment name (`gpt-5.1-chat`) |
| `TOOL_ACL_PATH` | Recommended | Path to YAML ACL file |

---

## Observability

### OpenTelemetry Tracing

The adapter emits OTel spans following
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

**`invoke_agent {agent_name}` span** (kind: `CLIENT`) — wraps the entire streaming session
per the [GenAI agent span semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/#invoke-agent-span):
- `gen_ai.operation.name`: `"invoke_agent"`
- `gen_ai.agent.name`: agent identifier
- `gen_ai.request.model`: requested model
- `gen_ai.response.model`: actual model used
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`

**`tools/call {tool_name}` child spans** (kind: `CLIENT`) — one per tool execution,
following [MCP semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/):
- `mcp.method.name`: `"tools/call"`
- `gen_ai.tool.name`: tool being executed
- `gen_ai.tool.call.id`: unique call ID
- `gen_ai.tool.call.arguments`: JSON arguments
- `gen_ai.tool.call.result`: truncated result

### Logging

All Copilot SDK events are logged at INFO level with event type and content
length.  Full payloads are available at DEBUG level.  Session errors are
logged at WARNING level with full details.

```
2026-02-22 12:34:56 INFO copilot_adapter: Copilot event #001: ASSISTANT_TURN_START
2026-02-22 12:34:57 INFO copilot_adapter: Copilot event #002: ASSISTANT_MESSAGE_DELTA content_len=42
2026-02-22 12:34:57 INFO copilot_adapter: ACL allowed tool request: kind=shell
2026-02-22 12:34:58 INFO copilot_adapter: Copilot event #003: TOOL_EXECUTION_START
```

---

## Security Considerations

### Tool ACL Best Practices

1. **Default deny** — Always use `default_action: deny` in production
2. **Allowlist shell commands** — Explicitly list permitted commands
3. **Restrict paths** — Limit `read`/`write` to specific directories
4. **Validate URLs** — Only allow trusted domains for `url` fetches
5. **Audit logs** — Monitor ACL decisions in production logs

### Container Security

- Run as non-root user in production
- Use read-only filesystem where possible
- Limit container capabilities
- Scan images for vulnerabilities before deployment

---

## RAPI Proxy Gaps — Copilot SDK Features Without RAPI Equivalents

This section documents Copilot SDK capabilities that **cannot be expressed**
through the RAPI (OpenAI Responses API) streaming protocol.  These gaps are
relevant when evaluating whether an alternative specification like
[A2A (Agent-to-Agent)](https://google.github.io/A2A/) would better represent
Copilot agent capabilities.

### 1. Tool Execution Visibility

**Copilot SDK events:**
- `TOOL_EXECUTION_START` — tool name, arguments, call ID, MCP server/tool names
- `TOOL_EXECUTION_PROGRESS` — progress messages during long-running tools
- `TOOL_EXECUTION_PARTIAL_RESULT` — intermediate outputs from tools
- `TOOL_EXECUTION_COMPLETE` — final result, detailed content
- `TOOL_USER_REQUESTED` — user-initiated tool invocation

**RAPI equivalent:** None for hosted agents.  RAPI has `function_call` events
for *requesting* the client execute a tool, but no events for server-side tool
execution.  The adapter currently logs these events for OTel tracing but
clients have zero visibility into what tools ran, when, or what they returned.

**Impact:** Clients cannot show "Searching files…" or "Running shell command…"
progress indicators; cannot display tool outputs; cannot audit which tools
were invoked on their behalf.

### 2. Sub-Agent / Delegation Events

**Copilot SDK events:**
- `SUBAGENT_SELECTED` — which sub-agent was chosen
- `SUBAGENT_STARTED` — sub-agent invocation began
- `SUBAGENT_COMPLETED` — sub-agent finished successfully
- `SUBAGENT_FAILED` — sub-agent execution failed

**RAPI equivalent:** None.  RAPI assumes a single-agent response model with
no concept of agent delegation or composition.

**Impact:** In Copilot's agentic mode, work is often delegated to specialised
sub-agents (e.g. code search, memory retrieval).  Clients cannot observe this
delegation, show which agent is responding, or understand failure boundaries.

### 3. Session Lifecycle & Context Events

**Copilot SDK events:**
- `SESSION_START`, `SESSION_RESUME`, `SESSION_SHUTDOWN`
- `SESSION_CONTEXT_CHANGED` — workspace, files, or context updated
- `SESSION_MODEL_CHANGE` — model switched (e.g. GPT-4 → o3)
- `SESSION_MODE_CHANGED` — mode switched (ask/edit/agent)
- `SESSION_TITLE_CHANGED` — conversation title updated
- `SESSION_PLAN_CHANGED` — execution plan modified
- `SESSION_TRUNCATION`, `SESSION_COMPACTION_*` — context window management
- `SESSION_SNAPSHOT_REWIND` — state rollback
- `SESSION_HANDOFF` — handoff to another system

**RAPI equivalent:** None.  RAPI treats each request as stateless; there's no
concept of a persistent session with lifecycle events.

**Impact:** Clients cannot react to model changes, understand context
truncation, or observe session state transitions.

### 4. Reasoning & Intent Transparency

**Copilot SDK events:**
- `ASSISTANT_REASONING` — full reasoning text
- `ASSISTANT_REASONING_DELTA` — streaming reasoning chunks
- `ASSISTANT_INTENT` — classified user intent

**RAPI equivalent:** Partial.  RAPI has `response.reasoning.delta/done` and
`response.reasoning_summary.*` events, but these are model-generated summaries,
not the raw reasoning trace.  Intent classification has no equivalent.

**Impact:** Clients cannot display chain-of-thought reasoning or understand
how the agent interpreted the user's request.

### 5. Rich Content Types

**Copilot SDK content types:**
- `text`, `image`, `audio`, `terminal`, `resource`, `resource_link`
- Tool results can include multiple content items with different types

**RAPI equivalent:** Only `text` and `image` (via code interpreter outputs).
Terminal output, file resources, and resource links have no RAPI representation.

**Impact:** Copilot can return rich, structured tool outputs (e.g. terminal
sessions with ANSI, file downloads, resource links) that collapse to plain
text strings when proxied through RAPI.

### 6. Approval Workflows

**Copilot SDK events:**
- `MCP_APPROVAL_REQUEST` — tool requests user approval before execution
- ACL system can defer approval to client

**RAPI equivalent:** None.  RAPI's `function_call` expects the *client* to
execute tools, so approval is implicit.  There's no "server wants to run this
tool — do you approve?" flow.

**Impact:** The adapter's tool ACL system currently auto-approves or rejects
tools.  Interactive approval (show user what tool wants to run, let them
decide) isn't possible through RAPI.

### 7. Hooks & Skills

**Copilot SDK events:**
- `HOOK_START`, `HOOK_END` — lifecycle hooks with input/output
- `SKILL_INVOKED` — skill system invocation

**RAPI equivalent:** None.

**Impact:** Extension points and skill invocations are invisible to clients.

### 8. Multi-Turn State & Usage

**Copilot SDK data:**
- `cache_read_tokens`, `cache_write_tokens` — KV cache statistics
- `cost` — dollar cost of the request
- `model` — actual model used (may differ from requested)
- Per-turn usage breakdown

**RAPI equivalent:** Partial.  RAPI usage only includes `input_tokens`,
`output_tokens`, `total_tokens`.  Cache stats, cost, and per-turn breakdown
are lost.

**Impact:** Clients cannot accurately track costs or understand caching
efficiency.

### 9. Platform Infrastructure Bug (Streaming)

**Observed behaviour:** The RAPI gateway (Foundry + Container Apps
ingress) drops the final events of every SSE stream.  Events
`response.output_text.done` through `response.completed` and `data: [DONE]`
are consistently truncated.  This affects *all* hosted agent adapters
(Copilot, LangGraph, AgentFramework) identically.

**Workaround status:** No adapter-side workaround found.  Tested sleep delays
up to 2 seconds with no effect.  Non-streaming mode works correctly.

**Impact:** Streaming responses never receive a `response.completed` event,
preventing clients from knowing when the response is truly finished.

---

## A2A Comparison Notes

The [A2A (Agent-to-Agent) protocol](https://google.github.io/A2A/) addresses
several of these gaps:

| Gap | RAPI | A2A |
|-----|------|-----|
| Tool execution visibility | ❌ | ✅ `tool/progress`, `tool/result` messages |
| Sub-agent delegation | ❌ | ✅ `agent/handoff`, nested agent contexts |
| Session lifecycle | ❌ | ✅ Session management is core to spec |
| Rich content types | Partial | ✅ MIME-typed content parts |
| Approval workflows | ❌ | ✅ `approval/request`, `approval/response` |
| Streaming completion | ✅ | ✅ But different wire format (JSON-RPC) |

A2A's JSON-RPC-based protocol is more verbose but naturally supports the
bidirectional, stateful, multi-agent communication patterns that Copilot
exhibits.  RAPI was designed for single-turn LLM completions and has been
extended for tool calling, but its unidirectional SSE model struggles with
complex agent workflows.

**Consider A2A if:**
- Tool execution transparency is required
- Sub-agent delegation needs to be visible
- Interactive approval workflows are needed
- Rich content types beyond text/image are important
- Session-level events (context changes, mode switches) matter

**Consider RAPI if:**
- OpenAI client compatibility is paramount
- Simple text-in/text-out flows dominate
- Tool execution can remain opaque
- Ecosystem tooling (SDKs, integrations) is a priority
