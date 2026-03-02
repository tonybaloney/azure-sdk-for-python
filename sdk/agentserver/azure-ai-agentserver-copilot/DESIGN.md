# Copilot Adapter — Streaming Design

This document describes how the Copilot adapter converts Copilot SDK session
events into the OpenAI Responses API (RAPI) streaming protocol.

## Overview

The adapter sits between the **Copilot SDK** (which drives an LLM session with
tool-calling capabilities) and the **RAPI streaming interface** that clients
consume.  Its job is to translate the Copilot event stream into the correct
sequence of RAPI Server-Sent Events so that any OpenAI-compatible client can
consume agent responses.

```
Client  ←──  RAPI SSE  ←──  Adapter  ←──  Copilot SDK session events
```

## Copilot SDK Event Ordering

The Copilot SDK emits events in a **strict, synchronous order**.  Events are
never reordered, never interleaved across turns, and never duplicated (the
adapter deduplicates reconnect artefacts at the ingestion layer).

### Single-turn sequence

A single LLM turn always produces this exact event order:

```
ASSISTANT_TURN_START
ASSISTANT_MESSAGE_DELTA  ×N      (streaming text chunks; 0 if no text)
ASSISTANT_USAGE                  (token counts — always before MESSAGE)
ASSISTANT_MESSAGE                (authoritative full text — always emitted)
ASSISTANT_TURN_END               (always emitted, even on error)
```

After all turns are finished, the session emits:

```
SESSION_IDLE                     (session is done processing)
```

### Multi-turn (tool-calling) sequence

When the agent calls tools, the session runs multiple turns.  Each turn follows
the same `TURN_START → … → TURN_END` pattern.  Tool-calling turns typically
contain no text content in `ASSISTANT_MESSAGE` — the message carries tool
request metadata instead.  Only the final turn produces user-visible text.

```
── Turn 1: tool-calling ──
ASSISTANT_TURN_START
ASSISTANT_USAGE
ASSISTANT_MESSAGE          (tool requests, no text content)
ASSISTANT_TURN_END

── Tool execution ──
TOOL_EXECUTION_START       (per tool call)
TOOL_EXECUTION_COMPLETE    (per tool call)

── Turn 2: final answer ──
ASSISTANT_TURN_START
ASSISTANT_MESSAGE_DELTA  ×N
ASSISTANT_USAGE
ASSISTANT_MESSAGE          (final text answer)
ASSISTANT_TURN_END

SESSION_IDLE
```

Additional events may appear between or alongside the core turn events:

- `ASSISTANT_REASONING` / `ASSISTANT_REASONING_DELTA` — chain-of-thought text
- `ASSISTANT_INTENT` — classified user intent
- `MCP_APPROVAL_REQUEST` — tool requests approval (handled by the permission system)
- `TOOL_EXECUTION_PROGRESS` — progress messages during long-running tools
- `TOOL_EXECUTION_PARTIAL_RESULT` — intermediate outputs from tools
- `SUBAGENT_SELECTED` / `SUBAGENT_STARTED` / `SUBAGENT_COMPLETED` / `SUBAGENT_FAILED` — sub-agent delegation
- `SESSION_COMPACTION_START` / `SESSION_COMPACTION_COMPLETE` — context window compaction
- `SESSION_TRUNCATION` — message truncation after context overflow
- `SESSION_WARNING` — non-fatal warnings (e.g. rate limits, degraded mode)

None of these affect the core RAPI conversion flow.

### Key ordering guarantees

| Guarantee | Detail |
|-----------|--------|
| `ASSISTANT_USAGE` before `ASSISTANT_MESSAGE` | Token counts are always available before the message event |
| `ASSISTANT_MESSAGE` always emitted | There is no "delta-only" model path; the full message is always sent |
| `ASSISTANT_TURN_END` always emitted | Emitted from a `finally` block, even on errors |
| `SESSION_IDLE` always last | Always fires after all turns are processed |
| Events are synchronous | No concurrent emission; safe to process sequentially |

## RAPI Event Mapping

The adapter converts Copilot events into the following RAPI streaming events.
A well-formed RAPI stream for a single text response looks like this:

```
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta       ×N
response.output_text.done
response.content_part.done
response.output_item.done
response.completed
```

### Event-by-event mapping

#### `ASSISTANT_TURN_START` → structure events

On the **first turn** of a session:

- `response.created` — contains the full response skeleton with `status: "in_progress"`
- `response.in_progress` — signals processing has begun

On **every turn** (including the first):

- `response.output_item.added` — announces a new output item (assistant message)
- `response.content_part.added` — announces the text content part within that item

Each turn gets a unique item ID so that multi-turn responses produce distinct
output items.

#### `ASSISTANT_MESSAGE_DELTA` → text delta

Each streaming text chunk produces:

- `response.output_text.delta` — contains the incremental text fragment

The adapter accumulates all delta text for use in the done events.

#### `ASSISTANT_USAGE` → (no RAPI event)

Token usage data is stored internally.  It does not produce any RAPI event
on its own.  The stored usage is included in the `response.completed` event
that follows.

#### `ASSISTANT_MESSAGE` → done events + completion

This is the **single authoritative event** that drives all completion logic.
When the message contains text content, it emits:

1. **Synthetic delta** (only if no `ASSISTANT_MESSAGE_DELTA` events arrived) —
   ensures clients always receive at least one `response.output_text.delta`
2. `response.output_text.done` — final accumulated text
3. `response.content_part.done` — completed content part
4. `response.output_item.done` — completed output item with full content
5. `response.completed` — final response with output items and usage data

When the message has no text content (tool-calling turns), no RAPI events
are emitted.

#### `ASSISTANT_TURN_END` → (no-op)

Since all completion logic is handled by `ASSISTANT_MESSAGE`, the turn-end
event produces no RAPI events.

#### `SESSION_IDLE` → safety net

If `SESSION_IDLE` arrives and `response.completed` has **not** yet been emitted
(e.g. an error caused `ASSISTANT_MESSAGE` to be skipped), the adapter forces
a `response.completed` event to prevent the client from hanging.

In the normal case (completion already emitted), this event is a no-op.

## Non-streaming Mode

When the client requests a non-streaming response, the adapter iterates over
all Copilot events but only extracts the text content from the
`ASSISTANT_MESSAGE` event.  All other events are ignored.  The final text is
wrapped in a single `Response` object and returned as JSON.

If no text content was produced, a fallback message is substituted so the
response is never blank.

## State Management

The streaming converter maintains minimal state:

| Field | Purpose |
|-------|---------|
| `_sequence` | Monotonically increasing RAPI event sequence number |
| `_created_at` | Unix timestamp, consistent across all events in one response |
| `_accumulated_text` | Delta text accumulated so far (for done events) |
| `_turn_count` | Number of turns seen (controls `response.created` emission) |
| `_item_id` | Current output item ID (regenerated each turn) |
| `_usage` | Token usage dict from `ASSISTANT_USAGE` |
| `_completed` | Whether `response.completed` has been emitted |

There are no deferred flags, pending queues, or conditional completion logic.
Each Copilot event maps directly to a fixed set of RAPI events.

## Sequence Numbers

Every RAPI event carries a `sequence_number` that increments monotonically
from 0.  The sequence is shared across all events in a response — it does not
reset between turns.  Clients can use this to detect gaps or reordering (though
neither should occur under normal operation).

## Observability

The adapter emits OpenTelemetry spans following the
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

- **`chat {agent_name}`** span (kind: `CLIENT`) — wraps the entire streaming
  session, with `gen_ai.operation.name = "chat"`.  Enriched with token usage
  and model information from `ASSISTANT_USAGE` events.
- **`tools/call {tool_name}`** child spans (kind: `CLIENT`) — one per tool
  execution, opened on `TOOL_EXECUTION_START` and closed on
  `TOOL_EXECUTION_COMPLETE`.  Follows
  [MCP semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/).

All Copilot SDK events are logged at INFO level with event type and content
length, with full data payloads available at DEBUG level.  `SESSION_ERROR`
events are additionally logged at WARNING level.  `SESSION_WARNING` events
are logged at WARNING level.

## Session Reuse

For multi-turn conversations, the adapter caches Copilot sessions keyed by
conversation ID.  Subsequent messages in the same conversation reuse the
existing session, preserving context.  The event listener is unsubscribed
after each message exchange to prevent stale listener accumulation.

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

