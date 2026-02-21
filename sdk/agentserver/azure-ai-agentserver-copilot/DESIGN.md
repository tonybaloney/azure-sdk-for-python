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

Additional tool-related events (`MCP_APPROVAL_REQUEST`, `ASSISTANT_REASONING`)
may appear but are not part of the core conversion flow.

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
[MCP semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/):

- **`invoke_agent`** span — wraps the entire streaming session, enriched with
  token usage and model information from `ASSISTANT_USAGE`
- **`tools/call`** child spans — one per tool execution, opened on
  `TOOL_EXECUTION_START` and closed on `TOOL_EXECUTION_COMPLETE`

All Copilot SDK events are logged at INFO level with event type and content
length, with full data payloads available at DEBUG level.

## Session Reuse

For multi-turn conversations, the adapter caches Copilot sessions keyed by
conversation ID.  Subsequent messages in the same conversation reuse the
existing session, preserving context.  The event listener is unsubscribed
after each message exchange to prevent stale listener accumulation.
