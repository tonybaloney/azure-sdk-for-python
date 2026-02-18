# GitHub Copilot SDK Adapter Sample

This sample demonstrates how to use the agents hosting adapter with the GitHub Copilot SDK.

## Prerequisites

> **GitHub authentication:** Ensure you are authenticated with `gh auth login` and have
> an active Copilot subscription so the Copilot CLI can start.

### Environment Variables

No additional environment variables are required. The Copilot CLI is discovered
automatically from `$PATH` or via `$COPILOT_CLI_PATH`.

## Running the Sample

Follow these steps from this folder:

1) Start the agent server (defaults to 0.0.0.0:8088):

```bash
python minimal_example.py
```

2) Send a non-streaming request (returns a single JSON response):

```bash
curl -sS \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d "{\"input\":\"What is the capital of France?\",\"stream\":false}"
```

3) Send a streaming request (server-sent events). Use -N to disable curl buffering:

```bash
curl -N \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d "{\"input\":\"Explain Python decorators briefly.\",\"stream\":true}"
```
