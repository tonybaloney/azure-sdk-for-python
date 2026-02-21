"""
Temporary debug script to log raw PermissionRequest structures from the Copilot SDK.
Run this, send a few prompts that trigger tools, and observe the request dicts.
"""
import asyncio
import json
import os
import sys

from copilot import CopilotClient, MessageOptions, SessionConfig

PROMPTS = [
    "Run the shell command: echo hello",
    "Read the file /etc/hostname",
    "Fetch the URL https://example.com",
    "List files in /tmp",
]


async def main() -> None:
    client = CopilotClient()
    await client.start()

    def on_permission(req, ctx):
        print("=" * 60)
        print("PERMISSION REQUEST")
        print(json.dumps(dict(req), indent=2, default=str))
        print("CTX:", json.dumps(dict(ctx), indent=2, default=str))
        print("=" * 60)
        return {"kind": "approved"}

    config = SessionConfig(
        model=os.getenv("COPILOT_MODEL", "gpt-4.1-mini"),
        on_permission_request=on_permission,
    )
    session = await client.create_session(config)

    queue: asyncio.Queue = asyncio.Queue()

    def on_event(ev):
        queue.put_nowait(ev)
        from copilot.generated.session_events import SessionEventType
        if ev.type == SessionEventType.SESSION_IDLE:
            queue.put_nowait(None)

    unsub = session.on(on_event)

    for prompt in PROMPTS:
        print(f"\n>>> {prompt}")
        await session.send(MessageOptions(prompt=prompt))
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=60)
            if ev is None:
                break
            from copilot.generated.session_events import SessionEventType
            if ev.type == SessionEventType.ASSISTANT_MESSAGE and ev.data and ev.data.content:
                print(f"RESPONSE: {ev.data.content[:200]}")

    unsub()
    await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
