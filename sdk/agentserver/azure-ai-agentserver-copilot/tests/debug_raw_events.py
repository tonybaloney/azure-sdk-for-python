#!/usr/bin/env python3
# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Raw Copilot SDK event probe.

Sends one or more prompts directly to the Copilot SDK (no HTTP server layer)
and dumps every SessionEvent with all non-None fields.  Use this to observe
what events and data shapes arrive so we can build out the match statement in
CopilotStreamingResponseConverter.

Usage
-----
    uv run tests/debug_raw_events.py
    uv run tests/debug_raw_events.py --prompt "Write a short poem"
    uv run tests/debug_raw_events.py --turns "Turn 1 prompt" "Turn 2 prompt" "Turn 3"
    uv run tests/debug_raw_events.py --prompt "Use a tool" --max-wait 60

Set a breakpoint on the `breakpoint()` call inside `on_event` to inspect live
events in the debugger.
"""
import argparse
import asyncio
import dataclasses
import json
import os
import sys
from typing import Any

from copilot import CopilotClient, MessageOptions, SessionConfig
from copilot.generated.session_events import SessionEvent, SessionEventType
from copilot.types import PermissionRequest, PermissionRequestResult


def _dump_event(event: SessionEvent) -> dict[str, Any]:
    """Return a dict of all non-None fields from the event's data payload."""
    result: dict[str, Any] = {"type": event.type.name if event.type else "UNKNOWN"}
    if event.data is not None:
        try:
            raw = dataclasses.asdict(event.data)
            # Only include non-None fields to keep output readable
            filtered = {k: v for k, v in raw.items() if v is not None}
            result["data"] = filtered
        except Exception:
            result["data"] = repr(event.data)
    return result


def _pp(d: dict[str, Any]) -> str:
    """Pretty-print a dict as compact JSON."""
    return json.dumps(d, indent=2, default=str)


async def run_probe(prompt: str, model: str, max_wait: int) -> None:
    """Single-prompt probe — delegates to run_multiturn_probe."""
    await run_multiturn_probe([prompt], model=model, max_wait=max_wait)


async def run_multiturn_probe(turns: list[str], model: str, max_wait: int) -> None:
    print(f"\n{'='*70}")
    print(f"MODEL : {model}")
    print(f"TURNS : {len(turns)}")
    for i, t in enumerate(turns):
        print(f"  [{i+1}] {t!r}")
    print(f"{'='*70}\n")

    client = CopilotClient()
    await client.start()

    def on_permission(req: PermissionRequest, _ctx: dict) -> PermissionRequestResult:
        print(f"\n[PERMISSION REQUEST] kind={req.get('kind')} → denying (probe mode)")
        return PermissionRequestResult(
            kind="denied-no-approval-rule-and-could-not-request-from-user"
        )

    session_config = SessionConfig(
        model=model,
        on_permission_request=on_permission,
    )
    session = await client.create_session(session_config)
    print(f"Session created: {session.session_id}\n")

    all_events_by_turn: list[list[SessionEvent]] = []

    for turn_idx, prompt in enumerate(turns):
        print(f"\n{'#'*70}")
        print(f"# TURN {turn_idx + 1}/{len(turns)}: {prompt!r}")
        print(f"{'#'*70}")

        turn_events: list[SessionEvent] = []
        done = asyncio.Event()
        last_key: tuple | None = None

        def on_event(event: SessionEvent, _turn=turn_idx, _events=turn_events, _done=done) -> None:
            nonlocal last_key

            text = ""
            if event.data and hasattr(event.data, "content") and event.data.content:
                text = event.data.content
            key = (event.type, text)
            if key == last_key:
                print(f"  [DEDUP SKIP] {event.type.name if event.type else '?'}")
                return
            last_key = key

            _events.append(event)
            dump = _dump_event(event)
            print(f"\n{'─'*60}")
            print(f"T{_turn+1} EVENT #{len(_events):03d}  {dump['type']}")
            print('─'*60)
            if "data" in dump and dump["data"]:
                for field, value in dump["data"].items():
                    val_str = repr(value)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "…"
                    print(f"  {field}: {val_str}")
            else:
                print("  (no data)")

            # ← SET A BREAKPOINT HERE to inspect `event` live in the debugger
            breakpoint()  # noqa: T100

            if event.type == SessionEventType.SESSION_IDLE:
                _done.set()
            elif event.type in (SessionEventType.SESSION_SHUTDOWN, SessionEventType.ABORT):
                _done.set()

        unsubscribe = session.on(on_event)
        print(f"Sending turn {turn_idx + 1}…\n")
        await session.send(MessageOptions(prompt=prompt))
        try:
            await asyncio.wait_for(done.wait(), timeout=max_wait)
        except asyncio.TimeoutError:
            print(f"\n[TIMEOUT after {max_wait}s]")
        finally:
            unsubscribe()

        all_events_by_turn.append(turn_events)

    # Grand summary
    print(f"\n{'='*70}")
    print(f"GRAND SUMMARY: {len(all_events_by_turn)} turns, {sum(len(e) for e in all_events_by_turn)} total events")
    print(f"{'='*70}")
    for i, events in enumerate(all_events_by_turn):
        by_type: dict[str, int] = {}
        for e in events:
            name = e.type.name if e.type else "UNKNOWN"
            by_type[name] = by_type.get(name, 0) + 1
        print(f"\nTurn {i+1} ({len(events)} events):")
        for name, count in sorted(by_type.items()):
            print(f"  {count:3d}x  {name}")

    await session.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw Copilot SDK event probe")
    parser.add_argument("--prompt", default="Reply with exactly one word: hello", help="Prompt to send (single turn)")
    parser.add_argument("--turns", nargs="+", metavar="PROMPT", help="List of prompts for multi-turn (overrides --prompt)")
    parser.add_argument("--model", default=os.getenv("COPILOT_MODEL", "gpt-4.1-mini"), help="Copilot model")
    parser.add_argument("--max-wait", type=int, default=120, help="Max seconds to wait per turn")
    parser.add_argument("--no-break", action="store_true", help="Disable breakpoint() calls (for non-interactive use)")
    args = parser.parse_args()

    if args.no_break:
        import builtins
        builtins.breakpoint = lambda *a, **kw: None  # type: ignore[assignment]

    turns = args.turns if args.turns else [args.prompt]
    asyncio.run(run_multiturn_probe(turns, model=args.model, max_wait=args.max_wait))


if __name__ == "__main__":
    main()
