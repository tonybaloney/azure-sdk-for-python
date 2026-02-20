#!/usr/bin/env python3
# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""RAPI (Responses API) streaming client for local testing.

Sends a request to the locally running Copilot adapter server and prints
every Server-Sent Event (SSE) it receives.  Use this to validate the full
HTTP → Copilot SDK → RAPI translation pipeline.

Usage
-----
Start the server first::

    uv run samples/hosted_agent/main.py

Then in another terminal (or via the "RAPI Client" launch config)::

    uv run tests/rapi_client.py
    uv run tests/rapi_client.py --prompt "Write a short poem" --stream
    uv run tests/rapi_client.py --prompt "What is 2+2?" --no-stream
    uv run tests/rapi_client.py --host 127.0.0.1 --port 8088

Multi-turn (each prompt gets the same conversation_id so the adapter reuses the same Copilot session)::

    uv run tests/rapi_client.py --multi-turn "My name is Alice." "What is my name?" "Say goodbye."
"""
import argparse
import json
import sys
import time
import uuid

import httpx


def _pp(obj: dict) -> str:
    return json.dumps(obj, indent=2)


def send_streaming(base_url: str, prompt: str) -> None:
    url = f"{base_url}/responses"
    payload = {"input": prompt, "stream": True}

    print(f"\n{'='*70}")
    print(f"POST {url}  (streaming SSE)")
    print(f"PROMPT: {prompt!r}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()
    event_count = 0
    event_types: list[str] = []

    with httpx.Client(timeout=120) as client:
        with client.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                if raw_line.startswith("data: "):
                    payload_str = raw_line[len("data: "):]
                    if payload_str.strip() == "[DONE]":
                        print(f"\n{'─'*60}")
                        print("[DONE]")
                        break
                    try:
                        event = json.loads(payload_str)
                    except json.JSONDecodeError:
                        print(f"[RAW] {raw_line}")
                        continue

                    event_count += 1
                    etype = event.get("type", "?")
                    event_types.append(etype)
                    elapsed = time.perf_counter() - t0

                    print(f"\n{'─'*60}")
                    print(f"EVENT #{event_count:03d}  +{elapsed:.3f}s  {etype}")
                    print('─'*60)
                    for k, v in event.items():
                        if k == "type":
                            continue
                        val = repr(v)
                        if len(val) > 300:
                            val = val[:300] + "…"
                        print(f"  {k}: {val}")
                elif raw_line.startswith("event: "):
                    # SSE event type line (before the data line)
                    pass
                else:
                    print(f"[RAW] {raw_line}")

    elapsed_total = time.perf_counter() - t0
    print(f"\n{'='*70}")
    print(f"DONE: {event_count} events in {elapsed_total:.2f}s")
    print(f"{'='*70}")
    for et in event_types:
        print(f"  {et}")


def send_non_streaming(base_url: str, prompt: str) -> None:
    url = f"{base_url}/responses"
    payload = {"input": prompt, "stream": False}

    print(f"\n{'='*70}")
    print(f"POST {url}  (non-streaming)")
    print(f"PROMPT: {prompt!r}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()
    with httpx.Client(timeout=120) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        elapsed = time.perf_counter() - t0
        body = r.json()
        print(f"Response ({elapsed:.2f}s):")
        print(_pp(body))


def wait_for_server(base_url: str, retries: int = 30) -> None:
    url = f"{base_url}/liveness"
    print(f"Waiting for server at {url} …", end="", flush=True)
    for _ in range(retries):
        try:
            r = httpx.get(url, timeout=1)
            if r.status_code == 200:
                print(" ready!")
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(0.5)
    print(" TIMEOUT")
    sys.exit(1)


def send_multi_turn(base_url: str, prompts: list[str]) -> None:
    """Send multiple prompts in a single conversation (same conversation_id each turn)."""
    conversation_id = f"conv_{uuid.uuid4().hex[:16]}"
    url = f"{base_url}/responses"

    print(f"\n{'='*70}")
    print(f"MULTI-TURN CONVERSATION  (conversation_id={conversation_id})")
    print(f"Turns: {len(prompts)}")
    print(f"{'='*70}")

    for turn_idx, prompt in enumerate(prompts):
        print(f"\n{'#'*70}")
        print(f"# TURN {turn_idx + 1}/{len(prompts)}: {prompt!r}")
        print(f"{'#'*70}")

        payload = {
            "input": prompt,
            "stream": True,
            "conversation": {"id": conversation_id},
        }

        t0 = time.perf_counter()
        event_count = 0
        event_types: list[str] = []
        reply_text = ""

        with httpx.Client(timeout=120) as client:
            with client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    if raw_line.startswith("data: "):
                        payload_str = raw_line[len("data: "):]
                        if payload_str.strip() == "[DONE]":
                            print(f"\n[DONE]")
                            break
                        try:
                            event = json.loads(payload_str)
                        except json.JSONDecodeError:
                            print(f"[RAW] {raw_line}")
                            continue

                        event_count += 1
                        etype = event.get("type", "?")
                        event_types.append(etype)
                        elapsed = time.perf_counter() - t0

                        print(f"\n{'─'*60}")
                        print(f"EVENT #{event_count:03d}  +{elapsed:.3f}s  {etype}")
                        print('─'*60)
                        for k, v in event.items():
                            if k == "type":
                                continue
                            val = repr(v)
                            if len(val) > 300:
                                val = val[:300] + "…"
                            print(f"  {k}: {val}")

                        # Collect assistant text from delta events for summary
                        if etype == "response.output_text.delta":
                            reply_text += event.get("delta", "")

        elapsed_total = time.perf_counter() - t0
        print(f"\nTurn {turn_idx + 1} summary: {event_count} events in {elapsed_total:.2f}s")
        print(f"  Event types: {event_types}")
        if reply_text:
            print(f"  Reply: {reply_text!r}")

    print(f"\n{'='*70}")
    print(f"CONVERSATION COMPLETE  (conversation_id={conversation_id})")
    print(f"{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAPI streaming client")
    parser.add_argument("--prompt", default="Reply with exactly one word: hello", help="Prompt to send")
    parser.add_argument("--multi-turn", nargs="+", metavar="PROMPT", help="Multi-turn: list of prompts sent in the same conversation")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8088, help="Server port")
    parser.add_argument("--stream", dest="stream", action="store_true", default=True, help="Use streaming (default)")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Use non-streaming")
    parser.add_argument("--no-wait", action="store_true", help="Skip server liveness check")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    if not args.no_wait:
        wait_for_server(base_url)

    if args.multi_turn:
        send_multi_turn(base_url, args.multi_turn)
    elif args.stream:
        send_streaming(base_url, args.prompt)
    else:
        send_non_streaming(base_url, args.prompt)


if __name__ == "__main__":
    main()
