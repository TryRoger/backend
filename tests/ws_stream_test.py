#!/usr/bin/env python3
"""
WebSocket client test for /ws/stream endpoint.
Exercises the full copilot flow: first_step -> step_figure_agent -> subsequent_steps -> roger interruption.

Usage:
    python tests/ws_stream_test.py
"""

import asyncio
import base64
import json
import pathlib
import random

import websockets

WS_URL = "ws://localhost:8000/ws/stream"
TRIAL_DIR = pathlib.Path(__file__).resolve().parent.parent / "controllers" / "trial"

STEP_IMAGES = [
    TRIAL_DIR / "20260227_171245_step1_original.png",
    TRIAL_DIR / "20260227_171251_step2_original.png",
    TRIAL_DIR / "20260227_171257_step3_original.png",
    TRIAL_DIR / "20260227_171308_step4_original.png",
    TRIAL_DIR / "20260227_171334_step5_original.png",
]

TASK = "please help me crop a video in circular format"


def load_image_b64(path: pathlib.Path) -> str:
    image_bytes = path.read_bytes()
    print(f"Loaded image: {path.name} ({len(image_bytes)} bytes)")
    return base64.b64encode(image_bytes).decode("utf-8")


async def send_and_recv_all(ws, payload: dict, label: str):
    """Send a payload and receive all responses until server stops sending."""
    print(f"\n{'='*60}")
    print(f">>> {label}")
    print(f"{'='*60}")
    await ws.send(json.dumps(payload))

    # Collect responses - server may send multiple messages per request
    # Use a short timeout to know when the server is done
    responses = []
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            data = json.loads(raw)
            responses.append(data)
            tag = data.get("tag", "?")
            msg_type = data.get("type", "")

            if tag == "roger" and msg_type == "delta":
                # Stream chunk - print inline
                print(data.get("text", ""), end="", flush=True)
            elif tag == "roger" and msg_type == "complete":
                print()  # newline after streaming
                print(f"  [{tag}] stream complete")
                print(f"  data: {json.dumps(data.get('data', {}), indent=2)}")
            else:
                print(f"  [{tag}] {json.dumps(data, indent=2)}")

            # If we got a terminal message, break
            if tag in ("task_complete", "error"):
                break
            if tag == "roger" and msg_type == "complete":
                # After roger completes, there may be a stepper message following
                continue
            if tag in ("first_step", "stepper"):
                # Might have more messages (background results won't come here)
                continue

        except asyncio.TimeoutError:
            print("  (no more messages)")
            break

    return responses


async def main():
    step_images_b64 = [load_image_b64(p) for p in STEP_IMAGES]

    print(f"\nConnecting to {WS_URL}...")
    async with websockets.connect(WS_URL, max_size=50 * 1024 * 1024, ping_timeout=None) as ws:
        print("Connected!\n")

        # 1. First step - new task (step 1 image)
        responses = await send_and_recv_all(ws, {
            "tag": "first_step",
            "text": TASK,
            "image": step_images_b64[0],
        }, "FIRST STEP")

        # 2. Check step progress via step_figure_agent (step 2 image)
        responses = await send_and_recv_all(ws, {
            "tag": "step_figure_agent",
            "image": step_images_b64[1],
        }, "STEP FIGURE AGENT - check progress")

        # 3. Explicitly request next step (step 3 image)
        responses = await send_and_recv_all(ws, {
            "tag": "subsequent_steps",
            "image": step_images_b64[2],
            "text": "",
        }, "SUBSEQUENT STEPS - advance to next")

        # 4. User interruption mid-flow (step 4 image)
        responses = await send_and_recv_all(ws, {
            "tag": "roger",
            "text": random.choice([
                "I am good at this software, just give me major steps",
                "This is my first time using the software i am bit nervous can you give instructions in hindi please",
            ]),
            "image": step_images_b64[3],
        }, "USER INTERRUPTION")

        # 5. Resume flow after interruption (step 5 image)
        responses = await send_and_recv_all(ws, {
            "tag": "subsequent_steps",
            "image": step_images_b64[4],
            "text": "",
        }, "SUBSEQUENT STEPS - resume after interruption")

        print(f"\n{'='*60}")
        print("Test complete.")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
