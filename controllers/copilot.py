import json
import asyncio
import base64
import os
import uuid
import pathlib
import time
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from dotenv import load_dotenv

from controllers.v2.roger_agent import RogerSession
from controllers.v2.task_agents import (
    stepper_agent,
    software_website_used_agent,
    task_completion_agent,
)
from controllers.v2.step_figure_agent import StepFigureSession

load_dotenv()

GEMINI_LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
_genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

IMAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

LOG_FILE = pathlib.Path(__file__).resolve().parent.parent / "ws_messages.log"


def _redact_for_log(obj):
    """Replace large base64 image strings with their length."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str) and len(v) > 256 and k in ("image", "image_b64", "screenshot", "img", "audio"):
                out[k] = f"<image {len(v)} chars>"
            else:
                out[k] = _redact_for_log(v)
        return out
    if isinstance(obj, list):
        return [_redact_for_log(item) for item in obj]
    return obj


def _log_message(direction: str, tag: str, data: dict):
    """Append a log line to the shared log file."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    redacted = _redact_for_log(data)
    line = json.dumps({"ts": ts, "dir": direction, "tag": tag, **redacted})
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

sessions: dict[int, RogerSession] = {}
session_uuids: dict[int, str] = {}
audio_sessions: dict[int, object] = {}       # ws_id -> AsyncSession
audio_ctx_managers: dict[int, object] = {}   # ws_id -> async context manager
audio_receiver_tasks: dict[int, asyncio.Task] = {}


def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


async def _send(ws: WebSocket, tag: str, data: dict):
    msg = json.dumps({"tag": tag, **data})
    preview = msg[:200] + "..." if len(msg) > 200 else msg
    print(f"[{_ts()}][WS] Sending tag={tag} {preview}")
    _log_message("OUT", tag, data)
    await ws.send_text(msg)


async def _stream_roger(ws: WebSocket, roger_gen):
    """Relay roger stream chunks to ws, return final data."""
    final = None
    async for chunk in roger_gen:
        if chunk["type"] == "stream_chunk":
            _log_message("OUT", "roger", {"type": "delta", "text": chunk["delta"]})
            await ws.send_text(json.dumps({"tag": "roger", "type": "delta", "text": chunk["delta"]}))
        elif chunk["type"] == "stream_complete":
            final = chunk["data"]
            await _send(ws, "roger", {"type": "complete", "data": final})
    return final


async def _plan_with_context(ws: WebSocket, roger: RogerSession, image_b64: str):
    """
    Phase 1: Generate plan + detect software in parallel.
    Phase 2: Get first step bounding box from stepper agent.
    Sends state updates to the client throughout.
    """
    t0 = time.perf_counter()
    print(f"[{_ts()}][copilot] Plan+context started")

    # Send "planning" state to client
    await _send(ws, "state", {"state": "planning"})

    # Phase 1: Plan + software detection in parallel
    async def _detect_software():
        sw_info = await software_website_used_agent(image_b64, roger.task_description)
        if isinstance(sw_info, dict):
            await roger.add_software_context(sw_info)
            print(f"[{_ts()}][copilot] Software context: {sw_info.get('software_name', 'unknown')} ({time.perf_counter()-t0:.2f}s)")
        return sw_info

    async def _plan():
        plan_result = await roger.generate_plan(roger.task_description, image_b64)
        if isinstance(plan_result, dict) and plan_result.get("steps_state"):
            print(f"[{_ts()}][copilot] Plan ready: {len(plan_result['steps_state'])} steps ({time.perf_counter()-t0:.2f}s)")
        return plan_result

    results = await asyncio.gather(_detect_software(), _plan(), return_exceptions=True)

    # Extract plan result
    plan_result = results[1] if not isinstance(results[1], Exception) else None
    if isinstance(results[1], Exception):
        print(f"[{_ts()}][copilot] Plan error: {results[1]}")

    # Phase 2: Get first step bounding box
    await _send(ws, "state", {"state": "getting_first_step"})

    first_step_info = roger.get_current_step()
    if first_step_info and image_b64:
        t1 = time.perf_counter()
        stepper_result = await stepper_agent(
            image_b64, first_step_info, roger.task_description, roger.steps_state
        )
        print(f"[{_ts()}][stepper] first step bounding box ({time.perf_counter()-t1:.2f}s)")
        if isinstance(stepper_result, dict):
            # Merge what/where from plan into stepper result for the first_step response
            stepper_result["what"] = first_step_info.get("what", stepper_result.get("what", ""))
            stepper_result["where"] = first_step_info.get("where", stepper_result.get("where", ""))
            # Wrap box_2d as array of arrays for first_step format
            if "box_2d" in stepper_result and isinstance(stepper_result["box_2d"], list):
                if stepper_result["box_2d"] and not isinstance(stepper_result["box_2d"][0], list):
                    stepper_result["box_2d"] = [stepper_result["box_2d"]]
            await _send(ws, "first_step", stepper_result)
        else:
            await _send(ws, "error", {"message": "Failed to get first step bounding box"})
    else:
        await _send(ws, "error", {"message": "No steps in plan"})

    print(f"[{_ts()}][copilot] Plan+first_step total: {time.perf_counter()-t0:.2f}s")


async def _start_audio_session(ws: WebSocket, ws_id: int):
    """Open a Gemini Live session for audio transcription and start the receiver."""
    config = {
        "response_modalities": ["AUDIO"],
        "input_audio_transcription": {},
    }
    ctx = _genai_client.aio.live.connect(
        model=GEMINI_LIVE_MODEL, config=config
    )
    session = await ctx.__aenter__()
    audio_ctx_managers[ws_id] = ctx
    audio_sessions[ws_id] = session
    audio_receiver_tasks[ws_id] = asyncio.create_task(
        _audio_receive_loop(ws, ws_id, session)
    )
    print(f"[{_ts()}][audio] Live session started for ws_id={ws_id}")
    return session


async def _audio_receive_loop(ws: WebSocket, ws_id: int, session):
    """Background task: read transcription events from Gemini and relay to client."""
    try:
        async for response in session.receive():
            content = response.server_content
            if not content:
                continue
            text = None
            if content.input_transcription and content.input_transcription.text:
                text = content.input_transcription.text
            if text:
                print(f"[{_ts()}][audio] Transcription: {text}")
                msg = json.dumps({"tag": "audio_transcription", "text": text})
                await ws.send_text(msg)
                _log_message("OUT", "audio_transcription", {"text": text})
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[{_ts()}][audio] Receive loop error: {e}")
    finally:
        print(f"[{_ts()}][audio] Receive loop ended for ws_id={ws_id}")


async def _stop_audio_session(ws_id: int):
    """Clean up the Gemini Live session and receiver task."""
    task = audio_receiver_tasks.pop(ws_id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    audio_sessions.pop(ws_id, None)
    ctx = audio_ctx_managers.pop(ws_id, None)
    if ctx:
        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            pass
    print(f"[{_ts()}][audio] Session cleaned up for ws_id={ws_id}")


async def handle_ws_stream(websocket: WebSocket):
    await websocket.accept()
    ws_id = id(websocket)
    task_uuid = str(uuid.uuid4())[:8]
    mode = "fast"  # will be set from first_step payload
    roger = RogerSession(mode=mode)
    step_figure = StepFigureSession()
    sessions[ws_id] = roger
    session_uuids[ws_id] = task_uuid

    try:
        while True:
            message = await websocket.receive()
            print(f"[{_ts()}][WS] Received: keys={list(message.keys())}")
            if message["type"] == "websocket.disconnect":
                break

            if "text" not in message:
                continue

            payload = json.loads(message["text"])
            tag = payload.get("tag", "")
            image_b64 = payload.get("image", "")
            text = payload.get("text", "")

            _log_message("IN", tag, payload)

            # Save image to images/<task_uuid>_<timestamp>.png
            if image_b64:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = IMAGES_DIR / f"{task_uuid}_{ts}.png"
                img_path.write_bytes(base64.b64decode(image_b64))
                print(f"[{_ts()}][WS] Saved image: {img_path.name}")

            # Log payload without the image data
            log_payload = {k: (f"<{k} {len(v)} chars>" if k in ("image", "audio") and isinstance(v, str) and len(v) > 256 else v) for k, v in payload.items()}
            print(f"[{_ts()}][WS] Received: {json.dumps(log_payload)}")

            action_start = time.perf_counter()

            if tag == "first_step":
                # Extract mode from payload and configure session
                mode = payload.get("mode", "fast")
                roger.mode = mode
                roger.task_description = text
                print(f"[{_ts()}][copilot] Mode set to: {mode}")

                # Plan first, then get first step bounding box
                # Software detection runs in parallel with planning
                await _plan_with_context(websocket, roger, image_b64)

            elif tag == "step_figure_agent":
                # step_figure_agent streams real-time course correction feedback
                current_step = roger.get_current_step()
                if not current_step:
                    print(f"[{_ts()}][step_figure] No current step, ignoring")
                    continue

                t0 = time.perf_counter()
                async for chunk in step_figure.check_step(image_b64, current_step, roger.task_description):
                    if chunk["type"] == "delta":
                        await websocket.send_text(json.dumps({
                            "tag": "step_figure_agent", "type": "delta", "text": chunk["text"]
                        }))
                    elif chunk["type"] == "complete":
                        await websocket.send_text(json.dumps({
                            "tag": "step_figure_agent", "type": "complete", "text": chunk["text"]
                        }))
                print(f"[{_ts()}][step_figure] streaming completed ({time.perf_counter()-t0:.2f}s)")

            elif tag == "subsequent_steps":
                step_figure = StepFigureSession()

                # Check if the overall task is complete before advancing
                t0 = time.perf_counter()
                task_result = await task_completion_agent(
                    image_b64, roger.task_description, roger.completed_steps
                )
                print(f"[{_ts()}][task_completion] completed ({time.perf_counter()-t0:.2f}s)")
                if task_result.get("is_task_completed"):
                    await _send(websocket, "task_complete", task_result)
                    continue

                # Client explicitly requests next step
                next_info = roger.get_next_step_info()
                if next_info and image_b64:
                    t0 = time.perf_counter()
                    stepper_coro = stepper_agent(image_b64, next_info, roger.task_description, roger.steps_state)
                    roger_gen = roger.handle_next_step(image_b64, text)
                    stepper_result, _ = await asyncio.gather(
                        stepper_coro,
                        _stream_roger(websocket, roger_gen),
                        return_exceptions=True,
                    )
                    print(f"[{_ts()}][stepper+roger] subsequent_steps completed ({time.perf_counter()-t0:.2f}s)")
                    if isinstance(stepper_result, dict):
                        await _send(websocket, "stepper", stepper_result)
                else:
                    t0 = time.perf_counter()
                    roger_gen = roger.handle_next_step(image_b64, text)
                    await _stream_roger(websocket, roger_gen)
                    print(f"[{_ts()}][roger] handle_next_step completed ({time.perf_counter()-t0:.2f}s)")

            elif tag == "roger":
                # User interruption
                t0 = time.perf_counter()
                roger_gen = roger.handle_user_interruption(image_b64, text)
                await _stream_roger(websocket, roger_gen)
                print(f"[{_ts()}][roger] handle_user_interruption completed ({time.perf_counter()-t0:.2f}s)")

                # After interruption, get stepper for new current step
                current_step = roger.get_current_step()
                if current_step and image_b64:
                    t0 = time.perf_counter()
                    stepper_result = await stepper_agent(
                        image_b64, current_step, roger.task_description, roger.steps_state
                    )
                    print(f"[{_ts()}][stepper] post-interruption completed ({time.perf_counter()-t0:.2f}s)")
                    await _send(websocket, "stepper", stepper_result)

            elif tag == "audio_start":
                # Close any stale session from a previous recording cycle
                if ws_id in audio_sessions:
                    print(f"[{_ts()}][audio] Closing stale session before new start")
                    await _stop_audio_session(ws_id)
                print(f"[{_ts()}][audio] audio_start received, opening fresh Live session...")
                await _start_audio_session(websocket, ws_id)

            elif tag == "audio":
                audio_b64 = payload.get("audio", "")
                if not audio_b64:
                    continue
                session = audio_sessions.get(ws_id)
                if not session:
                    print(f"[{_ts()}][audio] No active session, ignoring audio chunk")
                    continue
                # Send raw PCM bytes to Gemini
                pcm_bytes = base64.b64decode(audio_b64)
                print(f"[{_ts()}][audio] Sending {len(pcm_bytes)} bytes to Gemini")
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=pcm_bytes,
                        mime_type="audio/pcm;rate=16000",
                    )
                )

            elif tag == "audio_stop":
                print(f"[{_ts()}][audio] Stop requested, closing Live session")
                await _stop_audio_session(ws_id)

            print(f"[{_ts()}][copilot] tag={tag} total elapsed: {time.perf_counter()-action_start:.2f}s")

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    finally:
        await _stop_audio_session(ws_id)
        sessions.pop(ws_id, None)
        session_uuids.pop(ws_id, None)
