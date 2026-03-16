"""
Step Figure Agent - stateful course correction coach using direct streaming API.

Maintains conversation history of screenshots and gives real-time
supportive/corrective feedback as the user works through steps.
"""

import os
import base64
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

FLASH_LITE_MODEL = "gemini-3.1-flash-lite-preview"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _make_image_part(image_b64: str) -> types.Part:
    return types.Part.from_bytes(
        data=base64.b64decode(image_b64),
        mime_type="image/png",
    )


SYSTEM_INSTRUCTION = (
    "You are a friendly, encouraging coach watching a user's screen as they "
    "complete a task step-by-step. You see their screenshots in real time.\n\n"
    "Your job:\n"
    "- If the user is on the right track, cheer them on briefly and naturally. "
    "Examples: 'Nice, you found it!', 'Looking good, keep going!', "
    "'You're right there, just click it!'\n"
    "- If the user is going off-track or doing something unrelated, gently "
    "redirect them. Examples: 'Hmm, that's the Settings menu — you want the "
    "File menu instead, it's at the top left!', 'Whoops, looks like you "
    "clicked the wrong spot — try looking for the blue Save button.'\n\n"
    "Rules:\n"
    "- Keep responses to 1-2 short sentences MAX. Be concise.\n"
    "- Sound like a helpful friend, not a robot. Use casual language.\n"
    "- Do NOT return JSON. Just plain text.\n"
    "- Do NOT repeat the step instructions back. The user already sees them.\n"
    "- If this is the first screenshot and the user hasn't done anything yet, "
    "just say something brief and encouraging to start."
)


class StepFigureSession:
    """
    Stateful course correction coach using streaming generate_content API.
    Yields text chunks as they arrive for real-time display in the tooltip.
    """

    def __init__(self):
        self.history: list = []

    async def check_step(
        self,
        image_b64: str,
        current_step: dict,
        task_description: str,
    ):
        """
        Async generator that yields text chunks as they stream in.
        Yields dicts: {"type": "delta", "text": "..."} and {"type": "complete", "text": "..."}
        """
        prompt = (
            f'The user is working on: "{task_description}"\n\n'
            f'Current step they should be doing:\n'
            f'- What: {current_step["what"]}\n'
            f'- Where: {current_step["where"]}\n\n'
            f'Here is their latest screenshot. Give brief feedback.'
        )

        image_part = _make_image_part(image_b64)

        # Build contents: history + current prompt + screenshot
        contents = [prompt, image_part]

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        full_text = ""
        for chunk in client.models.generate_content_stream(
            model=FLASH_LITE_MODEL,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                full_text += chunk.text
                yield {"type": "delta", "text": chunk.text}

        # Add current screenshot to history for next call
        self.history.append(image_part)

        yield {"type": "complete", "text": full_text}
