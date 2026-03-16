"""
Roger Agent - stateful orchestrator using Gemini.

Maintains conversation state across the user journey, plans steps,
and coordinates task agents. Uses conversation history (messages list)
for context management instead of Interactions API.
"""

import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

ROGER_MODEL_FAST = "gemini-3-flash-preview"
ROGER_MODEL_THINKING = "gemini-2.5-pro"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

ROGER_SYSTEM_INSTRUCTION = """You are Roger, an AI assistant that guides users through tasks on their computer step-by-step.

Your responsibilities:
- Plan multi-step workflows for the user's task
- Maintain context as the user progresses through steps
- Adapt the plan when new information arrives (screenshots, user activity, software context)
- Respond to user interruptions and modify the plan accordingly

For planning requests, respond with:
{
    "current_step": {"what": "...", "where": "..."},
    "steps_state": [
        {"step_number": 1, "what": "...", "where": "...", "status": "current"},
        {"step_number": 2, "what": "...", "where": "...", "status": "pending"}
    ],
    "is_plan_changed": false,
    "thinking": "brief reasoning"
}

For context updates or interruptions, use the planning format above."""


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"instruction_text": text.strip(), "current_step": None, "steps_state": [], "is_plan_changed": False}


def _make_image_part(image_b64: str) -> types.Part:
    import base64
    return types.Part.from_bytes(
        data=base64.b64decode(image_b64),
        mime_type="image/png",
    )


def _make_content(role: str, parts: list) -> types.Content:
    """Build a types.Content from role and a list of mixed str/Part items."""
    typed_parts = []
    for p in parts:
        if isinstance(p, str):
            typed_parts.append(types.Part.from_text(text=p))
        else:
            typed_parts.append(p)
    return types.Content(role=role, parts=typed_parts)


class RogerSession:
    """
    Manages a single user session with the Roger agent.
    Uses conversation history (types.Content list) for context.
    """

    def __init__(self, mode: str = "fast"):
        self.mode: str = mode
        self.messages: list[types.Content] = []  # conversation history
        self.steps_state: list = []
        self.current_step_index: int = 0
        self.task_description: str = ""
        self.software_context: str = ""
        self.completed_steps: list = []

    @property
    def _model(self) -> str:
        return ROGER_MODEL_THINKING if self.mode == "thinking" else ROGER_MODEL_FAST

    def _config(self) -> types.GenerateContentConfig:
        budget = 1024 if self.mode == "thinking" else 0
        return types.GenerateContentConfig(
            system_instruction=ROGER_SYSTEM_INSTRUCTION,
            thinking_config=types.ThinkingConfig(thinking_budget=budget),
        )

    def _call(self, user_parts: list) -> str:
        """Make a generate_content call with full conversation history."""
        self.messages.append(_make_content("user", user_parts))

        response = client.models.generate_content(
            model=self._model,
            contents=self.messages,
            config=self._config(),
        )

        response_text = response.text
        self.messages.append(_make_content("model", [response_text]))
        return response_text

    def _call_stream(self, user_parts: list):
        """Make a streaming generate_content call with full conversation history."""
        self.messages.append(_make_content("user", user_parts))

        response_stream = client.models.generate_content_stream(
            model=self._model,
            contents=self.messages,
            config=self._config(),
        )

        return response_stream

    async def generate_plan(self, task_description: str, image_b64: str) -> dict:
        """
        First call: generate the full step plan for the task.
        Returns dict with steps_state, current_step, thinking.
        """
        self.task_description = task_description

        user_parts = [
            f"""New task: "{task_description}"

Screenshot attached. Create a detailed step-by-step plan for this task.
Each step needs step_number, what, where, status.
Mark the first step as "current" and the rest as "pending".
Respond with the planning JSON format.""",
            _make_image_part(image_b64),
        ]

        response_text = self._call(user_parts)
        print(f"[roger_agent] Plan: {response_text}")

        result = _parse_json_response(response_text)
        if result.get("steps_state"):
            self.steps_state = result["steps_state"]
        self.current_step_index = 0
        return result

    async def add_software_context(self, software_info: dict):
        """Add software/website context from software_website_used_agent."""
        self.software_context = json.dumps(software_info)

        user_parts = [
            f"""Context update - user is using:
{self.software_context}

Incorporate this. If it changes the plan, update steps_state.
Respond with the planning JSON format.""",
        ]

        response_text = self._call(user_parts)
        print(f"[roger_agent] Software context: {response_text}")

        result = _parse_json_response(response_text)
        if result.get("steps_state") and result.get("is_plan_changed"):
            self.steps_state = result["steps_state"]
        return result

    async def handle_next_step(self, image_b64: str, user_activity_text: str = ""):
        """
        Next step flow (step 2+, streaming).
        Advances step index, sends screenshot + activity to Roger.
        Yields stream chunks.
        """
        self.current_step_index += 1

        if self.steps_state and self.current_step_index > 0:
            prev_idx = self.current_step_index - 1
            if prev_idx < len(self.steps_state):
                self.steps_state[prev_idx]["status"] = "completed"
                self.completed_steps.append(self.steps_state[prev_idx])
            if self.current_step_index < len(self.steps_state):
                self.steps_state[self.current_step_index]["status"] = "current"

        activity_context = f"\nUser activity: {user_activity_text}" if user_activity_text else ""

        user_parts = [
            f"""Next step (step {self.current_step_index + 1}).{activity_context}

Plan state:
{json.dumps(self.steps_state, indent=2)}

Screenshot attached. Provide next step instruction + updated plan if needed.
Respond with the planning JSON format.""",
            _make_image_part(image_b64),
        ]

        stream = self._call_stream(user_parts)

        accumulated_text = ""
        for chunk in stream:
            if chunk.text:
                accumulated_text += chunk.text
                yield {"tag": "roger", "delta": chunk.text, "type": "stream_chunk"}

        # Store response in conversation history
        self.messages.append(_make_content("model", [accumulated_text]))

        result = _parse_json_response(accumulated_text)
        if result.get("steps_state") and result.get("is_plan_changed"):
            self.steps_state = result["steps_state"]

        yield {"tag": "roger", "type": "stream_complete", "data": result}

    async def handle_user_interruption(self, image_b64: str, user_text: str):
        """
        User interrupts the flow with a question/request (streaming).
        Roger responds, provides next step, recreates plan if needed.
        """
        user_parts = [
            f"""User interruption: "{user_text}"

Task: "{self.task_description}"
Step: {self.current_step_index + 1}
Plan:
{json.dumps(self.steps_state, indent=2)}

Screenshot attached. Address the question, provide next step, update plan if needed.
Respond with the planning JSON format.""",
            _make_image_part(image_b64),
        ]

        stream = self._call_stream(user_parts)

        accumulated_text = ""
        for chunk in stream:
            if chunk.text:
                accumulated_text += chunk.text
                yield {"tag": "roger", "delta": chunk.text, "type": "stream_chunk"}

        # Store response in conversation history
        self.messages.append(_make_content("model", [accumulated_text]))

        result = _parse_json_response(accumulated_text)
        if result.get("steps_state"):
            self.steps_state = result["steps_state"]
            for i, step in enumerate(self.steps_state):
                if step.get("status") == "current":
                    self.current_step_index = i
                    break

        yield {"tag": "roger", "type": "stream_complete", "data": result}

    def get_current_step(self) -> dict | None:
        if not self.steps_state or self.current_step_index >= len(self.steps_state):
            return None
        return self.steps_state[self.current_step_index]

    def get_next_step_info(self) -> dict | None:
        next_idx = self.current_step_index + 1
        if self.steps_state and next_idx < len(self.steps_state):
            return self.steps_state[next_idx]
        return None
