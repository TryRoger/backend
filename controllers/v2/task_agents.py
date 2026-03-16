"""
Task agents - lightweight Flash-based agents for specific tasks.

Four task agents:
1. stepper_agent: Gives bounding box + next step info (what/where)
2. step_figure_agent: Figures out if step is complete + what user is doing
3. task_agent: Figures out if the entire task is complete
4. software_website_used_agent: Identifies which software/website is being used
"""

import os
import json
import base64
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

FLASH_MODEL = "gemini-3-flash-preview"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _parse_json_response(text: str) -> dict:
    """Strip markdown fencing and parse JSON from model response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _make_image_part(image_b64: str) -> types.Part:
    """Create an image Part from base64 data."""
    return types.Part.from_bytes(
        data=base64.b64decode(image_b64),
        mime_type="image/png",
    )


async def stepper_agent(
    image_b64: str,
    step_info: dict,
    task_description: str,
    steps_state: list | None = None,
) -> dict:
    """
    Gives bounding box telling the user about the next step (what/where).

    Args:
        image_b64: Base64-encoded screenshot
        step_info: Dict with 'what' and 'where' describing the step to locate
        task_description: The overall user task
        steps_state: Optional full plan steps for context

    Returns:
        dict with box_2d, label, confidence, what, where
    """
    steps_context = ""
    if steps_state:
        steps_context = f"\nFull plan context:\n{json.dumps(steps_state, indent=2)}\n"

    prompt = f"""You are a UI automation assistant. Find the bounding box for a specific UI element.

Task: "{task_description}"
{steps_context}
CURRENT STEP TO LOCATE:
- What: {step_info['what']}
- Where: {step_info['where']}

Find the EXACT bounding box for the element described.

IMPORTANT:
- box_2d coordinates must be in 0-1000 scale relative to image dimensions
- box_2d format is [y0, x0, y1, x1] where (y0,x0) is top-left and (y1,x1) is bottom-right
- Be precise about the UI element location

Return ONLY a JSON object:
{{
    "box_2d": [y0, x0, y1, x1],
    "label": "element label",
    "confidence": "high",
    "what": "{step_info['what']}",
    "where": "{step_info['where']}"
}}"""

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    image_part = _make_image_part(image_b64)
    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=[prompt, image_part],
        config=config,
    )
    print(f"[stepper_agent] Response: {response.text}")
    return _parse_json_response(response.text)


async def step_figure_agent(
    image_b64: str,
    current_step: dict,
    task_description: str,
) -> dict:
    """
    Figures out if the step is complete and what the user is doing.

    Args:
        image_b64: Base64-encoded screenshot
        current_step: Dict with 'what' and 'where' for the current step
        task_description: The overall user task

    Returns:
        dict with in_progress (bool), is_step_completed (bool), user_activity_text (str)
    """
    prompt = f"""You are a UI monitoring assistant. Analyze the screenshot to determine user progress.

Task: "{task_description}"

CURRENT STEP THE USER SHOULD BE DOING:
- What: {current_step['what']}
- Where: {current_step['where']}

Determine:
1. Is the user currently in the process of doing this step?
2. Has the step been completed?
3. What is the user actually doing right now? (brief description)

Return ONLY a JSON object:
{{
    "in_progress": true,
    "is_step_completed": false,
    "user_activity_text": "User has clicked on the search bar and is typing a URL"
}}"""

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    image_part = _make_image_part(image_b64)
    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=[prompt, image_part],
        config=config,
    )
    print(f"[step_figure_agent] Response: {response.text}")
    return _parse_json_response(response.text)


async def task_completion_agent(
    image_b64: str,
    task_description: str,
    steps_completed: list,
) -> dict:
    """
    Figures out if the entire task is complete.

    Args:
        image_b64: Base64-encoded screenshot
        task_description: The overall user task
        steps_completed: List of steps already completed

    Returns:
        dict with is_task_completed (bool), reasoning (str)
    """
    steps_text = "\n".join(
        f"  Step {i+1}: {s.get('what', 'unknown')}" for i, s in enumerate(steps_completed)
    )

    prompt = f"""You are a task completion evaluator. Determine if the user's task is fully done.

Task: "{task_description}"

Steps completed so far:
{steps_text}

Look at the current screenshot and determine if the overall task is complete.

Return ONLY a JSON object:
{{
    "is_task_completed": false,
    "reasoning": "The user has opened the website but hasn't submitted the form yet"
}}"""

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    image_part = _make_image_part(image_b64)
    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=[prompt, image_part],
        config=config,
    )
    print(f"[task_completion_agent] Response: {response.text}")
    return _parse_json_response(response.text)


async def software_website_used_agent(
    image_b64: str,
    task_description: str,
) -> dict:
    """
    Figures out which software/website is being used.

    Args:
        image_b64: Base64-encoded screenshot
        task_description: The overall user task

    Returns:
        dict with software_name, website_url (if applicable), context_info
    """
    prompt = f"""You are a software/website identification assistant. Look at the screenshot and the task.

Task: "{task_description}"

Identify:
1. What software application or website is currently open/visible?
2. If it's a website, what is the URL or domain?
3. Provide any relevant context about this software/website that would help guide the user
   (e.g., common navigation patterns, key UI elements, tips)

Return ONLY a JSON object:
{{
    "software_name": "Google Chrome - IRCTC Website",
    "website_url": "www.irctc.co.in",
    "context_info": "IRCTC is the Indian Railway ticketing website. The main booking flow involves: login -> search trains -> select train -> fill passenger details -> payment. The login button is in the top right corner."
}}"""

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    image_part = _make_image_part(image_b64)
    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=[prompt, image_part],
        config=config,
    )
    print(f"[software_website_agent] Response: {response.text}")
    return _parse_json_response(response.text)
