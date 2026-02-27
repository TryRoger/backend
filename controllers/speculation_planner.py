"""
Speculation Planner - Optimized for low latency.

SYNC calls (Step 1 & Step 2+): SMALL prompts, fast responses
BACKGROUND tasks: Detailed prompts, pre-compute future steps

Flow:
    Step 1 (sync):  Small prompt -> box_2d, info, total_steps
    Step 1 (bg):    Big prompt -> generate steps 2-N, store in cache

    Step 2+ (sync): Small prompt -> box_2d only (where/info from cache)
"""

import json
import os
from typing import Optional, List, Dict, Any, Union

import PIL.Image
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv

from controllers.speculation_cache import StepPlan

load_dotenv()

MODEL = "gemini-3-flash-preview"
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _parse_json(text: str) -> Union[dict, list]:
    """Parse JSON, strip markdown fencing. Unwrap single-element arrays."""
    t = text.strip()
    if t.startswith("```json"): t = t[7:]
    if t.startswith("```"): t = t[3:]
    if t.endswith("```"): t = t[:-3]
    result = json.loads(t.strip())
    # Model sometimes returns [{...}] instead of {...} — unwrap single-element arrays
    if isinstance(result, list) and len(result) == 1:
        return result[0]
    return result


def _img_content(img_or_url):
    """Convert image to Gemini format (thread-safe).

    PIL Images are converted to PNG bytes upfront so multiple threads
    can safely use the result without sharing a stream.
    """
    if isinstance(img_or_url, str):
        data = requests.get(img_or_url).content
        return types.Part.from_bytes(data=data, mime_type="image/jpeg")
    # PIL Image -> bytes for thread safety (avoids shared stream issues)
    import io as _io
    buf = _io.BytesIO()
    img_or_url.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


# =============================================================================
# SYNC: Step 1 - Small prompt, fast response
# =============================================================================

def get_step1(img_or_url, task: str) -> Dict[str, Any]:
    """
    SYNC Step 1: Get box_2d + info + total_steps estimate.

    SMALL PROMPT - optimized for speed.

    Returns:
        {
            "box_2d": [y0, x0, y1, x1],
            "info": "Click Safari icon",
            "total_steps": 5
        }
    """
    prompt = f"""Task: "{task}"
Find step 1 UI element. Return JSON only:
{{"box_2d": [y0,x0,y1,x1], "action": "user action 3-7 words" , "info": "2 line info about bounded box", "type":"click|drag|type|scroll" "total_steps": N}}
box_2d should highlight region for visual guide with generous padding
box_2d: 0-1000 scale, [y0,x0,y1,x1] format.

IMPORTANT:
1.Combine trivial sequential actions into ONE step. Examples:
- "Click search bar" + "Type query" → "Click search bar and type 'query'"
- "Scroll down" + "Click button" → "Scroll down to 'Submit' button and click it"
- "Click dropdown" + "Select option" → "Click dropdown and select 'Option X'"
2. box_2d coordinates must be in 0-1000 scale for most appropiate screen element visual guide.
3. Return ONLY valid JSON array, no other text
"""

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, _img_content(img_or_url)],
        config=config
    )

    print(f"[Step1 Sync]: {response.text}")
    return _parse_json(response.text)


def get_next_step(img_or_url, task: str, step_number: int) -> Dict[str, Any]:
    """
    SYNC Step N fallback: Analyze the current screenshot and determine the
    NEXT action the user should take to progress toward the task goal.

    Used when the cached plan is stale or missing for step 2+.
    Unlike get_step1, this is step-aware and won't restart from the beginning.

    Returns:
        {
            "box_2d": [y0, x0, y1, x1],
            "action": "user action 3-7 words",
            "info": "2 line info about bounded box",
            "type": "click|drag|type|scroll",
            "total_steps": N
        }
    """
    prompt = f"""Task: "{task}"
The user is on step {step_number}. Look at the CURRENT screenshot and determine the NEXT action needed to progress toward the task goal.
Do NOT restart from the beginning. Identify what should be done RIGHT NOW based on what is visible on screen.
Return JSON only:
{{"box_2d": [y0,x0,y1,x1], "action": "user action 3-7 words", "info": "2 line info about bounded box", "type":"click|drag|type|scroll", "total_steps": N}}
box_2d should highlight region for visual guide with generous padding
box_2d: 0-1000 scale, [y0,x0,y1,x1] format.

IMPORTANT:
1.Combine trivial sequential actions into ONE step. Examples:
- "Click search bar" + "Type query" → "Click search bar and type 'query'"
- "Scroll down" + "Click button" → "Scroll down to 'Submit' button and click it"
- "Click dropdown" + "Select option" → "Click dropdown and select 'Option X'"
2. box_2d coordinates must be in 0-1000 scale for most appropiate screen element visual guide.
3. Return ONLY valid JSON array, no other text
"""

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, _img_content(img_or_url)],
        config=config
    )

    print(f"[StepN Fallback]: {response.text}")
    return _parse_json(response.text)


# =============================================================================
# SYNC: Step 2+ - Smallest prompt, just find box_2d
# =============================================================================

def get_box_only(img_or_url, where: str) -> Dict[str, Any]:
    """
    SYNC Step 2+: Find box_2d for known element.

    SMALLEST PROMPT - where/info already known from cache.

    Args:
        img_or_url: New screenshot
        where: Element description from cache

    Returns:
        {"box_2d": [y0, x0, y1, x1]}
    """
    prompt = f"""Where should user click for: "{where}"
box_2d should highlight region for visual guide with generous padding
JSON only: {{"box_2d": [y0,x0,y1,x1], "action": "user action 3-7 words" , "info": "2 line info about bounded box", "type":"click|drag|type|scroll" "total_steps": N}},
0-1000 scale.
"""

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, _img_content(img_or_url)],
        config=config
    )

    print(f"[Box Lookup]: {response.text}")
    return _parse_json(response.text)


# =============================================================================
# BACKGROUND: Generate full plan (steps 2-N)
# =============================================================================

# =============================================================================
# BACKGROUND: Check if task is complete
# =============================================================================

def is_task_complete(img_or_url, task: str) -> bool:
    """
    BACKGROUND: Check if the user's task has been completed.

    Runs in parallel with step execution to detect task completion
    instead of relying on estimated step counts.

    Args:
        img_or_url: Current screenshot
        task: The user's original task description

    Returns:
        True if task appears complete, False otherwise
    """
    prompt = f"""Look at this screenshot and determine if the following task has been completed.

TASK: "{task}"

Answer with ONLY a JSON object:
{{"complete": true}} or {{"complete": false}}

Return true ONLY if the task is clearly finished and the goal has been achieved.
Return false if the task is still in progress or not yet started."""

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, _img_content(img_or_url)],
        config=config
    )

    print(f"[Task Complete Check]: {response.text}")

    try:
        result = _parse_json(response.text)
        return result.get("complete", False)
    except Exception as e:
        print(f"[Task Complete Check] Parse error: {e}")
        return False


def generate_full_plan_bg(
    img_or_url,
    task: str,
    total_steps: int,
    step1_info: str
) -> List[StepPlan]:
    """
    BACKGROUND: Generate complete plan for steps 2 to N.

    BIG PROMPT - runs async, doesn't block user.

    Args:
        img_or_url: Screenshot from step 1
        task: User's task
        total_steps: Estimate from step 1
        step1_info: What step 1 does (for context)

    Returns:
        List of StepPlan for steps 2 through total_steps
    """
    prompt = f"""Generate a detailed step-by-step plan.

TASK: "{task}"
TOTAL STEPS: {total_steps}
STEP 1 (already identified): {step1_info}

Generate steps 2 through {total_steps}. For each step, return JSON only with this exact format:

[
    {{"step_number": 2, "box_2d": [y0,x0,y1,x1], "action": "user action 3-7 words", "info": "2 line info about bounded box", "type": "click|drag|type|scroll", "total_steps": {total_steps}}},
    {{"step_number": 3, "box_2d": [y0,x0,y1,x1], "action": "user action 3-7 words", "info": "2 line info about bounded box", "type": "click|drag|type|scroll", "total_steps": {total_steps}}}
]

FIELD DESCRIPTIONS:
- step_number: The step number (2, 3, 4, etc.)
- box_2d: Bounding box [y0,x0,y1,x1] in 0-1000 scale for the UI element. Use generous padding.
- action: Short description of user action (3-7 words)
- info: 2 line description about the bounded box element and what it does
- type: One of: click, drag, type, scroll
- total_steps: Always {total_steps}

IMPORTANT:
1.Combine trivial sequential actions into ONE step. Examples:
- "Click search bar" + "Type query" → "Click search bar and type 'query'"
- "Scroll down" + "Click button" → "Scroll down to 'Submit' button and click it"
- "Click dropdown" + "Select option" → "Click dropdown and select 'Option X'"
2. box_2d coordinates must be in 0-1000 scale for most appropiate screen element visual guide.
3. Return ONLY valid JSON array, no other text

"""

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, _img_content(img_or_url)],
        config=config
    )

    print(f"[BG Full Plan]: {response.text}")

    steps_data = _parse_json(response.text)

    # Validate and convert to StepPlan objects
    validated_steps = []
    for s in steps_data:
        # Enforce required fields
        if "box_2d" not in s or not isinstance(s["box_2d"], list) or len(s["box_2d"]) != 4:
            print(f"[BG] Warning: Invalid box_2d for step {s.get('step_number')}, skipping")
            continue
        if "action" not in s or not s["action"]:
            print(f"[BG] Warning: Missing action for step {s.get('step_number')}, skipping")
            continue
        if "info" not in s or not s["info"]:
            print(f"[BG] Warning: Missing info for step {s.get('step_number')}, skipping")
            continue

        # Validate type field
        valid_types = {"click", "drag", "type", "scroll"}
        step_type = s.get("type", "click")
        if step_type not in valid_types:
            step_type = "click"

        validated_steps.append(StepPlan(
            step_number=s["step_number"],
            box_2d=s["box_2d"],
            action=s["action"],
            info=s["info"],
            type=step_type,
            total_steps=s.get("total_steps", total_steps)
        ))

    return validated_steps
