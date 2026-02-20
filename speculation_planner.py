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

from speculation_cache import StepPlan

load_dotenv()

MODEL = "gemini-3-flash-preview"
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _parse_json(text: str) -> dict:
    """Parse JSON, strip markdown fencing."""
    t = text.strip()
    if t.startswith("```json"): t = t[7:]
    if t.startswith("```"): t = t[3:]
    if t.endswith("```"): t = t[:-3]
    return json.loads(t.strip())


def _img_content(img_or_url):
    """Convert image to Gemini format."""
    if isinstance(img_or_url, str):
        data = requests.get(img_or_url).content
        return types.Part.from_bytes(data=data, mime_type="image/jpeg")
    return img_or_url


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
{{"box_2d": [y0,x0,y1,x1], "info": "1-line action", "total_steps": N}}
box_2d should highlight region for visual guide with generous padding
box_2d: 0-1000 scale, [y0,x0,y1,x1] format."""

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
JSON only: {{"box_2d": [y0,x0,y1,x1]}}, 
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
    prompt = f"""You are a UI automation planner. Generate a detailed step-by-step plan.

TASK: "{task}"
TOTAL STEPS: {total_steps}
STEP 1 (already identified): {step1_info}

Generate steps 2 through {total_steps}. For each step:
- step_number: The step number (2, 3, 4, etc.)
- where: EXACT description of the UI element to interact with
  - Be specific: "Safari icon in dock" not just "icon"
  - Include location hints: "at bottom", "top-right corner", "in the menu bar"
- info: Short 1-line description of the action to perform
- action: One of: click, type, scroll, right_click, double_click
- type_text: If action is "type", the exact text to enter (null otherwise)

IMPORTANT GUIDELINES:
1. Each step should be ONE atomic action (one click, one text entry, etc.)
2. Account for UI state changes between steps (menus opening, pages loading, etc.)
3. Be realistic about what's visible on screen at each step
4. Consider common UI patterns (dropdown menus, modal dialogs, form fields)

Return a JSON array:
[
    {{
        "step_number": 2,
        "where": "URL/address bar at the top center of Safari window",
        "info": "Click URL bar to focus it for typing",
        "action": "click",
        "type_text": null
    }},
    {{
        "step_number": 3,
        "where": "URL bar (now focused with cursor)",
        "info": "Type the website address",
        "action": "type",
        "type_text": "uidai.gov.in"
    }},
    {{
        "step_number": 4,
        "where": "Search/Go button or press Enter",
        "info": "Navigate to the website",
        "action": "click",
        "type_text": null
    }}
]

Think through the entire task flow before responding. Be thorough and precise."""

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

    # Convert to StepPlan objects
    return [
        StepPlan(
            step_number=s["step_number"],
            where=s["where"],
            info=s["info"],
            action=s.get("action", "click"),
            type_text=s.get("type_text")
        )
        for s in steps_data
    ]
