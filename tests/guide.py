"""
Dynamic Step-by-Step Guidance Module.

This module provides functions for generating step-by-step guidance
using Gemini 2.5 Pro with structured JSON output.

Supports dynamic plan updates via request_screen flag - when a step
has request_screen=true, the client should take a new screenshot
and call the API again with the existing plan to get updated guidance.
"""

import json
import os
from typing import List, Optional

import PIL.Image
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

# --- Configuration ---
GUIDANCE_MODEL = "gemini-2.5-pro"

# --- Initialize Gemini Client ---
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# --- Pydantic Models ---

class Step(BaseModel):
    """A single step in the guidance plan."""
    id: int = Field(description="Sequential step ID starting from 1")
    what: str = Field(description="Human-readable instruction for the user")
    where: str = Field(description="Specific UI element to interact with (include position, color, icon details)")
    request_screen: bool = Field(
        default=False,
        description="If true, client should take a new screenshot after this step and call the API again with updated context"
    )
    step_completed: bool = Field(
        default=False,
        description="The step has been completed end to end")


class GuidanceSteps(BaseModel):
    """Complete guidance response with steps and metadata."""
    steps: List[Step] = Field(description="List of steps to accomplish the task")
    task_complete: bool = Field(
        default=False,
        description="True if the task appears to be complete based on the current screenshot"
    )
    total_steps: int = Field(description="Total Number of approximated steps to accomplish the task end to end ")


# --- Guidance Prompt ---

GUIDANCE_PROMPT = """You are a UI automation assistant that analyzes screenshots and provides step-by-step guidance to help users accomplish tasks.

Given a screenshot and a task description, provide clear steps to accomplish it.

IMPORTANT RULES:
1. Each step must have:
   - 'id': Sequential step number starting from 1 (1, 2, 3, ...)
   - 'what': Clear instruction for what the user should do
   - 'where': Specific UI element to interact with (be precise - include position, color, text, icon descriptions). This will be used to detect the element's bounding box, so be very specific.
   - 'request_screen': Set to true ONLY when the screen will change significantly after this step
     (e.g., clicking a button that opens a new page/dialog, navigating to a new URL)

2. Be specific about UI elements in 'where':
   - "the blue 'Sign In' button in the top-right corner"
   - "the Chrome icon (circular red/yellow/green/blue icon) in the dock at the bottom"
   - "the search field with placeholder text 'Search...'"

3. Keep steps atomic - ONE action per step

4. Set 'task_complete: true' if the screenshot shows the task is already accomplished

5. Use 'request_screen: true' strategically - only when you genuinely need to see the result
   of an action to provide accurate next steps

6. You need to list all the steps best of your ability to accomplish the task end to end, if a step requires more information
then request_screen should be set to true for that step.

7. Steps must be numbered sequentially with 'id' field: 1, 2, 3, etc.
"""


def generate_guidance_steps(
    img: PIL.Image.Image,
    task: str,
    existing_plan: dict = None
) -> GuidanceSteps:
    """
    Generate step-by-step guidance using Gemini 2.5 Pro with structured JSON output.

    Args:
        img: Current screenshot as PIL Image
        task: User's task description
        existing_plan: Optional previous plan dict for continuation/updates

    Returns:
        GuidanceSteps object with steps and metadata
    """
    if existing_plan:
        prompt = f"""{GUIDANCE_PROMPT}

TASK: {task}

PREVIOUS PLAN:
{json.dumps(existing_plan, indent=2)}

This is an UPDATED screenshot. The user has progressed through some steps.
Analyze the current screen state and UPDATE the plan:
- Remove steps that appear to be completed based on what you see
- Adjust remaining steps based on the current screen context
- Add new steps if the current screen reveals additional requirements
- Set 'task_complete: true' if the task is now accomplished
- Provide an updated 'summary' of the current screen state
"""
    else:
        prompt = f"""{GUIDANCE_PROMPT}

TASK: {task}

Analyze this screenshot and provide step-by-step guidance to accomplish the task.
Start with a 'summary' describing what you see on the current screen.
"""
    print(f"The prompt being sent to AI is: {prompt}")
    response = client.models.generate_content(
        model=GUIDANCE_MODEL,
        contents=[prompt, img],
        config={
            "response_mime_type": "application/json",
            "response_schema": GuidanceSteps,
        },
    )


    print(f"\n[Gemini Response]:\n{response.text}\n")

    return GuidanceSteps.model_validate_json(response.text)



"""
curl -X POST "http://localhost:8000/guide_steps" \
      -F "task=Change Aadhar address" \
      -F "screenshot=@/Users/user/tryroger/mouse/mac_app/later_bump/mvp/gemi
  ni/trial/20260209_130219_original.png"




"""
