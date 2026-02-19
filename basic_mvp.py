#!/usr/bin/env python3
"""
Gemini Screen Assistant with Click Guidance.

This script:
1. Captures your screen and sends frames to Gemini
2. Accepts text input from CLI
3. Gemini responds with TEXT and suggests where to click
4. Displays a visual circle indicator at the suggested click location

Usage: python basic_mvp.py
"""

import io
import os
import sys
import json
import re
import subprocess
from datetime import datetime
import time
from dotenv import load_dotenv
load_dotenv()

import PIL.Image
from PIL import ImageDraw
import mss
import requests

from google import genai
from google.genai import types


from AppKit import NSScreen


# --- Model Configuration ---
MODEL = "gemini-3-flash-preview"

# --- Screen capture settings ---
SCREEN_SCALE = 0.5  # Scale factor for coordinate conversion

# --- Trial output directory ---
TRIAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trial")


def get_timestamp():
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_trial_dir():
    """Ensure the trial directory exists."""
    os.makedirs(TRIAL_DIR, exist_ok=True)

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# System instruction for click guidance
SYSTEM_INSTRUCTION = """You are a helpful AI assistant that can see the user's screen.
Your role is to guide the user by telling them exactly where to click.

IMPORTANT: When suggesting where to click, you MUST include coordinates in this exact JSON format somewhere in your response:
{"x": <number>, "y": <number>, "element": "<description>"}

The coordinates should be in pixels relative to the screenshot image (origin at top-left).
The screenshot dimensions will be provided with each image.

Example response:
"To open Safari, click on the Safari icon in the dock. {"x": 480, "y": 850, "element": "Safari icon"}"

Always:
1. Look at the current screen to understand the context
2. Identify the exact UI element the user should click
3. Estimate the x,y pixel coordinates of the CENTER of that element
4. Include the JSON with coordinates in your response

Be concise and helpful. Describe what you see and where to click."""


def get_screen_info():
    """Get screen dimensions using AppKit."""
    main_screen = NSScreen.mainScreen()
    frame = main_screen.frame()
    return {
        "width": frame.size.width,
        "height": frame.size.height,
        "scale": main_screen.backingScaleFactor(),
    }
    


def convert_box2d_to_pixels(box_2d, img_width=1727, img_height=1116):
    """
    Convert box_2d from Gemini's 0-1000 scale to pixel coordinates.

    Args:
        box_2d: [y0, x0, y1, x1] in 0-1000 scale
        img_width: Screenshot width in pixels
        img_height: Screenshot height in pixels

    Returns:
        [y0_px, x0_px, y1_px, x1_px] in pixels
    """
    y0_px = box_2d[0] * img_height / 1000
    x0_px = box_2d[1] * img_width / 1000
    y1_px = box_2d[2] * img_height / 1000
    x1_px = box_2d[3] * img_width / 1000
    return [y0_px, x0_px, y1_px, x1_px]


def capture_screen(save_original=False, timestamp=None):
    """Capture screen and return as PIL Image and dimensions.

    Args:
        save_original: If True, save the original image to trial directory
        timestamp: Timestamp string for file naming (required if save_original=True)

    Returns:
        tuple: (PIL.Image, width, height, original_path or None)
    """
    sct = mss.mss()
    monitor = sct.monitors[0]

    screenshot = sct.grab(monitor)
    img = PIL.Image.frombytes("RGB", screenshot.size, screenshot.rgb)

    # Resize for faster transmission
    new_width = int(img.width * SCREEN_SCALE)
    new_height = int(img.height * SCREEN_SCALE)
    img = img.resize((new_width, new_height), PIL.Image.LANCZOS)

    original_path = None
    if save_original and timestamp:
        ensure_trial_dir()
        original_path = os.path.join(TRIAL_DIR, f"{timestamp}_original.png")
        img.save(original_path)
        print(f"[Saved original screenshot: {original_path}]")

    return img, new_width, new_height, original_path


def capture_screen_bytes(save_original=False, timestamp=None):
    """Capture screen and return as bytes for API.

    Args:
        save_original: If True, save the original image to trial directory
        timestamp: Timestamp string for file naming

    Returns:
        tuple: (bytes, width, height, PIL.Image, original_path or None)
    """
    img, width, height, original_path = capture_screen(save_original, timestamp)

    image_io = io.BytesIO()
    img.save(image_io, format="JPEG", quality=80)
    image_io.seek(0)

    return image_io.read(), width, height, img, original_path


def get_step_action(img_or_url, task: str, step_number: int, current_what: str = None, current_where: str = None):
    """Get the current step's bounding box and next step preview in a single call.

    This function has two modes:

    1. STEP 1 (current_what/current_where not provided): Full analysis
       - Analyzes screenshot to determine current action + next step preview
       - Returns: current_step (what, where, box_2d), next_step (what, where), is_completed

    2. STEP 2+ (current_what/current_where provided): Focused analysis (faster)
       - Client already knows what to do from previous next_step
       - Just finds bounding box for known element + determines next step
       - Returns: box_2d, next_step (what, where), is_completed

    The client should:
    1. Call with step_number=1 (no current_what/where) -> get current + next
    2. Use box_2d to perform action, take new screenshot
    3. Call with step_number=2, current_what/where from prev next_step -> get box_2d + next
    4. Repeat until is_completed = true

    Args:
        img_or_url: PIL Image to analyze OR a URL string (e.g., S3 URL)
        task: The user's task description
        step_number: Current step number (1-indexed)
        current_what: (Optional) What action to perform - from previous next_step
        current_where: (Optional) Where to perform it - from previous next_step

    Returns:
        dict with box_2d, next_step, is_completed (and current_step for step 1)
    """

    # Step 1: Full analysis - determine current action + next step
    if current_what is None or current_where is None:
        prompt = f"""You are a UI automation assistant. Analyze this screenshot to help complete a task step-by-step.

Task: "{task}"
Current step number: {step_number}

Analyze the screenshot and determine:
1. CURRENT STEP: What action to perform NOW and identify the exact UI element with its bounding box
2. NEXT STEP: What will need to be done after this step (or null if this is the final step)
3. IS_COMPLETED: Set to true if after performing the current step, the task will be complete

IMPORTANT:
- The bounding box coordinates must be in 0-1000 scale relative to image dimensions
- box_2d format is [y0, x0, y1, x1] where (x0,y0) is top-left and (x1,y1) is bottom-right
- Be precise about the UI element location
- is_completed=true means after this step, the task is done (next_step will be null)

Return ONLY a JSON object in this exact format:
{{
    "current_step": {{
        "what": "click the Safari icon to open the browser",
        "where": "Safari icon in the dock at bottom of screen",
        "box_2d": [900, 480, 950, 520],
        "label": "Safari icon",
        "confidence": "high"
    }},
    "next_step": {{
        "what": "click the URL bar to enter a web address",
        "where": "URL/address bar at the top of Safari window"
    }},
    "is_completed": false
}}

When this is the final step:
{{
    "current_step": {{
        "what": "click submit button to complete",
        "where": "Submit button at bottom of form",
        "box_2d": [800, 400, 850, 500],
        "label": "Submit button",
        "confidence": "high"
    }},
    "next_step": null,
    "is_completed": true
}}
"""

    # Step 2+: Focused analysis - find bounding box for known element + get next step
    else:
        prompt = f"""You are a UI automation assistant. Find the bounding box for a specific UI element and determine the next step.

Task: "{task}"
Current step number: {step_number}

CURRENT ACTION TO PERFORM:
- What: {current_what}
- Where: {current_where}

Your job:
1. Find the EXACT bounding box for the element described in "Where"
2. Determine what the NEXT step will be after this action (or null if this completes the task)
3. Set is_completed=true if after this step, the task will be complete

IMPORTANT:
- The bounding box coordinates must be in 0-1000 scale relative to image dimensions
- box_2d format is [y0, x0, y1, x1] where (x0,y0) is top-left and (x1,y1) is bottom-right
- Be precise about the UI element location

Return ONLY a JSON object in this exact format:
{{
    "box_2d": [900, 480, 950, 520],
    "label": "element label",
    "confidence": "high",
    "next_step": {{
        "what": "next action description",
        "where": "next element description"
    }},
    "is_completed": false
}}

When this is the final step:
{{
    "box_2d": [800, 400, 850, 500],
    "label": "element label",
    "confidence": "high",
    "next_step": null,
    "is_completed": true
}}
"""

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    # Handle URL string vs PIL Image
    if isinstance(img_or_url, str):
        # Fetch image from URL and create Part
        image_bytes = requests.get(img_or_url).content
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        contents = [prompt, image_part]
    else:
        # PIL Image passed directly
        contents = [prompt, img_or_url]

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config
    )

    # Print the entire raw response from Gemini
    print(f"\n[Step Action Response from Gemini]:\n{response.text}\n")

    # Parse JSON response
    response_text = response.text.strip()
    # Remove markdown fencing if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    return json.loads(response_text.strip())


def get_segmentation_bounding_box(img_or_url, user_task: str):
    """Request bounding box segmentation from Gemini for the user's task.

    Args:
        img_or_url: PIL Image to analyze OR a URL string (e.g., S3 URL)
        user_task: The user's task description to identify what to segment

    Returns:
        dict: Bounding box info with keys 'box_2d' (y0, x0, y1, x1 in 0-1000 scale),
              'label', and optionally 'mask', or None if not found
    """
    prompt = f"""
    Based on the user's task: "{user_task}"

    Identify the UI element the user needs to interact with and provide its bounding box.
    Output a JSON object with:
    - "box_2d": [y0, x0, y1, x1] coordinates in 0-1000 scale relative to image dimensions
    - "label": descriptive label of the element
    - "confidence": your confidence level (high/medium/low)

    Only return the JSON object, no other text.
    Example: {{"box_2d": [100, 200, 150, 300], "label": "Safari icon in dock", "confidence": "high"}}
    """

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    # Handle URL string vs PIL Image
    if isinstance(img_or_url, str):
        # Fetch image from URL and create Part
        image_bytes = requests.get(img_or_url).content
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        contents = [prompt, image_part]
    else:
        # PIL Image passed directly
        contents = [prompt, img_or_url]

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config
    )

    # Print the entire raw response from Gemini
    print(f"\n[Segmentation Response from Gemini]:\n{response.text}\n")

    # Parse JSON response
    response_text = response.text.strip()
    # Remove markdown fencing if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    return json.loads(response_text.strip())
    

def draw_bounding_box(img, box_info, timestamp=None):
    """Draw bounding box on image and optionally save it.

    Args:
        img: PIL Image to draw on (will create a copy)
        box_info: Dict with 'box_2d' key containing [y0, x0, y1, x1] in 0-1000 scale
        timestamp: If provided, save the image to trial directory

    Returns:
        tuple: (annotated PIL.Image, saved_path or None)
    """
    # Create a copy to avoid modifying the original
    annotated_img = img.copy().convert('RGBA')
    draw = ImageDraw.Draw(annotated_img)

    box = box_info.get("box_2d", [])
    if len(box) != 4:
        return annotated_img, None

    # Convert from 0-1000 scale to pixel coordinates
    y0 = int(box[0] / 1000 * img.height)
    x0 = int(box[1] / 1000 * img.width)
    y1 = int(box[2] / 1000 * img.height)
    x1 = int(box[3] / 1000 * img.width)

    # Draw rectangle with red outline
    outline_color = (255, 0, 0, 255)  # Red
    line_width = 3

    # Draw the bounding box
    draw.rectangle([x0, y0, x1, y1], outline=outline_color, width=line_width)

    # Draw label if present
    label = box_info.get("label", "")
    if label:
        # Draw label background
        text_bbox = draw.textbbox((x0, y0 - 20), label)
        draw.rectangle(text_bbox, fill=(255, 0, 0, 200))
        draw.text((x0, y0 - 20), label, fill=(255, 255, 255, 255))

    saved_path = None
    if timestamp:
        ensure_trial_dir()
        saved_path = os.path.join(TRIAL_DIR, f"{timestamp}_segmented.png")
        # Convert back to RGB for saving as PNG
        annotated_img.convert('RGB').save(saved_path)
        print(f"[Saved segmented screenshot: {saved_path}]")

    return annotated_img, saved_path


def draw_bounding_box_on_screen(pixel_coords, duration=3.0):
    """Draw a red bounding box overlay on screen via subprocess."""
    y0, x0, y1, x1 = pixel_coords
    scale = 1.0 / SCREEN_SCALE
    x0, y0, x1, y1 = int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)

    width = x1 - x0
    height = y1 - y0

    overlay_script = os.path.join(os.path.dirname(__file__), "overlay.py")
    subprocess.Popen(["python3", overlay_script, str(x0), str(y0), str(width), str(height), str(duration)])


class ScreenAssistant:
    def __init__(self):
        self.conversation_history = []

    def send_message(self, user_text: str) -> str:
        """Send a message with the current screenshot to Gemini.

        Performs segmentation to get bounding box and saves both
        original and segmented images to trial directory.
        """
        # Generate timestamp for this interaction
        timestamp = get_timestamp()

        # Capture current screen and save original
        
        start_time = time.time()
        image_bytes, width, height, img, original_path = capture_screen_bytes(
            save_original=True,
            timestamp=timestamp
        )

        # Get segmentation bounding box for the user's task
        print("[Getting segmentation bounding box...]")
        box_info = get_segmentation_bounding_box(img, user_text)
        elapsed_time = time.time() - start_time
        print(f"[Total time taken: {elapsed_time:.2f} seconds]")
        print(f"Box info as returned from Gemini: {box_info}")
        if box_info:
            print(f"[Found element: {box_info.get('label', 'unknown')} "
                  f"(confidence: {box_info.get('confidence', 'unknown')})]")
            # Draw and save segmented image
            draw_bounding_box(img, box_info, timestamp=timestamp)

            # Convert box_2d to pixel coordinates and print
            box = box_info.get("box_2d", [])
            pixel_coords = convert_box2d_to_pixels(box, width, height)
            print(f"Showing you the bounding box!")
            draw_bounding_box_on_screen(pixel_coords)
            print(f"\n[Coordinate Conversion]")
            print(f"  Normalized (0-1000): {box}")
            print(f"  Pixels: {pixel_coords}") 
        else:
            print("[No bounding box found for task]")

        return ""

    def run(self):
        """Run the interactive assistant loop."""
        print("=" * 60)
        print("Gemini Screen Assistant with Click Guidance")
        print("=" * 60)
        print()
        print("This assistant can see your screen and guide you where to click.")
        print("Type your request and press Enter. Gemini will respond with text")
        print("and show a red circle where you should click.")
        print()
        print("Type 'q' to quit.")
        print("=" * 60)
        print()

        while True:
            user_input = input("message > ").strip()

            if user_input.lower() == 'q':
                print("Exiting...")
                break

            if not user_input:
                print("Please enter a message.")
                continue

            # Get response from Gemini
            response_text = self.send_message(user_input)

            # Print the response
            print(f"\nAssistant: {response_text}")
                

if __name__ == "__main__":
    print("Starting Gemini Screen Assistant...")    

    # Display coordinate mapping info

    assistant = ScreenAssistant()
    assistant.run()
