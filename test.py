#!/usr/bin/env python3
"""
Test script to call get_step_action directly (without server).
Uses a local image file and saves the response to trial/out.json.
"""

import json
import os
import PIL.Image

from basic_mvp import get_step_action, convert_box2d_to_pixels, TRIAL_DIR, ensure_trial_dir

def main():
    # Input image path
    image_path = os.path.join(TRIAL_DIR, "20260215_222420_step1_original.png")

    # Task description
    task = "crop a video"

    # Step number (starting at 1 for full analysis)
    step_number = 1

    # Load the image
    print(f"Loading image: {image_path}")
    img = PIL.Image.open(image_path)
    img_width, img_height = img.size
    print(f"Image dimensions: {img_width}x{img_height}")

    # Call get_step_action directly
    print(f"\nTask: {task}")
    print(f"Step: {step_number}")
    print("Calling get_step_action...")

    step_result = get_step_action(img, task, step_number)

    # Extract data based on step 1 response format
    current_step = step_result.get("current_step", {})
    box_2d = current_step.get("box_2d", [])
    label = current_step.get("label", "")
    confidence = current_step.get("confidence", "unknown")
    current_what = current_step.get("what")
    current_where = current_step.get("where")

    # Convert box_2d to pixel coordinates
    pixel_coords = convert_box2d_to_pixels(box_2d, img_width, img_height) if box_2d else []

    # Build next_step if present
    next_step_data = step_result.get("next_step")
    next_step = None
    if next_step_data:
        next_step = {
            "what": next_step_data.get("what", ""),
            "where": next_step_data.get("where", "")
        }

    # Build response in API format
    response = {
        "step_number": step_number,
        "current_step_what": current_what,
        "current_step_where": current_where,
        "box_2d": box_2d,
        "pixel_coords": pixel_coords,
        "label": label,
        "confidence": confidence,
        "next_step": next_step,
        "is_completed": step_result.get("is_completed", False)
    }

    # Ensure trial directory exists
    ensure_trial_dir()

    # Save response to out.json
    output_path = os.path.join(TRIAL_DIR, "out.json")
    with open(output_path, "w") as f:
        json.dump(response, f, indent=2)

    print(f"\nResponse saved to: {output_path}")
    print("\nResponse:")
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
