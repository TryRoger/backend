"""
Step execution endpoint for task execution with UI element detection.

This module contains the /execute_step endpoint which handles:
- Step 1: Full analysis + background plan generation
- Step 2+: Fast cached lookup with fallback
"""

import io
import os
import time
import threading
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field
import PIL.Image

from controllers.basic_mvp import convert_box2d_to_pixels, draw_bounding_box, ensure_trial_dir, get_timestamp, TRIAL_DIR
from controllers.speculation_cache import speculation_cache
from controllers.speculation_planner import get_step1, get_box_only, generate_full_plan_bg, is_task_complete
from controllers import user_manager

# =============================================================================
# DEBUG FLAG: Set to True to save original and segmented images to trial folder
# =============================================================================
DEBUG_SAVE_IMAGES = True

router = APIRouter()


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class NextStep(BaseModel):
    """Preview of the next step to be executed."""
    box_2d: List[int] = Field(description="Bounding box [y0, x0, y1, x1] in 0-1000 scale")
    action: str = Field(description="User action 3-7 words")
    info: str = Field(description="2 line info about bounded box")
    type: str = Field(description="Action type: click, drag, type, scroll")


class ExecuteStepResponse(BaseModel):
    """Response for the execute_step endpoint.

    JSON Format:
    {
        "task_id": "abc123",
        "step_number": 1,
        "box_2d": [y0, x0, y1, x1],
        "action": "user action 3-7 words",
        "info": "2 line info about bounded box",
        "type": "click|drag|type|scroll",
        "total_steps": N,
        "pixel_coords": [y0, x0, y1, x1],
        "next_step": {...} | null,
        "is_completed": false
    }
    """
    # Task tracking
    task_id: Optional[str] = Field(default=None, description="Task ID for tracking across steps")
    step_number: int = Field(description="Current step number being executed")

    # Current step details (matches StepPlan format)
    box_2d: List[int] = Field(description="Bounding box [y0, x0, y1, x1] in 0-1000 scale")
    action: str = Field(description="User action 3-7 words")
    info: str = Field(description="2 line info about bounded box")
    type: str = Field(description="Action type: click, drag, type, scroll")
    total_steps: Optional[int] = Field(default=None, description="Estimated total steps for the task")

    # Pixel coordinates for client convenience
    pixel_coords: List[float] = Field(description="Bounding box converted to pixel coordinates")

    # Next step preview (null when task is complete)
    next_step: Optional[NextStep] = Field(default=None, description="Preview of the next step, null if task complete")

    # Completion status
    is_completed: bool = Field(description="True when task is fully complete after this step")


# =============================================================================
# BACKGROUND PLAN GENERATION HELPER
# =============================================================================

def _run_bg_plan_generation(img, task_id: str, task: str, total_steps: int, step1_info: str):
    """
    Background thread to generate full plan for steps 2-N.

    This runs asynchronously after Step 1 returns, so the user can start
    their action while we pre-compute future steps.

    Args:
        img: PIL Image from step 1 screenshot
        task_id: Unique task identifier
        task: The user's task description
        total_steps: Estimated total steps from step 1
        step1_info: What step 1 does (for context)
    """
    try:
        print(f"\n[BG] Starting full plan generation for task {task_id}...")
        start = time.time()

        steps = generate_full_plan_bg(img, task, total_steps, step1_info)

        # Store in cache
        speculation_cache.store_plan(task_id, steps)

        elapsed = time.time() - start
        print(f"[BG] Plan generated and cached in {elapsed:.2f}s")
        print(f"[BG] Cached steps: {[s.step_number for s in steps]}")

    except Exception as e:
        print(f"[BG] Error generating plan: {e}")
        # Plan generation failed, but step 2+ will fallback to full analysis


def _run_bg_completion_check(img, task_id: str, task: str):
    """
    Background thread to check if task is complete.

    This runs asynchronously after each step returns, checking if the
    user's task has been accomplished based on the current screenshot.

    Args:
        img: PIL Image from current screenshot
        task_id: Unique task identifier
        task: The user's task description
    """
    try:
        print(f"\n[BG] Checking task completion for {task_id}...")
        start = time.time()

        complete = is_task_complete(img, task)

        elapsed = time.time() - start
        print(f"[BG] Completion check done in {elapsed:.2f}s - complete: {complete}")

        if complete:
            speculation_cache.mark_complete(task_id)
            print(f"[BG] Task {task_id} marked as COMPLETE")

    except Exception as e:
        print(f"[BG] Error checking completion: {e}")


# =============================================================================
# TASK EXECUTION ENDPOINT
# =============================================================================

@router.post("/execute_step", response_model=ExecuteStepResponse)
async def execute_step(
    task: str = Form(..., description="The user's task to accomplish"),
    screenshot: UploadFile = File(..., description="Current screenshot image"),
    step_number: int = Form(1, description="Current step number (1-indexed, defaults to 1)"),
    task_id: Optional[str] = Form(None, description="Task ID from step 1 response (required for step 2+)"),
    user_uuid: str = Form(..., description="User UUID for tracking and rate limiting"),
):
    """
    Step-by-step task execution with Parallel Speculation for optimized latency.

    **Requires user_uuid** for task tracking and rate limiting.
    - Free users: 5 tasks total
    - Premium users: Unlimited tasks

    **Parallel Speculation Flow:**

    1. **Step 1** (no task_id): Full analysis + background plan generation
       - SYNC: Get box_2d, info, total_steps (~0.8s)
       - BACKGROUND: Generate full plan for steps 2-N (runs during user action)
       - Returns: task_id, total_steps, box_2d, current_step_what/where

    2. **Step 2+** (with task_id): Fast cached lookup (~0.3s)
       - Get where/info from cache (pre-computed in background)
       - Just find box_2d for known element (tiny prompt)
       - Falls back to full analysis if cache miss or element not found
       - Returns: box_2d, current_step_what (from cache)

    **Performance:**
    - Step 1: ~0.8s sync + background plan generation during user action
    - Step 2+: ~0.3s (box_2d only, info from cache)
    - ~55% faster than full analysis each step

    **curl examples:**

    Step 1 (new task):
    ```bash
    curl -X POST "http://localhost:8000/execute_step" \\
      -F "task=Change Aadhar address" \\
      -F "screenshot=@screenshot.png" \\
      -F "step_number=1" \\
      -F "user_uuid=test-user-123"
    ```

    Step 2+ (with task_id):
    ```bash
    curl -X POST "http://localhost:8000/execute_step" \\
      -F "task=Change Aadhar address" \\
      -F "screenshot=@screenshot2.png" \\
      -F "step_number=2" \\
      -F "task_id=abc12345" \\
      -F "user_uuid=test-user-123"
    ```

    **Errors:**
        - 402: Task limit reached (free tier)
        - 404: Element not found
    """
    total_start = time.time()

    # Check user task limits (only count new tasks, i.e., step 1)
    is_new_task = step_number == 1
    if is_new_task:
        can_execute, user_info = user_manager.can_execute_task(user_uuid)
        if not can_execute:
            raise HTTPException(
                status_code=402,  # Payment Required
                detail={
                    "error": "task_limit_reached",
                    "message": f"Free task limit ({user_info['free_task_limit']}) reached. Upgrade to premium for unlimited tasks.",
                    "tasks_used": user_info["tasks_used"],
                    "tasks_remaining": user_info["tasks_remaining"],
                    "is_premium": user_info["is_premium"],
                }
            )

    # Read and validate image
    image_bytes = await screenshot.read()
    img = PIL.Image.open(io.BytesIO(image_bytes))
    img_width, img_height = img.size

    # Initialize response variables
    box_2d = []
    response_action = ""
    response_info = ""
    response_type = "click"
    response_task_id = task_id
    response_total_steps = None
    next_step = None
    is_completed = False

    print("\n" + "=" * 60)

    # =========================================================================
    # STEP 1: Full analysis + background plan generation
    # =========================================================================
    if step_number == 1:
        print(f"[EXECUTE_STEP] Step 1 - SPECULATION MODE for task: {task}")
        print("=" * 60)

        # SYNC: Get step 1 result (fast, small prompt)
        detection_start = time.time()
        result_1 = get_step1(img, task)
        detection_end = time.time()
        print(f"[TIMING] Step 1 detection: {detection_end - detection_start:.3f} seconds")

        box_2d = result_1.get("box_2d", [])
        response_action = result_1.get("action", "")
        response_info = result_1.get("info", "")
        response_type = result_1.get("type", "click")
        response_total_steps = result_1.get("total_steps", 5)

        # Create task in cache and get task_id
        response_task_id = speculation_cache.create_task(
            task=task,
            total_steps=response_total_steps
        )
        print(f"[CACHE] Created task: {response_task_id}")

        # BACKGROUND: Start plan generation in separate thread
        # This runs while the user performs their action
        bg_thread = threading.Thread(
            target=_run_bg_plan_generation,
            args=(img, response_task_id, task, response_total_steps, response_info),
            daemon=True
        )
        bg_thread.start()
        print("[BG] Background plan generation started")

    # =========================================================================
    # STEP 2+: Use cached plan + find box_2d (FAST)
    # =========================================================================
    else:
        print(f"[EXECUTE_STEP] Step {step_number} - CACHE LOOKUP MODE")
        print(f"  task_id: {task_id}")
        print("=" * 60)

        # Check if task was already marked complete by background check
        if task_id and speculation_cache.is_task_complete(task_id):
            print(f"[COMPLETE] Task {task_id} already marked complete by background check")
            response = ExecuteStepResponse(
                task_id=task_id,
                step_number=step_number,
                box_2d=[0, 0, 0, 0],
                action="Task completed",
                info="The task has been completed successfully.",
                type="click",
                total_steps=speculation_cache.get_total_steps(task_id),
                pixel_coords=[0, 0, 0, 0],
                next_step=None,
                is_completed=True,
            )
            # Print full response to console
            print("\n" + "=" * 60)
            print("[RESPONSE] Sending to client:")
            print(response.model_dump_json(indent=2))
            print("=" * 60 + "\n")
            return response

        # Try to get cached step
        cached_step = None
        if task_id:
            cached_step = speculation_cache.get_step(task_id, step_number)
            response_total_steps = speculation_cache.get_total_steps(task_id)

        if cached_step:
            print(f"[CACHE HIT] Step {step_number} found in cache:")
            print(f"  action: {cached_step.action}")
            print(f"  info: {cached_step.info}")

            # SYNC: Just find box_2d (FAST - small prompt)
            detection_start = time.time()
            try:
                result = get_box_only(img, cached_step.info)
                detection_end = time.time()
                print(f"[TIMING] Box lookup: {detection_end - detection_start:.3f} seconds")

                box_2d = result.get("box_2d", [])

                # Check if box_2d is empty (element not found)
                if not box_2d or len(box_2d) != 4:
                    raise ValueError("Element not found - empty box_2d")

                response_action = cached_step.action
                response_info = cached_step.info
                response_type = cached_step.type

                # Note: is_completed is determined by background is_task_complete check,
                # not by step count. The check runs after each step and marks cache.
                # Next request will detect completion via speculation_cache.is_task_complete()

                # Build next_step from cache, or calculate if not available
                next_cached_step = speculation_cache.get_step(task_id, step_number + 1)
                if next_cached_step:
                    next_step = NextStep(
                        box_2d=next_cached_step.box_2d,
                        action=next_cached_step.action,
                        info=next_cached_step.info,
                        type=next_cached_step.type
                    )
                else:
                    # Cache miss for next step - calculate from current screenshot
                    print(f"[CACHE MISS] Next step {step_number + 1} not in cache, calculating...")
                    try:
                        next_result = get_step1(img, task)
                        next_step = NextStep(
                            box_2d=next_result.get("box_2d", []),
                            action=next_result.get("action", ""),
                            info=next_result.get("info", ""),
                            type=next_result.get("type", "click")
                        )
                        print(f"[CALCULATED] Next step: {next_step.action}")
                    except Exception as e:
                        print(f"[ERROR] Failed to calculate next step: {e}")
                        next_step = None

            except Exception as e:
                print(f"[ERROR] Cache lookup failed: {e}")
                print("[FALLBACK] Running full analysis...")
                cached_step = None  # Trigger fallback below

        # FALLBACK: Cache miss or element not found - run full analysis
        if not cached_step:
            print(f"[CACHE MISS] Step {step_number} not in cache - running fallback...")

            detection_start = time.time()
            result_fallback = get_step1(img, task)
            detection_end = time.time()
            print(f"[TIMING] Fallback detection: {detection_end - detection_start:.3f} seconds")

            box_2d = result_fallback.get("box_2d", [])
            response_action = result_fallback.get("action", "")
            response_info = result_fallback.get("info", "")
            response_type = result_fallback.get("type", "click")
            response_total_steps = result_fallback.get("total_steps", 5)

            # Create new task in cache (plan diverged)
            response_task_id = speculation_cache.create_task(
                task=task,
                total_steps=response_total_steps
            )
            print(f"[CACHE] Created new task after fallback: {response_task_id}")

            # Start new background plan generation
            bg_thread = threading.Thread(
                target=_run_bg_plan_generation,
                args=(img, response_task_id, task, response_total_steps, response_info),
                daemon=True
            )
            bg_thread.start()
            print("[BG] New background plan generation started")

    # Validate box_2d
    if not box_2d or len(box_2d) != 4:
        raise HTTPException(
            status_code=404,
            detail=f"Could not detect element for step {step_number}"
        )

    # Debug: save images if enabled
    if DEBUG_SAVE_IMAGES:
        timestamp = get_timestamp()
        ensure_trial_dir()
        original_path = os.path.join(TRIAL_DIR, f"{timestamp}_step{step_number}_original.png")
        img.save(original_path)
        print(f"[DEBUG] Saved original: {original_path}")
        box_info_for_draw = {"box_2d": box_2d, "label": response_action}
        draw_bounding_box(img, box_info_for_draw, timestamp=f"{timestamp}_step{step_number}")
        print(f"[DEBUG] Saved segmented image")

    # Convert box_2d to pixel coordinates
    pixel_coords = convert_box2d_to_pixels(box_2d, img_width, img_height)

    total_end = time.time()
    print(f"\n[TIMING] Total API time: {total_end - total_start:.3f} seconds")
    print("=" * 60 + "\n")

    # Increment task count on successful step 1 (new task started)
    if is_new_task:
        user_manager.increment_task_count(user_uuid)
        user_manager.record_task(user_uuid, task)
        print(f"[USER] Incremented task count for user {user_uuid}")

    # BACKGROUND: Launch task completion check
    # This runs in parallel while user performs current step action
    if response_task_id and not is_completed:
        completion_thread = threading.Thread(
            target=_run_bg_completion_check,
            args=(img, response_task_id, task),
            daemon=True
        )
        completion_thread.start()
        print("[BG] Background completion check started")

    # Build response
    response = ExecuteStepResponse(
        task_id=response_task_id,
        step_number=step_number,
        box_2d=box_2d,
        action=response_action,
        info=response_info,
        type=response_type,
        total_steps=response_total_steps,
        pixel_coords=pixel_coords,
        next_step=next_step,
        is_completed=is_completed,
    )

    # Print full response to console
    print("\n" + "=" * 60)
    print("[RESPONSE] Sending to client:")
    print(response.model_dump_json(indent=2))
    print("=" * 60 + "\n")

    return response
