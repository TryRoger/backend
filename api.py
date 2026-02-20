"""
FastAPI server for step-by-step task execution with UI element detection.

Endpoints:
- /execute_step: Single-prompt endpoint for bounding box detection + next step preview
- /health: Health check

Model: gemini-3-flash-preview

Flow:
1. Step 1: Full analysis - returns current_step (what, where, box_2d) + next_step preview
2. Step 2+: Focused analysis - client sends known what/where, gets box_2d + next_step preview
   (Faster because model just finds bounding box for known element)

Usage:
    uvicorn api:app --port 8000
"""

import io
import json
import time
import threading
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import PIL.Image
from starlette.middleware.base import BaseHTTPMiddleware

import os
from dotenv import load_dotenv
from basic_mvp import convert_box2d_to_pixels, get_step_action, draw_bounding_box, ensure_trial_dir, get_timestamp, TRIAL_DIR, MODEL
import user_manager
import stripe_handler

# Speculation imports for parallel plan generation
from speculation_cache import speculation_cache, StepPlan
from speculation_planner import get_step1, get_box_only, generate_full_plan_bg

# Load environment variables
load_dotenv()

# =============================================================================
# DEBUG FLAG: Set to True to save original and segmented images to trial folder
# Comment out or set to False in production
DEBUG_SAVE_IMAGES = True
# =============================================================================

app = FastAPI(
    title="Screen Element Detection & Guidance API",
    description="Detects UI elements and provides step-by-step guidance using Gemini",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Pre-initialize caches on server startup for faster first requests."""
    try:
        stripe_handler.initialize_stripe_cache()
    except Exception as e:
        print(f"[STARTUP] Warning: Could not initialize Stripe cache: {e}")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Print request details
        print("\n" + "=" * 60)
        print("INCOMING REQUEST")
        print("=" * 60)
        print(f"Method: {request.method}")
        print(f"URL: {request.url}")
        print(f"Path: {request.url.path}")
        print(f"Query Params: {dict(request.query_params)}")
        print(f"Client: {request.client.host}:{request.client.port}" if request.client else "Client: Unknown")

        # Print headers
        print("\nHeaders:")
        for name, value in request.headers.items():
            print(f"  {name}: {value}")

        # Read and print body
        body = await request.body()
        if body:
            print(f"\nBody ({len(body)} bytes):")
            try:
                # Try to decode as text
                body_text = body.decode('utf-8')
                # Try to parse as JSON for pretty printing
                try:
                    body_json = json.loads(body_text)
                    print(json.dumps(body_json, indent=2))
                except json.JSONDecodeError:
                    print(body_text)
            except UnicodeDecodeError:
                print(f"  [Binary data, {len(body)} bytes]")
        else:
            print("\nBody: (empty)")

        print("=" * 60 + "\n")

        # Restore body for downstream handlers
        async def receive():
            return {"type": "http.request", "body": body}
        request._receive = receive

        response = await call_next(request)
        return response


app.add_middleware(RequestLoggingMiddleware)
    

class NextStep(BaseModel):
    """Preview of the next step to be executed."""
    what: str = Field(description="What action will be performed next")
    where: str = Field(description="UI element to interact with next")


class UserResponse(BaseModel):
    """Response with user information."""
    user_uuid: str
    subscription_status: str
    is_premium: bool
    tasks_used: int
    tasks_remaining: int  # -1 for unlimited (premium)
    free_task_limit: int


class CheckoutResponse(BaseModel):
    """Response with Stripe checkout URL."""
    checkout_url: str
    session_id: str


class PortalResponse(BaseModel):
    """Response with Stripe billing portal URL."""
    portal_url: str


class ExecuteStepResponse(BaseModel):
    """Response for the execute_step endpoint.

    Flow (Parallel Speculation):
    1. Step 1: Call with screenshot, task, step_number=1 (no task_id)
       -> Returns task_id, total_steps, box_2d, current_step_what/where
       -> Background: Generates full plan for steps 2-N and caches it
    2. Step 2+: Call with screenshot, task, step_number, task_id
       -> Uses cached plan for where/info, just finds box_2d (FAST ~0.3s)
       -> Returns box_2d, current_step_what (from cache)
    3. Repeat until is_completed = true

    Performance:
    - Step 1: ~0.8s (sync) + background plan generation during user action
    - Step 2+: ~0.3s (box_2d lookup only, info from cache)
    - ~55% faster than full analysis each step
    """
    # Task tracking
    task_id: Optional[str] = Field(default=None, description="Task ID for tracking across steps (returned on step 1, pass back on step 2+)")
    total_steps: Optional[int] = Field(default=None, description="Estimated total steps for the task")

    # Current step execution details
    step_number: int = Field(description="Current step number being executed")
    current_step_what: Optional[str] = Field(default=None, description="What action to perform NOW")
    current_step_where: Optional[str] = Field(default=None, description="UI element to interact with NOW")
    box_2d: List[int] = Field(description="Bounding box [y0, x0, y1, x1] in 0-1000 scale for current step")
    pixel_coords: List[float] = Field(description="Bounding box converted to pixel coordinates")
    label: str = Field(description="Label of the detected element")
    confidence: str = Field(description="Confidence level of the detection")

    # Next step preview (null when task is complete)
    next_step: Optional[NextStep] = Field(default=None, description="Preview of the next step, null if task complete")

    # Completion status
    is_completed: bool = Field(description="True when task is fully complete after this step")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL}


# =============================================================================
# USER MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/user/register", response_model=UserResponse)
async def register_user(user_uuid: Optional[str] = Form(None)):
    """
    Register a new user or get existing user info.

    If user_uuid is provided and exists, returns existing user.
    If user_uuid is provided and doesn't exist, creates user with that UUID.
    If user_uuid is not provided, generates a new UUID.

    Returns user info including task limits and subscription status.
    """
    try:
        if user_uuid:
            user = user_manager.get_or_create_user(user_uuid)
        else:
            user = user_manager.create_user()

        return UserResponse(
            user_uuid=user["user_uuid"],
            subscription_status=user["subscription_status"],
            is_premium=user["is_premium"],
            tasks_used=user["tasks_used"],
            tasks_remaining=user["tasks_remaining"],
            free_task_limit=user["free_task_limit"],
        )
    except Exception as e:
        print(f"[ERROR] Failed to register user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_uuid}", response_model=UserResponse)
async def get_user(user_uuid: str):
    """Get user info by UUID."""
    user = user_manager.get_user(user_uuid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        user_uuid=user["user_uuid"],
        subscription_status=user["subscription_status"],
        is_premium=user["is_premium"],
        tasks_used=user["tasks_used"],
        tasks_remaining=user["tasks_remaining"],
        free_task_limit=user["free_task_limit"],
    )


# =============================================================================
# STRIPE SUBSCRIPTION ENDPOINTS
# =============================================================================

@app.post("/subscribe/checkout", response_model=CheckoutResponse)
async def create_checkout(
    user_uuid: str = Form(..., description="User UUID"),
    plan: str = Form("monthly", description="Plan: 'monthly' ($14/mo) or 'annual' ($10/mo billed yearly)"),
    success_url: str = Form("https://app.tryroger.xyz/success", description="URL to redirect on success"),
    cancel_url: str = Form("https://app.tryroger.xyz/cancel", description="URL to redirect on cancel"),
):
    """
    Create a Stripe Checkout session for subscription.

    Plans:
    - monthly: $14/month
    - annual: $10/month (billed as $120/year)

    Returns checkout URL to redirect user to Stripe payment page.
    """
    try:
        # Ensure user exists
        user_manager.get_or_create_user(user_uuid)
        print(user_manager)

        result = stripe_handler.create_checkout_session(
            user_uuid=user_uuid,
            plan=plan,
            success_url=success_url,
            cancel_url=cancel_url,
        )
        print(result)

        return CheckoutResponse(
            checkout_url=result["checkout_url"],
            session_id=result["session_id"],
        )
    except Exception as e:
        print(f"[ERROR] Failed to create checkout: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/subscribe/portal", response_model=PortalResponse)
async def create_portal(
    user_uuid: str = Form(..., description="User UUID"),
    return_url: str = Form("https://app.tryroger.xyz/settings", description="URL to return to after portal"),
):
    """
    Create a Stripe Billing Portal session for managing subscription.

    Allows users to:
    - Update payment method
    - Cancel subscription
    - View billing history
    """
    user = user_manager.get_user(user_uuid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.get("stripe_customer_id"):
        raise HTTPException(status_code=400, detail="User has no subscription")

    try:
        result = stripe_handler.create_billing_portal_session(
            stripe_customer_id=user["stripe_customer_id"],
            return_url=return_url,
        )
        return PortalResponse(portal_url=result["portal_url"])
    except Exception as e:
        print(f"[ERROR] Failed to create portal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """
    Stripe webhook endpoint for subscription events.

    Handles:
    - checkout.session.completed: New subscription created
    - customer.subscription.updated: Subscription status changed
    - customer.subscription.deleted: Subscription cancelled
    - invoice.payment_failed: Payment failed
    """
    payload = await request.body()
    signature = request.headers.get("stripe-signature")

    if not signature:
        raise HTTPException(status_code=400, detail="Missing Stripe signature")

    try:
        event = stripe_handler.verify_webhook_signature(payload, signature)
    except ValueError as e:
        print(f"[WEBHOOK] Invalid signature: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    print(f"[WEBHOOK] Received event: {event['type']}")

    # Handle the event
    result = stripe_handler.handle_subscription_event(event)

    if result:
        user_uuid = result.get("user_uuid")
        status = result.get("status")

        if user_uuid and status:
            user_manager.update_subscription(
                user_uuid=user_uuid,
                status=status,
                stripe_customer_id=result.get("stripe_customer_id"),
                subscription_id=result.get("subscription_id"),
            )
            print(f"[WEBHOOK] Updated user {user_uuid} to {status}")

    return {"status": "ok"}


@app.get("/pricing")
async def get_pricing():
    """Get current pricing information."""
    return {
        "monthly": {
            "price": 14,
            "currency": "USD",
            "interval": "month",
            "description": "$14/month",
        },
        "annual": {
            "price": 10,
            "currency": "USD",
            "interval": "month",
            "billed": "yearly",
            "total": 120,
            "description": "$10/month (billed as $120/year)",
            "savings": "Save $48/year",
        },
        "free": {
            "task_limit": int(os.getenv("FREE_TASK_LIMIT", "5")),
            "description": "5 free tasks",
        }
    }


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


# =============================================================================
# TASK EXECUTION ENDPOINT
# =============================================================================

@app.post("/execute_step", response_model=ExecuteStepResponse)
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
    label = ""
    confidence = "unknown"
    response_what = None
    response_where = None
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
        response_what = result_1.get("info", "")
        response_where = response_what  # In step 1, info serves as both what and where
        label = response_what
        confidence = "high"
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
            args=(img, response_task_id, task, response_total_steps, response_what),
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

        # Try to get cached step
        cached_step = None
        if task_id:
            cached_step = speculation_cache.get_step(task_id, step_number)
            response_total_steps = speculation_cache.get_total_steps(task_id)

        if cached_step:
            print(f"[CACHE HIT] Step {step_number} found in cache:")
            print(f"  where: {cached_step.where}")
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

                response_what = cached_step.info
                response_where = cached_step.where
                label = cached_step.info
                confidence = "high"

                # Check if this is the last step
                if response_total_steps and step_number >= response_total_steps:
                    is_completed = True
                else:
                    # Build next_step from cache
                    next_cached_step = speculation_cache.get_step(task_id, step_number + 1)
                    if next_cached_step:
                        next_step = NextStep(
                            what=next_cached_step.info,
                            where=next_cached_step.where
                        )

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
            response_what = result_fallback.get("info", "")
            response_where = response_what
            label = response_what
            confidence = "high"
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
                args=(img, response_task_id, task, response_total_steps, response_what),
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
        box_info_for_draw = {"box_2d": box_2d, "label": label}
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

    # Build response
    return ExecuteStepResponse(
        task_id=response_task_id,
        total_steps=response_total_steps,
        step_number=step_number,
        current_step_what=response_what,
        current_step_where=response_where,
        box_2d=box_2d,
        pixel_coords=pixel_coords,
        label=label,
        confidence=confidence,
        next_step=next_step,
        is_completed=is_completed,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

