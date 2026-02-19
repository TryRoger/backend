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

    Flow:
    1. Step 1: Call with screenshot, task, step_number=1 (no current_what/where)
       -> Returns current_step_what, current_step_where, box_2d, next_step preview
    2. Step 2+: Call with screenshot, task, step_number, current_what, current_where
       -> Returns box_2d, next_step preview (current_step_what/where are null - client already knows)
    3. Repeat until is_completed = true
    """
    # Current step execution details
    step_number: int = Field(description="Current step number being executed")
    current_step_what: Optional[str] = Field(default=None, description="What action to perform NOW (only for step 1)")
    current_step_where: Optional[str] = Field(default=None, description="UI element to interact with NOW (only for step 1)")
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
# TASK EXECUTION ENDPOINT
# =============================================================================

@app.post("/execute_step", response_model=ExecuteStepResponse)
async def execute_step(
    task: str = Form(..., description="The user's task to accomplish"),
    screenshot: UploadFile = File(..., description="Current screenshot image"),
    step_number: int = Form(1, description="Current step number (1-indexed, defaults to 1)"),
    current_what: Optional[str] = Form(None, description="What action to perform (from previous next_step, required for step 2+)"),
    current_where: Optional[str] = Form(None, description="Where to perform it (from previous next_step, required for step 2+)"),
    user_uuid: str = Form(..., description="User UUID for tracking and rate limiting"),
):
    """
    Step-by-step task execution endpoint with optimized latency.

    **Requires user_uuid** for task tracking and rate limiting.
    - Free users: 5 tasks total
    - Premium users: Unlimited tasks

    **Two modes:**

    1. **Step 1** (no current_what/current_where): Full analysis
       - Model analyzes screenshot to determine current action + next step preview
       - Returns: current_step_what, current_step_where, box_2d, next_step

    2. **Step 2+** (with current_what/current_where): Focused analysis (FASTER)
       - Client already knows what to do from previous next_step
       - Model just finds bounding box for known element + determines next step
       - Returns: box_2d, next_step (current_step_what/where are null)

    **Flow:**
    1. Call with step_number=1 -> get current action + box_2d + next_step preview
    2. Perform action, take new screenshot
    3. Call with step_number=2, current_what/where from prev next_step -> get box_2d + next_step
    4. Repeat until is_completed = true

    **Args:**
        task: Description of what the user wants to accomplish
        screenshot: Current screenshot image file (multipart upload)
        step_number: Current step number (1-indexed, defaults to 1)
        current_what: What action to perform (from previous next_step, for step 2+)
        current_where: Where to perform it (from previous next_step, for step 2+)
        user_uuid: User UUID for tracking and rate limiting

    **Returns:**
        ExecuteStepResponse with bounding box, next step preview, and completion status

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

    # Determine mode
    is_step_1 = current_what is None or current_where is None

    print("\n" + "=" * 60)
    if is_step_1:
        print(f"[EXECUTE_STEP] Step {step_number} - FULL ANALYSIS for task: {task}")
    else:
        print(f"[EXECUTE_STEP] Step {step_number} - FOCUSED ANALYSIS")
        print(f"  Action: {current_what}")
        print(f"  Target: {current_where}")
    print("=" * 60)

    # Call get_step_action with appropriate parameters
    detection_start = time.time()
    step_result = get_step_action(img, task, step_number, current_what, current_where)
    detection_end = time.time()
    print(f"[TIMING] Step action detection: {detection_end - detection_start:.3f} seconds")

    # Extract box_2d based on response format
    if is_step_1:
        # Step 1: box_2d is inside current_step
        current_step = step_result.get("current_step", {})
        box_2d = current_step.get("box_2d", [])
        label = current_step.get("label", "")
        confidence = current_step.get("confidence", "unknown")
        response_what = current_step.get("what")
        response_where = current_step.get("where")
    else:
        # Step 2+: box_2d is at top level
        box_2d = step_result.get("box_2d", [])
        label = step_result.get("label", "")
        confidence = step_result.get("confidence", "unknown")
        response_what = None  # Client already knows from previous next_step
        response_where = None

    if not box_2d or len(box_2d) != 4:
        target = current_where if current_where else "unknown"
        raise HTTPException(
            status_code=404,
            detail=f"Could not detect element for step {step_number}: {target}"
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

    # Build next_step if present
    next_step_data = step_result.get("next_step")
    next_step = None
    if next_step_data:
        next_step = NextStep(
            what=next_step_data.get("what", ""),
            where=next_step_data.get("where", "")
        )

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
        step_number=step_number,
        current_step_what=response_what,
        current_step_where=response_where,
        box_2d=box_2d,
        pixel_coords=pixel_coords,
        label=label,
        confidence=confidence,
        next_step=next_step,
        is_completed=step_result.get("is_completed", False),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

