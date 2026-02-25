"""
Stripe subscription and payment endpoints.

This module contains endpoints for:
- Creating checkout sessions
- Managing billing portal
- Handling Stripe webhooks
- Pricing information
"""

import os

from fastapi import APIRouter, Form, HTTPException, Request
from pydantic import BaseModel

from controllers import user_manager
from controllers import stripe_handler

router = APIRouter(tags=["payments"])


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class CheckoutResponse(BaseModel):
    """Response with Stripe checkout URL."""
    checkout_url: str
    session_id: str


class PortalResponse(BaseModel):
    """Response with Stripe billing portal URL."""
    portal_url: str


# =============================================================================
# STRIPE SUBSCRIPTION ENDPOINTS
# =============================================================================

@router.post("/subscribe/checkout", response_model=CheckoutResponse)
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


@router.post("/subscribe/portal", response_model=PortalResponse)
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


@router.post("/webhook/stripe")
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


@router.get("/pricing")
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
