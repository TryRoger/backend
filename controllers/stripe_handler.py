"""
Stripe payment handling module.

Handles:
- Creating checkout sessions for subscriptions
- Processing webhooks for subscription events
- Managing subscription status
"""

import os
import stripe
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from controllers import user_manager

# Load environment variables
load_dotenv()

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")

# Pricing (in cents)
MONTHLY_PRICE_USD = int(os.getenv("MONTHLY_PRICE_USD", "14")) * 100  # $14/month
ANNUAL_MONTHLY_PRICE_USD = int(os.getenv("ANNUAL_MONTHLY_PRICE_USD", "10")) * 100  # $10/month billed annually

# Product/Price IDs will be created dynamically or you can set them here after creating in Stripe Dashboard
MONTHLY_PRICE_ID = os.getenv("STRIPE_MONTHLY_PRICE_ID")
ANNUAL_PRICE_ID = os.getenv("STRIPE_ANNUAL_PRICE_ID")
PRODUCT_ID = os.getenv("STRIPE_PRODUCT_ID")

# In-memory cache for price IDs (initialized on first use)
_cached_prices: Optional[Dict[str, str]] = None
_cache_initialized = False


def create_or_get_product() -> str:
    """Create or get the Roger Premium product."""
    # Use cached product ID if available
    if PRODUCT_ID:
        return PRODUCT_ID

    products = stripe.Product.list(limit=10)
    for product in products.data:
        if product.name == "Roger Premium":
            print(f"[STRIPE] Found existing product: {product.id} - Add STRIPE_PRODUCT_ID={product.id} to .env for faster startup")
            return product.id

    # Create new product
    product = stripe.Product.create(
        name="Roger Premium",
        description="Unlimited task executions with Roger AI assistant",
    )
    print(f"[STRIPE] Created product: {product.id} - Add STRIPE_PRODUCT_ID={product.id} to .env for faster startup")
    return product.id


def create_or_get_prices(product_id: str) -> Dict[str, str]:
    """Create or get price IDs for monthly and annual plans."""
    global MONTHLY_PRICE_ID, ANNUAL_PRICE_ID

    if MONTHLY_PRICE_ID and ANNUAL_PRICE_ID:
        return {"monthly": MONTHLY_PRICE_ID, "annual": ANNUAL_PRICE_ID}

    prices = stripe.Price.list(product=product_id, active=True, limit=10)

    monthly_price_id = None
    annual_price_id = None

    for price in prices.data:
        if price.recurring:
            if price.recurring.interval == "month" and price.recurring.interval_count == 1:
                if price.unit_amount == MONTHLY_PRICE_USD:
                    monthly_price_id = price.id
            elif price.recurring.interval == "year":
                if price.unit_amount == ANNUAL_MONTHLY_PRICE_USD * 12:
                    annual_price_id = price.id

    # Create monthly price if not exists
    if not monthly_price_id:
        monthly_price = stripe.Price.create(
            product=product_id,
            unit_amount=MONTHLY_PRICE_USD,
            currency="usd",
            recurring={"interval": "month"},
        )
        monthly_price_id = monthly_price.id
        print(f"[STRIPE] Created monthly price: {monthly_price_id} - Add STRIPE_MONTHLY_PRICE_ID={monthly_price_id} to .env")
    else:
        print(f"[STRIPE] Found monthly price: {monthly_price_id} - Add STRIPE_MONTHLY_PRICE_ID={monthly_price_id} to .env for faster startup")

    # Create annual price if not exists
    if not annual_price_id:
        annual_price = stripe.Price.create(
            product=product_id,
            unit_amount=ANNUAL_MONTHLY_PRICE_USD * 12,  # $120/year ($10/month)
            currency="usd",
            recurring={"interval": "year"},
        )
        annual_price_id = annual_price.id
        print(f"[STRIPE] Created annual price: {annual_price_id} - Add STRIPE_ANNUAL_PRICE_ID={annual_price_id} to .env")
    else:
        print(f"[STRIPE] Found annual price: {annual_price_id} - Add STRIPE_ANNUAL_PRICE_ID={annual_price_id} to .env for faster startup")

    MONTHLY_PRICE_ID = monthly_price_id
    ANNUAL_PRICE_ID = annual_price_id

    return {"monthly": monthly_price_id, "annual": annual_price_id}


def get_price_ids() -> Dict[str, str]:
    """Get price IDs, creating product/prices if needed. Uses in-memory cache after first call."""
    global _cached_prices, _cache_initialized

    # Return cached prices if available (fast path - no API calls)
    if _cache_initialized and _cached_prices:
        return _cached_prices

    # First request: fetch/create prices from Stripe (slow path)
    product_id = create_or_get_product()
    prices = create_or_get_prices(product_id)

    # Cache for subsequent requests
    _cached_prices = prices
    _cache_initialized = True
    print(f"[STRIPE] Prices cached in memory - subsequent requests will be fast")

    return prices


def initialize_stripe_cache():
    """
    Pre-initialize the Stripe price cache at server startup.
    Call this during app initialization to avoid slow first request.
    """
    print("[STRIPE] Pre-initializing price cache...")
    get_price_ids()
    print("[STRIPE] Price cache initialized")


def create_checkout_session(
    user_uuid: str,
    plan: str = "monthly",
    success_url: str = "https://your-app.com/success",
    cancel_url: str = "https://your-app.com/cancel",
) -> Dict[str, Any]:
    """
    Create a Stripe Checkout session for subscription.

    Args:
        user_uuid: The user's UUID (stored in metadata)
        plan: "monthly" or "annual"
        success_url: URL to redirect on successful payment
        cancel_url: URL to redirect on cancelled payment

    Returns:
        Dict with checkout session URL and ID
    """
    prices = get_price_ids()
    price_id = prices.get(plan, prices["monthly"])

    # Get or create Stripe customer using our database (avoids Stripe Search API which is not available in all regions)
    customer = None
    user = user_manager.get_user(user_uuid)

    if user and user.get("stripe_customer_id"):
        # User already has a Stripe customer ID stored in our database
        try:
            customer = stripe.Customer.retrieve(user["stripe_customer_id"])
            print(f"[STRIPE] Retrieved existing customer {customer.id} for user {user_uuid}")
        except stripe.error.InvalidRequestError:
            # Customer was deleted in Stripe, create a new one
            customer = None

    if not customer:
        # Create new Stripe customer
        customer = stripe.Customer.create(
            metadata={"user_uuid": user_uuid}
        )
        print(f"[STRIPE] Created customer {customer.id} for user {user_uuid}")
        # Store the customer ID in our database for future lookups
        user_manager.update_subscription(
            user_uuid=user_uuid,
            status=user.get("subscription_status", "free") if user else "free",
            stripe_customer_id=customer.id
        )

    # Create checkout session
    session = stripe.checkout.Session.create(
        customer=customer.id,
        payment_method_types=["card"],
        line_items=[{
            "price": price_id,
            "quantity": 1,
        }],
        mode="subscription",
        success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=cancel_url,
        metadata={"user_uuid": user_uuid},
        subscription_data={
            "metadata": {"user_uuid": user_uuid}
        },
    )

    return {
        "checkout_url": session.url,
        "session_id": session.id,
    }


def create_billing_portal_session(
    stripe_customer_id: str,
    return_url: str = "https://your-app.com/settings",
) -> Dict[str, str]:
    """
    Create a Stripe Billing Portal session for managing subscription.

    Args:
        stripe_customer_id: The Stripe customer ID
        return_url: URL to return to after portal

    Returns:
        Dict with portal URL
    """
    session = stripe.billing_portal.Session.create(
        customer=stripe_customer_id,
        return_url=return_url,
    )

    return {"portal_url": session.url}


def verify_webhook_signature(payload: bytes, signature: str) -> Dict[str, Any]:
    """
    Verify and parse a Stripe webhook event.

    Args:
        payload: Raw request body
        signature: Stripe-Signature header value

    Returns:
        Parsed event object

    Raises:
        ValueError: If signature verification fails
    """
    try:
        event = stripe.Webhook.construct_event(
            payload, signature, STRIPE_WEBHOOK_SECRET
        )
        return event
    except stripe.error.SignatureVerificationError as e:
        raise ValueError(f"Invalid signature: {e}")


def handle_subscription_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Handle subscription-related webhook events.

    Args:
        event: Stripe event object

    Returns:
        Dict with user_uuid and new status, or None if not a subscription event
    """
    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        # Subscription created via checkout
        user_uuid = data.get("metadata", {}).get("user_uuid")
        customer_id = data.get("customer")
        subscription_id = data.get("subscription")

        if user_uuid:
            return {
                "user_uuid": user_uuid,
                "status": "premium",
                "stripe_customer_id": customer_id,
                "subscription_id": subscription_id,
            }

    elif event_type == "customer.subscription.created":
        user_uuid = data.get("metadata", {}).get("user_uuid")
        if user_uuid:
            return {
                "user_uuid": user_uuid,
                "status": "premium",
                "stripe_customer_id": data.get("customer"),
                "subscription_id": data.get("id"),
            }

    elif event_type == "customer.subscription.updated":
        user_uuid = data.get("metadata", {}).get("user_uuid")
        status = data.get("status")

        if user_uuid:
            # active, past_due, canceled, unpaid, etc.
            is_active = status in ["active", "trialing"]
            return {
                "user_uuid": user_uuid,
                "status": "premium" if is_active else "free",
                "subscription_id": data.get("id"),
            }

    elif event_type == "customer.subscription.deleted":
        user_uuid = data.get("metadata", {}).get("user_uuid")
        if user_uuid:
            return {
                "user_uuid": user_uuid,
                "status": "free",
                "subscription_id": None,
            }

    elif event_type == "invoice.payment_failed":
        subscription_id = data.get("subscription")
        if subscription_id:
            # Get subscription to find user
            subscription = stripe.Subscription.retrieve(subscription_id)
            user_uuid = subscription.get("metadata", {}).get("user_uuid")
            if user_uuid:
                return {
                    "user_uuid": user_uuid,
                    "status": "payment_failed",
                    "subscription_id": subscription_id,
                }

    return None


def get_subscription_info(stripe_customer_id: str) -> Optional[Dict[str, Any]]:
    """
    Get subscription info for a customer.

    Args:
        stripe_customer_id: Stripe customer ID

    Returns:
        Dict with subscription details or None
    """
    subscriptions = stripe.Subscription.list(
        customer=stripe_customer_id,
        status="active",
        limit=1,
    )

    if subscriptions.data:
        sub = subscriptions.data[0]
        return {
            "subscription_id": sub.id,
            "status": sub.status,
            "current_period_end": sub.current_period_end,
            "cancel_at_period_end": sub.cancel_at_period_end,
        }

    return None
