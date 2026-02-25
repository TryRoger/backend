"""
User management module using Firebase Firestore.

Tracks:
- User UUIDs
- Task usage counts
- Subscription status (free/premium)
- Stripe customer IDs
"""

import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Free tier task limit
FREE_TASK_LIMIT = int(os.getenv("FREE_TASK_LIMIT", "5"))

# Initialize Firebase
_firebase_initialized = False


def init_firebase():
    """Initialize Firebase Admin SDK."""
    global _firebase_initialized
    if _firebase_initialized:
        return

    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "firebase-service-account.json")

    # Check if path is absolute, otherwise make it relative to backend dir
    if not os.path.isabs(service_account_path):
        # __file__ is in controllers/, so go up one level to backend/
        backend_dir = os.path.dirname(os.path.dirname(__file__))
        service_account_path = os.path.join(backend_dir, service_account_path)

    if os.path.exists(service_account_path):
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
        print(f"[USER_MANAGER] Firebase initialized with service account")
    else:
        # Try to initialize with default credentials (for Cloud Run, etc.)
        try:
            firebase_admin.initialize_app()
            print("[USER_MANAGER] Firebase initialized with default credentials")
        except Exception as e:
            print(f"[USER_MANAGER] Warning: Could not initialize Firebase: {e}")
            print(f"[USER_MANAGER] Please add your Firebase service account JSON to: {service_account_path}")
            raise

    _firebase_initialized = True


def get_db():
    """Get Firestore client."""
    init_firebase()
    return firestore.client()


# Collection name
USERS_COLLECTION = "users"
TASKS_COLLECTION = "task_history"


def create_user(user_uuid: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new user with a UUID.

    Args:
        user_uuid: Optional UUID to use. If None, generates a new one.

    Returns:
        Dict with user info
    """
    if user_uuid is None:
        user_uuid = str(uuid.uuid4())

    db = get_db()
    user_ref = db.collection(USERS_COLLECTION).document(user_uuid)

    # Check if user already exists
    existing = user_ref.get()
    if existing.exists:
        return get_user(user_uuid)

    # Create new user
    user_data = {
        "user_uuid": user_uuid,
        "subscription_status": "free",
        "stripe_customer_id": None,
        "subscription_id": None,
        "subscription_end_date": None,
        "tasks_used": 0,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    user_ref.set(user_data)
    print(f"[USER_MANAGER] Created new user: {user_uuid}")

    return format_user_response(user_data)


def get_user(user_uuid: str) -> Optional[Dict[str, Any]]:
    """
    Get user info by UUID.

    Returns:
        Dict with user info or None if not found
    """
    db = get_db()
    user_ref = db.collection(USERS_COLLECTION).document(user_uuid)
    doc = user_ref.get()

    if not doc.exists:
        return None

    return format_user_response(doc.to_dict())


def format_user_response(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format user data for API response."""
    is_premium = user_data.get("subscription_status") == "premium"
    tasks_used = user_data.get("tasks_used", 0)
    tasks_remaining = -1 if is_premium else max(0, FREE_TASK_LIMIT - tasks_used)

    return {
        "user_uuid": user_data.get("user_uuid"),
        "subscription_status": user_data.get("subscription_status", "free"),
        "stripe_customer_id": user_data.get("stripe_customer_id"),
        "subscription_id": user_data.get("subscription_id"),
        "subscription_end_date": user_data.get("subscription_end_date"),
        "tasks_used": tasks_used,
        "tasks_remaining": tasks_remaining,
        "is_premium": is_premium,
        "free_task_limit": FREE_TASK_LIMIT,
        "created_at": user_data.get("created_at"),
    }


def get_or_create_user(user_uuid: str) -> Dict[str, Any]:
    """
    Get existing user or create new one.

    Args:
        user_uuid: The user's UUID

    Returns:
        Dict with user info
    """
    user = get_user(user_uuid)
    if user:
        return user
    return create_user(user_uuid)


def can_execute_task(user_uuid: str) -> tuple[bool, Dict[str, Any]]:
    """
    Check if user can execute a task.

    Returns:
        Tuple of (can_execute: bool, user_info: dict)
    """
    user = get_or_create_user(user_uuid)

    # Premium users have unlimited tasks
    if user["is_premium"]:
        return True, user

    # Free users have limited tasks
    can_execute = user["tasks_remaining"] > 0
    return can_execute, user


def increment_task_count(user_uuid: str) -> Dict[str, Any]:
    """
    Increment the task count for a user.

    Args:
        user_uuid: The user's UUID

    Returns:
        Updated user info
    """
    db = get_db()
    user_ref = db.collection(USERS_COLLECTION).document(user_uuid)

    # Use transaction to safely increment
    user_ref.update({
        "tasks_used": firestore.Increment(1),
        "updated_at": datetime.utcnow().isoformat(),
    })

    return get_user(user_uuid)


def record_task(user_uuid: str, task_description: str):
    """Record a task in history for analytics."""
    db = get_db()
    db.collection(TASKS_COLLECTION).add({
        "user_uuid": user_uuid,
        "task_description": task_description,
        "created_at": datetime.utcnow().isoformat(),
    })


def update_subscription(
    user_uuid: str,
    status: str,
    stripe_customer_id: Optional[str] = None,
    subscription_id: Optional[str] = None,
    subscription_end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update user's subscription status.

    Args:
        user_uuid: The user's UUID
        status: 'free' or 'premium'
        stripe_customer_id: Stripe customer ID
        subscription_id: Stripe subscription ID
        subscription_end_date: When subscription ends (ISO format)

    Returns:
        Updated user info
    """
    db = get_db()
    user_ref = db.collection(USERS_COLLECTION).document(user_uuid)

    update_data = {
        "subscription_status": status,
        "updated_at": datetime.utcnow().isoformat(),
    }

    if stripe_customer_id is not None:
        update_data["stripe_customer_id"] = stripe_customer_id

    if subscription_id is not None:
        update_data["subscription_id"] = subscription_id

    if subscription_end_date is not None:
        update_data["subscription_end_date"] = subscription_end_date

    user_ref.update(update_data)
    print(f"[USER_MANAGER] Updated subscription for {user_uuid}: {status}")

    return get_user(user_uuid)


def reset_tasks_for_user(user_uuid: str) -> Dict[str, Any]:
    """Reset task count for a user (e.g., for monthly reset)."""
    db = get_db()
    user_ref = db.collection(USERS_COLLECTION).document(user_uuid)

    user_ref.update({
        "tasks_used": 0,
        "updated_at": datetime.utcnow().isoformat(),
    })

    return get_user(user_uuid)


def get_user_by_stripe_customer(stripe_customer_id: str) -> Optional[Dict[str, Any]]:
    """Find user by Stripe customer ID."""
    db = get_db()
    users_ref = db.collection(USERS_COLLECTION)
    query = users_ref.where("stripe_customer_id", "==", stripe_customer_id).limit(1)

    docs = query.get()
    for doc in docs:
        return format_user_response(doc.to_dict())

    return None
