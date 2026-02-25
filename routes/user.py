"""
User management endpoints.

This module contains endpoints for:
- User registration
- User info retrieval
"""

from typing import Optional

from fastapi import APIRouter, Form, HTTPException
from pydantic import BaseModel

from controllers import user_manager

router = APIRouter(prefix="/user", tags=["user"])


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class UserResponse(BaseModel):
    """Response with user information."""
    user_uuid: str
    subscription_status: str
    is_premium: bool
    tasks_used: int
    tasks_remaining: int  # -1 for unlimited (premium)
    free_task_limit: int


# =============================================================================
# USER MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/register", response_model=UserResponse)
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


@router.get("/{user_uuid}", response_model=UserResponse)
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
