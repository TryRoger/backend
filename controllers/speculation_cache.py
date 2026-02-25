"""
Speculation Cache for Parallel Step Planning.

Simple in-memory cache for storing pre-computed step plans.
No Redis or Pub/Sub needed - just a thread-safe dict with TTL.

Usage:
    cache = SpeculationCache()

    # Step 1: Create task and store plan
    task_id = cache.create_task(task="Open Safari", total_steps=5)
    cache.store_plan(task_id, [step2, step3, step4, step5])

    # Step 2+: Get pre-computed step
    step = cache.get_step(task_id, step_number=2)
"""

import time
import threading
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import uuid


@dataclass
class StepPlan:
    """A single step in the plan."""
    step_number: int
    box_2d: List[int]  # Bounding box [y0, x0, y1, x1] in 0-1000 scale
    action: str  # User action description (3-7 words)
    info: str  # 2 line info about the bounded box
    type: str = "click"  # click, drag, type, scroll
    total_steps: Optional[int] = None  # Total steps estimate


@dataclass
class TaskPlan:
    """Full plan for a task."""
    task_id: str
    task: str  # Original task description
    total_steps: int
    steps: Dict[int, StepPlan] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 300)  # 5 min TTL
    is_complete: bool = False


class SpeculationCache:
    """Thread-safe in-memory cache for step speculation."""

    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds (default 5 minutes)
        """
        self._cache: Dict[str, TaskPlan] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl

    def generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return str(uuid.uuid4())[:8]

    def create_task(self, task: str, total_steps: int, task_id: Optional[str] = None) -> str:
        """
        Create a new task entry in cache.

        Args:
            task: The task description
            total_steps: Estimated total number of steps
            task_id: Optional task ID (generates one if not provided)

        Returns:
            task_id
        """
        if task_id is None:
            task_id = self.generate_task_id()

        with self._lock:
            self._cache[task_id] = TaskPlan(
                task_id=task_id,
                task=task,
                total_steps=total_steps,
                expires_at=time.time() + self._default_ttl
            )

        return task_id

    def store_plan(self, task_id: str, steps: List[StepPlan]) -> bool:
        """
        Store a full plan (steps 2+) for a task.

        Args:
            task_id: The task identifier
            steps: List of StepPlan objects

        Returns:
            True if stored successfully, False if task not found
        """
        with self._lock:
            if task_id not in self._cache:
                return False

            plan = self._cache[task_id]
            for step in steps:
                plan.steps[step.step_number] = step

            # Update total steps if we have more
            if steps:
                max_step = max(s.step_number for s in steps)
                plan.total_steps = max(plan.total_steps, max_step)

        return True

    def store_step(self, task_id: str, step: StepPlan) -> bool:
        """Store a single step in the plan."""
        with self._lock:
            if task_id not in self._cache:
                return False
            self._cache[task_id].steps[step.step_number] = step
            return True

    def get_step(self, task_id: str, step_number: int) -> Optional[StepPlan]:
        """
        Get a specific step from the plan.

        Args:
            task_id: The task identifier
            step_number: Step number to retrieve

        Returns:
            StepPlan if found and not expired, None otherwise
        """
        with self._lock:
            plan = self._cache.get(task_id)

            if plan is None:
                return None

            # Check expiration
            if time.time() > plan.expires_at:
                del self._cache[task_id]
                return None

            return plan.steps.get(step_number)

    def get_plan(self, task_id: str) -> Optional[TaskPlan]:
        """Get the full plan for a task."""
        with self._lock:
            plan = self._cache.get(task_id)

            if plan is None:
                return None

            # Check expiration
            if time.time() > plan.expires_at:
                del self._cache[task_id]
                return None

            return plan

    def get_total_steps(self, task_id: str) -> Optional[int]:
        """Get total steps for a task."""
        plan = self.get_plan(task_id)
        return plan.total_steps if plan else None

    def update_total_steps(self, task_id: str, total_steps: int) -> bool:
        """Update the total steps count."""
        with self._lock:
            if task_id not in self._cache:
                return False
            self._cache[task_id].total_steps = total_steps
            return True

    def is_plan_ready(self, task_id: str, step_number: int) -> bool:
        """Check if a step is available in cache."""
        return self.get_step(task_id, step_number) is not None

    def mark_complete(self, task_id: str) -> bool:
        """Mark a task as complete."""
        with self._lock:
            if task_id not in self._cache:
                return False
            self._cache[task_id].is_complete = True
            return True

    def is_task_complete(self, task_id: str) -> bool:
        """Check if a task is marked as complete."""
        with self._lock:
            plan = self._cache.get(task_id)
            if plan is None:
                return False
            return plan.is_complete

    def get_task_description(self, task_id: str) -> Optional[str]:
        """Get the original task description."""
        with self._lock:
            plan = self._cache.get(task_id)
            return plan.task if plan else None

    def extend_ttl(self, task_id: str, additional_seconds: int = 300) -> bool:
        """Extend the TTL for a task."""
        with self._lock:
            if task_id not in self._cache:
                return False
            self._cache[task_id].expires_at = time.time() + additional_seconds
            return True

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        removed = 0

        with self._lock:
            expired = [k for k, v in self._cache.items() if now > v.expires_at]
            for key in expired:
                del self._cache[key]
                removed += 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "total_tasks": len(self._cache),
                "tasks": {
                    tid: {
                        "total_steps": p.total_steps,
                        "cached_steps": list(p.steps.keys()),
                        "is_complete": p.is_complete,
                        "ttl_remaining": max(0, int(p.expires_at - time.time()))
                    }
                    for tid, p in self._cache.items()
                }
            }


# Global cache instance
speculation_cache = SpeculationCache()
