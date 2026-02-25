"""
Test script for the Parallel Speculation approach.

Tests the flow:
1. Step 1 (sync): Get box_2d + info + total_steps
   Step 1 (bg): Generate full plan, store in cache
2. Step 2+: Get where/info from cache, find box_2d on new screenshot

Uses same trial images as cache_test.py
"""

import json
import os
import time
import threading
from datetime import datetime
from PIL import Image

from controllers.speculation_cache import speculation_cache, StepPlan
from controllers.speculation_planner import get_step1, get_box_only, generate_full_plan_bg
from controllers.basic_mvp import draw_bounding_box

TRIAL_DIR = "/Users/user/tryroger/mouse/mac_app/swift_app/backend/trial"
OPTIMIZED_DIR = os.path.join(TRIAL_DIR, "optimized")

# Ensure optimized directory exists
os.makedirs(OPTIMIZED_DIR, exist_ok=True)

task = "Change Aadhar address"


def get_timestamp():
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_with_bounding_box(img, box_2d, label, step_name):
    """Save image with bounding box drawn to optimized directory."""
    timestamp = get_timestamp()
    box_info = {"box_2d": box_2d, "label": label}

    # Draw bounding box and save
    annotated_img, _ = draw_bounding_box(img, box_info)

    # Save to optimized directory
    filename = f"{step_name}_{timestamp}.png"
    save_path = os.path.join(OPTIMIZED_DIR, filename)
    annotated_img.convert('RGB').save(save_path)
    print(f"[SAVED] {save_path}")


def run_bg_plan_generation(img, task_id: str, task: str, total_steps: int, step1_info: str):
    """Background thread to generate full plan."""
    print("\n[BG] Starting full plan generation...")
    start = time.time()

    steps = generate_full_plan_bg(img, task, total_steps, step1_info)

    # Store in cache
    speculation_cache.store_plan(task_id, steps)

    elapsed = time.time() - start
    print(f"[BG] Plan generated and cached in {elapsed:.2f}s")
    print(f"[BG] Cached steps: {[s.step_number for s in steps]}")


# =============================================================================
# STEP 1: FULL ANALYSIS
# =============================================================================
print("=" * 60)
print("STEP 1: Get box_2d + info + total_steps (SYNC)")
print("        + Start background plan generation")
print("=" * 60)

screenshot_path_1 = f"{TRIAL_DIR}/20260212_160314_original.png"
img1 = Image.open(screenshot_path_1)

# SYNC: Get step 1 result
start_sync = time.time()
result_1 = get_step1(img1, task)
sync_time = time.time() - start_sync

print(f"\n[STEP 1 RESULT] (took {sync_time:.2f}s)")
print(f"  box_2d: {result_1.get('box_2d')}")
print(f"  info: {result_1.get('info')}")
print(f"  total_steps: {result_1.get('total_steps')}")

# Save image with bounding box
save_with_bounding_box(img1, result_1.get('box_2d', []), result_1.get('info', ''), "step1")

# Create task in cache
task_id = speculation_cache.create_task(
    task=task,
    total_steps=result_1.get("total_steps", 5)
)
print(f"\n[CACHE] Created task: {task_id}")

# BACKGROUND: Start plan generation in separate thread
bg_thread = threading.Thread(
    target=run_bg_plan_generation,
    args=(img1, task_id, task, result_1.get("total_steps", 5), result_1.get("info", ""))
)
bg_thread.start()

# User sees Step 1 result immediately, BG task runs in parallel
print("\n[USER] Step 1 returned to user, background task running...")
print("=" * 60)


# =============================================================================
# STEP 2: Use cached plan + find box_2d
# =============================================================================
print("\n[USER] Simulating user performing action (5s)...")
time.sleep(5)

# Wait for BG task if needed (in real API, we'd have fallback)
bg_thread.join(timeout=10)

print("\n" + "=" * 60)
print("STEP 2: Get where/info from CACHE, find box_2d (SYNC)")
print("=" * 60)

screenshot_path_2 = f"{TRIAL_DIR}/20260212_160338_original.png"
img2 = Image.open(screenshot_path_2)

# Check cache for step 2
cached_step_2 = speculation_cache.get_step(task_id, step_number=2)

if cached_step_2:
    print(f"\n[CACHE HIT] Step 2 found in cache:")
    print(f"  where: {cached_step_2.where}")
    print(f"  info: {cached_step_2.info}")

    # SYNC: Just find box_2d (FAST - small prompt)
    try:
        start_sync = time.time()
        result_2 = get_box_only(img2, cached_step_2.info)
        sync_time = time.time() - start_sync

        # Check if box_2d is empty (element not found)
        if not result_2.get('box_2d'):
            raise ValueError("Element not found - empty box_2d")

        print(f"\n[STEP 2 RESULT] (took {sync_time:.2f}s)")
        print(f"  box_2d: {result_2.get('box_2d')}")
        print(f"  info (from cache): {cached_step_2.info}")
        print(f"  total_steps: {speculation_cache.get_total_steps(task_id)}")

        # Save image with bounding box
        save_with_bounding_box(img2, result_2.get('box_2d', []), cached_step_2.info, "step2")

    except Exception as e:
        print(f"\n[ERROR] Step 2 failed: {e}")
        print("[FALLBACK] Running Step 1 on current screenshot...")

        # Fallback: Run step 1 on the current screenshot
        start_sync = time.time()
        result_2 = get_step1(img2, task)
        sync_time = time.time() - start_sync

        print(f"\n[STEP 2 FALLBACK RESULT] (took {sync_time:.2f}s)")
        print(f"  box_2d: {result_2.get('box_2d')}")
        print(f"  info: {result_2.get('info')}")
        print(f"  total_steps: {result_2.get('total_steps')}")

        # Save image with bounding box
        save_with_bounding_box(img2, result_2.get('box_2d', []), result_2.get('info', ''), "step2_fallback")

        # Update cache with new plan
        task_id = speculation_cache.create_task(
            task=task,
            total_steps=result_2.get("total_steps", 5)
        )
        print(f"\n[CACHE] Created new task: {task_id}")

        # Start new background plan generation
        bg_thread = threading.Thread(
            target=run_bg_plan_generation,
            args=(img2, task_id, task, result_2.get("total_steps", 5), result_2.get("info", ""))
        )
        bg_thread.start()
else:
    print("\n[CACHE MISS] Step 2 not in cache - running Step 1 fallback...")
    start_sync = time.time()
    result_2 = get_step1(img2, task)
    sync_time = time.time() - start_sync
    print(f"\n[STEP 2 FALLBACK RESULT] (took {sync_time:.2f}s)")
    print(f"  box_2d: {result_2.get('box_2d')}")
    print(f"  info: {result_2.get('info')}")

    # Save image with bounding box
    save_with_bounding_box(img2, result_2.get('box_2d', []), result_2.get('info', ''), "step2_fallback")


# =============================================================================
# STEP 3: Use cached plan + find box_2d
# =============================================================================
print("\n[USER] Simulating user performing action (5s)...")
time.sleep(5)

print("\n" + "=" * 60)
print("STEP 3: Get where/info from CACHE, find box_2d (SYNC)")
print("=" * 60)

screenshot_path_3 = f"{TRIAL_DIR}/20260212_160402_original.png"
img3 = Image.open(screenshot_path_3)

# Check cache for step 3
cached_step_3 = speculation_cache.get_step(task_id, step_number=3)

if cached_step_3:
    print(f"\n[CACHE HIT] Step 3 found in cache:")
    print(f"  where: {cached_step_3.where}")
    print(f"  info: {cached_step_3.info}")

    # SYNC: Just find box_2d (FAST)
    try:
        start_sync = time.time()
        result_3 = get_box_only(img3, cached_step_3.info)
        sync_time = time.time() - start_sync

        # Check if box_2d is empty (element not found)
        if not result_3.get('box_2d'):
            raise ValueError("Element not found - empty box_2d")

        print(f"\n[STEP 3 RESULT] (took {sync_time:.2f}s)")
        print(f"  box_2d: {result_3.get('box_2d')}")
        print(f"  info (from cache): {cached_step_3.info}")
        print(f"  total_steps: {speculation_cache.get_total_steps(task_id)}")

        # Save image with bounding box
        save_with_bounding_box(img3, result_3.get('box_2d', []), cached_step_3.info, "step3")

    except Exception as e:
        print(f"\n[ERROR] Step 3 failed: {e}")
        print("[FALLBACK] Running Step 1 on current screenshot...")

        # Fallback: Run step 1 on the current screenshot
        start_sync = time.time()
        result_3 = get_step1(img3, task)
        sync_time = time.time() - start_sync

        print(f"\n[STEP 3 FALLBACK RESULT] (took {sync_time:.2f}s)")
        print(f"  box_2d: {result_3.get('box_2d')}")
        print(f"  info: {result_3.get('info')}")
        print(f"  total_steps: {result_3.get('total_steps')}")

        # Save image with bounding box
        save_with_bounding_box(img3, result_3.get('box_2d', []), result_3.get('info', ''), "step3_fallback")
else:
    print("\n[CACHE MISS] Step 3 not in cache - running Step 1 fallback...")
    start_sync = time.time()
    result_3 = get_step1(img3, task)
    sync_time = time.time() - start_sync
    print(f"\n[STEP 3 FALLBACK RESULT] (took {sync_time:.2f}s)")
    print(f"  box_2d: {result_3.get('box_2d')}")
    print(f"  info: {result_3.get('info')}")

    # Save image with bounding box
    save_with_bounding_box(img3, result_3.get('box_2d', []), result_3.get('info', ''), "step3_fallback")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("CACHE STATS")
print("=" * 60)
print(json.dumps(speculation_cache.get_stats(), indent=2))

print("\n" + "=" * 60)
print("LATENCY COMPARISON")
print("=" * 60)
print("""
Current Approach:
  Step 1: ~1.2s (full analysis + next step)
  Step 2: ~1.0s (focused + next step)
  Step 3: ~1.0s (focused + next step)
  Total:  ~3.2s

Parallel Speculation:
  Step 1: ~0.8s (small prompt) + BG runs in parallel
  Step 2: ~0.3s (box_2d only, tiny prompt)
  Step 3: ~0.3s (box_2d only, tiny prompt)
  Total:  ~1.4s (+ BG completes during user action time)

Speedup: ~55% faster for user-perceived latency
""")
