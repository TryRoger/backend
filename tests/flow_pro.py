"""
Test script for the execute_step API endpoint with PARALLEL SPECULATION.

Flow:
1. Step 1: Get box_2d + info + total_steps, receive task_id
   Background: Full plan for steps 2-N is generated and cached
2. Step 2+: Pass task_id, get box_2d from cache lookup (FAST ~0.3s)

Performance:
- Step 1: ~0.8s sync + background plan generation
- Step 2+: ~0.3s (box_2d only, info from cache)
- ~55% faster than full analysis each step
"""

import json
import requests

API_URL = "http://localhost:8000/execute_step"
TRIAL_DIR = "/Users/user/tryroger/mouse/mac_app/swift_app/backend/trial"

# User UUID for rate limiting (required parameter)
USER_UUID = "test-user-123"

task = "Change Aadhar address"


# --- Step 1: SPECULATION MODE - get box_2d + info + task_id ---
# Background plan generation starts automatically
print("=" * 60)
print("STEP 1: SPECULATION MODE (returns task_id for subsequent steps)")
print("=" * 60)

screenshot_path_1 = f"{TRIAL_DIR}/20260212_160314_original.png"

with open(screenshot_path_1, "rb") as f:
    response_1 = requests.post(
        API_URL,
        data={"task": task, "step_number": 1, "user_uuid": USER_UUID},
        files={"screenshot": ("screenshot.png", f, "image/png")}
    )

print(f"Status Code: {response_1.status_code}")
result_1 = response_1.json()
print(f"Response:\n{json.dumps(result_1, indent=2)}\n")
print(f"→ Task ID: {result_1.get('task_id')} (save this for step 2+)")
print(f"→ Total Steps: {result_1.get('total_steps')}")
print(f"→ Current Action: {result_1.get('current_step_what')}")
print(f"→ Box: {result_1.get('box_2d')}")
if result_1.get('next_step'):
    print(f"→ Next What: {result_1['next_step'].get('what')}")
    print(f"→ Next Where: {result_1['next_step'].get('where')}")
print(f"→ Completed: {result_1.get('is_completed')}\n")

# Store task_id for subsequent steps
task_id = result_1.get('task_id')


# --- Step 2: CACHE LOOKUP MODE - pass task_id, get box_2d fast ---
print("=" * 60)
print("STEP 2: CACHE LOOKUP MODE (uses task_id, ~0.3s)")
print("=" * 60)

screenshot_path_2 = f"{TRIAL_DIR}/20260212_160338_original.png"

# Pass task_id from step 1 - info comes from cache
step_2_data = {
    "task": task,
    "step_number": 2,
    "task_id": task_id,
    "user_uuid": USER_UUID
}
print(f"Sending: task_id={step_2_data['task_id']}\n")

with open(screenshot_path_2, "rb") as f:
    response_2 = requests.post(
        API_URL,
        data=step_2_data,
        files={"screenshot": ("screenshot.png", f, "image/png")}
    )

print(f"Status Code: {response_2.status_code}")
result_2 = response_2.json()
print(f"Response:\n{json.dumps(result_2, indent=2)}\n")
print(f"→ Task ID: {result_2.get('task_id')}")
print(f"→ Current Action: {result_2.get('current_step_what')} (from cache)")
print(f"→ Current Target: {result_2.get('current_step_where')} (from cache)")
print(f"→ Box: {result_2.get('box_2d')}")
if result_2.get('next_step'):
    print(f"→ Next What: {result_2['next_step'].get('what')}")
    print(f"→ Next Where: {result_2['next_step'].get('where')}")
print(f"→ Completed: {result_2.get('is_completed')}\n")

# Update task_id in case it changed (fallback scenario)
task_id = result_2.get('task_id') or task_id


# --- Step 3: CACHE LOOKUP MODE - pass task_id, get box_2d fast ---
print("=" * 60)
print("STEP 3: CACHE LOOKUP MODE (uses task_id, ~0.3s)")
print("=" * 60)

screenshot_path_3 = f"{TRIAL_DIR}/20260212_160402_original.png"

# Pass task_id - info comes from cache
step_3_data = {
    "task": task,
    "step_number": 3,
    "task_id": task_id,
    "user_uuid": USER_UUID
}
print(f"Sending: task_id={step_3_data['task_id']}\n")

with open(screenshot_path_3, "rb") as f:
    response_3 = requests.post(
        API_URL,
        data=step_3_data,
        files={"screenshot": ("screenshot.png", f, "image/png")}
    )

print(f"Status Code: {response_3.status_code}")
result_3 = response_3.json()
print(f"Response:\n{json.dumps(result_3, indent=2)}\n")
print(f"→ Task ID: {result_3.get('task_id')}")
print(f"→ Current Action: {result_3.get('current_step_what')} (from cache)")
print(f"→ Current Target: {result_3.get('current_step_where')} (from cache)")
print(f"→ Box: {result_3.get('box_2d')}")
if result_3.get('next_step'):
    print(f"→ Next What: {result_3['next_step'].get('what')}")
    print(f"→ Next Where: {result_3['next_step'].get('where')}")
else:
    print(f"→ Next: None (this is the final step)")
print(f"→ Completed: {result_3.get('is_completed')}\n")

print("=" * 60)
if result_3.get('is_completed'):
    print("TASK COMPLETE! (is_completed=true)")
else:
    print("FLOW COMPLETE (task may need more steps)")
print("=" * 60)
