"""
Test script for the execute_step API endpoint with OPTIMIZED latency.


"""

import json
import requests

API_URL = "http://localhost:8000/execute_step"
TRIAL_DIR = "/Users/user/tryroger/mouse/mac_app/swift_app/backend/trial"

task = "Change Aadhar address"


# --- Step 1: FULL ANALYSIS - get current action + box_2d + next_step preview ---
print("=" * 60)
print("STEP 1: FULL ANALYSIS (no current_what/where)")
print("=" * 60)

screenshot_path_1 = f"{TRIAL_DIR}/20260212_160314_original.png"

with open(screenshot_path_1, "rb") as f:
    response_1 = requests.post(
        API_URL,
        data={"task": task, "step_number": 1},
        files={"screenshot": ("screenshot.png", f, "image/png")}
    )

print(f"Status Code: {response_1.status_code}")
result_1 = response_1.json()
print(f"Response:\n{json.dumps(result_1, indent=2)}\n")
print(f"→ Current Action: {result_1.get('current_step_what')}")
print(f"→ Current Target: {result_1.get('current_step_where')}")
print(f"→ Box: {result_1.get('box_2d')}")
if result_1.get('next_step'):
    print(f"→ Next What: {result_1['next_step'].get('what')}")
    print(f"→ Next Where: {result_1['next_step'].get('where')}")
print(f"→ Completed: {result_1.get('is_completed')}\n")

# Store next_step for step 2
next_step_1 = result_1.get('next_step')


# --- Step 2: FOCUSED ANALYSIS - pass current_what/where from step 1's next_step ---
print("=" * 60)
print("STEP 2: FOCUSED ANALYSIS (with current_what/where from step 1)")
print("=" * 60)

screenshot_path_2 = f"{TRIAL_DIR}/20260212_160338_original.png"

# Pass current_what/where from previous next_step
step_2_data = {
    "task": task,
    "step_number": 2,
    "current_what": next_step_1.get("what") if next_step_1 else "",
    "current_where": next_step_1.get("where") if next_step_1 else ""
}
print(f"Sending: current_what={step_2_data['current_what']}")
print(f"Sending: current_where={step_2_data['current_where']}\n")

with open(screenshot_path_2, "rb") as f:
    response_2 = requests.post(
        API_URL,
        data=step_2_data,
        files={"screenshot": ("screenshot.png", f, "image/png")}
    )

print(f"Status Code: {response_2.status_code}")
result_2 = response_2.json()
print(f"Response:\n{json.dumps(result_2, indent=2)}\n")
print(f"→ Current Action: {result_2.get('current_step_what')} (null - client already knows)")
print(f"→ Current Target: {result_2.get('current_step_where')} (null - client already knows)")
print(f"→ Box: {result_2.get('box_2d')}")
if result_2.get('next_step'):
    print(f"→ Next What: {result_2['next_step'].get('what')}")
    print(f"→ Next Where: {result_2['next_step'].get('where')}")
print(f"→ Completed: {result_2.get('is_completed')}\n")

# Store next_step for step 3
next_step_2 = result_2.get('next_step')


# --- Step 3: FOCUSED ANALYSIS - pass current_what/where from step 2's next_step ---
print("=" * 60)
print("STEP 3: FOCUSED ANALYSIS (with current_what/where from step 2)")
print("=" * 60)

screenshot_path_3 = f"{TRIAL_DIR}/20260212_160402_original.png"

# Pass current_what/where from previous next_step
step_3_data = {
    "task": task,
    "step_number": 3,
    "current_what": next_step_2.get("what") if next_step_2 else "",
    "current_where": next_step_2.get("where") if next_step_2 else ""
}
print(f"Sending: current_what={step_3_data['current_what']}")
print(f"Sending: current_where={step_3_data['current_where']}\n")

with open(screenshot_path_3, "rb") as f:
    response_3 = requests.post(
        API_URL,
        data=step_3_data,
        files={"screenshot": ("screenshot.png", f, "image/png")}
    )

print(f"Status Code: {response_3.status_code}")
result_3 = response_3.json()
print(f"Response:\n{json.dumps(result_3, indent=2)}\n")
print(f"→ Current Action: {result_3.get('current_step_what')} (null - client already knows)")
print(f"→ Current Target: {result_3.get('current_step_where')} (null - client already knows)")
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
