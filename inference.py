import os
import sys
import json
import subprocess

# Auto-install missing dependencies
for pkg in ["requests", "openai"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import requests
from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "<your-active-api-base-url>")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "<your-active-model-name>")
HF_TOKEN     = os.environ.get("HF_TOKEN")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

BASE_URL = os.environ.get("ENV_BASE_URL", "https://kavyanagar-lastmileops-env.hf.space")

TASKS     = ["easy", "medium", "hard"]
MAX_STEPS = {"easy": 10, "medium": 15, "hard": 20}

TASK_NAMES = {
    "easy":   "Single-site remote fix",
    "medium": "Cabinet failure dispatch",
    "hard":   "Regional storm response",
}

TASK_PROMPTS = {
    "easy":   "A business customer is offline. Identify the affected node, run the right remote diagnostic, reboot the correct device, then close the ticket.",
    "medium": "A neighborhood cabinet is down. Diagnose the correct cabinet, reserve the needed spare part, dispatch the right technician, and close the ticket only after repair.",
    "hard":   "A storm caused multiple outages. Prioritize the critical customer, inspect the overloaded aggregation node, reroute traffic to the backup, reserve the correct part for the cabinet outage, dispatch the right technician, send the right update, restore service, and close all incidents safely.",
}


def reset_env(task_id: str) -> dict:
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def step_env(action_type: str, params: dict) -> dict:
    payload = {"action_type": action_type, "params": params}
    resp = requests.post(f"{BASE_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_prompt(obs: dict) -> str:
    return f"""You are a Telecom NOC engineer. Resolve all incidents efficiently.

INCIDENTS:
{json.dumps(obs.get("incidents", []), indent=2)}

NETWORK NODES:
{json.dumps(obs.get("network", []), indent=2)}

TECHNICIANS:
{json.dumps(obs.get("technicians", []), indent=2)}

INVENTORY:
{json.dumps(obs.get("inventory", []), indent=2)}

RECENT ACTIONS:
{json.dumps(obs.get("action_log", []), indent=2)}

AVAILABLE ACTIONS:
- run_diagnostic: {{"node_id": "<id>"}}
- reboot_device: {{"node_id": "<id>"}}
- dispatch_technician: {{"tech_id": "<id>", "location": "<site>"}}
- reserve_part: {{"part_id": "<id>"}}
- reroute_traffic: {{"from_node_id": "<id>", "to_node_id": "<id>"}}
- send_customer_update: {{"incident_id": "<id>", "message_type": "update"}}
- close_ticket: {{"incident_id": "<id>"}}
- noop: {{}}

Respond with ONLY a JSON object like:
{{"action_type": "run_diagnostic", "params": {{"node_id": "ONT-007"}}}}
"""


def get_llm_action(obs: dict) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert NOC engineer. Always respond with valid JSON only."},
                {"role": "user",   "content": build_prompt(obs)},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"LLM ERROR: {e}", file=sys.stderr, flush=True)
        return {"action_type": "noop", "params": {}}


def run_task(task_id: str) -> float:
    obs = reset_env(task_id)

    # EXACT required [START] format
    print("[START] " + json.dumps({
        "task_id": task_id,
        "task":    TASK_NAMES[task_id],
        "prompt":  TASK_PROMPTS[task_id],
    }), flush=True)

    step_num  = 0
    done      = False
    score     = 0.0

    while not done and step_num < MAX_STEPS[task_id]:
        action = get_llm_action(obs)
        result = step_env(action.get("action_type", "noop"), action.get("params", {}))

        step_num += 1
        reward   = result.get("reward", 0.0)
        done     = result.get("done", False)
        score    = result.get("score", 0.0)
        message  = result.get("info", {}).get("message", "")
        obs      = result.get("observation", obs)

        # EXACT required [STEP] format
        print("[STEP] " + json.dumps({
            "task_id": task_id,
            "step":    step_num,
            "action":  action,
            "reward":  reward,
            "done":    done,
            "score":   score,
            "message": message,
        }), flush=True)

    # EXACT required [END] format
    print("[END] " + json.dumps({
        "task_id": task_id,
        "steps":   step_num,
        "score":   score,
        "done":    done,
    }), flush=True)

    return score


if __name__ == "__main__":
    results = {}
    for task in TASKS:
        try:
            results[task] = run_task(task)
        except Exception as e:
            print(f"Task {task} failed: {e}", file=sys.stderr, flush=True)
            results[task] = 0.0