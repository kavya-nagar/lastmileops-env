import os
import sys
import json
import subprocess

for pkg in ["requests", "openai"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "<your-active-api-base-url>")
MODEL_NAME = os.environ.get("MODEL_NAME", "<your-active-model-name>")
HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_URL = os.environ.get("ENV_BASE_URL", "https://kavyanagar-lastmileops-env.hf.space")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = {"easy": 10, "medium": 15, "hard": 20}

TASK_NAMES = {
    "easy": "Single-site remote fix",
    "medium": "Cabinet failure dispatch",
    "hard": "Regional storm response",
}

TASK_PROMPTS = {
    "easy": "A business customer is offline. Identify the affected node, run the right remote diagnostic, reboot the correct device, then close the ticket.",
    "medium": "A neighborhood cabinet is down. Diagnose the correct cabinet, reserve the needed spare part, dispatch the right technician, and close the ticket only after repair.",
    "hard": "A storm caused multiple outages. Prioritize the critical customer, inspect the overloaded aggregation node, reroute traffic to the backup, reserve the correct part for the cabinet outage, dispatch the right technician, send the right update, restore service, and close all incidents safely.",
}


def clamp_score(x):
    try:
        x = float(x)
    except Exception:
        x = 0.01
    return max(0.01, min(0.99, x))


def reset_env(task_id: str) -> dict:
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def step_env(action_type: str, params: dict) -> dict:
    resp = requests.post(
        f"{BASE_URL}/step",
        json={"action_type": action_type, "params": params},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def build_prompt(obs: dict) -> str:
    return f"""You are a Telecom NOC engineer. Resolve all incidents efficiently.

INCIDENTS:
{json.dumps(obs.get("incidents", []), indent=2)}

NETWORK:
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

Return ONLY valid JSON like:
{{"action_type":"run_diagnostic","params":{{"node_id":"ONT-007"}}}}
"""


def get_llm_action(obs: dict) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert NOC engineer. Respond with JSON only.",
                },
                {
                    "role": "user",
                    "content": build_prompt(obs),
                },
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action = json.loads(raw.strip())
        if not isinstance(action, dict):
            return {"action_type": "noop", "params": {}}
        return {
            "action_type": action.get("action_type", "noop"),
            "params": action.get("params", {}) or {},
        }
    except Exception as e:
        print(f"LLM ERROR: {e}", file=sys.stderr, flush=True)
        return {"action_type": "noop", "params": {}}


def run_task(task_id: str) -> float:
    obs = reset_env(task_id)

    print("[START] " + json.dumps({
        "task_id": task_id,
        "task": TASK_NAMES[task_id],
        "prompt": TASK_PROMPTS[task_id],
    }), flush=True)

    step_num = 0
    done = False
    score = 0.01

    while not done and step_num < MAX_STEPS[task_id]:
        action = get_llm_action(obs)

        try:
            result = step_env(
                action.get("action_type", "noop"),
                action.get("params", {}),
            )
        except Exception as e:
            print(f"STEP ERROR: {e}", file=sys.stderr, flush=True)
            result = {
                "reward": 0.0,
                "done": True,
                "score": 0.01,
                "info": {"message": f"step failed: {e}"},
                "observation": obs,
            }

        step_num += 1
        reward = result.get("reward", 0.0)
        done = bool(result.get("done", False))
        score = clamp_score(result.get("score", 0.01))
        message = result.get("info", {}).get("message", "")
        obs = result.get("observation", obs)

        print("[STEP] " + json.dumps({
            "task_id": task_id,
            "step": step_num,
            "action": action,
            "reward": reward,
            "done": done,
            "score": score,
            "message": message,
        }), flush=True)

    print("[END] " + json.dumps({
        "task_id": task_id,
        "steps": step_num,
        "score": clamp_score(score),
        "done": done,
    }), flush=True)

    return clamp_score(score)


if __name__ == "__main__":
    results = {}
    for task in TASKS:
        try:
            results[task] = clamp_score(run_task(task))
        except Exception as e:
            print(f"TASK ERROR: {e}", file=sys.stderr, flush=True)
            results[task] = 0.01
            print("[START] " + json.dumps({
                "task_id": task,
                "task": TASK_NAMES[task],
                "prompt": TASK_PROMPTS[task],
            }), flush=True)
            print("[END] " + json.dumps({
                "task_id": task,
                "steps": 0,
                "score": 0.01,
                "done": True,
            }), flush=True)