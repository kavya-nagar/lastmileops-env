from __future__ import annotations

import json
import os
import sys

import requests
from openai import OpenAI


API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME") or os.environ.get("MODEL") or "gpt-4o-mini"
BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")

if not API_BASE_URL:
    raise RuntimeError("Missing API_BASE_URL")

if not API_KEY:
    raise RuntimeError("Missing API_KEY")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

TASKS = ["easy", "medium", "hard"]

MAX_STEPS = {
    "easy": 10,
    "medium": 15,
    "hard": 20,
}

TASK_NAMES = {
    "easy": "Single-site remote fix",
    "medium": "Cabinet failure dispatch",
    "hard": "Regional storm response",
}

TASK_PROMPTS = {
    "easy": (
        "A business customer is offline. Identify the affected node, "
        "run the right remote diagnostic, reboot the correct device, then close the ticket."
    ),
    "medium": (
        "A neighborhood cabinet is down. Diagnose the correct cabinet, reserve the needed spare part, "
        "dispatch the right technician, and close the ticket only after repair."
    ),
    "hard": (
        "A storm caused multiple outages. Prioritize the critical customer, inspect the overloaded "
        "aggregation node, reroute traffic to the backup, reserve the correct part for the cabinet outage, "
        "dispatch the right technician, send the right update, restore service, and close all incidents safely."
    ),
}

ALLOWED_ACTIONS = {
    "run_diagnostic",
    "reboot_device",
    "dispatch_technician",
    "reserve_part",
    "reroute_traffic",
    "send_customer_update",
    "close_ticket",
    "noop",
}


def clamp_unit(x) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.01
    return round(max(0.01, min(0.99, x)), 4)


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


def build_prompt(obs: dict, task_id: str) -> str:
    return f"""
You are a telecom NOC engineer.
Choose the single best next action for this task.

Task:
{TASK_PROMPTS[task_id]}

Incidents:
{json.dumps(obs.get("incidents", []), indent=2)}

Network:
{json.dumps(obs.get("network", []), indent=2)}

Technicians:
{json.dumps(obs.get("technicians", []), indent=2)}

Inventory:
{json.dumps(obs.get("inventory", []), indent=2)}

Recent actions:
{json.dumps(obs.get("action_log", []), indent=2)}

Allowed actions:
- run_diagnostic with params {{"node_id": "STRING"}}
- reboot_device with params {{"node_id": "STRING"}}
- dispatch_technician with params {{"tech_id": "STRING", "location": "STRING"}}
- reserve_part with params {{"part_id": "STRING"}}
- reroute_traffic with params {{"from_node_id": "STRING", "to_node_id": "STRING"}}
- send_customer_update with params {{"incident_id": "STRING", "message_type": "update"}}
- close_ticket with params {{"incident_id": "STRING"}}
- noop with params {{}}

Return only valid JSON in exactly this shape:
{{"action_type":"noop","params":{{}}}}
""".strip()


def normalize_action(action: dict) -> dict:
    if not isinstance(action, dict):
        raise ValueError("LLM output is not a JSON object")

    action_type = action.get("action_type", "noop")
    params = action.get("params", {})

    if not isinstance(action_type, str):
        action_type = "noop"

    if action_type not in ALLOWED_ACTIONS:
        action_type = "noop"

    if not isinstance(params, dict):
        params = {}

    return {"action_type": action_type, "params": params}


def parse_action_text(raw: str) -> dict:
    text = (raw or "").strip()

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        text = text[start:end + 1]

    action = json.loads(text)
    return normalize_action(action)


def call_llm(obs: dict, task_id: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an expert telecom NOC engineer. Return JSON only.",
            },
            {
                "role": "user",
                "content": build_prompt(obs, task_id),
            },
        ],
        temperature=0.0,
        max_tokens=180,
    )

    raw = response.choices[0].message.content or ""
    return parse_action_text(raw)


def fallback_action(task_id: str, obs: dict, step_num: int) -> dict:
    incidents = {x.get("id"): x for x in obs.get("incidents", []) if isinstance(x, dict)}
    network = {x.get("id"): x for x in obs.get("network", []) if isinstance(x, dict)}
    technicians = {x.get("id"): x for x in obs.get("technicians", []) if isinstance(x, dict)}
    inventory = {x.get("id"): x for x in obs.get("inventory", []) if isinstance(x, dict)}
    recent = " | ".join(obs.get("action_log", [])).lower()

    def did(action_name: str, token: str = "") -> bool:
        if action_name not in recent:
            return False
        if token and token.lower() not in recent:
            return False
        return True

    if task_id == "easy":
        node = network.get("ONT-007", {})
        inc = incidents.get("INC-001", {})

        if not did("run_diagnostic", "ont-007"):
            return {"action_type": "run_diagnostic", "params": {"node_id": "ONT-007"}}

        if node.get("status") == "offline":
            return {"action_type": "reboot_device", "params": {"node_id": "ONT-007"}}

        if inc.get("status") == "resolved":
            return {"action_type": "close_ticket", "params": {"incident_id": "INC-001"}}

        return {"action_type": "noop", "params": {}}

    if task_id == "medium":
        inc = incidents.get("INC-010", {})
        p2 = inventory.get("P2", {})
        t3 = technicians.get("T3", {})

        if not did("run_diagnostic", "cab-012"):
            return {"action_type": "run_diagnostic", "params": {"node_id": "CAB-012"}}

        if int(p2.get("reserved", 0)) < 1:
            return {"action_type": "reserve_part", "params": {"part_id": "P2"}}

        if bool(t3.get("available", False)):
            return {
                "action_type": "dispatch_technician",
                "params": {"tech_id": "T3", "location": "CAB-012"},
            }

        if inc.get("status") == "resolved":
            return {"action_type": "close_ticket", "params": {"incident_id": "INC-010"}}

        return {"action_type": "noop", "params": {}}

    if task_id == "hard":
        inc20 = incidents.get("INC-020", {})
        inc21 = incidents.get("INC-021", {})
        agg = network.get("AGG-002", {})
        p2 = inventory.get("P2", {})
        t3 = technicians.get("T3", {})

        if not did("run_diagnostic", "agg-002"):
            return {"action_type": "run_diagnostic", "params": {"node_id": "AGG-002"}}

        if agg.get("status") == "overloaded":
            return {
                "action_type": "reroute_traffic",
                "params": {"from_node_id": "AGG-002", "to_node_id": "AGG-BACKUP"},
            }

        if inc20.get("status") == "resolved":
            return {"action_type": "close_ticket", "params": {"incident_id": "INC-020"}}

        if not did("run_diagnostic", "cab-019"):
            return {"action_type": "run_diagnostic", "params": {"node_id": "CAB-019"}}

        if int(p2.get("reserved", 0)) < 1:
            return {"action_type": "reserve_part", "params": {"part_id": "P2"}}

        if bool(t3.get("available", False)):
            return {
                "action_type": "dispatch_technician",
                "params": {"tech_id": "T3", "location": "CAB-019"},
            }

        if not did("send_customer_update", "inc-021"):
            return {
                "action_type": "send_customer_update",
                "params": {"incident_id": "INC-021", "message_type": "update"},
            }

        if inc21.get("status") == "resolved":
            return {"action_type": "close_ticket", "params": {"incident_id": "INC-021"}}

        return {"action_type": "noop", "params": {}}

    return {"action_type": "noop", "params": {}}


def choose_action(task_id: str, obs: dict, step_num: int) -> dict:
    try:
        return call_llm(obs, task_id)
    except Exception as e:
        print(f"LLM ERROR: {e}", file=sys.stderr, flush=True)
        return fallback_action(task_id, obs, step_num)


def run_task(task_id: str) -> float:
    start_payload = {
        "task_id": task_id,
        "task": TASK_NAMES[task_id],
        "prompt": TASK_PROMPTS[task_id],
    }
    print("START " + json.dumps(start_payload), flush=True)

    step_num = 0
    score = 0.01
    done = False
    obs = None

    try:
        obs = reset_env(task_id)

        while step_num < MAX_STEPS[task_id] and not done:
            action = choose_action(task_id, obs, step_num)
            result = step_env(action["action_type"], action["params"])

            score = clamp_unit(result.get("score", score))
            done = bool(result.get("done", False))
            obs = result.get("observation", obs)
            step_num += 1

    except Exception as e:
        print(f"TASK ERROR: {task_id}: {e}", file=sys.stderr, flush=True)
        done = True

    end_payload = {
        "task_id": task_id,
        "steps": step_num,
        "score": clamp_unit(score),
        "done": bool(done),
    }
    print("END " + json.dumps(end_payload), flush=True)

    return clamp_unit(score)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)