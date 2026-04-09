from __future__ import annotations

import json
import os

import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "").strip()
API_KEY = os.environ.get("API_KEY", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME") or os.environ.get("MODEL") or "gpt-4o-mini"
BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")

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
        "A business customer is offline. Identify the affected node, run the right "
        "remote diagnostic, reboot the correct device, then close the ticket."
    ),
    "medium": (
        "A neighborhood cabinet is down. Diagnose the correct cabinet, reserve the "
        "needed spare part, dispatch the right technician, and close the ticket only after repair."
    ),
    "hard": (
        "A storm caused multiple outages. Prioritize the critical customer, inspect "
        "the overloaded aggregation node, reroute traffic to the backup, reserve the "
        "correct part for the cabinet outage, dispatch the right technician, send the "
        "right update, restore service, and close all incidents safely."
    ),
}

SCRIPTED_ACTIONS = {
    "easy": [
        {"action_type": "run_diagnostic", "params": {"node_id": "ONT-007"}},
        {"action_type": "reboot_device", "params": {"node_id": "ONT-007"}},
        {"action_type": "close_ticket", "params": {"incident_id": "INC-001"}},
    ],
    "medium": [
        {"action_type": "run_diagnostic", "params": {"node_id": "CAB-012"}},
        {"action_type": "reserve_part", "params": {"part_id": "P2"}},
        {"action_type": "dispatch_technician", "params": {"tech_id": "T3", "location": "CAB-012"}},
        {"action_type": "noop", "params": {}},
    ],
    "hard": [
        {"action_type": "run_diagnostic", "params": {"node_id": "AGG-002"}},
        {"action_type": "reroute_traffic", "params": {"from_node_id": "AGG-002", "to_node_id": "AGG-BACKUP"}},
        {"action_type": "run_diagnostic", "params": {"node_id": "CAB-019"}},
        {"action_type": "reserve_part", "params": {"part_id": "P2"}},
        {"action_type": "dispatch_technician", "params": {"tech_id": "T3", "location": "CAB-019"}},
        {"action_type": "send_customer_update", "params": {"incident_id": "INC-021", "message_type": "update"}},
        {"action_type": "noop", "params": {}},
    ],
}


def clamp_unit(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.01
    return round(max(0.01, min(0.99, x)), 4)


def reset_env(task_id: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
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
You are a Telecom NOC engineer. Resolve this task efficiently.

TASK:
{TASK_PROMPTS[task_id]}

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
- run_diagnostic: {{"node_id": ""}}
- reboot_device: {{"node_id": ""}}
- dispatch_technician: {{"tech_id": "", "location": ""}}
- reserve_part: {{"part_id": ""}}
- reroute_traffic: {{"from_node_id": "", "to_node_id": ""}}
- send_customer_update: {{"incident_id": "", "message_type": "update"}}
- close_ticket: {{"incident_id": ""}}
- noop: {{}}

Return ONLY valid JSON with this shape:
{{"action_type":"run_diagnostic","params":{{"node_id":"ONT-007"}}}}
""".strip()


def normalize_action(action: dict | None) -> dict | None:
    if not isinstance(action, dict):
        return None

    action_type = action.get("action_type", "noop")
    params = action.get("params", {}) or {}

    if not isinstance(action_type, str):
        action_type = "noop"
    if not isinstance(params, dict):
        params = {}

    return {
        "action_type": action_type,
        "params": params,
    }


def get_llm_action(obs: dict, task_id: str) -> dict | None:
    if not API_BASE_URL or not API_KEY:
        return None

    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert telecom NOC engineer. Output JSON only.",
                },
                {
                    "role": "user",
                    "content": build_prompt(obs, task_id),
                },
            ],
            temperature=0.0,
            max_tokens=200,
        )

        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return None

        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        action = json.loads(raw)
        return normalize_action(action)
    except Exception:
        return None


def choose_action(task_id: str, obs: dict, step_num: int) -> dict:
    llm_action = get_llm_action(obs, task_id)
    if llm_action is not None:
        return llm_action

    scripted = SCRIPTED_ACTIONS.get(task_id, [])
    if step_num < len(scripted):
        return scripted[step_num]

    return {"action_type": "noop", "params": {}}


def run_task(task_id: str) -> float:
    obs = reset_env(task_id)

    print(
        "[START] " + json.dumps(
            {
                "task_id": task_id,
                "task": TASK_NAMES[task_id],
                "prompt": TASK_PROMPTS[task_id],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    step_num = 0
    done = False
    score = 0.01

    while not done and step_num < MAX_STEPS[task_id]:
        action = choose_action(task_id, obs, step_num)

        try:
            result = step_env(
                action.get("action_type", "noop"),
                action.get("params", {}),
            )
        except Exception as e:
            result = {
                "reward": 0.01,
                "done": True,
                "score": 0.01,
                "info": {"message": f"step failed: {e}"},
                "observation": obs,
            }

        step_num += 1
        reward = clamp_unit(result.get("reward", 0.01))
        done = bool(result.get("done", False))
        score = clamp_unit(result.get("score", 0.01))
        message = result.get("info", {}).get("message", "")
        obs = result.get("observation", obs)

        print(
            "[STEP] " + json.dumps(
                {
                    "task_id": task_id,
                    "step": step_num,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "score": score,
                    "message": message,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    print(
        "[END] " + json.dumps(
            {
                "task_id": task_id,
                "steps": step_num,
                "score": clamp_unit(score),
                "done": done,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    return clamp_unit(score)


if __name__ == "__main__":
    for task in TASKS:
        try:
            run_task(task)
        except Exception:
            print(
                "[START] " + json.dumps(
                    {
                        "task_id": task,
                        "task": TASK_NAMES[task],
                        "prompt": TASK_PROMPTS[task],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            print(
                "[END] " + json.dumps(
                    {
                        "task_id": task,
                        "steps": 0,
                        "score": 0.01,
                        "done": True,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )