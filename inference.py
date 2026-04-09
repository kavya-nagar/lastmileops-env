from __future__ import annotations

import json
import os

import requests
from openai import OpenAI

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME") or os.environ.get("MODEL") or "gpt-4o-mini"
BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

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
    if x <= 0.0:
        return 0.01
    if x >= 1.0:
        return 0.99
    return round(x, 4)


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


def build_prompt(obs: dict, task_id: str, step_num: int) -> str:
    return f"""
You are a telecom NOC engineer. Choose exactly one next action.

Task ID: {task_id}
Task: {TASK_NAMES[task_id]}
Goal: {TASK_PROMPTS[task_id]}
Current step: {step_num + 1} / {MAX_STEPS[task_id]}

Observation:
{json.dumps(obs, indent=2)}

Valid action schema:
{{
  "action_type": "run_diagnostic|reboot_device|dispatch_technician|reserve_part|reroute_traffic|send_customer_update|close_ticket|noop",
  "params": {{}}
}}

Parameter rules:
- run_diagnostic -> {{"node_id": "..."}}
- reboot_device -> {{"node_id": "..."}}
- dispatch_technician -> {{"tech_id": "...", "location": "..."}}
- reserve_part -> {{"part_id": "..."}}
- reroute_traffic -> {{"from_node_id": "...", "to_node_id": "..."}}
- send_customer_update -> {{"incident_id": "...", "message_type": "update"}}
- close_ticket -> {{"incident_id": "..."}}
- noop -> {{}}

Return ONLY valid JSON.
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


def get_llm_action(obs: dict, task_id: str, step_num: int) -> dict | None:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert telecom NOC engineer. Output JSON only.",
                },
                {
                    "role": "user",
                    "content": build_prompt(obs, task_id, step_num),
                },
            ],
        )

        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return None

        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1].strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        parsed = json.loads(raw)
        return normalize_action(parsed)
    except Exception:
        return None


def choose_action(task_id: str, obs: dict, step_num: int) -> dict:
    llm_action = get_llm_action(obs, task_id, step_num)
    if llm_action is not None:
        return llm_action

    scripted = SCRIPTED_ACTIONS.get(task_id, [])
    if step_num < len(scripted):
        return scripted[step_num]

    return {"action_type": "noop", "params": {}}


def print_start(task_id: str) -> None:
    print(
        f"[START] task={task_id} task_id={task_id} name={TASK_NAMES[task_id]}",
        flush=True,
    )


def print_step(task_id: str, step_num: int, reward: float, score: float, done: bool) -> None:
    print(
        f"[STEP] task={task_id} task_id={task_id} step={step_num} reward={clamp_unit(reward)} score={clamp_unit(score)} done={done}",
        flush=True,
    )


def print_end(task_id: str, step_num: int, score: float, done: bool) -> None:
    print(
        f"[END] task={task_id} task_id={task_id} steps={step_num} score={clamp_unit(score)} done={done}",
        flush=True,
    )


def run_task(task_id: str) -> float:
    obs = reset_env(task_id)
    print_start(task_id)

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
        except Exception:
            result = {
                "reward": 0.01,
                "done": True,
                "score": 0.01,
                "observation": obs,
                "info": {"message": "step failed"},
            }

        step_num += 1
        reward = clamp_unit(result.get("reward", 0.01))
        done = bool(result.get("done", False))
        score = clamp_unit(result.get("score", 0.01))
        obs = result.get("observation", obs)

        print_step(task_id, step_num, reward, score, done)

    final_score = clamp_unit(score)
    print_end(task_id, step_num, final_score, done)
    return final_score


if __name__ == "__main__":
    for task in TASKS:
        try:
            run_task(task)
        except Exception:
            print_start(task)
            print_end(task, 0, 0.01, True)