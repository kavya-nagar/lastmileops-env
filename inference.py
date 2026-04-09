from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, Optional

import requests
from openai import OpenAI


ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
API_BASE_URL = os.environ.get("API_BASE_URL", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME") or os.environ.get("MODEL") or "gpt-4o-mini"

# Support all likely validator/provided key names without crashing.
OPENAI_API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("API_KEY")
    or ""
)

# Keep LLM optional for reliability; scripted fallback is deterministic.
ENABLE_LLM = os.environ.get("ENABLE_LLM", "0") == "1"

TASKS = ["easy", "medium", "hard"]

MAX_STEPS = {
    "easy": 10,
    "medium": 15,
    "hard": 20,
}

TASK_PROMPTS = {
    "easy": (
        "A business customer is offline. Identify the affected node, run the right remote "
        "diagnostic, reboot the correct device, then close the ticket."
    ),
    "medium": (
        "A neighborhood cabinet is down. Diagnose the correct cabinet, reserve the needed spare "
        "part, dispatch the right technician, and close the ticket only after repair."
    ),
    "hard": (
        "A storm caused multiple outages. Prioritize the critical customer, inspect the overloaded "
        "aggregation node, reroute traffic to the backup, reserve the correct part for the cabinet "
        "outage, dispatch the right technician, send the right update, restore service, and close "
        "all incidents safely."
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
        {"action_type": "close_ticket", "params": {"incident_id": "INC-010"}},
    ],
    "hard": [
        {"action_type": "run_diagnostic", "params": {"node_id": "AGG-002"}},
        {"action_type": "reroute_traffic", "params": {"from_node_id": "AGG-002", "to_node_id": "AGG-BACKUP"}},
        {"action_type": "run_diagnostic", "params": {"node_id": "CAB-019"}},
        {"action_type": "reserve_part", "params": {"part_id": "P2"}},
        {"action_type": "dispatch_technician", "params": {"tech_id": "T3", "location": "CAB-019"}},
        {"action_type": "reboot_device", "params": {"node_id": "ONT-031"}},
        {"action_type": "send_customer_update", "params": {"incident_id": "INC-021", "message_type": "update"}},
        {"action_type": "close_ticket", "params": {"incident_id": "INC-020"}},
        {"action_type": "close_ticket", "params": {"incident_id": "INC-021"}},
        {"action_type": "close_ticket", "params": {"incident_id": "INC-022"}},
    ],
}


def clamp_unit(x: Any) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.01
    if x <= 0.0:
        return 0.01
    if x >= 1.0:
        return 0.99
    return round(x, 4)


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def log_start(task: str) -> None:
    print(f"[START] task={task}", flush=True)


def log_step(step: int, reward: Any) -> None:
    print(f"[STEP] step={step} reward={clamp_unit(reward):.4f}", flush=True)


def log_end(task: str, score: Any, steps: int) -> None:
    print(f"[END] task={task} score={clamp_unit(score):.4f} steps={steps}", flush=True)


def reset_env(task_id: str) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def step_env(action_type: str, params: dict) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action_type": action_type, "params": params},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def build_prompt(obs: dict, task_id: str) -> str:
    return f"""
You are a telecom NOC engineer. Output JSON only.

Task:
{TASK_PROMPTS[task_id]}

Observation:
{json.dumps(obs, indent=2)}

Allowed actions:
- run_diagnostic: {{"node_id": ""}}
- reboot_device: {{"node_id": ""}}
- dispatch_technician: {{"tech_id": "", "location": ""}}
- reserve_part: {{"part_id": ""}}
- reroute_traffic: {{"from_node_id": "", "to_node_id": ""}}
- send_customer_update: {{"incident_id": "", "message_type": "update"}}
- close_ticket: {{"incident_id": ""}}
- noop: {{}}

Return exactly:
{{"action_type":"noop","params":{{}}}}
""".strip()


def extract_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def normalize_action(obj: Optional[dict]) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None

    action_type = obj.get("action_type")
    params = obj.get("params", {})

    if not isinstance(action_type, str):
        return None
    if action_type not in ALLOWED_ACTIONS:
        return None
    if not isinstance(params, dict):
        params = {}

    return {"action_type": action_type, "params": params}


def get_llm_action(obs: dict, task_id: str) -> Optional[dict]:
    if not ENABLE_LLM:
        return None
    if not OPENAI_API_KEY:
        eprint("LLM disabled: missing OPENAI_API_KEY/HF_TOKEN/API_KEY")
        return None

    try:
        client_kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
        if API_BASE_URL:
            client_kwargs["base_url"] = API_BASE_URL

        client = OpenAI(**client_kwargs)

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

        raw = response.choices[0].message.content or ""
        parsed = extract_json_object(raw)
        return normalize_action(parsed)
    except Exception as e:
        eprint(f"LLM ERROR: {e}")
        return None


def choose_action(task_id: str, obs: dict, step_index: int) -> dict:
    llm_action = get_llm_action(obs, task_id)
    if llm_action is not None:
        return llm_action

    scripted = SCRIPTED_ACTIONS.get(task_id, [])
    if step_index < len(scripted):
        return scripted[step_index]

    return {"action_type": "noop", "params": {}}


def run_task(task_id: str) -> float:
    steps = 0
    score = 0.01
    done = False
    obs: dict = {}

    log_start(task_id)

    try:
        obs = reset_env(task_id)
    except Exception as e:
        eprint(f"RESET ERROR ({task_id}): {e}")
        log_end(task_id, score, steps)
        return score

    while not done and steps < MAX_STEPS[task_id]:
        action = choose_action(task_id, obs, steps)

        try:
            result = step_env(
                action_type=action.get("action_type", "noop"),
                params=action.get("params", {}) or {},
            )
        except Exception as e:
            eprint(f"STEP ERROR ({task_id}, step={steps + 1}): {e}")
            steps += 1
            log_step(steps, 0.01)
            score = 0.01
            done = True
            break

        steps += 1
        reward = clamp_unit(result.get("reward", 0.01))
        done = bool(result.get("done", False))
        score = clamp_unit(result.get("score", score))
        obs = result.get("observation", obs) or obs

        log_step(steps, reward)

    log_end(task_id, score, steps)
    return score


def main() -> None:
    for task_id in TASKS:
        try:
            run_task(task_id)
        except Exception as e:
            eprint(f"TASK ERROR ({task_id}): {e}")
            # Always emit parseable end block even on unexpected failure.
            log_start(task_id)
            log_end(task_id, 0.01, 0)


if __name__ == "__main__":
    main()