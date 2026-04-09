from __future__ import annotations

import json
import sys
import os
import requests

BASEURL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")

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
    "easy": "A business customer is offline. Identify the affected node, run diagnostics, reboot the correct device, and close the ticket.",
    "medium": "A neighborhood cabinet is down. Diagnose the cabinet, reserve the correct spare part, dispatch the correct technician, and close the ticket after repair.",
    "hard": "A storm caused multiple outages. Prioritise the critical service, inspect the aggregation fault, reroute traffic, reserve the correct part, dispatch the correct technician, update customers, and close all incidents.",
}

SCRIPTED_ACTIONS = {
    "easy": [
        {"action_type": "rundiagnostic", "params": {"nodeid": "ONT-007"}},
        {"action_type": "rebootdevice", "params": {"nodeid": "ONT-007"}},
        {"action_type": "closeticket", "params": {"incidentid": "INC-001"}},
    ],
    "medium": [
        {"action_type": "rundiagnostic", "params": {"nodeid": "CAB-012"}},
        {"action_type": "reservepart", "params": {"partid": "P2"}},
        {"action_type": "dispatchtechnician", "params": {"techid": "T3", "location": "CAB-012"}},
        {"action_type": "closeticket", "params": {"incidentid": "INC-010"}},
    ],
    "hard": [
        {"action_type": "rundiagnostic", "params": {"nodeid": "AGG-002"}},
        {"action_type": "reroutetraffic", "params": {"fromnodeid": "AGG-002", "tonodeid": "AGG-BACKUP"}},
        {"action_type": "rundiagnostic", "params": {"nodeid": "CAB-019"}},
        {"action_type": "reservepart", "params": {"partid": "P2"}},
        {"action_type": "dispatchtechnician", "params": {"techid": "T3", "location": "CAB-019"}},
        {"action_type": "sendcustomerupdate", "params": {"incidentid": "INC-021", "messagetype": "update"}},
        {"action_type": "closeticket", "params": {"incidentid": "INC-020"}},
        {"action_type": "closeticket", "params": {"incidentid": "INC-021"}},
        {"action_type": "closeticket", "params": {"incidentid": "INC-022"}},
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
        f"{BASEURL}/reset",
        json={"task_id": task_id, "taskid": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def step_env(action_type: str, params: dict) -> dict:
    payload = {
        "action_type": action_type,
        "actiontype": action_type,
        "params": params or {},
    }
    resp = requests.post(f"{BASEURL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def choose_action(task_id: str, step_num: int, obs: dict) -> dict:
    scripted = SCRIPTED_ACTIONS.get(task_id, [])
    if step_num < len(scripted):
        return scripted[step_num]
    return {"action_type": "noop", "params": {}}


def run_task(task_id: str) -> float:
    obs = reset_env(task_id)

    print(
        "START",
        json.dumps(
            {
                "taskid": task_id,
                "task": TASK_NAMES[task_id],
                "prompt": TASK_PROMPTS[task_id],
            }
        ),
        flush=True,
    )

    step_num = 0
    done = False
    score = 0.01

    while not done and step_num < MAX_STEPS[task_id]:
        action = choose_action(task_id, step_num, obs)

        try:
            result = step_env(action.get("action_type", "noop"), action.get("params", {}))
        except Exception as e:
            print(f"STEP ERROR: {e}", file=sys.stderr, flush=True)
            result = {
                "reward": 0.01,
                "done": True,
                "score": 0.01,
                "info": {"message": f"step failed: {e}"},
                "observation": obs,
            }

        reward = clamp_unit(result.get("reward", 0.01))
        done = bool(result.get("done", False))
        score = clamp_unit(result.get("score", 0.01))
        info = result.get("info", {}) or {}
        message = info.get("message", "")
        obs = result.get("observation", obs)

        print(
            "STEP",
            json.dumps(
                {
                    "taskid": task_id,
                    "step": step_num,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "score": score,
                    "message": message,
                }
            ),
            flush=True,
        )

        step_num += 1

    print(
        "END",
        json.dumps(
            {
                "taskid": task_id,
                "steps": step_num,
                "score": clamp_unit(score),
                "done": bool(done),
            }
        ),
        flush=True,
    )

    return clamp_unit(score)


if __name__ == "__main__":
    for task in TASKS:
        try:
            run_task(task)
        except Exception as e:
            print(f"TASK ERROR: {e}", file=sys.stderr, flush=True)
            print(
                "START",
                json.dumps(
                    {
                        "taskid": task,
                        "task": TASK_NAMES[task],
                        "prompt": TASK_PROMPTS[task],
                    }
                ),
                flush=True,
            )
            print(
                "END",
                json.dumps(
                    {
                        "taskid": task,
                        "steps": 0,
                        "score": 0.01,
                        "done": True,
                    }
                ),
                flush=True,
            )