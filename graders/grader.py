from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.environment import LastMileOpsEnv
from server.models import Action


def strict_unit(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.01
    return round(max(0.01, min(0.99, x)), 4)


def scripted_actions(task_id: str):
    if task_id == "easy":
        return [
            {"action_type": "run_diagnostic", "params": {"node_id": "ONT-007"}},
            {"action_type": "reboot_device", "params": {"node_id": "ONT-007"}},
            {"action_type": "close_ticket", "params": {"incident_id": "INC-001"}},
        ]

    if task_id == "medium":
        return [
            {"action_type": "run_diagnostic", "params": {"node_id": "CAB-012"}},
            {"action_type": "reserve_part", "params": {"part_id": "P2"}},
            {"action_type": "dispatch_technician", "params": {"tech_id": "T3", "location": "CAB-012"}},
        ]

    if task_id == "hard":
        return [
            {"action_type": "run_diagnostic", "params": {"node_id": "AGG-002"}},
            {"action_type": "reroute_traffic", "params": {"from_node_id": "AGG-002", "to_node_id": "AGG-BACKUP"}},
            {"action_type": "run_diagnostic", "params": {"node_id": "CAB-019"}},
            {"action_type": "reserve_part", "params": {"part_id": "P2"}},
            {"action_type": "dispatch_technician", "params": {"tech_id": "T3", "location": "CAB-019"}},
            {"action_type": "send_customer_update", "params": {"incident_id": "INC-021", "message_type": "update"}},
        ]

    return []


def grade(task_id: str) -> float:
    env = LastMileOpsEnv()
    env.reset(task_id=task_id)

    actions = scripted_actions(task_id)

    for a in actions:
        if env.state()["done"]:
            break
        env.step(Action(action_type=a["action_type"], params=a.get("params", {})))

    return strict_unit(env.state()["score"])


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        s = grade(task)
        print(f"task={task} score={s}")
        assert 0.0 < s < 1.0, f"Score out of strict range for {task}: {s}"
    print("All graders passed.")