from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.environment import LastMileOpsEnvironment


def grade(task_id: str) -> float:
    env = LastMileOpsEnvironment()
    env.reset(task_id=task_id)

    if task_id == "easy":
        actions = [
            {"action_type": "list_incidents"},
            {"action_type": "inspect_node", "node_id": "ONT-7"},
            {"action_type": "run_remote_test", "node_id": "ONT-7", "test_type": "ping"},
            {"action_type": "reboot_device", "node_id": "ONT-7"},
            {"action_type": "send_customer_update", "ticket_id": "INC-1001", "message_template": "service_restored"},
            {"action_type": "close_incident", "ticket_id": "INC-1001"},
        ]
    elif task_id == "medium":
        actions = [
            {"action_type": "list_incidents"},
            {"action_type": "inspect_node", "node_id": "CAB-22"},
            {"action_type": "run_remote_test", "node_id": "CAB-22", "test_type": "optical_power"},
            {"action_type": "reserve_part", "depot_id": "D-2", "part_name": "sfp_module", "qty": 1},
            {"action_type": "dispatch_technician", "tech_id": "T-02", "ticket_id": "INC-2002"},
            {"action_type": "send_customer_update", "ticket_id": "INC-2002", "message_template": "tech_dispatched"},
            {"action_type": "close_incident", "ticket_id": "INC-2002"},
        ]
    elif task_id == "hard":
        actions = [
            {"action_type": "list_incidents"},
            {"action_type": "inspect_node", "node_id": "AGG-9"},
            {"action_type": "run_remote_test", "node_id": "AGG-9", "test_type": "cpu_profile"},
            {"action_type": "reroute_traffic", "node_id": "AGG-9", "backup_node_id": "AGG-9B"},
            {"action_type": "reboot_device", "node_id": "AGG-9"},
            {"action_type": "inspect_node", "node_id": "CAB-41"},
            {"action_type": "run_remote_test", "node_id": "CAB-41", "test_type": "optical_power"},
            {"action_type": "reserve_part", "depot_id": "D-3", "part_name": "sfp_module", "qty": 1},
            {"action_type": "dispatch_technician", "tech_id": "T-04", "ticket_id": "INC-3002"},
            {"action_type": "send_customer_update", "ticket_id": "INC-3001", "message_template": "service_restored"},
            {"action_type": "close_incident", "ticket_id": "INC-3001"},
            {"action_type": "close_incident", "ticket_id": "INC-3002"},
        ]
    else:
        return 0.0

    from models import LastMileOpsAction as Action
    for a in actions:
        if env.done:
            break
        env.step(Action(**a))

    score = round(max(0.0, min(1.0, env.score)), 3)
    return score


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        s = grade(task)
        print(f"task={task} score={s}")
        assert 0.0 <= s <= 1.0, f"Score out of range for {task}"
    print("All graders passed.")