from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

from models import (
    IncidentSummary,
    InventorySummary,
    LastMileOpsAction,
    LastMileOpsObservation,
    LastMileOpsState,
    NodeSummary,
    ResetResult,
    StepResult,
    TechnicianSummary,
)


TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "task_name": "Single-site remote fix",
        "prompt": "A business customer is offline. Identify the affected node, run the right remote diagnostic, reboot the correct device, then close the ticket.",
        "max_steps": 8,
        "incidents": [
            {
                "ticket_id": "INC-1001",
                "customer": "BlueCart Grocery",
                "priority": "high",
                "sla_minutes_remaining": 90,
                "status": "open",
                "symptom": "ONT not responding",
                "affected_node": "ONT-7",
                "service_restored": False,
            }
        ],
        "nodes": [
            {"node_id": "ONT-7", "node_type": "ont", "status": "hung", "cpu_load": 98, "power_ok": True, "linked_backup": None},
            {"node_id": "OLT-A1", "node_type": "olt", "status": "healthy", "cpu_load": 36, "power_ok": True, "linked_backup": None},
        ],
        "technicians": [
            {"tech_id": "T-01", "name": "Ravi", "skills": ["fiber", "router"], "region": "north", "available": True, "eta_minutes": 35},
        ],
        "inventory": [{"depot_id": "D-1", "items": {"sfp_module": 2, "ont": 1, "power_supply": 3}}],
        "logs": {"ONT-7": "device heartbeat lost; watchdog timeout; recommended action: reboot_device", "OLT-A1": "all optical levels nominal"},
        "target": {"ticket_id": "INC-1001", "node_id": "ONT-7", "test_type": "ping"},
    },
    "medium": {
        "task_name": "Cabinet failure dispatch",
        "prompt": "A neighborhood cabinet is down. Diagnose the correct cabinet, reserve the needed spare part, dispatch the right technician, and close the ticket only after repair.",
        "max_steps": 10,
        "incidents": [
            {
                "ticket_id": "INC-2002",
                "customer": "Maple Residency Block C",
                "priority": "normal",
                "sla_minutes_remaining": 160,
                "status": "open",
                "symptom": "cabinet optical module failure",
                "affected_node": "CAB-22",
                "service_restored": False,
            }
        ],
        "nodes": [
            {"node_id": "CAB-22", "node_type": "cabinet", "status": "degraded", "cpu_load": 41, "power_ok": True, "linked_backup": None},
            {"node_id": "DIST-5", "node_type": "distribution_switch", "status": "healthy", "cpu_load": 52, "power_ok": True, "linked_backup": None},
        ],
        "technicians": [
            {"tech_id": "T-02", "name": "Neha", "skills": ["fiber", "splicing"], "region": "east", "available": True, "eta_minutes": 25},
            {"tech_id": "T-03", "name": "Amit", "skills": ["router"], "region": "east", "available": True, "eta_minutes": 20},
        ],
        "inventory": [{"depot_id": "D-2", "items": {"sfp_module": 1, "cabinet_fan": 2, "ont": 2}}],
        "logs": {"CAB-22": "rx power unstable; sfp fault detected; field replacement required", "DIST-5": "healthy"},
        "target": {"ticket_id": "INC-2002", "node_id": "CAB-22", "test_type": "optical_power", "part_name": "sfp_module", "depot_id": "D-2", "tech_id": "T-02"},
    },
    "hard": {
        "task_name": "Regional storm response",
        "prompt": "A storm caused multiple outages. Prioritize the critical customer, inspect the overloaded aggregation node, reroute traffic to the backup, reserve the correct part for the cabinet outage, dispatch the right technician, send the right update, restore service, and close all incidents safely.",
        "max_steps": 12,
        "incidents": [
            {
                "ticket_id": "INC-3001",
                "customer": "CityCare Hospital",
                "priority": "critical",
                "sla_minutes_remaining": 45,
                "status": "open",
                "symptom": "aggregation overload after storm",
                "affected_node": "AGG-9",
                "service_restored": False,
            },
            {
                "ticket_id": "INC-3002",
                "customer": "GreenHeights Apartments",
                "priority": "normal",
                "sla_minutes_remaining": 150,
                "status": "open",
                "symptom": "cabinet optic failure",
                "affected_node": "CAB-41",
                "service_restored": False,
            },
        ],
        "nodes": [
            {"node_id": "AGG-9", "node_type": "aggregation_router", "status": "overloaded", "cpu_load": 100, "power_ok": True, "linked_backup": "AGG-9B"},
            {"node_id": "AGG-9B", "node_type": "aggregation_router", "status": "healthy", "cpu_load": 44, "power_ok": True, "linked_backup": None},
            {"node_id": "CAB-41", "node_type": "cabinet", "status": "degraded", "cpu_load": 57, "power_ok": True, "linked_backup": None},
        ],
        "technicians": [
            {"tech_id": "T-04", "name": "Sara", "skills": ["fiber", "splicing"], "region": "south", "available": True, "eta_minutes": 18},
            {"tech_id": "T-05", "name": "Karan", "skills": ["router", "fiber"], "region": "south", "available": True, "eta_minutes": 30},
        ],
        "inventory": [{"depot_id": "D-3", "items": {"sfp_module": 1, "generator": 0, "fiber_patch": 4}}],
        "logs": {
            "AGG-9": "storm surge traffic spike; recommended action: reroute_traffic to AGG-9B before reboot_device",
            "CAB-41": "optic alarm; sfp fault detected; replace sfp_module",
        },
        "target": {
            "critical_ticket": "INC-3001",
            "normal_ticket": "INC-3002",
            "node_id": "AGG-9",
            "backup_node_id": "AGG-9B",
            "cabinet_id": "CAB-41",
            "part_name": "sfp_module",
            "depot_id": "D-3",
            "tech_id": "T-04",
        },
    },
}


class LastMileOpsEnvironment:
    def __init__(self) -> None:
        self.reset()

    def reset(self, task_id: str = "easy") -> ResetResult:
        if task_id not in TASKS:
            task_id = "easy"
        data = copy.deepcopy(TASKS[task_id])
        self.episode_id = str(uuid.uuid4())
        self.task_id = task_id
        self.task_name = data["task_name"]
        self.prompt = data["prompt"]
        self.max_steps = data["max_steps"]
        self.step_count = 0
        self.done = False
        self.score = 0.0
        self.logs = data["logs"]
        self.incidents = {x["ticket_id"]: x for x in data["incidents"]}
        self.nodes = {x["node_id"]: x for x in data["nodes"]}
        self.techs = {x["tech_id"]: x for x in data["technicians"]}
        self.inventory = {x["depot_id"]: x["items"] for x in data["inventory"]}
        self.target = data["target"]
        self.history: List[str] = [f"Episode started for task={task_id}"]
        self.visible_logs: Dict[str, str] = {}
        self.milestones = {
            "listed": False,
            "inspected_target": False,
            "ran_test": False,
            "rerouted": False,
            "rebooted": False,
            "reserved": False,
            "dispatched": False,
            "updated": False,
            "closed": False,
        }
        return ResetResult(observation=self._observation("Environment reset."), info={"task_id": task_id})

    def step(self, action: LastMileOpsAction) -> StepResult:
        if self.done:
            return StepResult(observation=self._observation("Episode already finished."), reward=0.0, done=True, info={"warning": "already_done"})

        self.step_count += 1
        reward = -0.01
        info: Dict[str, Any] = {"task_id": self.task_id}
        msg = "Action processed."

        if action.action_type == "noop":
            reward -= 0.03
            msg = "No-op used."

        elif action.action_type == "list_incidents":
            if not self.milestones["listed"]:
                reward += 0.10
                self.milestones["listed"] = True
            msg = f"Listed {len(self.incidents)} incidents."

        elif action.action_type == "inspect_node":
            node_id = action.node_id or ""
            if node_id in self.nodes:
                self.visible_logs[node_id] = self.logs.get(node_id, "No logs available")
                if self.task_id == "easy" and node_id == self.target["node_id"]:
                    reward += 0.20
                    self.milestones["inspected_target"] = True
                elif self.task_id == "medium" and node_id == self.target["node_id"]:
                    reward += 0.20
                    self.milestones["inspected_target"] = True
                elif self.task_id == "hard" and node_id in {self.target["node_id"], self.target["cabinet_id"]}:
                    reward += 0.10
                    if node_id == self.target["node_id"]:
                        self.milestones["inspected_target"] = True
                else:
                    reward -= 0.02
                msg = f"Inspected {node_id}."
            else:
                reward -= 0.10
                msg = f"Unknown node {node_id}."

        elif action.action_type == "run_remote_test":
            node_id = action.node_id or self._ticket_node(action.ticket_id)
            test_type = action.test_type
            if node_id in self.nodes and test_type:
                correct = False
                if self.task_id == "easy":
                    correct = node_id == self.target["node_id"] and test_type == self.target["test_type"]
                elif self.task_id == "medium":
                    correct = node_id == self.target["node_id"] and test_type == self.target["test_type"]
                elif self.task_id == "hard":
                    correct = (node_id == self.target["node_id"] and test_type in {"cpu_profile", "route_trace"}) or (node_id == self.target["cabinet_id"] and test_type == "optical_power")
                reward += 0.18 if correct else -0.04
                if correct:
                    self.milestones["ran_test"] = True
                msg = f"Ran {test_type} on {node_id}."
            else:
                reward -= 0.08
                msg = "Invalid test request."

        elif action.action_type == "reboot_device":
            node_id = action.node_id or ""
            if node_id not in self.nodes:
                reward -= 0.10
                msg = "Unknown node for reboot."
            else:
                if self.task_id == "easy" and node_id == self.target["node_id"] and self.milestones["ran_test"]:
                    self.nodes[node_id]["status"] = "healthy"
                    self.nodes[node_id]["cpu_load"] = 22
                    self.incidents[self.target["ticket_id"]]["service_restored"] = True
                    reward += 0.40
                    self.milestones["rebooted"] = True
                    msg = f"Rebooted {node_id}; service restored."
                elif self.task_id == "hard" and node_id == self.target["node_id"] and self.milestones["rerouted"]:
                    self.nodes[node_id]["status"] = "healthy"
                    self.nodes[node_id]["cpu_load"] = 39
                    self.incidents[self.target["critical_ticket"]]["service_restored"] = True
                    reward += 0.20
                    self.milestones["rebooted"] = True
                    msg = f"Rebooted {node_id} after traffic reroute."
                else:
                    reward -= 0.06
                    msg = f"Reboot of {node_id} had no useful effect."

        elif action.action_type == "reroute_traffic":
            node_id = action.node_id or ""
            backup = action.backup_node_id or ""
            if self.task_id == "hard" and node_id == self.target["node_id"] and backup == self.target["backup_node_id"]:
                self.nodes[node_id]["status"] = "stable"
                self.nodes[node_id]["cpu_load"] = 72
                reward += 0.25
                self.milestones["rerouted"] = True
                msg = f"Traffic rerouted from {node_id} to {backup}."
            else:
                reward -= 0.08
                msg = "Invalid reroute request."

        elif action.action_type == "reserve_part":
            depot = action.depot_id or ""
            part = action.part_name or ""
            qty = int(action.qty or 1)
            available = self.inventory.get(depot, {}).get(part, 0)
            if available >= qty:
                self.inventory[depot][part] -= qty
                correct = False
                if self.task_id == "medium":
                    correct = depot == self.target["depot_id"] and part == self.target["part_name"]
                elif self.task_id == "hard":
                    correct = depot == self.target["depot_id"] and part == self.target["part_name"]
                reward += 0.22 if correct else -0.04
                if correct:
                    self.milestones["reserved"] = True
                msg = f"Reserved {qty} {part} from {depot}."
            else:
                reward -= 0.12
                msg = f"Part {part} unavailable at {depot}."

        elif action.action_type == "dispatch_technician":
            tech_id = action.tech_id or ""
            ticket_id = action.ticket_id or ""
            tech = self.techs.get(tech_id)
            if tech and ticket_id in self.incidents and tech["available"]:
                tech["available"] = False
                correct = False
                if self.task_id == "medium":
                    correct = tech_id == self.target["tech_id"] and ticket_id == self.target["ticket_id"] and self.milestones["reserved"]
                elif self.task_id == "hard":
                    correct = tech_id == self.target["tech_id"] and ticket_id == self.target["normal_ticket"] and self.milestones["reserved"]
                reward += 0.24 if correct else -0.05
                if correct:
                    self.milestones["dispatched"] = True
                    self.incidents[ticket_id]["service_restored"] = True
                msg = f"Dispatched {tech_id} to {ticket_id}."
            else:
                reward -= 0.10
                msg = "Invalid technician dispatch."

        elif action.action_type == "send_customer_update":
            ticket_id = action.ticket_id or ""
            template = action.message_template
            if ticket_id in self.incidents and template:
                correct = False
                if self.task_id == "hard":
                    correct = ticket_id == self.target["critical_ticket"] and template in {"investigating", "service_restored"}
                else:
                    correct = template in {"investigating", "service_restored", "tech_dispatched"}
                reward += 0.08 if correct else -0.02
                if correct:
                    self.milestones["updated"] = True
                msg = f"Sent {template} update for {ticket_id}."
            else:
                reward -= 0.06
                msg = "Invalid customer update."

        elif action.action_type == "close_incident":
            ticket_id = action.ticket_id or ""
            incident = self.incidents.get(ticket_id)
            if incident:
                if incident["service_restored"]:
                    incident["status"] = "closed"
                    reward += 0.20
                    msg = f"Closed {ticket_id}."
                else:
                    reward -= 0.20
                    msg = f"Cannot close {ticket_id}; service not restored."
                self.milestones["closed"] = all(x["status"] == "closed" for x in self.incidents.values())
            else:
                reward -= 0.08
                msg = "Unknown ticket."

        else:
            reward -= 0.10
            msg = "Unsupported action."

        reward = round(reward, 3)
        self.score = round(max(0.0, min(1.0, self.score + max(reward, -0.2))), 3)
        self.history.append(f"step={self.step_count} action={action.action_type} reward={reward} msg={msg}")

        if self.task_id == "medium" and self.milestones["dispatched"]:
            self.incidents[self.target["ticket_id"]]["status"] = "resolved_pending_close"

        if self.step_count >= self.max_steps:
            self.done = True
            self.history.append("Episode ended: max steps reached")

        if self._all_success_conditions_met():
            self.done = True
            self.score = max(self.score, 1.0)
            self.history.append("Episode ended: task complete")

        return StepResult(observation=self._observation(msg), reward=reward, done=self.done, info=info)

    def state(self) -> LastMileOpsState:
        return LastMileOpsState(
            episode_id=self.episode_id,
            task_id=self.task_id,
            task_name=self.task_name,
            done=self.done,
            score=self.score,
            step_count=self.step_count,
            max_steps=self.max_steps,
            milestones=self.milestones,
            raw_state={
                "incidents": self.incidents,
                "nodes": self.nodes,
                "techs": self.techs,
                "inventory": self.inventory,
                "visible_logs": self.visible_logs,
            },
        )

    def _all_success_conditions_met(self) -> bool:
        if self.task_id == "easy":
            inc = self.incidents[self.target["ticket_id"]]
            return inc["service_restored"] and inc["status"] == "closed"
        if self.task_id == "medium":
            inc = self.incidents[self.target["ticket_id"]]
            return self.milestones["reserved"] and self.milestones["dispatched"] and inc["status"] == "closed"
        crit = self.incidents[self.target["critical_ticket"]]
        norm = self.incidents[self.target["normal_ticket"]]
        return (
            self.milestones["rerouted"]
            and self.milestones["rebooted"]
            and self.milestones["reserved"]
            and self.milestones["dispatched"]
            and self.milestones["updated"]
            and crit["status"] == "closed"
            and norm["status"] == "closed"
        )

    def _ticket_node(self, ticket_id: str | None) -> str | None:
        if not ticket_id or ticket_id not in self.incidents:
            return None
        return self.incidents[ticket_id]["affected_node"]

    def _observation(self, message: str) -> LastMileOpsObservation:
        return LastMileOpsObservation(
            task_id=self.task_id,
            task_name=self.task_name,
            prompt=self.prompt,
            message=message,
            incidents=[IncidentSummary(**x) for x in self.incidents.values()],
            nodes=[NodeSummary(**x) for x in self.nodes.values()],
            technicians=[TechnicianSummary(**x) for x in self.techs.values()],
            inventory=[InventorySummary(depot_id=depot, items=items) for depot, items in self.inventory.items()],
            visible_logs=self.visible_logs,
            history=self.history[-8:],
            progress_score=self.score,
            step_count=self.step_count,
            max_steps=self.max_steps,
        )