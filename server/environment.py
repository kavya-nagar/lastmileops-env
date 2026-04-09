from __future__ import annotations

from typing import Any, Dict, Tuple

from .models import Action, Observation, StepResult, Incident, NetworkNode, Technician, SparePart


def strict_unit(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.01
    return round(max(0.01, min(0.99, x)), 4)


ACTIONSPACE = {
    "rundiagnostic": "Run remote diagnostic on a node. params: nodeid",
    "rebootdevice": "Reboot a network device. params: nodeid",
    "dispatchtechnician": "Send technician to location. params: techid, location",
    "reservepart": "Reserve spare part. params: partid",
    "reroutetraffic": "Reroute traffic between nodes. params: fromnodeid, tonodeid",
    "sendcustomerupdate": "Notify customer. params: incidentid, messagetype",
    "closeticket": "Close resolved incident. params: incidentid",
    "noop": "No action.",
}


def _easy_scenario() -> Dict[str, Any]:
    return {
        "task_id": "easy",
        "max_steps": 10,
        "incidents": [
            Incident(
                id="INC-001",
                customer_id="BIZ-42",
                customer_tier="business",
                symptom="No connectivity",
                affected_node="ONT-007",
                status="open",
                priority=2,
            ),
        ],
        "network": [
            NetworkNode(id="ONT-007", type="ont", status="offline", load_pct=0),
            NetworkNode(id="CAB-003", type="cabinet", status="online", load_pct=45),
        ],
        "technicians": [
            Technician(id="T1", name="Alice", skill="fiber", available=True, location="depot"),
            Technician(id="T2", name="Bob", skill="network", available=True, location="depot"),
        ],
        "inventory": [
            SparePart(id="P1", name="ONT unit", quantity=5, reserved=0),
            SparePart(id="P2", name="Line card", quantity=2, reserved=0),
        ],
        "grader": {
            "diagnostic_run": False,
            "reboot_done": False,
            "ticket_closed": False,
        },
    }


def _medium_scenario() -> Dict[str, Any]:
    return {
        "task_id": "medium",
        "max_steps": 15,
        "incidents": [
            Incident(
                id="INC-010",
                customer_id="AREA-05",
                customer_tier="residential",
                symptom="Neighbourhood offline",
                affected_node="CAB-012",
                status="open",
                priority=3,
            ),
        ],
        "network": [
            NetworkNode(id="CAB-012", type="cabinet", status="offline", load_pct=0),
            NetworkNode(id="AGG-001", type="aggregation", status="online", load_pct=60),
        ],
        "technicians": [
            Technician(id="T1", name="Alice", skill="fiber", available=True, location="depot"),
            Technician(id="T3", name="Carol", skill="electronics", available=True, location="north"),
            Technician(id="T4", name="Dave", skill="network", available=False, location="south"),
        ],
        "inventory": [
            SparePart(id="P2", name="Line card", quantity=3, reserved=0),
            SparePart(id="P3", name="Power module", quantity=1, reserved=0),
            SparePart(id="P4", name="Splice kit", quantity=4, reserved=0),
        ],
        "grader": {
            "diagnostic_run": False,
            "correct_part_reserved": False,
            "correct_tech_dispatched": False,
            "ticket_closed": False,
        },
    }


def _hard_scenario() -> Dict[str, Any]:
    return {
        "task_id": "hard",
        "max_steps": 20,
        "incidents": [
            Incident(
                id="INC-020",
                customer_id="CRIT-01",
                customer_tier="critical",
                symptom="Service down",
                affected_node="AGG-002",
                status="open",
                priority=1,
            ),
            Incident(
                id="INC-021",
                customer_id="AREA-09",
                customer_tier="residential",
                symptom="Cabinet offline",
                affected_node="CAB-019",
                status="open",
                priority=3,
            ),
            Incident(
                id="INC-022",
                customer_id="BIZ-77",
                customer_tier="business",
                symptom="Degraded service",
                affected_node="ONT-031",
                status="open",
                priority=2,
            ),
        ],
        "network": [
            NetworkNode(id="AGG-002", type="aggregation", status="overloaded", load_pct=98),
            NetworkNode(id="AGG-BACKUP", type="backup", status="online", load_pct=20),
            NetworkNode(id="CAB-019", type="cabinet", status="offline", load_pct=0),
            NetworkNode(id="ONT-031", type="ont", status="degraded", load_pct=70),
        ],
        "technicians": [
            Technician(id="T1", name="Alice", skill="fiber", available=True, location="depot"),
            Technician(id="T3", name="Carol", skill="electronics", available=True, location="north"),
            Technician(id="T5", name="Eve", skill="network", available=False, location="south"),
        ],
        "inventory": [
            SparePart(id="P1", name="ONT unit", quantity=2, reserved=0),
            SparePart(id="P2", name="Line card", quantity=1, reserved=0),
            SparePart(id="P3", name="Power module", quantity=0, reserved=0),
        ],
        "grader": {
            "critical_prioritised": False,
            "agg_inspected": False,
            "traffic_rerouted": False,
            "correct_part_reserved": False,
            "correct_tech_dispatched": False,
            "customer_updated": False,
            "all_tickets_closed": False,
        },
    }


SCENARIOS = {
    "easy": _easy_scenario,
    "medium": _medium_scenario,
    "hard": _hard_scenario,
}


class LastMileOpsEnv:
    def __init__(self) -> None:
        self.state_data: Dict[str, Any] = {}
        self.step_count = 0
        self.score = 0.01
        self.done = False
        self.action_log: list[str] = []

    def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in SCENARIOS:
            raise ValueError(f"Unknown task '{task_id}'. Choose from: {list(SCENARIOS)}")
        self.state_data = SCENARIOS[task_id]()
        self.step_count = 0
        self.score = 0.01
        self.done = False
        self.action_log = []
        return self.build_obs("Environment reset.")

    def step(self, action: Action) -> StepResult:
        if not self.state_data:
            self.reset("easy")

        if self.done:
            return StepResult(
                observation=self.build_obs("Episode already done."),
                reward=strict_unit(0.01),
                done=True,
                score=strict_unit(self.score),
                info={"message": "Episode already done."},
            )

        self.step_count += 1
        reward, message = self.apply_action(action)
        reward = strict_unit(reward)
        self.score = strict_unit(self.compute_grader_score())
        self.done = self._all_incidents_closed() or self.step_count >= self.state_data.get("max_steps", 0)

        obs = self.build_obs(message)
        return StepResult(
            observation=obs,
            reward=reward,
            done=self.done,
            score=strict_unit(self.score),
            info={
                "step": self.step_count,
                "grader": self.state_data.get("grader", {}),
                "message": message,
            },
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.state_data.get("task_id", ""),
            "step": self.step_count,
            "max_steps": self.state_data.get("max_steps", 0),
            "done": self.done,
            "score": strict_unit(self.score),
            "grader_checkpoints": self.state_data.get("grader", {}),
            "incidents": [i.model_dump() for i in self.state_data.get("incidents", [])],
            "network": [n.model_dump() for n in self.state_data.get("network", [])],
            "technicians": [t.model_dump() for t in self.state_data.get("technicians", [])],
            "inventory": [p.model_dump() for p in self.state_data.get("inventory", [])],
            "action_log": list(self.action_log),
        }

    def apply_action(self, action: Action) -> Tuple[float, str]:
        action_type = (
            getattr(action, "action_type", None)
            or getattr(action, "actiontype", None)
            or "noop"
        )
        params = getattr(action, "params", {}) or {}

        action_type = str(action_type).strip().lower()
        s = self.state_data
        g = s["grader"]

        if action_type == "noop":
            self._log_action(action_type, params, "No-op used.")
            return 0.01, "No-op used."

        if action_type == "rundiagnostic":
            node = self.find_node(params.get("nodeid"))
            if not node:
                self._log_action(action_type, params, "Node not found.")
                return 0.01, "Node not found."

            if s["task_id"] == "easy" and node.id == "ONT-007" and node.status == "offline":
                g["diagnostic_run"] = True
                msg = f"Diagnostic complete on {node.id}. Fault confirmed."
                self._log_action(action_type, params, msg)
                return 0.15, msg

            if s["task_id"] == "medium" and node.id == "CAB-012" and node.status == "offline":
                g["diagnostic_run"] = True
                msg = f"Diagnostic complete on {node.id}. Line card failure suspected."
                self._log_action(action_type, params, msg)
                return 0.15, msg

            if s["task_id"] == "hard" and node.id == "AGG-002" and node.status == "overloaded":
                g["agg_inspected"] = True
                g["critical_prioritised"] = True
                msg = f"Diagnostic complete on {node.id}. Overload confirmed."
                self._log_action(action_type, params, msg)
                return 0.15, msg

            if s["task_id"] == "hard" and node.id == "CAB-019" and node.status == "offline":
                msg = f"Diagnostic complete on {node.id}. Cabinet hardware fault confirmed."
                self._log_action(action_type, params, msg)
                return 0.12, msg

            msg = f"Diagnostic on {node.id} found no actionable fault."
            self._log_action(action_type, params, msg)
            return 0.05, msg

        if action_type == "rebootdevice":
            node = self.find_node(params.get("nodeid"))
            if not node:
                self._log_action(action_type, params, "Node not found.")
                return 0.01, "Node not found."

            if s["task_id"] == "easy" and node.id == "ONT-007" and node.status == "offline":
                node.status = "online"
                node.load_pct = 30
                self._resolve_incident("INC-001")
                g["reboot_done"] = True
                msg = f"Rebooted {node.id}. Service restored."
                self._log_action(action_type, params, msg)
                return 0.25, msg

            msg = f"Reboot had no effect on {node.id}."
            self._log_action(action_type, params, msg)
            return 0.02, msg

        if action_type == "reservepart":
            part = self.find_part(params.get("partid"))
            if not part:
                self._log_action(action_type, params, "Part not found.")
                return 0.01, "Part not found."

            available_qty = part.quantity - part.reserved
            if available_qty <= 0:
                msg = f"{part.name} out of stock."
                self._log_action(action_type, params, msg)
                return 0.01, msg

            part.reserved += 1
            if part.id == "P2" and s["task_id"] in {"medium", "hard"}:
                g["correct_part_reserved"] = True
                msg = f"Reserved correct part: {part.name}."
                self._log_action(action_type, params, msg)
                return 0.20, msg

            msg = f"Reserved part: {part.name}."
            self._log_action(action_type, params, msg)
            return 0.08, msg

        if action_type == "dispatchtechnician":
            tech = self.find_tech(params.get("techid"))
            if not tech:
                self._log_action(action_type, params, "Technician not found.")
                return 0.01, "Technician not found."

            if not tech.available:
                msg = f"{tech.name} is unavailable."
                self._log_action(action_type, params, msg)
                return 0.01, msg

            tech.available = False
            tech.location = params.get("location", tech.location)

            if tech.id == "T3" and tech.skill == "electronics" and s["task_id"] in {"medium", "hard"}:
                g["correct_tech_dispatched"] = True

            if s["task_id"] == "medium" and tech.id == "T3" and g.get("correct_part_reserved"):
                cab = self.find_node("CAB-012")
                if cab:
                    cab.status = "online"
                    cab.load_pct = 35
                self._consume_reserved_part("P2")
                self._resolve_incident("INC-010")
                msg = f"Dispatched {tech.name}. Cabinet repaired and service restored."
                self._log_action(action_type, params, msg)
                return 0.25, msg

            if s["task_id"] == "hard" and tech.id == "T3" and g.get("correct_part_reserved"):
                cab = self.find_node("CAB-019")
                ont = self.find_node("ONT-031")
                if cab:
                    cab.status = "online"
                    cab.load_pct = 30
                if ont:
                    ont.status = "online"
                    ont.load_pct = 35
                self._consume_reserved_part("P2")
                self._resolve_incident("INC-021")
                self._resolve_incident("INC-022")
                msg = f"Dispatched {tech.name}. Field repair completed for cabinet and downstream ONT."
                self._log_action(action_type, params, msg)
                return 0.25, msg

            msg = f"Dispatched {tech.name} to {tech.location}."
            self._log_action(action_type, params, msg)
            return 0.10, msg

        if action_type == "reroutetraffic":
            from_node = self.find_node(params.get("fromnodeid"))
            to_node = self.find_node(params.get("tonodeid"))
            if not from_node or not to_node:
                self._log_action(action_type, params, "Nodes not found.")
                return 0.01, "Nodes not found."

            if (
                s["task_id"] == "hard"
                and from_node.id == "AGG-002"
                and to_node.id == "AGG-BACKUP"
                and from_node.status == "overloaded"
                and to_node.type == "backup"
            ):
                from_node.status = "online"
                from_node.load_pct = 55
                to_node.load_pct = min(to_node.load_pct + 35, 85)
                g["traffic_rerouted"] = True
                g["critical_prioritised"] = True
                self._resolve_incident("INC-020")
                msg = f"Traffic rerouted from {from_node.id} to {to_node.id}. Critical service restored."
                self._log_action(action_type, params, msg)
                return 0.30, msg

            msg = "Reroute had no effect."
            self._log_action(action_type, params, msg)
            return 0.02, msg

        if action_type == "sendcustomerupdate":
            inc = self.find_incident(params.get("incidentid"))
            if not inc:
                self._log_action(action_type, params, "Incident not found.")
                return 0.01, "Incident not found."

            if "customer_updated" in g and not g["customer_updated"]:
                g["customer_updated"] = True
                msg = f"Customer update sent for {inc.id}."
                self._log_action(action_type, params, msg)
                return 0.10, msg

            msg = f"Update noted for {inc.id}."
            self._log_action(action_type, params, msg)
            return 0.02, msg

        if action_type == "closeticket":
            inc = self.find_incident(params.get("incidentid"))
            if not inc:
                self._log_action(action_type, params, "Incident not found.")
                return 0.01, "Incident not found."

            if inc.status == "resolved":
                inc.status = "closed"
                if "ticket_closed" in g:
                    g["ticket_closed"] = True
                if self._all_incidents_closed():
                    if "all_tickets_closed" in g:
                        g["all_tickets_closed"] = True
                    msg = f"{inc.id} closed. All incidents closed."
                    self._log_action(action_type, params, msg)
                    return 0.20, msg
                msg = f"{inc.id} closed."
                self._log_action(action_type, params, msg)
                return 0.10, msg

            if inc.status == "closed":
                msg = f"{inc.id} already closed."
                self._log_action(action_type, params, msg)
                return 0.01, msg

            msg = f"Cannot close {inc.id}; status is {inc.status}."
            self._log_action(action_type, params, msg)
            return 0.01, msg

        msg = f"Unknown action '{action_type}'."
        self._log_action(action_type, params, msg)
        return 0.01, msg

    def compute_grader_score(self) -> float:
        grader = self.state_data.get("grader", {})
        if not grader:
            return 0.01
        achieved = sum(1 for v in grader.values() if v)
        progress = achieved / max(len(grader), 1)
        efficiency = max(0.0, 1.0 - (self.step_count / max(self.state_data.get("max_steps", 1), 1)))
        raw_score = 0.01 + (0.80 * progress) + (0.18 * efficiency)
        return strict_unit(raw_score)

    def build_obs(self, message: str = "") -> Observation:
        s = self.state_data
        return Observation(
            task_id=s.get("task_id", ""),
            step=self.step_count,
            max_steps=s.get("max_steps", 0),
            done=self.done,
            incidents=s.get("incidents", []),
            network=s.get("network", []),
            technicians=s.get("technicians", []),
            inventory=s.get("inventory", []),
            action_log=self.action_log[-5:],
            message=message,
        )

    def find_node(self, node_id: str | None) -> NetworkNode | None:
        return next((n for n in self.state_data.get("network", []) if n.id == node_id), None)

    def find_tech(self, tech_id: str | None) -> Technician | None:
        return next((t for t in self.state_data.get("technicians", []) if t.id == tech_id), None)

    def find_part(self, part_id: str | None) -> SparePart | None:
        return next((p for p in self.state_data.get("inventory", []) if p.id == part_id), None)

    def find_incident(self, incident_id: str | None) -> Incident | None:
        return next((i for i in self.state_data.get("incidents", []) if i.id == incident_id), None)

    def _resolve_incident(self, incident_id: str) -> None:
        inc = self.find_incident(incident_id)
        if inc and inc.status in {"open", "in_progress", "resolved"}:
            inc.status = "resolved"

    def _consume_reserved_part(self, part_id: str) -> None:
        part = self.find_part(part_id)
        if not part:
            return
        if part.reserved > 0:
            part.reserved -= 1
        if part.quantity > 0:
            part.quantity -= 1

    def _all_incidents_closed(self) -> bool:
        incidents = self.state_data.get("incidents", [])
        return bool(incidents) and all(i.status == "closed" for i in incidents)

    def _log_action(self, action_type: str, params: Dict[str, Any], message: str) -> None:
        self.action_log.append(f"{self.step_count}. {action_type} {params} -> {message}")