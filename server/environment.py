from __future__ import annotations
from typing import Any, Dict, Tuple
from .models import Action, Observation, StepResult, Incident, NetworkNode, Technician, SparePart

ACTION_SPACE = {
    "run_diagnostic":       "Run remote diagnostic on a node. params: {node_id}",
    "reboot_device":        "Reboot a network device. params: {node_id}",
    "dispatch_technician":  "Send technician to location. params: {tech_id, location}",
    "reserve_part":         "Reserve spare part. params: {part_id}",
    "reroute_traffic":      "Reroute traffic between nodes. params: {from_node_id, to_node_id}",
    "send_customer_update": "Notify customer. params: {incident_id, message_type}",
    "close_ticket":         "Close resolved incident. params: {incident_id}",
    "noop":                 "No action (penalised).",
}

SCENARIOS = {
    "easy": lambda: {
        "task_id": "easy", "max_steps": 10,
        "incidents": [
            Incident(id="INC-001", customer_id="BIZ-42", customer_tier="business",
                     symptom="No connectivity", affected_node="ONT-007", status="open", priority=2),
        ],
        "network": [
            NetworkNode(id="ONT-007", type="ont", status="offline", load_pct=0),
            NetworkNode(id="CAB-003", type="cabinet", status="online", load_pct=45),
        ],
        "technicians": [
            Technician(id="T1", name="Alice", skill="fiber",   available=True, location="depot"),
            Technician(id="T2", name="Bob",   skill="network", available=True, location="depot"),
        ],
        "inventory": [
            SparePart(id="P1", name="ONT unit",  quantity=5, reserved=0),
            SparePart(id="P2", name="Line card", quantity=2, reserved=0),
        ],
        "_grader": {"diagnostic_run": False, "reboot_done": False, "ticket_closed": False},
    },
    "medium": lambda: {
        "task_id": "medium", "max_steps": 15,
        "incidents": [
            Incident(id="INC-010", customer_id="AREA-05", customer_tier="residential",
                     symptom="Neighbourhood offline", affected_node="CAB-012", status="open", priority=3),
        ],
        "network": [
            NetworkNode(id="CAB-012", type="cabinet",     status="offline", load_pct=0),
            NetworkNode(id="AGG-001", type="aggregation", status="online",  load_pct=60),
        ],
        "technicians": [
            Technician(id="T1", name="Alice", skill="fiber",       available=True,  location="depot"),
            Technician(id="T3", name="Carol", skill="electronics", available=True,  location="north"),
            Technician(id="T4", name="Dave",  skill="network",     available=False, location="south"),
        ],
        "inventory": [
            SparePart(id="P2", name="Line card",    quantity=3, reserved=0),
            SparePart(id="P3", name="Power module", quantity=1, reserved=0),
            SparePart(id="P4", name="Splice kit",   quantity=4, reserved=0),
        ],
        "_grader": {
            "diagnostic_run": False,
            "correct_part_reserved": False,
            "correct_tech_dispatched": False,
            "ticket_closed": False,
        },
    },
    "hard": lambda: {
        "task_id": "hard", "max_steps": 20,
        "incidents": [
            Incident(id="INC-020", customer_id="CRIT-01", customer_tier="critical",
                     symptom="Service down",      affected_node="AGG-002",  status="open", priority=1),
            Incident(id="INC-021", customer_id="AREA-09", customer_tier="residential",
                     symptom="Cabinet offline",   affected_node="CAB-019",  status="open", priority=3),
            Incident(id="INC-022", customer_id="BIZ-77",  customer_tier="business",
                     symptom="Degraded service",  affected_node="ONT-031",  status="open", priority=2),
        ],
        "network": [
            NetworkNode(id="AGG-002",    type="aggregation", status="overloaded", load_pct=98),
            NetworkNode(id="AGG-BACKUP", type="backup",      status="online",     load_pct=20),
            NetworkNode(id="CAB-019",    type="cabinet",     status="offline",    load_pct=0),
            NetworkNode(id="ONT-031",    type="ont",         status="degraded",   load_pct=70),
        ],
        "technicians": [
            Technician(id="T1", name="Alice", skill="fiber",       available=True,  location="depot"),
            Technician(id="T3", name="Carol", skill="electronics", available=True,  location="north"),
            Technician(id="T5", name="Eve",   skill="network",     available=False, location="south"),
        ],
        "inventory": [
            SparePart(id="P1", name="ONT unit",    quantity=2, reserved=0),
            SparePart(id="P2", name="Line card",   quantity=1, reserved=0),
            SparePart(id="P3", name="Power module",quantity=0, reserved=0),
        ],
        "_grader": {
            "critical_prioritised":    False,
            "agg_inspected":           False,
            "traffic_rerouted":        False,
            "correct_part_reserved":   False,
            "correct_tech_dispatched": False,
            "customer_updated":        False,
            "all_tickets_closed":      False,
        },
    },
}


class LastMileOpsEnv:
    def __init__(self):
        self._state: Dict = {}
        self._step_count = 0
        self._score = 0.0
        self._done = False
        self._action_log = []

    def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in SCENARIOS:
            raise ValueError(f"Unknown task '{task_id}'. Choose: {list(SCENARIOS)}")
        self._state = SCENARIOS[task_id]()
        self._step_count = 0
        self._score = 0.0
        self._done = False
        self._action_log = []
        return self._build_obs()

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(observation=self._build_obs(), reward=0.0,
                              done=True, score=self._score, info={"message": "Episode done"})
        self._step_count += 1
        reward, msg = self._apply_action(action)
        self._action_log.append(f"[{self._step_count}] {action.action_type}({action.params}) → {msg}")
        self._score = self._compute_grader_score()
        all_closed = all(i.status == "closed" for i in self._state["incidents"])
        self._done = all_closed or self._step_count >= self._state["max_steps"]
        return StepResult(
            observation=self._build_obs(message=msg),
            reward=round(reward, 4),
            done=self._done,
            score=round(self._score, 4),
            info={"step": self._step_count, "grader": self._state["_grader"], "message": msg},
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task_id":  self._state.get("task_id", ""),
            "step":     self._step_count,
            "max_steps":self._state.get("max_steps", 0),
            "done":     self._done,
            "score":    round(self._score, 4),
            "grader_checkpoints": self._state.get("_grader", {}),
            "incidents":    [i.model_dump() for i in self._state.get("incidents", [])],
            "network":      [n.model_dump() for n in self._state.get("network", [])],
            "technicians":  [t.model_dump() for t in self._state.get("technicians", [])],
            "inventory":    [p.model_dump() for p in self._state.get("inventory", [])],
            "action_log":   self._action_log,
        }

    # ── Action logic ──────────────────────────────────────────────────────────
    def _apply_action(self, action: Action) -> Tuple[float, str]:
        t, p, s, g = action.action_type, action.params, self._state, self._state["_grader"]

        if t == "noop":
            return -0.04, "No-op used."

        if t == "run_diagnostic":
            node = self._find_node(p.get("node_id", ""))
            if not node: return -0.05, f"Node '{p.get('node_id')}' not found."
            if node.status in ("offline", "degraded", "overloaded"):
                if "diagnostic_run" in g and not g["diagnostic_run"]: g["diagnostic_run"] = True
                if "agg_inspected" in g and node.type == "aggregation": g["agg_inspected"] = True
                return 0.15, f"Diagnostic on {node.id}: {node.status}, load={node.load_pct}%"
            return 0.05, f"Diagnostic on {node.id}: {node.status} (no fault)."

        if t == "reboot_device":
            node = self._find_node(p.get("node_id", ""))
            if not node: return -0.05, f"Node not found."
            if node.type == "ont" and node.status == "offline":
                node.status = "online"; node.load_pct = 30
                if "reboot_done" in g and not g["reboot_done"]: g["reboot_done"] = True
                for inc in s["incidents"]:
                    if inc.affected_node == node.id and inc.status in ("open","in_progress"):
                        inc.status = "resolved"
                return 0.25, f"Rebooted {node.id} → online. Incident resolved."
            if node.status == "online": return -0.05, "Already online."
            return 0.0, f"Reboot had no effect on {node.id}."

        if t == "dispatch_technician":
            tech = self._find_tech(p.get("tech_id", ""))
            if not tech: return -0.05, "Technician not found."
            if not tech.available: return -0.08, f"{tech.name} unavailable."
            bonus = 0.0
            if tech.skill == "electronics" and s["task_id"] in ("medium","hard"):
                if "correct_tech_dispatched" in g and not g["correct_tech_dispatched"]:
                    g["correct_tech_dispatched"] = True; bonus = 0.15
            tech.available = False; tech.location = p.get("location","site")
            for inc in s["incidents"]:
                if inc.status == "open": inc.status = "in_progress"
            return 0.10 + bonus, f"Dispatched {tech.name} ({tech.skill})."

        if t == "reserve_part":
            part = self._find_part(p.get("part_id", ""))
            if not part: return -0.05, "Part not found."
            if part.quantity - part.reserved <= 0: return -0.08, f"{part.name} out of stock."
            part.reserved += 1
            if p.get("part_id") == "P2" and s["task_id"] in ("medium","hard"):
                if "correct_part_reserved" in g and not g["correct_part_reserved"]:
                    g["correct_part_reserved"] = True
                    return 0.20, f"Reserved correct part: {part.name}."
            return 0.08, f"Reserved {part.name}."

        if t == "reroute_traffic":
            fn = self._find_node(p.get("from_node_id",""))
            tn = self._find_node(p.get("to_node_id",""))
            if not fn or not tn: return -0.05, "Node(s) not found."
            if fn.status == "overloaded" and tn.type == "backup":
                fn.status = "online"; fn.load_pct = 55; tn.load_pct = min(tn.load_pct+35, 85)
                g["traffic_rerouted"] = True; g["critical_prioritised"] = True
                for inc in s["incidents"]:
                    if inc.priority == 1 and inc.status == "open": inc.status = "resolved"
                return 0.30, f"Traffic rerouted {fn.id} → {tn.id}. Critical incident resolved."
            return 0.0, "Reroute had no effect."

        if t == "send_customer_update":
            inc = self._find_incident(p.get("incident_id",""))
            if not inc: return -0.05, "Incident not found."
            if "customer_updated" in g and not g["customer_updated"]:
                g["customer_updated"] = True
                return 0.10, f"Customer update sent for {inc.id}."
            return 0.02, "Duplicate update."

        if t == "close_ticket":
            inc = self._find_incident(p.get("incident_id",""))
            if not inc: return -0.05, "Incident not found."
            if inc.status == "resolved":
                inc.status = "closed"
                if all(i.status == "closed" for i in s["incidents"]):
                    if "ticket_closed" in g: g["ticket_closed"] = True
                    if "all_tickets_closed" in g: g["all_tickets_closed"] = True
                    return 0.20, f"{inc.id} closed. ALL INCIDENTS RESOLVED ✓"
                return 0.10, f"{inc.id} closed."
            if inc.status == "closed": return -0.02, "Already closed."
            return -0.10, f"Cannot close — status is '{inc.status}', must be 'resolved' first."

        return -0.05, f"Unknown action '{t}'."

    def _compute_grader_score(self) -> float:
        g = self._state["_grader"]
        if not g: return 0.0
        achieved = sum(1 for v in g.values() if v)
        base = achieved / len(g)
        if base >= 1.0:
            efficiency = max(0.0, 1.0 - self._step_count / self._state["max_steps"])
            return min(1.0, base + efficiency * 0.15)
        return round(base, 4)

    def _build_obs(self, message="") -> Observation:
        s = self._state
        return Observation(
            task_id=s.get("task_id",""), step=self._step_count,
            max_steps=s.get("max_steps",0), done=self._done,
            incidents=s.get("incidents",[]), network=s.get("network",[]),
            technicians=s.get("technicians",[]), inventory=s.get("inventory",[]),
            action_log=list(self._action_log[-5:]), message=message,
        )

    def _find_node(self, nid):
        return next((n for n in self._state.get("network",[]) if n.id==nid), None)
    def _find_tech(self, tid):
        return next((t for t in self._state.get("technicians",[]) if t.id==tid), None)
    def _find_part(self, pid):
        return next((p for p in self._state.get("inventory",[]) if p.id==pid), None)
    def _find_incident(self, iid):
        return next((i for i in self._state.get("incidents",[]) if i.id==iid), None)