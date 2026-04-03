from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


ActionType = Literal[
    "list_incidents",
    "inspect_node",
    "run_remote_test",
    "reboot_device",
    "reroute_traffic",
    "reserve_part",
    "dispatch_technician",
    "send_customer_update",
    "close_incident",
    "noop",
]


class IncidentSummary(BaseModel):
    ticket_id: str
    customer: str
    priority: Literal["low", "normal", "high", "critical"]
    sla_minutes_remaining: int
    status: str
    symptom: str
    affected_node: str
    service_restored: bool = False


class NodeSummary(BaseModel):
    node_id: str
    node_type: str
    status: str
    cpu_load: int = 0
    power_ok: bool = True
    linked_backup: Optional[str] = None


class TechnicianSummary(BaseModel):
    tech_id: str
    name: str
    skills: List[str]
    region: str
    available: bool
    eta_minutes: int


class InventorySummary(BaseModel):
    depot_id: str
    items: Dict[str, int]


class LastMileOpsAction(BaseModel):
    action_type: ActionType
    ticket_id: Optional[str] = None
    node_id: Optional[str] = None
    test_type: Optional[Literal["ping", "optical_power", "throughput", "route_trace", "cpu_profile"]] = None
    backup_node_id: Optional[str] = None
    part_name: Optional[str] = None
    qty: Optional[int] = Field(default=1, ge=1)
    depot_id: Optional[str] = None
    tech_id: Optional[str] = None
    message_template: Optional[Literal["investigating", "tech_dispatched", "service_restored", "delay_expected"]] = None


class LastMileOpsObservation(BaseModel):
    task_id: str
    task_name: str
    prompt: str
    message: str
    incidents: List[IncidentSummary]
    nodes: List[NodeSummary]
    technicians: List[TechnicianSummary]
    inventory: List[InventorySummary]
    visible_logs: Dict[str, str] = Field(default_factory=dict)
    history: List[str] = Field(default_factory=list)
    progress_score: float = 0.0
    step_count: int = 0
    max_steps: int = 12


class LastMileOpsState(BaseModel):
    episode_id: str
    task_id: str
    task_name: str
    done: bool
    score: float
    step_count: int
    max_steps: int
    milestones: Dict[str, bool]
    raw_state: Dict[str, Any]


class StepResult(BaseModel):
    observation: LastMileOpsObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: LastMileOpsObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)