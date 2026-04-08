from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Incident(BaseModel):
    id: str
    customer_id: str
    customer_tier: str
    symptom: str
    affected_node: Optional[str] = None
    status: str
    priority: int


class NetworkNode(BaseModel):
    id: str
    type: str
    status: str
    load_pct: int


class Technician(BaseModel):
    id: str
    name: str
    skill: str
    available: bool
    location: str


class SparePart(BaseModel):
    id: str
    name: str
    quantity: int
    reserved: int


class Observation(BaseModel):
    task_id: str
    step: int
    max_steps: int
    done: bool
    incidents: List[Incident]
    network: List[NetworkNode]
    technicians: List[Technician]
    inventory: List[SparePart]
    action_log: List[str] = Field(default_factory=list)
    message: str = ""


class Action(BaseModel):
    action_type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(gt=0.0, lt=1.0)
    done: bool
    score: float = Field(gt=0.0, lt=1.0)
    info: Dict[str, Any] = Field(default_factory=dict)