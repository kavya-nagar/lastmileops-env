from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict

from .models import Action, Observation, StepResult
from .environment import LastMileOpsEnv, ACTION_SPACE

app = FastAPI(
    title="LastMileOps — Telecom NOC OpenEnv",
    description="Real-world telecom network operations environment for AI agent training.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = LastMileOpsEnv()


class ResetRequest(BaseModel):
    task_id: str = "easy"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Single-site remote fix",
                "difficulty": "easy",
                "max_steps": 10,
                "description": "Identify the fault, run diagnostics, reboot the ONT, close the ticket.",
            },
            {
                "id": "medium",
                "name": "Cabinet failure dispatch",
                "difficulty": "medium",
                "max_steps": 15,
                "description": "Diagnose cabinet fault, reserve correct spare, dispatch right technician, close ticket.",
            },
            {
                "id": "hard",
                "name": "Regional storm response",
                "difficulty": "hard",
                "max_steps": 20,
                "description": "Multi-incident storm response: reroute traffic, dispatch, reserve parts, update customers, close all.",
            },
        ]
    }


@app.get("/actions")
def list_actions():
    return {"action_space": ACTION_SPACE}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    try:
        obs = env.reset(task_id=req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action):
    return env.step(action)


@app.get("/state")
def state() -> Dict[str, Any]:
    return env.state()