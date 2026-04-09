from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

from .models import Action, Observation, StepResult
from .environment import LastMileOpsEnv, ACTION_SPACE

app = FastAPI(
    title="LastMileOps Telecom NOC OpenEnv",
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
    taskid: str = "easy"


@app.get("/")
def root():
    return {
        "name": "LastMileOps Telecom NOC OpenEnv",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
    }


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
                "maxsteps": 10,
            },
            {
                "id": "medium",
                "name": "Cabinet failure dispatch",
                "difficulty": "medium",
                "maxsteps": 15,
            },
            {
                "id": "hard",
                "name": "Regional storm response",
                "difficulty": "hard",
                "maxsteps": 20,
            },
        ]
    }


@app.get("/actions")
def list_actions():
    return {"actionspace": ACTION_SPACE}


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest(taskid="easy")
    try:
        obs = env.reset(task_id=req.taskid)
    except TypeError:
        obs = env.reset(req.taskid)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action):
    return env.step(action)


@app.get("/state")
def state() -> Dict[str, Any]:
    return env.state()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()