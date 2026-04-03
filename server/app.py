from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import LastMileOpsAction, ResetResult, StepResult, LastMileOpsState
from server.environment import LastMileOpsEnvironment

app = FastAPI(title="LastMileOps Environment", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = LastMileOpsEnvironment()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResult)
def reset(task_id: str = "easy"):
    try:
        result = env.reset(task_id=task_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: LastMileOpsAction):
    try:
        result = env.step(action)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=LastMileOpsState)
def state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))