# LastMileOps Environment

A telecom field-operations OpenEnv environment where an AI agent manages
network incidents, dispatches technicians, reroutes traffic, reserves parts,
and restores service for real-world ISP scenarios.

## Tasks

| Task | Difficulty | Max Steps |
|------|-----------|-----------|
| Single-site remote fix | Easy | 8 |
| Cabinet failure dispatch | Medium | 10 |
| Regional storm response | Hard | 12 |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Run Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Run Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=your-model-name
export HF_TOKEN=your-hf-token
python inference.py
```

## Run Graders

```bash
python graders/grader.py
```