# LastMileOps — Telecom NOC OpenEnv Environment

A real-world **Network Operations Center (NOC)** simulation where an AI agent
triages faults, dispatches technicians, manages spare parts, reroutes traffic,
and closes incidents — exactly as a human NOC engineer would.

## Environment Description

Telecom providers handle hundreds of field incidents daily: ONT reboots,
cabinet failures, storm outages. This environment models that workflow with
full state management, partial-progress rewards, and 3 difficulty tiers.

## Action Space

| Action | Params | Description |
|--------|--------|-------------|
| `run_diagnostic` | `{node_id}` | Run remote diagnostic on a node |
| `reboot_device` | `{node_id}` | Reboot an offline device |
| `dispatch_technician` | `{tech_id, location}` | Send technician to site |
| `reserve_part` | `{part_id}` | Reserve spare from inventory |
| `reroute_traffic` | `{from_node_id, to_node_id}` | Reroute overloaded traffic |
| `send_customer_update` | `{incident_id, message_type}` | Notify customer |
| `close_ticket` | `{incident_id}` | Close a resolved incident |
| `noop` | `{}` | No action (penalised −0.04) |

## Observation Space

JSON object containing:
- `incidents` — list of active incidents with customer tier, priority, status
- `network` — node statuses and load percentages
- `technicians` — availability, skill, location
- `inventory` — spare parts with quantities
- `action_log` — last 5 actions taken

## Tasks

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| `easy` | Easy | 10 | Single ONT offline — diagnose, reboot, close |
| `medium` | Medium | 15 | Cabinet failure — diagnose, reserve right part, dispatch right tech |
| `hard` | Hard | 20 | Storm response — reroute traffic, multi-incident, parts, dispatch, update, close all |

## Reward Function

- Correct diagnostic on faulty node: **+0.15**
- Successful reboot: **+0.25**
- Correct technician dispatched: **+0.10–0.25**
- Correct part reserved: **+0.08–0.20**
- Traffic rerouted correctly: **+0.30**
- Customer update sent: **+0.10**
- Ticket closed: **+0.10–0.20**
- Noop: **−0.04** | Wrong action: **−0.05–0.10**
- Efficiency bonus: up to **+0.15** for fast resolution

## Baseline Scores

| Task | Score |
|------|-------|
| easy | 0.75 |
| medium | 0.55 |
| hard | 0.35 |

## Setup

```bash
# Local
pip install fastapi uvicorn pydantic pyyaml openai requests
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Docker
docker build -t lastmileops .
docker run -p 7860:7860 lastmileops

# Inference
export API_BASE_URL=https://integrate.api.nvidia.com/v1
export MODEL_NAME=nvidia/llama-3.3-nemotron-super-49b-v1
export HF_TOKEN=your_key_here
export ENV_URL=http://localhost:7860
python inference.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| GET | `/actions` | List action space |
| POST | `/reset` | Reset with `{"task_id": "easy"}` |
| POST | `/step` | Step with `{"action_type": "...", "params": {...}}` |
| GET | `/state` | Full current state |