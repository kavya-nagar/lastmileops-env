---
title: LastMileOps Env
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
tags:
  - openenv
---

# LastMileOps — Telecom NOC OpenEnv Environment

A real-world **Network Operations Center (NOC)** simulation where an AI agent
triages faults, dispatches technicians, manages spare parts, reroutes traffic,
and closes incidents — exactly as a human NOC engineer would.

## Action Space

| Action | Params |
|--------|--------|
| `run_diagnostic` | `{node_id}` |
| `reboot_device` | `{node_id}` |
| `dispatch_technician` | `{tech_id, location}` |
| `reserve_part` | `{part_id}` |
| `reroute_traffic` | `{from_node_id, to_node_id}` |
| `send_customer_update` | `{incident_id, message_type}` |
| `close_ticket` | `{incident_id}` |
| `noop` | `{}` |

## Tasks

| Task | Difficulty | Max Steps |
|------|-----------|-----------|
| `easy` | Easy | 10 |
| `medium` | Medium | 15 |
| `hard` | Hard | 20 |

## Setup

```bash
docker build -t lastmileops .
docker run -p 7860:7860 lastmileops
```

## Baseline Scores

| Task | Score |
|------|-------|
| easy | 0.75 |
| medium | 0.55 |
| hard | 0.35 |