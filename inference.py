from __future__ import annotations

import json
import os
import sys
import time

import requests
from openai import OpenAI

# ── env variables ──────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "<your-active-api-base-url>")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "<your-active-model-name>")
HF_TOKEN     = os.environ.get("HF_TOKEN")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert telecom NOC engineer controlling a LastMileOps environment.
You receive a JSON observation and must return a single JSON action object.

Available action_types and their REQUIRED fields:
  list_incidents        → {}
  inspect_node          → {"node_id": "..."}
  run_remote_test       → {"node_id": "...", "test_type": "ping|optical_power|throughput|route_trace|cpu_profile"}
  reboot_device         → {"node_id": "..."}
  reroute_traffic       → {"node_id": "...", "backup_node_id": "..."}
  reserve_part          → {"depot_id": "...", "part_name": "...", "qty": 1}
  dispatch_technician   → {"tech_id": "...", "ticket_id": "..."}
  send_customer_update  → {"ticket_id": "...", "message_template": "investigating|tech_dispatched|service_restored|delay_expected"}
  close_incident        → {"ticket_id": "..."}
  noop                  → {}

STRICT RULES:
- Always use exactly "backup_node_id" for reroute_traffic. NEVER use "target_node".
- Always include "test_type" when using run_remote_test.
- Always list_incidents first.
- Inspect the affected node before running tests.
- For overloaded nodes: reroute_traffic FIRST, then reboot_device.
- Reserve parts before dispatching technicians.
- Send customer update before closing.
- Close ALL open incidents.
- Return ONLY valid JSON. No explanation. No markdown. No extra fields.

Examples:
{"action_type": "run_remote_test", "node_id": "ONT-7", "test_type": "ping"}
{"action_type": "reroute_traffic", "node_id": "AGG-9", "backup_node_id": "AGG-9B"}
{"action_type": "reserve_part", "depot_id": "D-2", "part_name": "sfp_module", "qty": 1}
"""


def call_env(method: str, path: str, **kwargs):
    url = f"{ENV_URL}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=30, **kwargs)
        else:
            r = requests.post(url, timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ENV ERROR] {method} {path} → {e}", file=sys.stderr)
        return None


def get_action(observation: dict, history: list[dict]) -> dict:
    obs_text = json.dumps(observation, indent=2)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-6:],
        {"role": "user", "content": f"Current observation:\n{obs_text}\n\nReturn your next action as JSON:"},
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        # find first { ... } block safely
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in LLM response")
        raw = raw[start:end]

        action = json.loads(raw)

        # fix common LLM field name mistakes
        if "target_node" in action:
            action["backup_node_id"] = action.pop("target_node")
        if "node" in action and "node_id" not in action:
            action["node_id"] = action.pop("node")
        if "ticket" in action and "ticket_id" not in action:
            action["ticket_id"] = action.pop("ticket")

        # safety: ensure action_type exists
        if "action_type" not in action:
            return {"action_type": "noop"}

        return action

    except Exception as e:
        print(f"[LLM ERROR] {e}", file=sys.stderr)
        return {"action_type": "noop"}


def run_episode(task_id: str) -> float:
    # ── RESET ──────────────────────────────────────────────────────────────────
    reset_data = call_env("POST", f"/reset?task_id={task_id}")
    if not reset_data:
        print(json.dumps({
            "event": "END",
            "task_id": task_id,
            "score": 0.0,
            "steps": 0,
            "error": "reset_failed",
        }))
        return 0.0

    obs   = reset_data.get("observation", reset_data)
    done  = reset_data.get("done", False)
    score = 0.0
    step  = 0
    history: list[dict] = []

    print(json.dumps({
        "event":   "[START]",
        "task_id": task_id,
        "task":    obs.get("task_name", task_id),
        "prompt":  obs.get("prompt", ""),
    }), flush=True)

    # ── STEP LOOP ──────────────────────────────────────────────────────────────
    while not done and step < obs.get("max_steps", 12):
        action = get_action(obs, history)
        step  += 1

        step_data = call_env("POST", "/step", json=action)
        if not step_data:
            break

        reward   = step_data.get("reward", 0.0)
        done     = step_data.get("done", False)
        obs      = step_data.get("observation", obs)
        score    = obs.get("progress_score", score)
        message  = obs.get("message", "")

        print(json.dumps({
            "event":       "[STEP]",
            "task_id":     task_id,
            "step":        step,
            "action":      action,
            "reward":      reward,
            "done":        done,
            "score":       score,
            "message":     message,
        }), flush=True)

        history.append({"role": "assistant", "content": json.dumps(action)})
        history.append({"role": "user",      "content": f"Result: {message} | reward={reward} | score={score}"})

        time.sleep(0.3)

    # ── END ────────────────────────────────────────────────────────────────────
    final_score = round(max(0.0, min(1.0, score)), 3)

    print(json.dumps({
        "event":   "[END]",
        "task_id": task_id,
        "steps":   step,
        "score":   final_score,
        "done":    done,
    }), flush=True)

    return final_score


def main():
    results = {}
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)
        score = run_episode(task_id)
        results[task_id] = score
        assert 0.0 <= score <= 1.0, f"Score out of range for {task_id}: {score}"

    print("\n" + json.dumps({"event": "SUMMARY", "results": results}), flush=True)
    print("All tasks completed successfully.", flush=True)


if __name__ == "__main__":
    main()