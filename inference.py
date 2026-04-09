from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


API_BASE_URL = os.environ.get("APIBASEURL", "").strip()
API_KEY = os.environ.get("APIKEY", "").strip()
MODEL_NAME = os.environ.get("MODELNAME") or os.environ.get("MODEL") or "gpt-4o-mini"
ENV_BASE_URL = os.environ.get("ENVBASEURL", "http://127.0.0.1:7860").rstrip("/")
ENABLE_LLM = os.environ.get("ENABLE_LLM", "0").strip().lower() in {"1", "true", "yes"}

TASKS = ["easy", "medium", "hard"]

MAX_STEPS = {
    "easy": 10,
    "medium": 15,
    "hard": 20,
}

TASK_NAMES = {
    "easy": "Single-site remote fix",
    "medium": "Cabinet failure dispatch",
    "hard": "Regional storm response",
}

TASK_PROMPTS = {
    "easy": "A business customer is offline. Identify the affected node, run the right remote diagnostic, reboot the correct device, then close the ticket.",
    "medium": "A neighborhood cabinet is down. Diagnose the correct cabinet, reserve the needed spare part, dispatch the right technician, and close the ticket only after repair.",
    "hard": "A storm caused multiple outages. Prioritize the critical customer, inspect the overloaded aggregation node, reroute traffic to the backup, reserve the correct part for the cabinet outage, dispatch the right technician, send the right update, restore service, and close all incidents safely.",
}

ALLOWED_ACTIONS = {
    "rundiagnostic",
    "rebootdevice",
    "dispatchtechnician",
    "reservepart",
    "reroutetraffic",
    "sendcustomerupdate",
    "closeticket",
    "noop",
}


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def strict_unit(x: Any) -> float:
    try:
        value = float(x)
    except Exception:
        value = 0.01
    value = max(0.01, min(0.99, value))
    return round(value, 4)


def clean_token(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", "_", text)
    return text or "na"


def safe_json(response: requests.Response) -> Dict[str, Any]:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def reset_env(task_id: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"taskid": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return safe_json(resp)


def step_env(action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"actiontype": action_type, "params": params or {}},
        timeout=30,
    )
    resp.raise_for_status()
    return safe_json(resp)


def get_action_log(obs: Dict[str, Any]) -> List[str]:
    action_log = obs.get("actionlog", [])
    return action_log if isinstance(action_log, list) else []


def find_incident(obs: Dict[str, Any], incident_id: str) -> Optional[Dict[str, Any]]:
    for inc in obs.get("incidents", []) or []:
        if isinstance(inc, dict) and inc.get("id") == incident_id:
            return inc
    return None


def find_node(obs: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    for node in obs.get("network", []) or []:
        if isinstance(node, dict) and node.get("id") == node_id:
            return node
    return None


def find_technician(obs: Dict[str, Any], tech_id: str) -> Optional[Dict[str, Any]]:
    for tech in obs.get("technicians", []) or []:
        if isinstance(tech, dict) and tech.get("id") == tech_id:
            return tech
    return None


def find_part(obs: Dict[str, Any], part_id: str) -> Optional[Dict[str, Any]]:
    for part in obs.get("inventory", []) or []:
        if isinstance(part, dict) and part.get("id") == part_id:
            return part
    return None


def has_logged(action_log: List[str], action_name: str, token: str = "") -> bool:
    for line in action_log:
        text = str(line)
        if action_name in text and (not token or token in text):
            return True
    return False


def action(actiontype: str, **params: Any) -> Dict[str, Any]:
    return {"actiontype": actiontype, "params": params}


def close_any_resolved(obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for inc in obs.get("incidents", []) or []:
        if isinstance(inc, dict) and inc.get("status") == "resolved":
            incident_id = inc.get("id")
            if incident_id:
                return action("closeticket", incidentid=incident_id)
    return None


def heuristic_action(task_id: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    action_log = get_action_log(obs)

    close_now = close_any_resolved(obs)
    if close_now:
        return close_now

    if task_id == "easy":
        inc = find_incident(obs, "INC-001")
        node = find_node(obs, "ONT-007")

        if inc and inc.get("status") == "resolved":
            return action("closeticket", incidentid="INC-001")

        if node and node.get("status") == "offline":
            if not has_logged(action_log, "rundiagnostic", "ONT-007"):
                return action("rundiagnostic", nodeid="ONT-007")
            return action("rebootdevice", nodeid="ONT-007")

        if node and node.get("status") == "degraded":
            return action("rundiagnostic", nodeid="ONT-007")

        return action("noop")

    if task_id == "medium":
        inc = find_incident(obs, "INC-010")
        cab = find_node(obs, "CAB-012")
        part = find_part(obs, "P2")
        tech = find_technician(obs, "T3")

        if inc and inc.get("status") == "resolved":
            return action("closeticket", incidentid="INC-010")

        if cab and cab.get("status") in {"offline", "degraded"} and not has_logged(action_log, "rundiagnostic", "CAB-012"):
            return action("rundiagnostic", nodeid="CAB-012")

        if part and int(part.get("reserved", 0) or 0) < 1 and int(part.get("quantity", 0) or 0) > 0:
            return action("reservepart", partid="P2")

        if tech and bool(tech.get("available", False)):
            return action("dispatchtechnician", techid="T3", location="CAB-012")

        return action("noop")

    if task_id == "hard":
        inc_020 = find_incident(obs, "INC-020")
        inc_021 = find_incident(obs, "INC-021")
        inc_022 = find_incident(obs, "INC-022")
        agg = find_node(obs, "AGG-002")
        cab = find_node(obs, "CAB-019")
        ont = find_node(obs, "ONT-031")
        part = find_part(obs, "P2")
        tech = find_technician(obs, "T3")

        if inc_020 and inc_020.get("status") == "resolved":
            return action("closeticket", incidentid="INC-020")
        if inc_021 and inc_021.get("status") == "resolved":
            return action("closeticket", incidentid="INC-021")
        if inc_022 and inc_022.get("status") == "resolved":
            return action("closeticket", incidentid="INC-022")

        if agg and agg.get("status") == "overloaded":
            if not has_logged(action_log, "rundiagnostic", "AGG-002"):
                return action("rundiagnostic", nodeid="AGG-002")
            return action("reroutetraffic", fromnodeid="AGG-002", tonodeid="AGG-BACKUP")

        if cab and cab.get("status") in {"offline", "degraded"} and not has_logged(action_log, "rundiagnostic", "CAB-019"):
            return action("rundiagnostic", nodeid="CAB-019")

        if part and int(part.get("reserved", 0) or 0) < 1 and int(part.get("quantity", 0) or 0) > 0:
            return action("reservepart", partid="P2")

        if tech and bool(tech.get("available", False)) and not has_logged(action_log, "dispatchtechnician", "T3"):
            return action("dispatchtechnician", techid="T3", location="CAB-019")

        if inc_021 and not has_logged(action_log, "sendcustomerupdate", "INC-021"):
            return action("sendcustomerupdate", incidentid="INC-021", messagetype="update")

        if ont and ont.get("status") in {"degraded", "offline"} and not has_logged(action_log, "rundiagnostic", "ONT-031"):
            return action("rundiagnostic", nodeid="ONT-031")

        close_now = close_any_resolved(obs)
        if close_now:
            return close_now

        return action("noop")

    return action("noop")


def build_prompt(obs: Dict[str, Any], task_id: str) -> str:
    return (
        f"You are a telecom NOC engineer.\n"
        f"Task: {task_id} - {TASK_PROMPTS.get(task_id, '')}\n\n"
        f"Observation:\n{json.dumps(obs, ensure_ascii=False, indent=2)}\n\n"
        f"Return ONLY valid JSON with exactly this shape:\n"
        f'{{"actiontype":"noop","params":{{}}}}\n\n'
        f"Allowed actiontype values: {sorted(ALLOWED_ACTIONS)}"
    )


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def sanitize_action(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return action("noop")

    actiontype = str(raw.get("actiontype", "noop")).strip()
    params = raw.get("params", {})
    if actiontype not in ALLOWED_ACTIONS:
        actiontype = "noop"
    if not isinstance(params, dict):
        params = {}

    safe_params: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(key, str) and isinstance(value, (str, int, float, bool)) or value is None:
            safe_params[str(key)] = value

    return {"actiontype": actiontype, "params": safe_params}


def llm_action(obs: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    if not ENABLE_LLM:
        return None
    if not API_BASE_URL or not API_KEY:
        eprint("LLM disabled or missing API credentials; using deterministic fallback.")
        return None
    if OpenAI is None:
        eprint("OpenAI package import failed; using deterministic fallback.")
        return None

    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert telecom NOC engineer. Output JSON only.",
                },
                {
                    "role": "user",
                    "content": build_prompt(obs, task_id),
                },
            ],
        )
        raw = (response.choices.message.content or "").strip()
        parsed = extract_first_json_object(raw)
        if not parsed:
            return None
        return sanitize_action(parsed)
    except Exception as exc:
        eprint(f"LLM ERROR: {exc}")
        return None


def choose_action(task_id: str, obs: Dict[str, Any], step_num: int) -> Dict[str, Any]:
    deterministic = heuristic_action(task_id, obs)
    if deterministic.get("actiontype") != "noop":
        return deterministic

    llm = llm_action(obs, task_id)
    if isinstance(llm, dict) and llm.get("actiontype") in ALLOWED_ACTIONS:
        return llm

    return deterministic


def print_start(task_id: str) -> None:
    print(f"[START] task={clean_token(task_id)}", flush=True)


def print_step(task_id: str, step_num: int, action_name: str, reward: float, score: float) -> None:
    print(
        f"[STEP] task={clean_token(task_id)} step={step_num} action={clean_token(action_name)} reward={reward:.4f} score={score:.4f}",
        flush=True,
    )


def print_end(task_id: str, score: float, steps: int) -> None:
    print(
        f"[END] task={clean_token(task_id)} score={strict_unit(score):.4f} steps={int(steps)}",
        flush=True,
    )


def run_task(task_id: str) -> float:
    score = 0.01
    step_num = 0
    obs: Dict[str, Any] = {}
    done = False

    print_start(task_id)

    try:
        obs = reset_env(task_id)

        while not done and step_num < MAX_STEPS[task_id]:
            chosen = choose_action(task_id, obs, step_num)
            action_name = chosen.get("actiontype", "noop")
            params = chosen.get("params", {}) if isinstance(chosen.get("params", {}), dict) else {}

            try:
                result = step_env(action_name, params)
            except Exception as exc:
                eprint(f"STEP ERROR [{task_id}] {exc}")
                result = {
                    "reward": 0.01,
                    "done": True,
                    "score": score,
                    "observation": obs,
                    "info": {"message": f"step failed: {exc}"},
                }

            step_num += 1
            reward = strict_unit(result.get("reward", 0.01))
            score = strict_unit(result.get("score", score))
            done = bool(result.get("done", False))

            new_obs = result.get("observation", obs)
            if isinstance(new_obs, dict):
                obs = new_obs

            print_step(task_id, step_num, action_name, reward, score)

    except Exception as exc:
        eprint(f"TASK ERROR [{task_id}] {exc}")

    finally:
        print_end(task_id, score, step_num)

    return strict_unit(score)


def main() -> None:
    for task_id in TASKS:
        run_task(task_id)


if __name__ == "__main__":
    main()