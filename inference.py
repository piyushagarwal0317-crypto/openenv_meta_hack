"""
Inference script for CloudScaleRL / AutoScaleOps OpenEnv.

Required environment variables for LLM:
- API_BASE_URL: OpenAI-compatible LLM endpoint
- MODEL_NAME: model identifier
- HF_TOKEN: API key

Optional:
- LOCAL_IMAGE_NAME: local Docker image name used by from_docker_image()
  (example: cloudscale-autoscaling:latest)

This script emits exactly these stdout line types:
- [START] ...
- [STEP]  ... (one per step)
- [END]   ... (always)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from client import CloudScaleEnv
from models import CloudScaleAction


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK_URL = os.getenv("BENCHMARK_URL", "http://localhost:8000")
BENCHMARK = os.getenv("BENCHMARK", "cloudscale_rl")
MAX_STEPS = int(os.getenv("MAX_STEPS", "0"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))
TASKS = ["easy", "medium", "hard"]


SYSTEM_PROMPT = (
    "You are a cloud infrastructure SRE agent managing Kubernetes-like autoscaling. "
    "You control three scaling dimensions:\n"
    "1. HPA (Horizontal Pod Autoscaler): scale_delta (-2 to +2) to add/remove pods\n"
    "2. Cluster Autoscaler: node_delta (-1, 0, +1) to add/remove worker nodes\n"
    "3. VPA (Vertical Pod Autoscaler): pod_size ('xs','sm','md','lg' or null) to resize pods\n\n"
    "Return exactly one JSON object: {\"scale_delta\": N, \"node_delta\": M, \"pod_size\": S}\n"
    "where N is -2,-1,0,1,2 and M is -1,0,1 and S is 'xs','sm','md','lg' or null.\n\n"
    "Key rules:\n"
    "- Scale pods up early when latency or queue pressure rises\n"
    "- Add nodes when pod scheduling fails or node CPU/memory > 85%\n"
    "- Use VPA to right-size pods: bigger pods = more capacity but higher cost\n"
    "- Nodes take much longer to provision than pods — plan ahead\n"
    "- Avoid oscillating between actions\n"
    "- Watch for infrastructure events (flash crowds, DDoS, node failures)"
)


# ---------------------------------------------------------------------------
# Logging helpers (strict format for evaluators)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: str | None
) -> None:
    err = _single_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    reward_csv = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={reward_csv}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


def extract_result_fields(
    result: Any,
) -> tuple[dict[str, Any], float, bool, str | None]:
    obs = result.observation
    if hasattr(obs, "model_dump"):
        observation = obs.model_dump()
    else:
        observation = dict(obs)
    reward = float(result.reward or 0.0)
    done = bool(result.done)
    return observation, reward, done, None


def action_to_str(action: dict[str, Any]) -> str:
    return _single_line(json.dumps(action, separators=(",", ":")))


def _single_line(text: str | None) -> str:
    return (text or "").replace("\n", " ").replace("\r", " ").strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_user_prompt(step: int, obs: dict[str, Any], rewards: list[float]) -> str:
    # Traffic snapshot
    traffic = obs.get("traffic_snapshot") or {}
    trend = traffic.get("trend", "stable")
    avg_rate = traffic.get("recent_avg_request_rate", 0)
    peak_rate = traffic.get("recent_peak_request_rate", 0)

    # Node info
    node_info = obs.get("node_info") or {}
    pod_res = obs.get("pod_resource_info") or {}

    # Recent events
    events = obs.get("recent_events") or []
    event_strs = [f"{e.get('event_type', '?')}: {e.get('details', '')}" for e in events[-3:]]

    summary = {
        "step": step,
        "task": obs.get("task_id"),
        "time_step": obs.get("time_step"),
        "horizon": obs.get("horizon"),
        "cpu_utilization_pct": round(float(obs.get("cpu_utilization", 0)) * 100, 1),
        "memory_utilization_pct": round(float(obs.get("memory_utilization", 0)) * 100, 1),
        "latency_ms": round(float(obs.get("latency_ms", 0)), 1),
        "request_rate": round(float(obs.get("request_rate", 0)), 1),
        "queue_length": int(obs.get("queue_length", 0)),
        "active_pods": int(obs.get("active_pods", 0)),
        "pending_scale_ups": int(obs.get("pending_scale_ups", 0)),
        "pending_scale_downs": int(obs.get("pending_scale_downs", 0)),
        "traffic_trend": trend,
        "avg_request_rate": round(float(avg_rate), 1),
        "peak_request_rate": round(float(peak_rate), 1),
        # node info
        "active_nodes": int(node_info.get("active_nodes", 0)),
        "pending_node_ups": int(node_info.get("pending_node_ups", 0)),
        "node_cpu_capacity": round(float(node_info.get("node_cpu_capacity", 0)), 1),
        "node_cpu_used": round(float(node_info.get("node_cpu_used", 0)), 1),
        "node_memory_capacity_gb": round(float(node_info.get("node_memory_capacity_gb", 0)), 1),
        "node_memory_used_gb": round(float(node_info.get("node_memory_used_gb", 0)), 1),
        # VPA info
        "pod_size": pod_res.get("pod_size", "sm"),
        "pod_capacity": round(float(pod_res.get("pod_capacity", 500)), 0),
        "vpa_restart_in_progress": pod_res.get("vpa_restart_in_progress", False),
        # KPIs
        "total_sla_violations": int(obs.get("total_sla_violations", 0)),
        "total_requests_dropped": int(obs.get("total_requests_dropped", 0)),
        "average_latency_ms": round(float(obs.get("average_latency_ms", 0)), 1),
        "recent_rewards": [round(r, 2) for r in rewards[-5:]],
        "recent_events": event_strs,
    }
    return (
        "Choose one scaling action as JSON: "
        "{\"scale_delta\": N, \"node_delta\": M, \"pod_size\": S}\n"
        + json.dumps(summary, ensure_ascii=True)
    )


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------


def parse_action(text: str) -> dict[str, Any]:
    text = (text or "").strip()

    # Try JSON extraction first
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            delta = int(obj.get("scale_delta", 0))
            delta = max(-2, min(2, delta))

            node_delta = int(obj.get("node_delta", 0))
            node_delta = max(-1, min(1, node_delta))

            pod_size = obj.get("pod_size", None)
            if pod_size not in ("xs", "sm", "md", "lg", None):
                pod_size = None

            result = {"scale_delta": delta, "node_delta": node_delta}
            if pod_size is not None:
                result["pod_size"] = pod_size
            return result
        except Exception:
            pass

    # Try bare integer (legacy fallback)
    match = re.search(r"(-?[0-2])", text)
    if match:
        return {"scale_delta": int(match.group(1))}

    return {"scale_delta": 0}


def action_key(action: dict[str, Any]) -> tuple:
    return (action.get("scale_delta", 0),)


def observation_signature(obs: dict[str, Any]) -> tuple:
    return (
        int(obs.get("time_step", 0)),
        int(obs.get("active_pods", 0)),
        int(obs.get("queue_length", 0)),
        int(obs.get("total_sla_violations", 0)),
    )


# ---------------------------------------------------------------------------
# LLM decision + fallback
# ---------------------------------------------------------------------------


def choose_action_with_llm(
    client: OpenAI, step: int, obs: dict[str, Any], rewards: list[float]
) -> tuple[dict[str, Any], bool]:
    prompt = build_user_prompt(step, obs, rewards)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        action = parse_action(content)
        return action, False
    except Exception:
        return choose_fallback_action(obs), True


def choose_fallback_action(obs: dict[str, Any]) -> dict[str, Any]:
    """Heuristic fallback when LLM fails or is unavailable."""
    cpu = float(obs.get("cpu_utilization", 0))
    latency = float(obs.get("latency_ms", 0))
    queue = int(obs.get("queue_length", 0))
    pending_ups = int(obs.get("pending_scale_ups", 0))
    pending_downs = int(obs.get("pending_scale_downs", 0))
    active_pods = int(obs.get("active_pods", 0))

    # Node info
    node_info = obs.get("node_info") or {}
    node_cpu_cap = float(node_info.get("node_cpu_capacity", 1))
    node_cpu_used = float(node_info.get("node_cpu_used", 0))
    active_nodes = int(node_info.get("active_nodes", 1))
    pending_node_ups = int(node_info.get("pending_node_ups", 0))

    scale_delta = 0
    node_delta = 0
    pod_size = None

    # --- Pod scaling (HPA) ---
    # Aggressive scale-up if latency spike or large queue
    if latency > 300 or queue > 2000:
        if pending_ups == 0:
            scale_delta = 2

    # Moderate scale-up if above SLA or high CPU
    elif cpu > 0.85 or latency > 150 or queue > 500:
        if pending_ups == 0:
            scale_delta = 1

    # Scale down if very low utilisation and no queue
    elif cpu < 0.2 and queue == 0 and active_pods > 1 and pending_downs == 0:
        scale_delta = -1

    # --- Node scaling (Cluster Autoscaler) ---
    if node_cpu_cap > 0:
        node_cpu_ratio = node_cpu_used / node_cpu_cap
        if node_cpu_ratio > 0.85 and pending_node_ups == 0:
            node_delta = 1
        elif node_cpu_ratio < 0.2 and active_nodes > 1:
            node_delta = -1

    result = {"scale_delta": scale_delta, "node_delta": node_delta}
    if pod_size is not None:
        result["pod_size"] = pod_size
    return result


def choose_unsticking_action(obs: dict[str, Any]) -> dict[str, Any]:
    """If the agent keeps repeating the same action with no effect, force a change."""
    cpu = float(obs.get("cpu_utilization", 0))
    if cpu > 0.7:
        return {"scale_delta": 1}
    elif cpu < 0.3:
        return {"scale_delta": -1}
    return {"scale_delta": 0}


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_score(obs: dict[str, Any]) -> float:
    """Compute a normalised [0, 1] score from final observation."""
    total_steps = max(1, int(obs.get("time_step", 1) or 1))
    sla_violations = int(obs.get("total_sla_violations", 0) or 0)
    avg_latency = float(obs.get("average_latency_ms", 0) or 0)
    dropped = int(obs.get("total_requests_dropped", 0) or 0)
    active_pods = int(obs.get("active_pods", 0) or 0)

    # SLA compliance (50%)
    sla_compliance = max(0.0, 1.0 - (sla_violations / total_steps))

    # Efficiency (30%) — lower pods is better, normalised against 20
    efficiency = max(0.0, 1.0 - (active_pods / 20.0))

    # Latency quality (10%) — reward staying well under SLA
    sla_target = 250.0  # conservative default
    latency_score = max(0.0, 1.0 - (avg_latency / (sla_target * 2.0)))

    # Drop penalty (10%)
    drop_score = 1.0 if dropped == 0 else max(0.0, 1.0 - (dropped / 1000.0))

    score = (
        0.50 * sla_compliance
        + 0.30 * efficiency
        + 0.10 * latency_score
        + 0.10 * drop_score
    )
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


def run_task(client: OpenAI, task_name: str) -> None:
    rewards: list[float] = []
    recent_action_counts: dict[tuple, int] = {}
    prev_sig: tuple | None = None
    stuck_counter = 0
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        if LOCAL_IMAGE_NAME:
            env_client = CloudScaleEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env_client = CloudScaleEnv(base_url=BENCHMARK_URL)

        with env_client.sync() as env:
            reset_result = env.reset(task=task_name)
            obs, _, done, _ = extract_result_fields(reset_result)
            horizon = int(obs.get("horizon", 0) or 0)
            max_steps = MAX_STEPS if MAX_STEPS > 0 else horizon
            if max_steps <= 0:
                max_steps = 240

            for step in range(1, max_steps + 1):
                if done:
                    break

                action, used_fallback = choose_action_with_llm(
                    client, step, obs, rewards
                )

                key = action_key(action)
                recent_action_counts[key] = recent_action_counts.get(key, 0) + 1

                # Loop breaker: avoid repeating identical action forever.
                if recent_action_counts[key] >= 5:
                    action = choose_fallback_action(obs)
                    key = action_key(action)
                    recent_action_counts[key] = recent_action_counts.get(key, 0) + 1

                # If observation keeps repeating, force an unsticking action.
                current_sig = observation_signature(obs)
                if prev_sig is not None and current_sig == prev_sig:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                prev_sig = current_sig
                if stuck_counter >= 3:
                    action = choose_unsticking_action(obs)
                    key = action_key(action)
                    recent_action_counts[key] = recent_action_counts.get(key, 0) + 1

                # Decay memory so old repetitions do not dominate.
                if step % 5 == 0:
                    for k in list(recent_action_counts.keys()):
                        recent_action_counts[k] = max(0, recent_action_counts[k] - 1)
                        if recent_action_counts[k] == 0:
                            recent_action_counts.pop(k, None)

                try:
                    result = env.step(CloudScaleAction(**action))
                    obs, reward, done, last_error = extract_result_fields(result)
                except Exception as exc:
                    reward = 0.0
                    done = True
                    last_error = _single_line(str(exc))

                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=action_to_str(action),
                    reward=reward,
                    done=done,
                    error=last_error,
                )

                if done:
                    break

            score = compute_score(obs)
            success = score >= 0.5
    except Exception:
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if os.getenv("HF_TOKEN") is None:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_name in TASKS:
        run_task(client, task_name)


if __name__ == "__main__":
    main()
