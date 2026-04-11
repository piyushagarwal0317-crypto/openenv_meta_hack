# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the CloudScaleRL / AutoScaleOps Environment.
Includes interactive dashboard and spike injection API.
"""

import json
import os
from fastapi import Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from typing import Dict, Any

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import CloudScaleAction, CloudScaleObservation
    from .cloudscale_rl_environment import CloudScaleEnvironment, SCENARIOS
    from .grader import grade_episode
except (ModuleNotFoundError, ImportError):
    from models import CloudScaleAction, CloudScaleObservation
    from server.cloudscale_rl_environment import CloudScaleEnvironment, SCENARIOS
    from server.grader import grade_episode


# Create the app with web interface
app = create_app(
    CloudScaleEnvironment,
    CloudScaleAction,
    CloudScaleObservation,
    env_name="cloudscale_rl",
    max_concurrent_envs=10,
)


@app.middleware("http")
async def _compat_step_payload(request, call_next):
    if request.method == "POST" and request.url.path == "/step":
        raw_body = await request.body()
        if raw_body:
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                payload = None

            if isinstance(payload, dict) and "action" not in payload:
                action_keys = {"scale_delta", "node_delta", "pod_size"}
                if action_keys.intersection(payload):
                    wrapped_body = json.dumps({"action": payload}).encode("utf-8")
                    request._body = wrapped_body
                    request._stream_consumed = True
    return await call_next(request)

# ---------------------------------------------------------------------------
# Dashboard environment (separate instance for interactive dashboard use)
# ---------------------------------------------------------------------------

_dashboard_env: CloudScaleEnvironment | None = None
_dashboard_history: list[dict] = []


def _get_dashboard_env(task: str = "easy") -> CloudScaleEnvironment:
    global _dashboard_env
    if _dashboard_env is None:
        _dashboard_env = CloudScaleEnvironment(task=task)
    return _dashboard_env


def _obs_to_dict(obs: CloudScaleObservation) -> dict:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    return dict(obs)


# ---------------------------------------------------------------------------
# Dashboard API endpoints
# ---------------------------------------------------------------------------

@app.post("/dashboard/reset")
async def dashboard_reset(body: Dict[str, Any] = Body(default={})):
    """Reset the dashboard environment."""
    global _dashboard_env, _dashboard_history
    task = body.get("task", "easy")
    _dashboard_env = CloudScaleEnvironment(task=task)
    _dashboard_history = []
    obs = _dashboard_env.reset(task=task)
    data = _obs_to_dict(obs)
    _dashboard_history.append(data)
    return {"observation": data, "history_length": len(_dashboard_history)}


@app.post("/dashboard/step")
async def dashboard_step(body: Dict[str, Any] = Body(default={})):
    """Take a step in the dashboard environment."""
    env = _get_dashboard_env()
    scale_delta = body.get("scale_delta", 0)
    node_delta = body.get("node_delta", 0)
    pod_size = body.get("pod_size", None)
    action = CloudScaleAction(scale_delta=scale_delta, node_delta=node_delta, pod_size=pod_size)
    obs = env.step(action)
    data = _obs_to_dict(obs)
    _dashboard_history.append(data)
    return {"observation": data, "history_length": len(_dashboard_history)}


@app.post("/dashboard/inject_spike")
async def dashboard_inject_spike(body: Dict[str, Any] = Body(...)):
    """Inject a traffic spike into the dashboard environment."""
    env = _get_dashboard_env()
    spike_type = body.get("spike_type", "flash_crowd")
    multiplier = float(body.get("multiplier", 0))
    duration = int(body.get("duration", 0))
    result = env.inject_spike(spike_type, multiplier, duration)
    return result


@app.get("/dashboard/history")
async def dashboard_history():
    """Get the full step history for charting."""
    return {"history": _dashboard_history}


@app.post("/dashboard/auto_run")
async def dashboard_auto_run(body: Dict[str, Any] = Body(default={})):
    """Run multiple steps automatically with the hybrid policy."""
    from decision import choose_heuristic
    env = _get_dashboard_env()
    steps = min(int(body.get("steps", 10)), 50)
    results = []
    for _ in range(steps):
        obs = env._build_observation(done=False)
        action = choose_heuristic("hybrid", obs)
        obs = env.step(action)
        data = _obs_to_dict(obs)
        _dashboard_history.append(data)
        results.append({
            "action": {"scale_delta": action.scale_delta, "node_delta": action.node_delta,
                        "pod_size": action.pod_size},
            "reward": data.get("reward", 0),
        })
        if data.get("done", False):
            break
    return {"steps_taken": len(results), "results": results, "history_length": len(_dashboard_history)}


# ---------------------------------------------------------------------------
# Serve dashboard HTML
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """Redirect the default entry point to the dashboard."""
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the interactive CloudScaleRL dashboard."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


# ---------------------------------------------------------------------------
# Existing endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def get_tasks():
    """Returns the list of available scaling tasks."""
    return {
        "tasks": [
            {
                "id": cfg.task_id,
                "horizon": cfg.horizon,
                "sla_latency_ms": cfg.sla_latency_ms,
                "initial_pods": cfg.initial_pods,
            }
            for cfg in SCENARIOS.values()
        ]
    }

@app.post("/grader")
async def post_grader(metrics: Dict[str, Any] = Body(...)):
    """Calculates the score for a finished episode."""
    try:
        results = grade_episode(metrics)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "environment": "cloudscale_rl"}

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()
