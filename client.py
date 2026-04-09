"""CloudScaleRL / AutoScaleOps — Cloud Autoscaling Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import (
        CloudScaleAction,
        CloudScaleObservation,
        NodeInfo,
        PodResourceInfo,
        InfraEvent,
    )
except ImportError:  # pragma: no cover
    from models import (
        CloudScaleAction,
        CloudScaleObservation,
        NodeInfo,
        PodResourceInfo,
        InfraEvent,
    )


class CloudScaleEnv(EnvClient[CloudScaleAction, CloudScaleObservation, State]):
    """
    Client for the CloudScaleRL / AutoScaleOps Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    The agent acts as a Site Reliability Engineer (SRE) managing cloud
    autoscaling decisions under uncertain traffic, delayed infrastructure
    changes, latency constraints, and cost pressure.

    Supports three scaling dimensions:
    - HPA: Horizontal pod scaling via scale_delta
    - VPA: Vertical pod scaling via pod_size
    - Cluster Autoscaler: Node scaling via node_delta

    Example:
        >>> with CloudScaleEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task="easy")
        ...     obs = result.observation
        ...     print(f"CPU: {obs.cpu_utilization}, Pods: {obs.active_pods}")
        ...     print(f"Nodes: {obs.node_info.active_nodes}")
        ...
        ...     result = client.step(CloudScaleAction(scale_delta=1, node_delta=0))
        ...     print(f"Latency: {result.observation.latency_ms} ms")

    Example with Docker:
        >>> client = CloudScaleEnv.from_docker_image("cloudscale-autoscaling:latest")
        >>> try:
        ...     result = client.reset(task="medium")
        ...     result = client.step(CloudScaleAction(scale_delta=0))
        ... finally:
        ...     client.close()
    """

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _step_payload(self, action: CloudScaleAction) -> Dict[str, Any]:
        """
        Convert CloudScaleAction to JSON payload for step message.

        Args:
            action: CloudScaleAction instance with scale_delta, node_delta,
                    and optional pod_size.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        payload = {"scale_delta": action.scale_delta}
        # only include optional fields if they differ from defaults
        if action.node_delta != 0:
            payload["node_delta"] = action.node_delta
        if action.pod_size is not None:
            payload["pod_size"] = action.pod_size
        return payload

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[CloudScaleObservation]:
        """
        Parse server response into StepResult[CloudScaleObservation].

        Args:
            payload: JSON response data from server.

        Returns:
            StepResult with CloudScaleObservation.
        """
        obs_data = payload.get("observation", {})

        # parse node info
        node_data = obs_data.get("node_info", {})
        node_info = NodeInfo(
            total_nodes=node_data.get("total_nodes", 0),
            active_nodes=node_data.get("active_nodes", 0),
            pending_node_ups=node_data.get("pending_node_ups", 0),
            pending_node_downs=node_data.get("pending_node_downs", 0),
            node_cpu_capacity=node_data.get("node_cpu_capacity", 0.0),
            node_memory_capacity_gb=node_data.get("node_memory_capacity_gb", 0.0),
            node_cpu_used=node_data.get("node_cpu_used", 0.0),
            node_memory_used_gb=node_data.get("node_memory_used_gb", 0.0),
            node_type=node_data.get("node_type", "medium"),
        )

        # parse pod resource info
        pod_res_data = obs_data.get("pod_resource_info", {})
        pod_resource_info = PodResourceInfo(
            pod_size=pod_res_data.get("pod_size", "sm"),
            pod_cpu_request=pod_res_data.get("pod_cpu_request", 0.5),
            pod_memory_request_mb=pod_res_data.get("pod_memory_request_mb", 512),
            pod_capacity=pod_res_data.get("pod_capacity", 500),
            vpa_restart_in_progress=pod_res_data.get("vpa_restart_in_progress", False),
            vpa_restart_remaining_steps=pod_res_data.get("vpa_restart_remaining_steps", 0),
        )

        # parse infra events
        events_data = obs_data.get("recent_events", [])
        recent_events = [
            InfraEvent(
                step=e.get("step", 0),
                event_type=e.get("event_type", "unknown"),
                details=e.get("details", ""),
            )
            for e in events_data
        ]

        observation = CloudScaleObservation(
            # ---- time ----
            task_id=obs_data.get("task_id", ""),
            time_step=obs_data.get("time_step", 0),
            horizon=obs_data.get("horizon", 0),

            # ---- infrastructure state ----
            cpu_utilization=obs_data.get("cpu_utilization", 0.0),
            memory_utilization=obs_data.get("memory_utilization", 0.0),
            latency_ms=obs_data.get("latency_ms", 0.0),
            request_rate=obs_data.get("request_rate", 0.0),
            queue_length=obs_data.get("queue_length", 0),
            active_pods=obs_data.get("active_pods", 0),
            pending_scale_ups=obs_data.get("pending_scale_ups", 0),
            pending_scale_downs=obs_data.get("pending_scale_downs", 0),

            # ---- node & resource info ----
            node_info=node_info,
            pod_resource_info=pod_resource_info,
            recent_events=recent_events,

            # ---- cumulative KPIs ----
            total_requests_processed=obs_data.get("total_requests_processed", 0),
            total_requests_dropped=obs_data.get("total_requests_dropped", 0),
            total_sla_violations=obs_data.get("total_sla_violations", 0),
            average_latency_ms=obs_data.get("average_latency_ms", 0.0),

            # ---- reward ----
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),

            # ---- episode control ----
            done=payload.get("done", obs_data.get("done", False)),

            # ---- optional metadata ----
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request.

        Returns:
            State object with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
