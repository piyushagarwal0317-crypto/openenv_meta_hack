"""CloudScaleRL / AutoScaleOps — Data Models.

Defines the action space, observation space, and all typed schemas used by
the cloud autoscaling environment.

Supports Kubernetes-realistic features:
- HPA (Horizontal Pod Autoscaler) via scale_delta
- VPA (Vertical Pod Autoscaler) via pod_size
- Cluster Autoscaler via node_delta
- Node/memory resource tracking
- Infrastructure event reporting
"""

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ScaleDelta = Literal[-2, -1, 0, 1, 2]
NodeDelta = Literal[-1, 0, 1]
PodSizeTier = Literal["xs", "sm", "md", "lg"]

# Pod size → resource mapping (used by environment)
POD_SIZE_SPECS = {
    "xs": {"cpu_request": 0.25, "memory_request_mb": 256, "capacity": 250, "cost_mult": 0.5},
    "sm": {"cpu_request": 0.50, "memory_request_mb": 512, "capacity": 500, "cost_mult": 1.0},
    "md": {"cpu_request": 1.00, "memory_request_mb": 1024, "capacity": 800, "cost_mult": 1.6},
    "lg": {"cpu_request": 2.00, "memory_request_mb": 2048, "capacity": 1200, "cost_mult": 2.4},
}

# Node type → resource mapping
NODE_TYPE_SPECS = {
    "small":  {"cpu": 2, "memory_gb": 4,  "cost_mult": 1.0, "provision_base": 10},
    "medium": {"cpu": 4, "memory_gb": 8,  "cost_mult": 1.8, "provision_base": 12},
    "large":  {"cpu": 8, "memory_gb": 16, "cost_mult": 3.2, "provision_base": 15},
}


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class CloudScaleAction(Action):
    """Autoscaling decision issued by the agent each time-step.

    The agent controls three dimensions of scaling:

    **HPA (Horizontal Pod Autoscaler)** — ``scale_delta``:
        -2  →  scale down by 2 pods
        -1  →  scale down by 1 pod
         0  →  hold current scale (no change)
        +1  →  scale up by 1 pod
        +2  →  scale up by 2 pods

    **Cluster Autoscaler** — ``node_delta``:
        -1  →  remove 1 worker node
         0  →  no node change
        +1  →  add 1 worker node

    **VPA (Vertical Pod Autoscaler)** — ``pod_size``:
        None  →  keep current pod size
        "xs"/"sm"/"md"/"lg"  →  resize pods (triggers rolling restart)

    Scaling requests are subject to provisioning delay and are *not*
    applied instantly.  Invalid requests (e.g. scaling below 1 pod) are
    clamped and penalised.
    """

    scale_delta: ScaleDelta = Field(
        ...,
        description=(
            "Number of pods to add (+) or remove (−). "
            "Must be one of: -2, -1, 0, 1, 2."
        ),
    )
    node_delta: NodeDelta = Field(
        default=0,
        description=(
            "Number of worker nodes to add (+1) or remove (-1). "
            "Node provisioning takes longer than pod provisioning. "
            "Must be one of: -1, 0, 1."
        ),
    )
    pod_size: Optional[PodSizeTier] = Field(
        default=None,
        description=(
            "Target pod resource tier for VPA. None means keep current size. "
            "Changing size triggers a rolling restart with temporary reduced capacity. "
            "Must be one of: 'xs', 'sm', 'md', 'lg', or null."
        ),
    )


# ---------------------------------------------------------------------------
# Observation — helper sub-models
# ---------------------------------------------------------------------------

class PendingScaleEvent(BaseModel):
    """Represents a single in-flight scaling operation that has not yet
    completed (e.g. a pod that is still starting up)."""

    direction: Literal["up", "down"] = Field(
        ..., description="Whether this event is a scale-up or scale-down."
    )
    pods: int = Field(
        ..., description="Number of pods being added or removed in this event."
    )
    remaining_steps: int = Field(
        ...,
        description="Time-steps remaining before this scaling event takes effect.",
    )


class TrafficSnapshot(BaseModel):
    """A lightweight summary of recent traffic behaviour, optionally
    provided in harder task difficulties to give the agent richer context."""

    recent_avg_request_rate: float = Field(
        0.0, description="Rolling average request rate over the last N steps."
    )
    recent_peak_request_rate: float = Field(
        0.0, description="Peak request rate observed in the last N steps."
    )
    trend: Literal["rising", "falling", "stable"] = Field(
        "stable", description="Short-term traffic trend direction."
    )


class NodeInfo(BaseModel):
    """Summary of the cluster's worker node state."""

    total_nodes: int = Field(0, ge=0, description="Total worker nodes in the cluster")
    active_nodes: int = Field(0, ge=0, description="Nodes that are ready to host pods")
    pending_node_ups: int = Field(0, ge=0, description="Nodes currently being provisioned")
    pending_node_downs: int = Field(0, ge=0, description="Nodes being drained/terminated")
    node_cpu_capacity: float = Field(0.0, ge=0.0, description="Total vCPU capacity across all active nodes")
    node_memory_capacity_gb: float = Field(0.0, ge=0.0, description="Total memory (GB) across all active nodes")
    node_cpu_used: float = Field(0.0, ge=0.0, description="vCPU consumed by all running pods")
    node_memory_used_gb: float = Field(0.0, ge=0.0, description="Memory (GB) consumed by all running pods")
    node_type: str = Field("medium", description="Current node instance type")


class PodResourceInfo(BaseModel):
    """Current pod resource configuration (VPA state)."""

    pod_size: PodSizeTier = Field("sm", description="Current pod resource tier")
    pod_cpu_request: float = Field(0.5, description="CPU request per pod (vCPU)")
    pod_memory_request_mb: float = Field(512, description="Memory request per pod (MB)")
    pod_capacity: float = Field(500, description="Request throughput capacity per pod (req/s)")
    vpa_restart_in_progress: bool = Field(False, description="Whether a VPA rolling restart is in progress")
    vpa_restart_remaining_steps: int = Field(0, ge=0, description="Steps remaining for VPA restart")


class InfraEvent(BaseModel):
    """An infrastructure event that occurred during the episode."""

    step: int = Field(..., description="Time-step when the event occurred")
    event_type: str = Field(..., description="Type of event (e.g. 'flash_crowd', 'node_failure', 'vpa_restart')")
    details: str = Field("", description="Human-readable description of the event")


# ---------------------------------------------------------------------------
# Observation — main schema
# ---------------------------------------------------------------------------

class CloudScaleObservation(Observation):
    """Complete observation returned by the environment after each step.

    Contains the full autoscaling state, cumulative KPIs, reward info,
    node/cluster state, VPA state, and episode-control flags.
    """

    # ---- time ----
    task_id: str = Field(..., description="Task difficulty id (easy / medium / hard)")
    time_step: int = Field(..., description="Current time-step of the episode")
    horizon: int = Field(..., description="Total episode length in time-steps")

    # ---- core infrastructure metrics ----
    cpu_utilization: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate CPU utilisation across active pods (0.0–1.0)",
    )
    memory_utilization: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate memory utilisation across active pods (0.0–1.0)",
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="Current average request latency in milliseconds"
    )
    request_rate: float = Field(
        ..., ge=0.0, description="Incoming requests per second at this time-step"
    )
    queue_length: int = Field(
        ..., ge=0, description="Number of requests currently waiting in the queue"
    )
    active_pods: int = Field(
        ..., ge=0, description="Number of pods currently running and serving traffic"
    )

    # ---- pending scaling info ----
    pending_scale_ups: int = Field(
        0,
        ge=0,
        description="Total number of pods currently being provisioned (not yet active)",
    )
    pending_scale_downs: int = Field(
        0,
        ge=0,
        description="Total number of pods currently being terminated (not yet removed)",
    )
    pending_events: list[PendingScaleEvent] = Field(
        default_factory=list,
        description="Detailed list of in-flight scaling events with countdown timers.",
    )

    # ---- optional traffic context ----
    traffic_snapshot: TrafficSnapshot | None = Field(
        default=None,
        description=(
            "Optional rolling traffic summary. "
            "Provided in some task difficulties to aid decision-making."
        ),
    )

    # ---- node / cluster info ----
    node_info: NodeInfo = Field(
        default_factory=NodeInfo,
        description="Worker node cluster state (Cluster Autoscaler context).",
    )

    # ---- pod resource info (VPA) ----
    pod_resource_info: PodResourceInfo = Field(
        default_factory=PodResourceInfo,
        description="Current pod resource tier and VPA state.",
    )

    # ---- infrastructure events ----
    recent_events: list[InfraEvent] = Field(
        default_factory=list,
        description="Recent infrastructure events (last 5) such as node failures, traffic spikes, VPA restarts.",
    )

    # ---- cumulative KPIs ----
    total_requests_processed: int = Field(
        0, ge=0, description="Cumulative count of successfully processed requests"
    )
    total_requests_dropped: int = Field(
        0, ge=0, description="Cumulative count of dropped/timed-out requests"
    )
    total_sla_violations: int = Field(
        0,
        ge=0,
        description="Cumulative count of time-steps where latency exceeded the SLA target",
    )
    average_latency_ms: float = Field(
        0.0, ge=0.0, description="Running average latency across all processed requests"
    )

    # ---- reward ----
    reward: float = Field(0.0, description="Reward received at this time-step")
    cumulative_reward: float = Field(
        0.0, description="Sum of all rewards received so far in this episode"
    )

    # ---- episode control ----
    done: bool = Field(
        False, description="Whether the episode has ended (horizon reached)"
    )

    # ---- metadata ----
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value metadata for debugging or extended info "
            "(e.g. cold-start flags, cost breakdown)."
        ),
    )
