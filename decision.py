"""
Decision logic for CloudScaleRL / AutoScaleOps.

Provides deterministic heuristic baseline policies for benchmark comparison.
These policies mirror the role that nearest/hybrid/noop play in the original baseline script.

Updated to support Kubernetes-realistic features:
- Node scaling decisions (Cluster Autoscaler)
- VPA pod size recommendations
- Memory-aware scaling
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from models import CloudScaleAction, CloudScaleObservation
except ImportError:
    from .models import CloudScaleAction, CloudScaleObservation


META_ACTIONS = [
    "threshold_cpu",
    "latency_queue",
    "hybrid",
    "noop",
    "emergency_scale",
]


@dataclass
class ActionChoice:
    action: CloudScaleAction
    label: str


def action_mask(obs: CloudScaleObservation) -> list[int]:
    """Return a binary mask for valid meta-actions."""
    is_emergency = obs.latency_ms > 300 or obs.queue_length > 2000
    # threshold_cpu, latency_queue, hybrid, and noop are always valid.
    # emergency_scale relies on there being actual severe pressure.
    return [1, 1, 1, 1, 1 if is_emergency else 0]


def _node_decision(obs: CloudScaleObservation) -> int:
    """Decide whether to add/remove a worker node based on cluster state."""
    node_info = obs.node_info
    if node_info is None:
        return 0

    # If pods can't be scheduled (node CPU near capacity), add a node
    if node_info.node_cpu_capacity > 0:
        cpu_ratio = node_info.node_cpu_used / node_info.node_cpu_capacity
        mem_ratio = node_info.node_memory_used_gb / max(0.1, node_info.node_memory_capacity_gb)

        # scale up node if resources are tight
        if cpu_ratio > 0.85 or mem_ratio > 0.85:
            if node_info.pending_node_ups == 0:
                return 1

        # scale down node if resources are very underutilised
        if cpu_ratio < 0.2 and mem_ratio < 0.2 and node_info.active_nodes > 1:
            if node_info.pending_node_downs == 0:
                return -1

    return 0


def _vpa_decision(obs: CloudScaleObservation) -> str | None:
    """Decide whether to change pod size based on resource utilisation."""
    pod_info = obs.pod_resource_info
    if pod_info is None or pod_info.vpa_restart_in_progress:
        return None

    current_size = pod_info.pod_size

    # If CPU is consistently high and latency is bad, scale up pod size
    if obs.cpu_utilization > 0.85 and obs.latency_ms > 150:
        size_order = ["xs", "sm", "md", "lg"]
        idx = size_order.index(current_size) if current_size in size_order else 1
        if idx < len(size_order) - 1:
            return size_order[idx + 1]

    # If CPU is very low and we're paying for big pods, scale down
    if obs.cpu_utilization < 0.2 and obs.queue_length == 0:
        size_order = ["xs", "sm", "md", "lg"]
        idx = size_order.index(current_size) if current_size in size_order else 1
        if idx > 0:
            return size_order[idx - 1]

    return None


def choose_heuristic(
    policy_id: str, obs: CloudScaleObservation
) -> CloudScaleAction:
    node_delta = _node_decision(obs)
    pod_size = _vpa_decision(obs)

    if policy_id == "noop":
        return CloudScaleAction(scale_delta=0, node_delta=node_delta, pod_size=pod_size)

    if policy_id == "threshold_cpu":
        scale_delta = 0
        if obs.cpu_utilization > 0.80:
            if obs.pending_scale_ups == 0:
                scale_delta = 1
        elif obs.cpu_utilization < 0.30:
            if obs.active_pods > 1 and obs.pending_scale_downs == 0:
                scale_delta = -1
        return CloudScaleAction(scale_delta=scale_delta, node_delta=node_delta, pod_size=pod_size)

    if policy_id == "latency_queue":
        scale_delta = 0
        if obs.latency_ms > 300 or obs.queue_length > 2000:
            if obs.pending_scale_ups == 0:
                scale_delta = 2
        elif obs.latency_ms > 150 or obs.queue_length > 500 or obs.cpu_utilization > 0.85:
            if obs.pending_scale_ups == 0:
                scale_delta = 1
        elif (
            obs.cpu_utilization < 0.20
            and obs.queue_length == 0
            and obs.active_pods > 1
            and obs.pending_scale_downs == 0
        ):
            scale_delta = -1
        return CloudScaleAction(scale_delta=scale_delta, node_delta=node_delta, pod_size=pod_size)

    # hybrid - considers traffic trend if available
    trend = "stable"
    if obs.traffic_snapshot is not None:
        trend = obs.traffic_snapshot.trend

    # Check for infra events that need attention
    has_spike = False
    if obs.recent_events:
        spike_types = {"flash_crowd", "ddos_attack", "thundering_herd", "gradual_ramp"}
        has_spike = any(e.event_type in spike_types for e in obs.recent_events)

    # Prevent oscillation
    if obs.pending_scale_ups > 0 and obs.pending_scale_downs > 0:
        return CloudScaleAction(scale_delta=0, node_delta=node_delta, pod_size=pod_size)

    scale_delta = 0

    if obs.latency_ms > 300 or obs.queue_length > 2000:
        scale_delta = 2
    elif has_spike and obs.pending_scale_ups == 0:
        # proactively scale up during detected traffic events
        scale_delta = 2
    elif trend == "rising" and obs.cpu_utilization > 0.60 and obs.pending_scale_ups == 0:
        scale_delta = 1
    elif obs.latency_ms > 150 or obs.queue_length > 500:
        if obs.pending_scale_ups == 0:
            scale_delta = 1
    elif obs.cpu_utilization > 0.85 and obs.pending_scale_ups == 0:
        scale_delta = 1
    elif (
        trend == "falling"
        and obs.cpu_utilization < 0.30
        and obs.queue_length == 0
        and obs.active_pods > 1
        and obs.pending_scale_downs == 0
    ):
        scale_delta = -1
    elif (
        obs.cpu_utilization < 0.15
        and obs.queue_length == 0
        and obs.active_pods > 1
        and obs.pending_scale_downs == 0
    ):
        scale_delta = -1

    return CloudScaleAction(scale_delta=scale_delta, node_delta=node_delta, pod_size=pod_size)


def choose_meta_action(action_id: int, obs: CloudScaleObservation) -> ActionChoice:
    action_id = max(0, min(action_id, len(META_ACTIONS) - 1))
    label = META_ACTIONS[action_id]

    if label in {"threshold_cpu", "latency_queue", "hybrid", "noop"}:
        return ActionChoice(action=choose_heuristic(label, obs), label=label)

    if label == "emergency_scale":
        if obs.latency_ms > 300 or obs.queue_length > 2000:
            if obs.pending_scale_ups < 2:
                return ActionChoice(
                    action=CloudScaleAction(scale_delta=2, node_delta=1), label="emergency_scale_up"
                )
        return ActionChoice(action=CloudScaleAction(scale_delta=0), label="noop_fallback")

    return ActionChoice(action=CloudScaleAction(scale_delta=0), label="default_noop")


class HybridPolicy:
    """Class wrapper for run_baseline script compatibility."""
    def __call__(self, obs: CloudScaleObservation) -> CloudScaleAction:
        return choose_heuristic("hybrid", obs)

class ThresholdCpuPolicy:
    """Class wrapper for run_baseline script compatibility."""
    def __call__(self, obs: CloudScaleObservation) -> CloudScaleAction:
        return choose_heuristic("threshold_cpu", obs)

class LatencyQueuePolicy:
    """Class wrapper for run_baseline script compatibility."""
    def __call__(self, obs: CloudScaleObservation) -> CloudScaleAction:
        return choose_heuristic("latency_queue", obs)

class NoopPolicy:
    """Class wrapper for run_baseline script compatibility."""
    def __call__(self, obs: CloudScaleObservation) -> CloudScaleAction:
        return choose_heuristic("noop", obs)

POLICIES = {
    "threshold_cpu": ThresholdCpuPolicy,
    "latency_queue": LatencyQueuePolicy,
    "hybrid": HybridPolicy,
    "noop": NoopPolicy,
}

def get_policy(name: str = "hybrid"):
    """Return an instantiated policy by name."""
    cls = POLICIES.get(name, HybridPolicy)
    return cls()
