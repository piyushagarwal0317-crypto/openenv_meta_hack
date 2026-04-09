from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Optional
from uuid import uuid4
import math
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        CloudScaleAction,
        CloudScaleObservation,
        InfraEvent,
        NodeInfo,
        PendingScaleEvent,
        PodResourceInfo,
        TrafficSnapshot,
        POD_SIZE_SPECS,
        NODE_TYPE_SPECS,
    )
except (ImportError, ModuleNotFoundError):
    from models import (
        CloudScaleAction,
        CloudScaleObservation,
        InfraEvent,
        NodeInfo,
        PendingScaleEvent,
        PodResourceInfo,
        TrafficSnapshot,
        POD_SIZE_SPECS,
        NODE_TYPE_SPECS,
    )


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    task_id: str
    horizon: int
    initial_pods: int
    min_pods: int
    max_pods: int
    base_seed: int
    # --- traffic ---
    base_request_rate: float          # mean requests/sec at baseline
    traffic_amplitude: float          # sinusoidal amplitude
    traffic_noise_std: float          # Gaussian noise std-dev
    burst_probability: float          # chance of a random traffic spike per step
    burst_multiplier: float           # how much a burst inflates traffic
    morning_peak_multiplier: float    # multiplier during morning peak
    afternoon_peak_multiplier: float  # multiplier during afternoon peak
    # --- enhanced spike config ---
    flash_crowd_probability: float    # chance of flash crowd event per step
    flash_crowd_multiplier: float     # traffic multiplier during flash crowd
    flash_crowd_duration: int         # how many steps a flash crowd lasts
    gradual_ramp_probability: float   # chance of a gradual ramp event
    gradual_ramp_peak: float          # peak multiplier for gradual ramp
    gradual_ramp_duration: int        # total ramp up + plateau + ramp down steps
    ddos_probability: float           # chance of DDoS-like burst cluster
    ddos_multiplier: float            # traffic multiplier during DDoS
    ddos_burst_count: int             # number of micro-bursts in a DDoS event
    thundering_herd_probability: float  # chance of thundering herd
    thundering_herd_multiplier: float   # instantaneous traffic multiplier
    # --- capacity ---
    pod_capacity: float               # default max req/s each pod can serve (sm tier)
    max_queue: int                    # max queue depth before drops
    # --- node config ---
    initial_nodes: int                # starting worker nodes
    min_nodes: int                    # minimum worker nodes
    max_nodes: int                    # maximum worker nodes
    node_type: str                    # default node instance type
    node_failure_probability: float   # chance of a random node failure per step
    node_recovery_steps: int          # steps to recover a failed node
    # --- provisioning ---
    provision_delay_mean: int         # mean steps for a pod to come online
    provision_delay_std: float        # std-dev for provisioning delay
    deprovision_delay: int            # steps for pod termination
    cold_start_latency_ms: float     # extra latency for a newly active pod
    cold_start_steps: int             # how many steps the cold-start lasts
    vpa_restart_steps: int            # steps for VPA rolling restart
    # --- latency model ---
    base_latency_ms: float            # baseline per-request latency
    queue_latency_factor: float       # ms added per queued request (normalised)
    cpu_latency_factor: float         # ms added at high CPU utilisation
    memory_latency_factor: float      # ms added at high memory pressure
    # --- SLA ---
    sla_latency_ms: int               # target p50 latency
    # --- reward weights ---
    w_sla_bonus: float
    w_sla_penalty: float
    w_cost: float
    w_queue: float
    w_oscillation: float
    w_invalid: float
    w_idle_pod: float
    w_node_cost: float                # per-node cost penalty
    w_vpa_churn: float                # penalty for VPA restarts
    w_node_failure: float             # penalty when a node fails


SCENARIOS: dict[str, ScenarioConfig] = {
    "easy": ScenarioConfig(
        task_id="easy",
        horizon=180,
        initial_pods=3,
        min_pods=1,
        max_pods=15,
        base_seed=101,
        base_request_rate=1000.0,
        traffic_amplitude=200.0,
        traffic_noise_std=50.0,
        burst_probability=0.02,
        burst_multiplier=1.6,
        morning_peak_multiplier=1.20,
        afternoon_peak_multiplier=1.15,
        # enhanced spikes (mild for easy)
        flash_crowd_probability=0.005,
        flash_crowd_multiplier=2.5,
        flash_crowd_duration=8,
        gradual_ramp_probability=0.008,
        gradual_ramp_peak=1.8,
        gradual_ramp_duration=30,
        ddos_probability=0.0,        # no DDoS on easy
        ddos_multiplier=1.0,
        ddos_burst_count=0,
        thundering_herd_probability=0.0,
        thundering_herd_multiplier=1.0,
        # capacity
        pod_capacity=500.0,
        max_queue=3000,
        # nodes
        initial_nodes=2,
        min_nodes=1,
        max_nodes=5,
        node_type="medium",
        node_failure_probability=0.0,  # no failures on easy
        node_recovery_steps=8,
        # provisioning
        provision_delay_mean=3,
        provision_delay_std=0.5,
        deprovision_delay=2,
        cold_start_latency_ms=30.0,
        cold_start_steps=3,
        vpa_restart_steps=3,
        # latency
        base_latency_ms=40.0,
        queue_latency_factor=150.0,
        cpu_latency_factor=80.0,
        memory_latency_factor=40.0,
        # SLA
        sla_latency_ms=250,
        # reward
        w_sla_bonus=2.0,
        w_sla_penalty=0.03,
        w_cost=0.08,
        w_queue=0.005,
        w_oscillation=0.4,
        w_invalid=5.0,
        w_idle_pod=0.003,
        w_node_cost=0.15,
        w_vpa_churn=1.0,
        w_node_failure=2.0,
    ),
    "medium": ScenarioConfig(
        task_id="medium",
        horizon=240,
        initial_pods=4,
        min_pods=1,
        max_pods=20,
        base_seed=202,
        base_request_rate=1500.0,
        traffic_amplitude=500.0,
        traffic_noise_std=150.0,
        burst_probability=0.05,
        burst_multiplier=1.8,
        morning_peak_multiplier=1.40,
        afternoon_peak_multiplier=1.50,
        # enhanced spikes
        flash_crowd_probability=0.015,
        flash_crowd_multiplier=3.0,
        flash_crowd_duration=10,
        gradual_ramp_probability=0.012,
        gradual_ramp_peak=2.2,
        gradual_ramp_duration=40,
        ddos_probability=0.005,
        ddos_multiplier=5.0,
        ddos_burst_count=3,
        thundering_herd_probability=0.005,
        thundering_herd_multiplier=4.0,
        # capacity
        pod_capacity=500.0,
        max_queue=4000,
        # nodes
        initial_nodes=2,
        min_nodes=1,
        max_nodes=8,
        node_type="medium",
        node_failure_probability=0.005,
        node_recovery_steps=10,
        # provisioning
        provision_delay_mean=5,
        provision_delay_std=1.0,
        deprovision_delay=3,
        cold_start_latency_ms=45.0,
        cold_start_steps=4,
        vpa_restart_steps=4,
        # latency
        base_latency_ms=45.0,
        queue_latency_factor=180.0,
        cpu_latency_factor=100.0,
        memory_latency_factor=60.0,
        # SLA
        sla_latency_ms=180,
        # reward
        w_sla_bonus=2.0,
        w_sla_penalty=0.04,
        w_cost=0.10,
        w_queue=0.008,
        w_oscillation=0.5,
        w_invalid=5.0,
        w_idle_pod=0.003,
        w_node_cost=0.20,
        w_vpa_churn=1.5,
        w_node_failure=3.0,
    ),
    "hard": ScenarioConfig(
        task_id="hard",
        horizon=300,
        initial_pods=4,
        min_pods=1,
        max_pods=25,
        base_seed=303,
        base_request_rate=2000.0,
        traffic_amplitude=1000.0,
        traffic_noise_std=300.0,
        burst_probability=0.10,
        burst_multiplier=2.0,
        morning_peak_multiplier=1.65,
        afternoon_peak_multiplier=1.90,
        # enhanced spikes (aggressive)
        flash_crowd_probability=0.025,
        flash_crowd_multiplier=4.0,
        flash_crowd_duration=12,
        gradual_ramp_probability=0.015,
        gradual_ramp_peak=2.8,
        gradual_ramp_duration=50,
        ddos_probability=0.012,
        ddos_multiplier=8.0,
        ddos_burst_count=5,
        thundering_herd_probability=0.01,
        thundering_herd_multiplier=6.0,
        # capacity
        pod_capacity=500.0,
        max_queue=5000,
        # nodes
        initial_nodes=3,
        min_nodes=1,
        max_nodes=10,
        node_type="medium",
        node_failure_probability=0.012,
        node_recovery_steps=12,
        # provisioning
        provision_delay_mean=8,
        provision_delay_std=2.0,
        deprovision_delay=4,
        cold_start_latency_ms=60.0,
        cold_start_steps=5,
        vpa_restart_steps=5,
        # latency
        base_latency_ms=50.0,
        queue_latency_factor=200.0,
        cpu_latency_factor=120.0,
        memory_latency_factor=80.0,
        # SLA
        sla_latency_ms=120,
        # reward
        w_sla_bonus=2.0,
        w_sla_penalty=0.05,
        w_cost=0.12,
        w_queue=0.01,
        w_oscillation=0.6,
        w_invalid=5.0,
        w_idle_pod=0.003,
        w_node_cost=0.25,
        w_vpa_churn=2.0,
        w_node_failure=4.0,
    ),
}


# ---------------------------------------------------------------------------
# Internal simulation state
# ---------------------------------------------------------------------------

@dataclass
class WorkerNode:
    """A Kubernetes worker node in the cluster."""
    node_id: str
    node_type: str = "medium"
    status: str = "active"            # active | provisioning | draining | failed
    created_step: int = 0
    active_step: int = 0
    cpu_capacity: float = 4.0         # vCPU
    memory_capacity_gb: float = 8.0   # GB
    pods_hosted: list[str] = field(default_factory=list)
    failure_recovery_remaining: int = 0  # steps until recovery from failure


@dataclass
class Pod:
    pod_id: str
    status: str = "active"           # active | provisioning | terminating | restarting
    created_step: int = 0
    active_step: int = 0             # step when pod became active
    cold_start_remaining: int = 0    # steps of cold-start penalty left
    requests_served: int = 0
    idle_steps: int = 0
    # VPA resource dimensions
    size_tier: str = "sm"
    cpu_request: float = 0.5         # vCPU
    memory_request_mb: float = 512   # MB
    capacity: float = 500.0          # req/s
    cost_multiplier: float = 1.0
    # scheduling
    node_id: str = ""                # which node this pod is on
    restart_remaining: int = 0       # VPA rolling restart countdown


@dataclass
class ScaleEvent:
    """In-flight scaling operation tracked internally."""
    direction: str                   # "up" | "down"
    pods_count: int
    remaining_steps: int
    pod_ids: list[str] = field(default_factory=list)


@dataclass
class NodeScaleEvent:
    """In-flight node scaling operation."""
    direction: str                   # "up" | "down"
    remaining_steps: int
    node_id: str = ""


@dataclass
class SpikeEvent:
    """Active traffic spike event."""
    spike_type: str                  # flash_crowd | gradual_ramp | ddos | thundering_herd
    remaining_steps: int
    total_steps: int
    multiplier: float
    burst_remaining: int = 0         # for DDoS micro-bursts


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CloudScaleEnvironment(Environment):
    """Cloud autoscaling simulation with Kubernetes-realistic features.

    The agent acts as an SRE deciding how many pods to add or remove (HPA),
    whether to add/remove worker nodes (Cluster Autoscaler), and what
    resource tier pods should run at (VPA).

    Traffic is stochastic with daily patterns, random bursts, flash crowds,
    DDoS-like attacks, gradual ramps, and thundering herd events.
    Scaling decisions are subject to provisioning delays.
    Nodes can fail randomly, requiring recovery time.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "easy"):
        self.task = task if task in SCENARIOS else "easy"
        self.config = SCENARIOS[self.task]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random(self.config.base_seed)

        # simulation state (will be initialised in _setup_episode)
        self._step_idx: int = 0
        self._pods: dict[str, Pod] = {}
        self._nodes: dict[str, WorkerNode] = {}
        self._pending_events: list[ScaleEvent] = []
        self._pending_node_events: list[NodeScaleEvent] = []
        self._active_spikes: list[SpikeEvent] = []
        self._queue_length: int = 0
        self._next_pod_id: int = 1
        self._next_node_id: int = 1

        # VPA state
        self._current_pod_size: str = "sm"
        self._vpa_restart_in_progress: bool = False
        self._vpa_restart_remaining: int = 0

        # counters
        self._cumulative_reward: float = 0.0
        self._step_reward: float = 0.0
        self._total_requests_processed: int = 0
        self._total_requests_dropped: int = 0
        self._total_sla_violations: int = 0
        self._latency_history: list[float] = []
        self._request_rate_history: list[float] = []
        self._action_history: list[int] = []

        # current-step metrics (set in _advance)
        self._current_cpu: float = 0.0
        self._current_memory: float = 0.0
        self._current_latency: float = 0.0
        self._current_request_rate: float = 0.0
        self._step_processed: int = 0
        self._step_dropped: int = 0

        # infrastructure events log
        self._infra_events: list[InfraEvent] = []

        self._setup_episode(seed=self.config.base_seed)

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task: Optional[str] = None) -> CloudScaleObservation:
        if task and task in SCENARIOS:
            self.task = task
            self.config = SCENARIOS[task]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._setup_episode(seed=self.config.base_seed)
        return self._build_observation(done=False)

    def step(self, action: CloudScaleAction) -> CloudScaleObservation:  # type: ignore[override]
        self._state.step_count += 1
        self._step_reward = 0.0

        self._apply_action(action)
        self._advance_one_step()

        done = self._step_idx >= self.config.horizon
        return self._build_observation(done=done)

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Episode bootstrap
    # ------------------------------------------------------------------

    def _setup_episode(self, seed: int):
        self._rng = random.Random(seed)
        self._step_idx = 0
        self._pods = {}
        self._nodes = {}
        self._pending_events = []
        self._pending_node_events = []
        self._active_spikes = []
        self._queue_length = 0
        self._next_pod_id = 1
        self._next_node_id = 1

        self._current_pod_size = "sm"
        self._vpa_restart_in_progress = False
        self._vpa_restart_remaining = 0

        self._cumulative_reward = 0.0
        self._step_reward = 0.0
        self._total_requests_processed = 0
        self._total_requests_dropped = 0
        self._total_sla_violations = 0
        self._latency_history = []
        self._request_rate_history = []
        self._action_history = []

        self._current_cpu = 0.0
        self._current_memory = 0.0
        self._current_latency = 0.0
        self._current_request_rate = 0.0
        self._step_processed = 0
        self._step_dropped = 0

        self._infra_events = []

        # spawn initial worker nodes
        node_spec = NODE_TYPE_SPECS[self.config.node_type]
        for _ in range(self.config.initial_nodes):
            nid = self._new_node_id()
            self._nodes[nid] = WorkerNode(
                node_id=nid,
                node_type=self.config.node_type,
                status="active",
                created_step=0,
                active_step=0,
                cpu_capacity=node_spec["cpu"],
                memory_capacity_gb=node_spec["memory_gb"],
            )

        # spawn initial pods (already active, no cold-start)
        pod_spec = POD_SIZE_SPECS[self._current_pod_size]
        for _ in range(self.config.initial_pods):
            pid = self._new_pod_id()
            node_id = self._find_node_for_pod(pod_spec["cpu_request"], pod_spec["memory_request_mb"])
            pod = Pod(
                pod_id=pid,
                status="active",
                created_step=0,
                active_step=0,
                cold_start_remaining=0,
                size_tier=self._current_pod_size,
                cpu_request=pod_spec["cpu_request"],
                memory_request_mb=pod_spec["memory_request_mb"],
                capacity=pod_spec["capacity"],
                cost_multiplier=pod_spec["cost_mult"],
                node_id=node_id,
            )
            self._pods[pid] = pod
            if node_id:
                self._nodes[node_id].pods_hosted.append(pid)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, done: bool) -> CloudScaleObservation:
        active_pods = self._active_pod_count()

        # pending events → public view
        pending_views = [
            PendingScaleEvent(
                direction=evt.direction,
                pods=evt.pods_count,
                remaining_steps=evt.remaining_steps,
            )
            for evt in self._pending_events
        ]
        pending_ups = sum(
            evt.pods_count for evt in self._pending_events if evt.direction == "up"
        )
        pending_downs = sum(
            evt.pods_count for evt in self._pending_events if evt.direction == "down"
        )

        # traffic snapshot
        recent = self._request_rate_history[-10:] if self._request_rate_history else [0.0]
        avg_rate = mean(recent)
        peak_rate = max(recent)
        trend = self._compute_trend(recent)
        traffic_snapshot = TrafficSnapshot(
            recent_avg_request_rate=round(avg_rate, 2),
            recent_peak_request_rate=round(peak_rate, 2),
            trend=trend,
        )

        avg_latency = mean(self._latency_history) if self._latency_history else 0.0

        self._cumulative_reward += self._step_reward

        # node info
        active_nodes = [n for n in self._nodes.values() if n.status == "active"]
        pending_node_ups = sum(1 for e in self._pending_node_events if e.direction == "up")
        pending_node_downs = sum(1 for e in self._pending_node_events if e.direction == "down")
        total_cpu_cap = sum(n.cpu_capacity for n in active_nodes)
        total_mem_cap = sum(n.memory_capacity_gb for n in active_nodes)

        # compute resource usage from pods
        active_pod_list = [p for p in self._pods.values() if p.status == "active"]
        cpu_used = sum(p.cpu_request for p in active_pod_list)
        mem_used = sum(p.memory_request_mb / 1024.0 for p in active_pod_list)

        node_info = NodeInfo(
            total_nodes=len(self._nodes),
            active_nodes=len(active_nodes),
            pending_node_ups=pending_node_ups,
            pending_node_downs=pending_node_downs,
            node_cpu_capacity=round(total_cpu_cap, 2),
            node_memory_capacity_gb=round(total_mem_cap, 2),
            node_cpu_used=round(cpu_used, 2),
            node_memory_used_gb=round(mem_used, 2),
            node_type=self.config.node_type,
        )

        # pod resource info
        pod_spec = POD_SIZE_SPECS[self._current_pod_size]
        pod_resource_info = PodResourceInfo(
            pod_size=self._current_pod_size,
            pod_cpu_request=pod_spec["cpu_request"],
            pod_memory_request_mb=pod_spec["memory_request_mb"],
            pod_capacity=pod_spec["capacity"],
            vpa_restart_in_progress=self._vpa_restart_in_progress,
            vpa_restart_remaining_steps=self._vpa_restart_remaining,
        )

        # recent events (last 5)
        recent_events = self._infra_events[-5:]

        return CloudScaleObservation(
            task_id=self.task,
            time_step=self._step_idx,
            horizon=self.config.horizon,
            cpu_utilization=round(self._current_cpu, 4),
            memory_utilization=round(self._current_memory, 4),
            latency_ms=round(self._current_latency, 2),
            request_rate=round(self._current_request_rate, 2),
            queue_length=self._queue_length,
            active_pods=active_pods,
            pending_scale_ups=pending_ups,
            pending_scale_downs=pending_downs,
            pending_events=pending_views,
            traffic_snapshot=traffic_snapshot,
            node_info=node_info,
            pod_resource_info=pod_resource_info,
            recent_events=recent_events,
            total_requests_processed=self._total_requests_processed,
            total_requests_dropped=self._total_requests_dropped,
            total_sla_violations=self._total_sla_violations,
            average_latency_ms=round(avg_latency, 2),
            reward=round(self._step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=done,
            metadata={
                "version": "3.0",
                "seed": self.config.base_seed,
                "step_processed": self._step_processed,
                "step_dropped": self._step_dropped,
                "traffic_multiplier": round(
                    self._time_of_day_multiplier(self._step_idx), 3
                ),
                "active_spike_types": [s.spike_type for s in self._active_spikes],
                "pod_size": self._current_pod_size,
                "node_type": self.config.node_type,
            },
        )

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------

    def _apply_action(self, action: CloudScaleAction):
        # --- HPA: pod scale_delta ---
        delta = action.scale_delta
        self._action_history.append(delta)

        active = self._active_pod_count()
        pending_up = sum(
            e.pods_count for e in self._pending_events if e.direction == "up"
        )
        pending_down = sum(
            e.pods_count for e in self._pending_events if e.direction == "down"
        )
        projected = active + pending_up - pending_down + delta

        # clamp to [min_pods, max_pods] — penalise the part that was clamped
        if projected < self.config.min_pods:
            wasted = self.config.min_pods - projected
            delta += wasted
            self._step_reward -= self.config.w_invalid * (wasted > 0)
        elif projected > self.config.max_pods:
            wasted = projected - self.config.max_pods
            delta -= wasted
            self._step_reward -= self.config.w_invalid * (wasted > 0)

        if delta != 0:
            if delta > 0:
                # Check if there's node capacity to schedule new pods
                pod_spec = POD_SIZE_SPECS[self._current_pod_size]
                schedulable = self._count_schedulable_pods(
                    pod_spec["cpu_request"], pod_spec["memory_request_mb"]
                )
                actual_delta = min(delta, schedulable)
                unschedulable = delta - actual_delta

                if actual_delta > 0:
                    delay = max(
                        1,
                        int(round(self._rng.gauss(
                            self.config.provision_delay_mean,
                            self.config.provision_delay_std,
                        ))),
                    )
                    pod_ids: list[str] = []
                    for _ in range(actual_delta):
                        pid = self._new_pod_id()
                        node_id = self._find_node_for_pod(
                            pod_spec["cpu_request"], pod_spec["memory_request_mb"]
                        )
                        self._pods[pid] = Pod(
                            pod_id=pid,
                            status="provisioning",
                            created_step=self._step_idx,
                            size_tier=self._current_pod_size,
                            cpu_request=pod_spec["cpu_request"],
                            memory_request_mb=pod_spec["memory_request_mb"],
                            capacity=pod_spec["capacity"],
                            cost_multiplier=pod_spec["cost_mult"],
                            node_id=node_id,
                        )
                        if node_id:
                            self._nodes[node_id].pods_hosted.append(pid)
                        pod_ids.append(pid)
                    self._pending_events.append(
                        ScaleEvent(
                            direction="up",
                            pods_count=actual_delta,
                            remaining_steps=delay,
                            pod_ids=pod_ids,
                        )
                    )

                if unschedulable > 0:
                    self._step_reward -= self.config.w_invalid * 0.5
                    self._infra_events.append(InfraEvent(
                        step=self._step_idx,
                        event_type="pod_unschedulable",
                        details=f"{unschedulable} pod(s) cannot be scheduled — insufficient node capacity. Consider adding nodes.",
                    ))
            else:
                # schedule scale-down (pick pods with fewest requests served)
                abs_delta = abs(delta)
                active_pods_list = sorted(
                    [p for p in self._pods.values() if p.status == "active"],
                    key=lambda p: p.requests_served,
                )
                to_remove = active_pods_list[:abs_delta]
                pod_ids_down = [p.pod_id for p in to_remove]
                for p in to_remove:
                    p.status = "terminating"
                self._pending_events.append(
                    ScaleEvent(
                        direction="down",
                        pods_count=abs_delta,
                        remaining_steps=self.config.deprovision_delay,
                        pod_ids=pod_ids_down,
                    )
                )

        # --- Cluster Autoscaler: node_delta ---
        node_delta = action.node_delta
        if node_delta != 0:
            active_node_count = sum(1 for n in self._nodes.values() if n.status == "active")
            pending_node_up = sum(1 for e in self._pending_node_events if e.direction == "up")
            pending_node_down = sum(1 for e in self._pending_node_events if e.direction == "down")
            projected_nodes = active_node_count + pending_node_up - pending_node_down + node_delta

            if projected_nodes < self.config.min_nodes:
                self._step_reward -= self.config.w_invalid
                node_delta = 0
            elif projected_nodes > self.config.max_nodes:
                self._step_reward -= self.config.w_invalid
                node_delta = 0

            if node_delta > 0:
                node_spec = NODE_TYPE_SPECS[self.config.node_type]
                nid = self._new_node_id()
                provision_delay = max(1, int(round(self._rng.gauss(
                    node_spec["provision_base"],
                    node_spec["provision_base"] * 0.2,
                ))))
                self._nodes[nid] = WorkerNode(
                    node_id=nid,
                    node_type=self.config.node_type,
                    status="provisioning",
                    created_step=self._step_idx,
                    cpu_capacity=node_spec["cpu"],
                    memory_capacity_gb=node_spec["memory_gb"],
                )
                self._pending_node_events.append(
                    NodeScaleEvent(direction="up", remaining_steps=provision_delay, node_id=nid)
                )
                self._infra_events.append(InfraEvent(
                    step=self._step_idx,
                    event_type="node_provisioning",
                    details=f"Node {nid} ({self.config.node_type}) provisioning — ETA {provision_delay} steps",
                ))
            elif node_delta < 0:
                # drain the node with fewest pods
                active_nodes = sorted(
                    [n for n in self._nodes.values() if n.status == "active"],
                    key=lambda n: len(n.pods_hosted),
                )
                if active_nodes:
                    node_to_drain = active_nodes[0]
                    node_to_drain.status = "draining"
                    # evict pods from this node
                    for pid in list(node_to_drain.pods_hosted):
                        pod = self._pods.get(pid)
                        if pod and pod.status == "active":
                            # try to reschedule to another node
                            new_node = self._find_node_for_pod(
                                pod.cpu_request, pod.memory_request_mb,
                                exclude_node=node_to_drain.node_id,
                            )
                            if new_node:
                                pod.node_id = new_node
                                self._nodes[new_node].pods_hosted.append(pid)
                            else:
                                # pod becomes unschedulable — terminate it
                                pod.status = "terminating"
                                self._infra_events.append(InfraEvent(
                                    step=self._step_idx,
                                    event_type="pod_evicted",
                                    details=f"Pod {pid} evicted from draining node {node_to_drain.node_id} — no node available",
                                ))
                    node_to_drain.pods_hosted.clear()
                    self._pending_node_events.append(
                        NodeScaleEvent(
                            direction="down",
                            remaining_steps=self.config.deprovision_delay + 2,
                            node_id=node_to_drain.node_id,
                        )
                    )
                    self._infra_events.append(InfraEvent(
                        step=self._step_idx,
                        event_type="node_draining",
                        details=f"Node {node_to_drain.node_id} draining and will be terminated",
                    ))

        # --- VPA: pod_size ---
        if action.pod_size is not None and action.pod_size != self._current_pod_size:
            if not self._vpa_restart_in_progress:
                old_size = self._current_pod_size
                new_size = action.pod_size
                self._current_pod_size = new_size
                self._vpa_restart_in_progress = True
                self._vpa_restart_remaining = self.config.vpa_restart_steps

                new_spec = POD_SIZE_SPECS[new_size]
                # mark active pods for rolling restart
                for pod in self._pods.values():
                    if pod.status == "active":
                        pod.status = "restarting"
                        pod.restart_remaining = self.config.vpa_restart_steps
                        pod.size_tier = new_size
                        pod.cpu_request = new_spec["cpu_request"]
                        pod.memory_request_mb = new_spec["memory_request_mb"]
                        pod.capacity = new_spec["capacity"]
                        pod.cost_multiplier = new_spec["cost_mult"]

                self._step_reward -= self.config.w_vpa_churn
                self._infra_events.append(InfraEvent(
                    step=self._step_idx,
                    event_type="vpa_restart",
                    details=f"VPA rolling restart: {old_size} → {new_size} ({self.config.vpa_restart_steps} steps)",
                ))

    # ------------------------------------------------------------------
    # Simulation tick
    # ------------------------------------------------------------------

    def _advance_one_step(self):
        # 1. Resolve pending pod scaling events
        still_pending: list[ScaleEvent] = []
        for evt in self._pending_events:
            evt.remaining_steps -= 1
            if evt.remaining_steps <= 0:
                if evt.direction == "up":
                    for pid in evt.pod_ids:
                        pod = self._pods.get(pid)
                        if pod and pod.status == "provisioning":
                            pod.status = "active"
                            pod.active_step = self._step_idx
                            pod.cold_start_remaining = self.config.cold_start_steps
                else:
                    for pid in evt.pod_ids:
                        if pid in self._pods:
                            pod = self._pods[pid]
                            # remove from node
                            if pod.node_id and pod.node_id in self._nodes:
                                node = self._nodes[pod.node_id]
                                if pid in node.pods_hosted:
                                    node.pods_hosted.remove(pid)
                            del self._pods[pid]
            else:
                still_pending.append(evt)
        self._pending_events = still_pending

        # 2. Resolve pending node scaling events
        still_pending_nodes: list[NodeScaleEvent] = []
        for evt in self._pending_node_events:
            evt.remaining_steps -= 1
            if evt.remaining_steps <= 0:
                if evt.direction == "up":
                    node = self._nodes.get(evt.node_id)
                    if node and node.status == "provisioning":
                        node.status = "active"
                        node.active_step = self._step_idx
                        self._infra_events.append(InfraEvent(
                            step=self._step_idx,
                            event_type="node_ready",
                            details=f"Node {evt.node_id} is now active and ready to host pods",
                        ))
                else:
                    if evt.node_id in self._nodes:
                        del self._nodes[evt.node_id]
            else:
                still_pending_nodes.append(evt)
        self._pending_node_events = still_pending_nodes

        # 3. Handle VPA rolling restart countdown
        if self._vpa_restart_in_progress:
            self._vpa_restart_remaining -= 1
            for pod in self._pods.values():
                if pod.status == "restarting":
                    pod.restart_remaining -= 1
                    if pod.restart_remaining <= 0:
                        pod.status = "active"
                        pod.cold_start_remaining = max(1, self.config.cold_start_steps // 2)
            if self._vpa_restart_remaining <= 0:
                self._vpa_restart_in_progress = False

        # 4. Tick cold-start counters
        for pod in self._pods.values():
            if pod.cold_start_remaining > 0:
                pod.cold_start_remaining -= 1

        # 5. Random node failures
        if self.config.node_failure_probability > 0:
            active_nodes = [n for n in self._nodes.values() if n.status == "active"]
            for node in active_nodes:
                if self._rng.random() < self.config.node_failure_probability:
                    node.status = "failed"
                    node.failure_recovery_remaining = self.config.node_recovery_steps
                    # pods on this node lose their host — try rescheduling
                    for pid in list(node.pods_hosted):
                        pod = self._pods.get(pid)
                        if pod and pod.status == "active":
                            new_node = self._find_node_for_pod(
                                pod.cpu_request, pod.memory_request_mb,
                                exclude_node=node.node_id,
                            )
                            if new_node:
                                pod.node_id = new_node
                                self._nodes[new_node].pods_hosted.append(pid)
                                pod.cold_start_remaining = 2  # brief disruption
                            else:
                                pod.status = "terminating"
                                self._infra_events.append(InfraEvent(
                                    step=self._step_idx,
                                    event_type="pod_lost",
                                    details=f"Pod {pid} lost — node {node.node_id} failed, no reschedule target",
                                ))
                    node.pods_hosted.clear()
                    self._step_reward -= self.config.w_node_failure
                    self._infra_events.append(InfraEvent(
                        step=self._step_idx,
                        event_type="node_failure",
                        details=f"Node {node.node_id} failed! Recovery in {self.config.node_recovery_steps} steps",
                    ))

        # 6. Recover failed nodes
        for node in self._nodes.values():
            if node.status == "failed":
                node.failure_recovery_remaining -= 1
                if node.failure_recovery_remaining <= 0:
                    node.status = "active"
                    self._infra_events.append(InfraEvent(
                        step=self._step_idx,
                        event_type="node_recovered",
                        details=f"Node {node.node_id} recovered and is active again",
                    ))

        # 7. Clean up terminated pods (from eviction/failure)
        terminated = [pid for pid, p in self._pods.items() if p.status == "terminating"
                      and pid not in {pid for evt in self._pending_events for pid in evt.pod_ids}]
        for pid in terminated:
            pod = self._pods[pid]
            if pod.node_id and pod.node_id in self._nodes:
                node = self._nodes[pod.node_id]
                if pid in node.pods_hosted:
                    node.pods_hosted.remove(pid)
            del self._pods[pid]

        # 8. Generate traffic (with enhanced spikes)
        request_rate = self._generate_traffic(self._step_idx)
        self._current_request_rate = request_rate
        self._request_rate_history.append(request_rate)

        # 9. Process requests
        active_pods = [p for p in self._pods.values() if p.status == "active"]
        restarting_pods = [p for p in self._pods.values() if p.status == "restarting"]
        num_active = len(active_pods)
        num_restarting = len(restarting_pods)

        if num_active == 0 and num_restarting == 0:
            # no pods — everything queues/drops
            self._queue_length += int(request_rate)
            total_capacity = 0
        else:
            # pods under cold-start have reduced capacity
            total_capacity = 0.0
            for pod in active_pods:
                if pod.cold_start_remaining > 0:
                    total_capacity += pod.capacity * 0.5
                else:
                    total_capacity += pod.capacity

            # restarting pods have 30% capacity (graceful degradation)
            for pod in restarting_pods:
                total_capacity += pod.capacity * 0.3

            available = request_rate + self._queue_length
            processed = min(available, total_capacity)
            remaining = available - processed

            self._queue_length = int(remaining)
            self._step_processed = int(processed)
            self._total_requests_processed += int(processed)

            # distribute processed requests across pods for accounting
            all_serving = active_pods + restarting_pods
            per_pod = int(processed) // max(1, len(all_serving))
            for pod in all_serving:
                pod.requests_served += per_pod

        # 10. Drop excess queue
        if self._queue_length > self.config.max_queue:
            dropped = self._queue_length - self.config.max_queue
            self._queue_length = self.config.max_queue
            self._step_dropped = dropped
            self._total_requests_dropped += dropped
        else:
            self._step_dropped = 0

        # 11. Calculate CPU utilisation
        if total_capacity > 0:
            cpu_util = min(1.0, (request_rate + max(0, self._queue_length - request_rate * 0.1)) / total_capacity)
        else:
            cpu_util = 1.0
        self._current_cpu = cpu_util

        # 12. Calculate memory utilisation
        active_nodes = [n for n in self._nodes.values() if n.status == "active"]
        total_mem_cap_mb = sum(n.memory_capacity_gb * 1024 for n in active_nodes)
        total_mem_used_mb = sum(p.memory_request_mb for p in self._pods.values()
                                if p.status in ("active", "restarting"))
        # memory also grows with queue pressure (buffering)
        queue_mem_mb = self._queue_length * 0.01  # ~10KB per queued request
        total_mem_used_mb += queue_mem_mb
        if total_mem_cap_mb > 0:
            mem_util = min(1.0, total_mem_used_mb / total_mem_cap_mb)
        else:
            mem_util = 1.0
        self._current_memory = mem_util

        # 13. Calculate latency
        latency = self._compute_latency(cpu_util, mem_util, num_active + num_restarting, total_capacity)
        self._current_latency = latency
        self._latency_history.append(latency)

        if latency > self.config.sla_latency_ms:
            self._total_sla_violations += 1

        # 14. Idle pod accounting
        for pod in active_pods:
            if cpu_util < 0.15:
                pod.idle_steps += 1

        # 15. Compute step reward
        self._compute_reward(latency, cpu_util, num_active + num_restarting)

        # 16. Advance clock
        self._step_idx += 1

    # ------------------------------------------------------------------
    # Traffic generation (enhanced with realistic spike patterns)
    # ------------------------------------------------------------------

    def _generate_traffic(self, step: int) -> float:
        """Stochastic traffic with sinusoidal base, daily pattern, noise, and
        multiple realistic spike types."""
        period = 60  # steps for a full cycle
        base = self.config.base_request_rate
        amp = self.config.traffic_amplitude
        noise_std = self.config.traffic_noise_std

        # base sinusoidal + seasonal waves (multi-frequency)
        sinusoidal = base + amp * math.sin(2 * math.pi * step / period)
        # add a slower seasonal wave
        sinusoidal += amp * 0.3 * math.sin(2 * math.pi * step / (period * 3.7))

        noise = self._rng.gauss(0, noise_std)
        time_mult = self._time_of_day_multiplier(step)

        rate = (sinusoidal + noise) * time_mult

        # --- Legacy simple burst ---
        if self._rng.random() < self.config.burst_probability:
            rate *= self.config.burst_multiplier

        # --- Enhanced spike events ---
        self._maybe_trigger_spikes(step)
        spike_mult = self._compute_spike_multiplier(step)
        rate *= spike_mult

        return max(0.0, rate)

    def _maybe_trigger_spikes(self, step: int):
        """Probabilistically trigger new spike events."""
        cfg = self.config

        # Flash crowd
        if cfg.flash_crowd_probability > 0 and self._rng.random() < cfg.flash_crowd_probability:
            # don't stack flash crowds
            if not any(s.spike_type == "flash_crowd" for s in self._active_spikes):
                self._active_spikes.append(SpikeEvent(
                    spike_type="flash_crowd",
                    remaining_steps=cfg.flash_crowd_duration,
                    total_steps=cfg.flash_crowd_duration,
                    multiplier=cfg.flash_crowd_multiplier,
                ))
                self._infra_events.append(InfraEvent(
                    step=step, event_type="flash_crowd",
                    details=f"Flash crowd detected! {cfg.flash_crowd_multiplier}x traffic for ~{cfg.flash_crowd_duration} steps",
                ))

        # Gradual ramp
        if cfg.gradual_ramp_probability > 0 and self._rng.random() < cfg.gradual_ramp_probability:
            if not any(s.spike_type == "gradual_ramp" for s in self._active_spikes):
                self._active_spikes.append(SpikeEvent(
                    spike_type="gradual_ramp",
                    remaining_steps=cfg.gradual_ramp_duration,
                    total_steps=cfg.gradual_ramp_duration,
                    multiplier=cfg.gradual_ramp_peak,
                ))
                self._infra_events.append(InfraEvent(
                    step=step, event_type="gradual_ramp",
                    details=f"Gradual traffic ramp starting — peak {cfg.gradual_ramp_peak}x over {cfg.gradual_ramp_duration} steps",
                ))

        # DDoS-like burst cluster
        if cfg.ddos_probability > 0 and self._rng.random() < cfg.ddos_probability:
            if not any(s.spike_type == "ddos" for s in self._active_spikes):
                self._active_spikes.append(SpikeEvent(
                    spike_type="ddos",
                    remaining_steps=cfg.ddos_burst_count * 3,
                    total_steps=cfg.ddos_burst_count * 3,
                    multiplier=cfg.ddos_multiplier,
                    burst_remaining=cfg.ddos_burst_count,
                ))
                self._infra_events.append(InfraEvent(
                    step=step, event_type="ddos_attack",
                    details=f"DDoS-like traffic pattern — {cfg.ddos_burst_count} micro-bursts at {cfg.ddos_multiplier}x",
                ))

        # Thundering herd
        if cfg.thundering_herd_probability > 0 and self._rng.random() < cfg.thundering_herd_probability:
            if not any(s.spike_type == "thundering_herd" for s in self._active_spikes):
                self._active_spikes.append(SpikeEvent(
                    spike_type="thundering_herd",
                    remaining_steps=3,  # very short but intense
                    total_steps=3,
                    multiplier=cfg.thundering_herd_multiplier,
                ))
                self._infra_events.append(InfraEvent(
                    step=step, event_type="thundering_herd",
                    details=f"Thundering herd! {cfg.thundering_herd_multiplier}x instantaneous spike",
                ))

    def _compute_spike_multiplier(self, step: int) -> float:
        """Compute the combined traffic multiplier from all active spikes."""
        mult = 1.0
        still_active: list[SpikeEvent] = []

        for spike in self._active_spikes:
            spike.remaining_steps -= 1
            if spike.remaining_steps <= 0:
                continue  # spike expired

            still_active.append(spike)
            progress = 1.0 - (spike.remaining_steps / spike.total_steps)

            if spike.spike_type == "flash_crowd":
                # sharp rise, sustain, sharp drop
                if progress < 0.2:
                    # ramp up
                    mult *= 1.0 + (spike.multiplier - 1.0) * (progress / 0.2)
                elif progress > 0.8:
                    # ramp down
                    tail = (1.0 - progress) / 0.2
                    mult *= 1.0 + (spike.multiplier - 1.0) * tail
                else:
                    mult *= spike.multiplier

            elif spike.spike_type == "gradual_ramp":
                # bell curve: slow ramp up, plateau at middle, slow decay
                # use sine for smooth shape
                phase = math.sin(math.pi * progress)
                mult *= 1.0 + (spike.multiplier - 1.0) * phase

            elif spike.spike_type == "ddos":
                # micro-bursts: alternating high/normal
                if spike.burst_remaining > 0 and spike.remaining_steps % 3 == 0:
                    mult *= spike.multiplier
                    spike.burst_remaining -= 1
                # between bursts: slight elevation
                else:
                    mult *= 1.0 + (spike.multiplier - 1.0) * 0.1

            elif spike.spike_type == "thundering_herd":
                # instant max, rapid decay
                decay = spike.remaining_steps / spike.total_steps
                mult *= 1.0 + (spike.multiplier - 1.0) * decay

        self._active_spikes = still_active
        return mult

    def _time_of_day_multiplier(self, step: int) -> float:
        """Simulate morning and afternoon peaks (like lunch/dinner for food delivery)."""
        # Treat step 0 as 08:00, each step ≈ 1 minute
        hour = 8 + (step / 60.0)
        multiplier = 1.0
        # morning peak: 10-12
        if 10 <= hour <= 12:
            multiplier *= self.config.morning_peak_multiplier
        # afternoon peak: 14-17
        if 14 <= hour <= 17:
            multiplier *= self.config.afternoon_peak_multiplier
        return multiplier

    # ------------------------------------------------------------------
    # Latency model
    # ------------------------------------------------------------------

    def _compute_latency(
        self, cpu_util: float, mem_util: float, num_serving: int, total_capacity: float
    ) -> float:
        """Realistic latency: base + queue pressure + CPU saturation + memory pressure + cold-start."""
        base = self.config.base_latency_ms

        # queue pressure
        if total_capacity > 0:
            queue_pressure = (self._queue_length / total_capacity) * self.config.queue_latency_factor
        else:
            queue_pressure = self.config.queue_latency_factor * 2.0

        # CPU saturation (exponential near 1.0)
        cpu_pressure = (cpu_util ** 3) * self.config.cpu_latency_factor

        # Memory pressure (quadratic near 1.0)
        mem_pressure = (mem_util ** 2) * self.config.memory_latency_factor

        # cold-start penalty
        cold_pods = sum(
            1 for p in self._pods.values()
            if p.status == "active" and p.cold_start_remaining > 0
        )
        cold_penalty = 0.0
        if num_serving > 0:
            cold_ratio = cold_pods / num_serving
            cold_penalty = cold_ratio * self.config.cold_start_latency_ms

        # VPA restart penalty (pods restarting add latency)
        restart_pods = sum(1 for p in self._pods.values() if p.status == "restarting")
        restart_penalty = 0.0
        if num_serving > 0 and restart_pods > 0:
            restart_ratio = restart_pods / num_serving
            restart_penalty = restart_ratio * self.config.cold_start_latency_ms * 1.5

        latency = base + queue_pressure + cpu_pressure + mem_pressure + cold_penalty + restart_penalty

        # add a small amount of noise
        latency += self._rng.gauss(0, 3.0)
        return max(0.0, latency)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, latency: float, cpu_util: float, num_serving: int):
        cfg = self.config

        # SLA compliance
        if latency <= cfg.sla_latency_ms:
            self._step_reward += cfg.w_sla_bonus
        else:
            late_ms = latency - cfg.sla_latency_ms
            self._step_reward += 0.8
            self._step_reward -= cfg.w_sla_penalty * late_ms

        # cost penalty (per active pod, weighted by pod size cost)
        active_pods = [p for p in self._pods.values() if p.status in ("active", "restarting")]
        pod_cost = sum(p.cost_multiplier for p in active_pods)
        self._step_reward -= cfg.w_cost * pod_cost

        # node cost penalty
        active_nodes = sum(1 for n in self._nodes.values() if n.status in ("active", "provisioning"))
        node_spec = NODE_TYPE_SPECS.get(cfg.node_type, {"cost_mult": 1.0})
        self._step_reward -= cfg.w_node_cost * active_nodes * node_spec["cost_mult"]

        # queue backlog penalty
        self._step_reward -= cfg.w_queue * self._queue_length

        # drops penalty
        if self._step_dropped > 0:
            self._step_reward -= 1.2

        # oscillation penalty (look at last 3 actions)
        if len(self._action_history) >= 3:
            last3 = self._action_history[-3:]
            # penalise sign changes (e.g. +1, -1, +1)
            sign_changes = sum(
                1 for i in range(1, len(last3))
                if last3[i] * last3[i - 1] < 0
            )
            self._step_reward -= cfg.w_oscillation * sign_changes

        # idle pod penalty
        idle_count = sum(
            1 for p in self._pods.values()
            if p.status == "active" and cpu_util < 0.15
        )
        self._step_reward -= cfg.w_idle_pod * idle_count

    # ------------------------------------------------------------------
    # Node scheduling helpers
    # ------------------------------------------------------------------

    def _find_node_for_pod(
        self, cpu_request: float, memory_request_mb: float,
        exclude_node: str = "",
    ) -> str:
        """Find the best node to schedule a pod on (bin-packing)."""
        best_node = ""
        best_remaining_cpu = float("inf")

        for node in self._nodes.values():
            if node.status != "active" or node.node_id == exclude_node:
                continue
            used_cpu = sum(
                self._pods[pid].cpu_request
                for pid in node.pods_hosted
                if pid in self._pods
            )
            used_mem = sum(
                self._pods[pid].memory_request_mb
                for pid in node.pods_hosted
                if pid in self._pods
            )
            avail_cpu = node.cpu_capacity - used_cpu
            avail_mem = (node.memory_capacity_gb * 1024) - used_mem

            if avail_cpu >= cpu_request and avail_mem >= memory_request_mb:
                if avail_cpu < best_remaining_cpu:
                    best_remaining_cpu = avail_cpu
                    best_node = node.node_id

        return best_node

    def _count_schedulable_pods(self, cpu_request: float, memory_request_mb: float) -> int:
        """Count how many additional pods can be scheduled on current nodes."""
        count = 0
        for node in self._nodes.values():
            if node.status != "active":
                continue
            used_cpu = sum(
                self._pods[pid].cpu_request
                for pid in node.pods_hosted
                if pid in self._pods
            )
            used_mem = sum(
                self._pods[pid].memory_request_mb
                for pid in node.pods_hosted
                if pid in self._pods
            )
            avail_cpu = node.cpu_capacity - used_cpu
            avail_mem = (node.memory_capacity_gb * 1024) - used_mem

            pods_by_cpu = int(avail_cpu / cpu_request) if cpu_request > 0 else 100
            pods_by_mem = int(avail_mem / memory_request_mb) if memory_request_mb > 0 else 100
            count += min(pods_by_cpu, pods_by_mem)
        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_pod_count(self) -> int:
        return sum(1 for p in self._pods.values() if p.status == "active")

    def _new_pod_id(self) -> str:
        pid = f"P{self._next_pod_id:04d}"
        self._next_pod_id += 1
        return pid

    def _new_node_id(self) -> str:
        nid = f"N{self._next_node_id:03d}"
        self._next_node_id += 1
        return nid

    # ------------------------------------------------------------------
    # Public API for dashboard spike injection
    # ------------------------------------------------------------------

    def inject_spike(self, spike_type: str, multiplier: float = 0, duration: int = 0) -> dict:
        """Manually inject a traffic spike event (for dashboard use).

        Args:
            spike_type: One of 'flash_crowd', 'gradual_ramp', 'ddos', 'thundering_herd'
            multiplier: Traffic multiplier (0 = use scenario default)
            duration: Duration in steps (0 = use scenario default)

        Returns:
            Dict with spike details.
        """
        cfg = self.config
        if spike_type == "flash_crowd":
            m = multiplier or cfg.flash_crowd_multiplier
            d = duration or cfg.flash_crowd_duration
            self._active_spikes.append(SpikeEvent(
                spike_type="flash_crowd", remaining_steps=d, total_steps=d, multiplier=m,
            ))
        elif spike_type == "gradual_ramp":
            m = multiplier or cfg.gradual_ramp_peak
            d = duration or cfg.gradual_ramp_duration
            self._active_spikes.append(SpikeEvent(
                spike_type="gradual_ramp", remaining_steps=d, total_steps=d, multiplier=m,
            ))
        elif spike_type == "ddos":
            m = multiplier or cfg.ddos_multiplier or 5.0
            bc = duration or cfg.ddos_burst_count or 5
            d = bc * 3
            self._active_spikes.append(SpikeEvent(
                spike_type="ddos", remaining_steps=d, total_steps=d,
                multiplier=m, burst_remaining=bc,
            ))
        elif spike_type == "thundering_herd":
            m = multiplier or cfg.thundering_herd_multiplier or 4.0
            d = duration or 3
            self._active_spikes.append(SpikeEvent(
                spike_type="thundering_herd", remaining_steps=d, total_steps=d, multiplier=m,
            ))
        else:
            return {"error": f"Unknown spike type: {spike_type}"}

        self._infra_events.append(InfraEvent(
            step=self._step_idx,
            event_type=spike_type,
            details=f"[INJECTED] {spike_type} — {m}x multiplier, {d} steps",
        ))
        return {"spike_type": spike_type, "multiplier": m, "duration": d, "step": self._step_idx}

    @staticmethod
    def _compute_trend(recent: list[float]) -> str:
        if len(recent) < 5:
            return "stable"
        if recent[-1] > recent[-5] * 1.1:
            return "rising"
        elif recent[-1] < recent[-5] * 0.9:
            return "falling"
        return "stable"
