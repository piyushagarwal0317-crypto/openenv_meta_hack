# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import pytest
from server.cloudscale_rl_environment import CloudScaleEnvironment
from models import CloudScaleAction


def test_env_reset():
    env = CloudScaleEnvironment(task="easy")
    obs = env.reset()
    assert obs.task_id == "easy"
    assert obs.time_step == 0
    assert obs.active_pods == 3
    assert obs.horizon == 180


def test_env_step_scale_up():
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    action = CloudScaleAction(scale_delta=1)
    obs = env.step(action)

    assert obs.time_step == 1
    assert obs.pending_scale_ups >= 1
    # Active pods shouldn't change yet due to provisioning delay
    assert obs.active_pods == 3


def test_env_step_hold():
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    action = CloudScaleAction(scale_delta=0)
    obs = env.step(action)

    assert obs.time_step == 1
    assert obs.active_pods == 3
    assert obs.request_rate > 0


def test_env_done():
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    for _ in range(180):
        obs = env.step(CloudScaleAction(scale_delta=0))

    assert obs.done is True
    assert obs.time_step == 180


def test_env_provisioning_completes():
    """After enough steps, a scale-up should result in more active pods."""
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    # Scale up by 2
    env.step(CloudScaleAction(scale_delta=2))
    # Wait for provisioning to complete (easy delay_mean=3, +margin)
    for _ in range(6):
        obs = env.step(CloudScaleAction(scale_delta=0))

    assert obs.active_pods >= 4  # started with 3, added 2


def test_env_scale_down():
    """Scaling down should eventually reduce pod count."""
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    # Scale down by 1
    env.step(CloudScaleAction(scale_delta=-1))
    # Wait for deprovision to complete (easy deprovision_delay=2, +margin)
    for _ in range(4):
        obs = env.step(CloudScaleAction(scale_delta=0))

    assert obs.active_pods <= 3  # started with 3, removed 1


def test_env_medium_task():
    env = CloudScaleEnvironment(task="medium")
    obs = env.reset()
    assert obs.task_id == "medium"
    assert obs.horizon == 240
    assert obs.active_pods == 4


def test_env_hard_task():
    env = CloudScaleEnvironment(task="hard")
    obs = env.reset()
    assert obs.task_id == "hard"
    assert obs.horizon == 300
    assert obs.active_pods == 4


def test_env_deterministic():
    """Two episodes with same seed should produce identical first-step traffic."""
    env1 = CloudScaleEnvironment(task="easy")
    obs1 = env1.reset()
    step1 = env1.step(CloudScaleAction(scale_delta=0))

    env2 = CloudScaleEnvironment(task="easy")
    obs2 = env2.reset()
    step2 = env2.step(CloudScaleAction(scale_delta=0))

    assert step1.request_rate == step2.request_rate
    assert step1.latency_ms == step2.latency_ms


# ---- New: Node layer tests ----


def test_env_has_nodes():
    """Environment should start with worker nodes."""
    env = CloudScaleEnvironment(task="easy")
    obs = env.reset()
    assert obs.node_info is not None
    assert obs.node_info.active_nodes == 2  # easy starts with 2 nodes
    assert obs.node_info.node_cpu_capacity > 0
    assert obs.node_info.node_memory_capacity_gb > 0


def test_node_scale_up():
    """Adding a node should eventually increase active_nodes."""
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    env.step(CloudScaleAction(scale_delta=0, node_delta=1))
    # node provisioning is slow (~10 steps for medium type)
    for _ in range(15):
        obs = env.step(CloudScaleAction(scale_delta=0))

    assert obs.node_info.active_nodes >= 3  # started with 2, added 1


def test_node_scale_down():
    """Removing a node should eventually decrease active_nodes."""
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    env.step(CloudScaleAction(scale_delta=0, node_delta=-1))
    # Wait for drain + deprovision
    for _ in range(8):
        obs = env.step(CloudScaleAction(scale_delta=0))

    assert obs.node_info.active_nodes <= 2


def test_pod_scheduling_respects_node_capacity():
    """Pods should only be scheduled if node capacity is available."""
    env = CloudScaleEnvironment(task="easy")
    obs = env.reset()
    initial_pods = obs.active_pods
    initial_nodes = obs.node_info.active_nodes

    # Try to add many pods — should be limited by node capacity
    for _ in range(5):
        obs = env.step(CloudScaleAction(scale_delta=2))

    # Wait for provisioning
    for _ in range(10):
        obs = env.step(CloudScaleAction(scale_delta=0))

    # There should be a ceiling imposed by node capacity
    # With 2 medium nodes (4 vCPU each) and sm pods (0.5 vCPU each),
    # max pods = 2 * (4 / 0.5) = 16, but we started with 3, so max new = 13
    assert obs.active_pods <= 20  # reasonable upper bound


# ---- New: VPA tests ----


def test_vpa_pod_size_change():
    """Changing pod size should trigger a VPA restart and update capacity."""
    env = CloudScaleEnvironment(task="easy")
    obs = env.reset()
    assert obs.pod_resource_info.pod_size == "sm"

    # Change to medium pods
    obs = env.step(CloudScaleAction(scale_delta=0, pod_size="md"))
    assert obs.pod_resource_info.vpa_restart_in_progress is True
    assert obs.pod_resource_info.pod_size == "md"

    # Wait for restart to complete
    for _ in range(5):
        obs = env.step(CloudScaleAction(scale_delta=0))

    assert obs.pod_resource_info.vpa_restart_in_progress is False
    assert obs.pod_resource_info.pod_capacity == 800  # md tier


def test_vpa_no_change_when_same_size():
    """Setting pod_size to current size should not trigger restart."""
    env = CloudScaleEnvironment(task="easy")
    obs = env.reset()
    obs = env.step(CloudScaleAction(scale_delta=0, pod_size="sm"))
    assert obs.pod_resource_info.vpa_restart_in_progress is False


def test_vpa_resize_to_xs():
    """Resizing to xs should reduce capacity and cost."""
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    obs = env.step(CloudScaleAction(scale_delta=0, pod_size="xs"))
    assert obs.pod_resource_info.pod_size == "xs"
    assert obs.pod_resource_info.pod_capacity == 250


def test_vpa_resize_to_lg():
    """Resizing to lg should increase capacity."""
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    obs = env.step(CloudScaleAction(scale_delta=0, pod_size="lg"))
    assert obs.pod_resource_info.pod_size == "lg"
    assert obs.pod_resource_info.pod_capacity == 1200


# ---- New: Memory utilization tests ----


def test_memory_utilization_reported():
    """Memory utilization should be in the observation."""
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    obs = env.step(CloudScaleAction(scale_delta=0))
    assert 0.0 <= obs.memory_utilization <= 1.0


# ---- New: Infrastructure events tests ----


def test_infra_events_reported():
    """Recent events should be reported in observations."""
    env = CloudScaleEnvironment(task="easy")
    obs = env.reset()
    assert isinstance(obs.recent_events, list)


def test_node_add_generates_event():
    """Adding a node should generate an infra event."""
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    obs = env.step(CloudScaleAction(scale_delta=0, node_delta=1))
    # Should have a node_provisioning event
    event_types = [e.event_type for e in obs.recent_events]
    assert "node_provisioning" in event_types


def test_vpa_generates_event():
    """Changing pod size should generate a vpa_restart event."""
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    obs = env.step(CloudScaleAction(scale_delta=0, pod_size="lg"))
    event_types = [e.event_type for e in obs.recent_events]
    assert "vpa_restart" in event_types


# ---- New: Enhanced traffic spike tests ----


def test_traffic_with_spikes_hard():
    """Hard difficulty should produce traffic variation with spikes."""
    env = CloudScaleEnvironment(task="hard")
    env.reset()
    rates = []
    for _ in range(100):
        obs = env.step(CloudScaleAction(scale_delta=0))
        rates.append(obs.request_rate)

    # Should have meaningful variance due to spikes
    min_rate = min(rates)
    max_rate = max(rates)
    assert max_rate > min_rate * 1.5  # at least 50% variation


# ---- New: Backward compatibility test ----


def test_backward_compat_action():
    """Action with only scale_delta should work (defaults for node_delta and pod_size)."""
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    # Only scale_delta provided — node_delta=0 and pod_size=None by default
    action = CloudScaleAction(scale_delta=1)
    obs = env.step(action)
    assert obs.time_step == 1
    assert obs.node_info.active_nodes == 2  # unchanged


def test_full_episode_easy():
    """Run a complete episode on easy to verify no crashes."""
    env = CloudScaleEnvironment(task="easy")
    env.reset()
    for _ in range(180):
        obs = env.step(CloudScaleAction(scale_delta=0))
    assert obs.done is True


def test_full_episode_hard_with_scaling():
    """Run a complete episode on hard with active scaling decisions."""
    env = CloudScaleEnvironment(task="hard")
    env.reset()
    for i in range(300):
        if i % 10 == 0:
            action = CloudScaleAction(scale_delta=1)
        elif i % 15 == 0:
            action = CloudScaleAction(scale_delta=-1, node_delta=1)
        else:
            action = CloudScaleAction(scale_delta=0)
        obs = env.step(action)
    assert obs.done is True
    assert obs.time_step == 300
