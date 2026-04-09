# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Grader for CloudScaleRL / AutoScaleOps.

Calculates a normalized score [0, 1] based on service reliability,
efficiency, resource utilization, and stability.

Updated to include node efficiency and resource utilization scoring
for Kubernetes-realistic environment features.
"""

from typing import Dict, Any

def grade_episode(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Grades an episode based on cumulative metrics.
    
    Expected metrics:
    - total_steps
    - total_sla_violations
    - average_latency_ms
    - average_pods
    - total_dropped_requests
    - sla_target_ms
    - average_nodes (optional, for node efficiency)
    - average_cpu_utilization (optional)
    - average_memory_utilization (optional)
    - total_node_failures (optional)
    """
    
    total_steps = metrics.get("total_steps", 1)
    sla_violations = metrics.get("total_sla_violations", 0)
    avg_latency = metrics.get("average_latency_ms", 0)
    avg_pods = metrics.get("average_pods", 0)
    dropped = metrics.get("total_dropped_requests", 0)
    sla_target = metrics.get("sla_target_ms", 250)
    avg_nodes = metrics.get("average_nodes", 2)
    avg_cpu = metrics.get("average_cpu_utilization", 0.5)
    avg_mem = metrics.get("average_memory_utilization", 0.5)
    node_failures = metrics.get("total_node_failures", 0)
    
    # 1. SLA Score (Service Reliability) — 40%
    # Higher is better. 1.0 if no violations.
    sla_compliance = 1.0 - (sla_violations / total_steps)
    sla_compliance = max(0.0, sla_compliance)
    
    # 2. Efficiency Score (Cost/Resource Usage) — 20%
    # We want to minimize pods while keeping SLA.
    efficiency = max(0.0, 1.0 - (avg_pods / 20.0))
    
    # 3. Latency Score — 10%
    # Bonus for being well under SLA.
    latency_score = max(0.0, 1.0 - (avg_latency / (sla_target * 2.0)))
    
    # 4. Dropped Requests Penalty — 10%
    dropped_penalty = 1.0 if dropped == 0 else max(0.0, 1.0 - (dropped / 1000.0))
    
    # 5. Node Efficiency — 10%
    # Reward for using fewer nodes efficiently
    node_efficiency = max(0.0, 1.0 - (avg_nodes / 10.0))
    
    # 6. Resource Utilization — 10%
    # Reward for keeping CPU in the sweet spot (40-75%)
    # Penalize both over-provisioning (low CPU) and under-provisioning (very high CPU)
    if 0.4 <= avg_cpu <= 0.75:
        util_score = 1.0
    elif avg_cpu < 0.4:
        util_score = avg_cpu / 0.4  # linear ramp up to sweet spot
    else:
        util_score = max(0.0, 1.0 - (avg_cpu - 0.75) / 0.25)
    
    # Factor in memory utilization similarly
    if 0.3 <= avg_mem <= 0.7:
        mem_score = 1.0
    elif avg_mem < 0.3:
        mem_score = avg_mem / 0.3
    else:
        mem_score = max(0.0, 1.0 - (avg_mem - 0.7) / 0.3)
    
    resource_score = 0.6 * util_score + 0.4 * mem_score
    
    # Combined score
    final_score = (
        0.40 * sla_compliance +
        0.20 * efficiency +
        0.10 * latency_score +
        0.10 * dropped_penalty +
        0.10 * node_efficiency +
        0.10 * resource_score
    )
    
    return {
        "score": round(final_score, 3),
        "sla_compliance": round(sla_compliance, 3),
        "efficiency": round(efficiency, 3),
        "latency_score": round(latency_score, 3),
        "dropped_penalty": round(dropped_penalty, 3),
        "node_efficiency": round(node_efficiency, 3),
        "resource_utilization": round(resource_score, 3),
    }
