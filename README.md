---
title: OpenEnv Cloud Autoscaling Environment Server
emoji: ☁️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /docs
tags:
  - openenv
  - cloud
  - autoscaling
  - reinforcement-learning
  - llm
  - sre
---

# CloudScaleRL / AutoScaleOps
**OpenEnv Cloud Autoscaling Environment Server**

A real-world inspired Kubernetes autoscaling simulation for reinforcement learning and LLM-based decision-making, built for OpenEnv and the Meta PyTorch OpenEnv Hackathon.

---

# 🧠 What is CloudScaleRL / AutoScaleOps?

CloudScaleRL (AutoScaleOps) is a reinforcement learning environment that simulates real-world Kubernetes cloud autoscaling decisions, where an agent acts like a **Site Reliability Engineer (SRE)** managing infrastructure under uncertainty.

The environment models realistic Kubernetes operational tradeoffs with:
- **Worker Nodes** with CPU/memory capacity constraints and failures.
- **Horizontal Pod Autoscaling (HPA)** for dynamic pod scaling.
- **Vertical Pod Autoscaling (VPA)** for adjusting pod resource sizes.
- **Realistic Traffic Spikes** including Flash Crowds 🔥, DDoS Attacks 💀, Gradual Ramps 📈, and Thundering Herds 🐘.
- **Delayed Provisioning** modelling real-world infrastructure latency.

**🔮 Future Roadmap:** Currently the project uses heuristic fallback policies. In the near future, we will implement an LLM-first **reasoning model** for the autoscaling workflow (instead of hardcoding rules), and fine-tune it on publicly available cloud-provider knowledge (for example Azure and Alibaba documentation) as well as open sources like Wikipedia to enable stronger real-world SRE decision-making.

---

# 🚀 Quick Start

## Run the server
```bash
uv sync
uv run python -m server.app
```

## 📊 Access the Interactive Dashboard
Once the server is running, experience the environment visually through the premium interactive dashboard!

1. Open your browser and go to: **[http://localhost:8000/dashboard](http://localhost:8000/dashboard)**
2. Click **"Run 10"** or **"Auto-Play"** to see live metrics and charts.
3. Inject real-world anomalies like **Flash Crowds** or **DDoS Attacks** using the control panel to see how the system reacts!

## Run baseline evaluation
In another terminal:
```bash
uv run python scripts/run_baseline.py --url http://localhost:8000 --task easy --policy hybrid
```

## Run tests
```bash
uv sync --dev
uv run pytest tests/ -v
```

---

# 📊 Task Descriptions and Difficulty

All tasks use fixed seeds for deterministic evaluation.

| Task   | Description                              | Nodes | Pods | Horizon | Latency SLA | Difficulty |
|--------|------------------------------------------|-------|-----:|--------:|------------:|-----------|
| easy   | Stable traffic, relaxed constraints      | 2     | 3    | 180     | 250 ms      | Intro / low volatility |
| medium | Bursty demand, tighter latency target    | 2     | 4    | 240     | 180 ms      | Moderate operational pressure |
| hard   | Adversarial spikes, node failures        | 3     | 4    | 300     | 120 ms      | High volatility and strong tradeoffs |

---

# 🎮 Action Space Definition

The agent controls 3 dimensions of autoscaling, outputting a JSON object.

| Field       | Type    | Description |
|------------|---------|-------------|
| `scale_delta` | integer | (HPA) Number of pods to add/remove; `[-2, -1, 0, 1, 2]` |
| `node_delta`  | integer | (Cluster Autoscaler) Add/remove nodes; `[-1, 0, 1]` |
| `pod_size`    | string  | (VPA) Set pod resource tier; `'xs', 'sm', 'md', 'lg'` or `null` |

---

# 👀 Observation Space Definition

Includes complete infrastructure state and KPI counters:
- **time_step / horizon**
- **cpu_utilization / memory_utilization**: Aggregate resource load
- **latency_ms & request_rate**: Real-time traffic KPIs
- **queue_length**: Current backlog of unprocessed requests
- **active_pods & node_info**: Infrastructure state and nodes provisioning limits
- **pod_resource_info**: Current VPA configuration
- **recent_events**: Array of infrastructure events (e.g. `node_failure`, `flash_crowd`)
- **totals**: processed, dropped, and SLA violations
- **reward**: current step reward and cumulative reward

---

# 🏆 Reward & Grader Design

The `/grader` returns a normalized score in `[0, 1]`. Dense reward is applied every step based on:
- **SLA Compliance (40%)**: Bonus for latency <= target, penalty for exceeding it.
- **Pod Efficiency (20%)**: Minimizing unnecessary pods.
- **Latency Quality (10%)**: Staying well under the SLA threshold.
- **Node Efficiency (10%)**: Minimizing unused worker nodes.
- **Resource Utilization (10%)**: Keeping CPU/Memory in the 40-75% sweet spot.
- **Drop Penalty (10%)**: Penalizing queue overflow.

---

# 🤖 LLM Inference Results (Example Runs)

Using `inference.py` with `Qwen2.5-72B-Instruct`:

| Task   | Steps | Success | Score |
|--------|------:|---------|------:|
| easy   | 15    | true    | 0.908 |
| medium | 15    | true    | 0.575 |
| hard   | 15    | false   | 0.240 |

---

# 🧠 Inference Script (Submission Path)

Root `inference.py` is the script used by evaluators.
- Uses OpenAI-compatible client
- Required variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`

## Reproduce Inference Runs
```bash
export API_BASE_URL="<open_ai_compat_url>"
export MODEL_NAME="<model_name>"
export HF_TOKEN="<your_key>"
export BENCHMARK_URL="http://localhost:8000"
uv run python inference.py
```

---

# ☁️ Hugging Face Spaces

Deploy using `openenv push`. After deployment, the following standard OpenEnv routes will be available:
- `POST /reset`
- `POST /step`
- `/tasks`, `/grader`, `/dashboard`