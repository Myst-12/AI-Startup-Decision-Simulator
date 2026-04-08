---
title: AI Startup Decision Simulator
emoji: 🚀
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
tags:
  - openenv
  - ai-agent
  - startup
  - decision-making
  - llm
  - benchmark
---


### OpenEnv-Compatible AI Agent Benchmark Environment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-green.svg)]()
[![Docker Ready](https://img.shields.io/badge/Docker-ready-blue.svg)]()

---

## 📋 Project Description

The **Startup Decision Simulator** is a production-ready, containerised benchmark environment where AI agents act as startup founders making sequential weekly decisions. It is designed to evaluate sophisticated multi-step reasoning, resource management under uncertainty, and long-horizon planning — capabilities central to general-purpose AI agents.

The environment is fully compliant with the **OpenEnv specification**, providing typed Pydantic schemas, a deterministic `reset/step/state` interface, three graded tasks of increasing difficulty, and a complete inference script compatible with any OpenAI-API-compatible LLM endpoint.

---

## 🌍 Real-World Motivation

Building a startup involves hard resource allocation tradeoffs over many months. Decisions about when to hire, what to build, and how much to spend on marketing have cascading consequences. This simulator captures that complexity:

- **Budget constraints** force prioritisation between competing goals.
- **Market dynamics** (demand, competition, trends) react to both agent decisions and random events.
- **Stochastic events** (viral growth, server crashes, competitor launches) test robustness.
- **Dense reward shaping** ensures agents receive useful learning signal at every step.

Unlike toy grid-world environments, this simulator evaluates judgment under realistic operational pressure.

---

## 🔭 Observation Space

Each week the agent receives a full `Observation` object:

| Field | Type | Description |
|---|---|---|
| `budget` | float | Remaining cash (USD) |
| `team.engineers` | int | Number of software engineers |
| `team.designers` | int | Number of designers |
| `team.marketers` | int | Number of marketers |
| `product.features_built` | list[str] | Released feature names |
| `product.quality` | float [0–1] | Overall product quality |
| `market.demand` | float [0–1] | Market demand for your category |
| `market.competition` | float [0–1] | Competitive intensity |
| `market.trend` | str | Current market trend label |
| `metrics.revenue` | float | Weekly revenue (USD) |
| `metrics.burn_rate` | float | Weekly cash burn (USD) |
| `metrics.user_growth` | float | Weekly user growth rate (%) |
| `time.current_week` | int | Current simulation week |
| `time.max_weeks` | int | Episode length |
| `pending_events` | list[str] | Active events this week |

---

## 🎮 Action Space

| Action | Payload | Effect |
|---|---|---|
| `hire` | `{role: "engineer"\|"designer"\|"marketer"}` | Adds headcount, costs salary + hiring fee |
| `fire` | `{role: "engineer"\|"designer"\|"marketer"}` | Reduces headcount, saves salary |
| `build_feature` | `{feature_name: str}` | Adds a feature, boosts quality, costs engineering sprint |
| `marketing` | `{budget: float ≥ 500}` | Spends on ads, boosts demand and user growth |
| `pivot` | `{new_trend: str}` | Repositions to a new market trend |
| `wait` | `{}` | No action this week |

All actions are validated before execution. Invalid actions incur a reward penalty.

---

## 📋 Task Descriptions

### 🟢 Task 1 — MVP Launch (Easy)
- **Budget:** $120,000 | **Max weeks:** 20 | **Competition:** 15%
- **Goal:** Ship at least one feature and generate revenue > $3,000/week.
- **Grader:** Weighted combination of product launch and revenue milestones.
- **Expected baseline score:** 0.45–0.65

### 🟡 Task 2 — Growth Phase (Medium)
- **Budget:** $350,000 | **Max weeks:** 36 | **Competition:** 40%
- **Starting state:** Product already launched with a core feature.
- **Goal:** Scale user growth (>20%/week) and revenue (>$20k/week).
- **Grader:** Normalised sum of revenue peak, user growth peak, features built, and survival.
- **Expected baseline score:** 0.30–0.50

### 🔴 Task 3 — Survival Mode (Hard)
- **Budget:** $80,000 | **Max weeks:** 48 | **Competition:** 70%
- **Random disruptive events** with higher frequency.
- **Goal:** Survive (avoid bankruptcy), maintain revenue, and grow users.
- **Grader:** Weighted combination of survival, revenue, user growth, and episode duration.
- **Expected baseline score:** 0.20–0.40

---

## 🏆 Reward Design

The reward function is **dense and continuous** — agents receive meaningful signal every step.

```
reward = 0.30 × revenue_component
       + 0.25 × user_growth_component
       + 0.25 × quality_component
       + 0.20 × efficiency_component
       − penalty
```

| Component | Normalisation | Rationale |
|---|---|---|
| Revenue | Revenue / $30,000 | Revenue is the primary success signal |
| User Growth | Growth / 20% | Users drive long-term value |
| Quality | Quality score [0–1] | Quality reduces churn |
| Efficiency | Revenue / Burn Rate | Efficient spending is sustainable |
| Penalty | 0.15 per invalid action | Discourages illegal/irrational actions |

All components are clamped to [0, 1]. The total is normalised to [0, 1].

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Docker (for containerised runs)

### Local Setup

```bash
# Clone / enter the project directory
cd startup-decision-simulator

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hugging_face_token"
```

---

## ▶️ Running Inference

### Local

```bash
python inference.py
```

### Docker

```bash
# Build
docker build -t startup-simulator .

# Run
docker run \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e HF_TOKEN="your_hugging_face_token" \
  startup-simulator
```

### Expected Log Format

```
[START] task=mvp_launch env=startup-decision-simulator model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"type":"build_feature","payload":{"feature_name":"mvp_week_1"}} reward=0.13 done=false error=null
[STEP] step=2 action={"type":"marketing","payload":{"budget":1000.0}} reward=0.19 done=false error=null
...
[END] success=true steps=18 score=0.624 rewards=0.13,0.19,...
```

---

## 📊 Expected Baseline Scores

| Task | Difficulty | Random Agent | Greedy Agent | GPT-4o-mini |
|---|---|---|---|---|
| MVP Launch | Easy | 0.10–0.20 | 0.40–0.55 | 0.45–0.65 |
| Growth Phase | Medium | 0.08–0.15 | 0.25–0.40 | 0.30–0.50 |
| Survival Mode | Hard | 0.05–0.12 | 0.15–0.30 | 0.20–0.40 |

---

## 🗂️ Project Structure

```
startup-decision-simulator/
├── environment/
│   ├── __init__.py        # Package exports
│   ├── models.py          # Pydantic schemas (Observation, Action, Reward)
│   ├── startup_env.py     # Core simulation engine
│   └── tasks.py           # Task configs and graders
├── inference.py           # LLM agent runner (OpenAI-compatible)
├── openenv.yaml           # OpenEnv specification file
├── requirements.txt       # Python dependencies
├── Dockerfile             # Production container
└── README.md              # This file
```

---

## 🔬 Programmatic Usage

```python
from environment import StartupEnv, Action
from environment.tasks import TASK_MVP_LAUNCH, grade_mvp_launch

# Initialise and reset
env = StartupEnv(config=TASK_MVP_LAUNCH["config"])
obs = env.reset()

# Step through the episode
action = Action.from_dict({"type": "build_feature", "payload": {"feature_name": "core_mvp"}})
obs, reward, done, info = env.step(action)

print(f"Reward: {reward.total:.4f}")
print(f"Budget: ${obs.budget:,.0f}")

# Access full state
state = env.state()
```

---

## ⚙️ Hugging Face Spaces Deployment

1. Push this repository to a Hugging Face Space (Docker SDK).
2. Set secrets: `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`.
3. The Space will automatically run `inference.py` on startup.

---

## 📜 License

MIT License — see `LICENSE` for details.
