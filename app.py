"""
app.py — Gradio web interface for the Startup Decision Simulator.
Runs on port 7860 for Hugging Face Spaces (Docker SDK).
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

import gradio as gr

from environment.startup_env import StartupEnv
from environment.models import Action
from environment.tasks import ALL_TASKS, get_task_by_name

# ---------------------------------------------------------------------------
# Load .env if present (local dev)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = (
    os.environ.get("GROQ_API_KEY")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# LLM client (lazy init so app loads even without key)
# ---------------------------------------------------------------------------
_client = None

def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=GROQ_API_KEY or "sk-placeholder", base_url=API_BASE_URL)
    return _client


SYSTEM_PROMPT = """You are an experienced startup founder making strategic weekly decisions.
Return ONLY a valid JSON object with keys "type" and "payload". No explanations.

Actions:
- {"type": "hire", "payload": {"role": "engineer"|"designer"|"marketer"}}
- {"type": "fire", "payload": {"role": "engineer"|"designer"|"marketer"}}
- {"type": "build_feature", "payload": {"feature_name": "<unique name>"}}
- {"type": "marketing", "payload": {"budget": <float >= 500>}}
- {"type": "pivot", "payload": {"new_trend": "AI/ML"|"sustainability"|"consumer_health"|"fintech"|"enterprise_saas"|"creator_economy"|"web3"|"edtech"|"stable"}}
- {"type": "wait", "payload": {}}

Build features before marketing. Watch burn rate. Respond to events intelligently."""


def get_llm_action(obs_dict: Dict, task_desc: str, week: int, max_weeks: int) -> Action:
    try:
        client = get_client()
        prompt = f"TASK: {task_desc}\nWEEK {week}/{max_weeks}\nSTATE:\n{json.dumps(obs_dict, indent=2)}\n\nReturn ONLY the JSON action:"
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=128,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return Action.from_dict(json.loads(raw))
    except Exception:
        return Action.from_dict({"type": "wait", "payload": {}})


# ---------------------------------------------------------------------------
# Run a task and yield step-by-step log lines
# ---------------------------------------------------------------------------

def run_task_streaming(task_name: str):
    task_cfg, grader_fn = get_task_by_name(task_name)
    config = task_cfg["config"]
    desc = task_cfg["description"]

    env = StartupEnv(config=config)
    obs = env.reset()

    log_lines = []
    history = []
    done = False
    step = 0

    yield f"🚀 **{task_cfg['display_name']}** | Difficulty: `{task_cfg['difficulty']}` | Budget: ${obs.budget:,.0f} | Max Weeks: {obs.time.max_weeks}\n\n"
    yield f"📋 *{desc}*\n\n"
    yield "---\n"
    yield f"| Week | Action | Reward | Budget | Revenue | User Growth | Quality | Event |\n"
    yield f"|------|--------|--------|--------|---------|-------------|---------|-------|\n"

    while not done and step < config["max_weeks"]:
        step += 1
        obs_dict = obs.model_dump()
        action = get_llm_action(obs_dict, desc, obs.time.current_week, obs.time.max_weeks)
        obs, reward, done, info = env.step(action)
        history.append({"obs": obs, "reward": reward, "action": action, "info": info})

        event = info.get("event", "none")
        event_icon = {"viral_growth": "🚀", "server_crash": "💥", "competitor_launch": "⚔️",
                      "press_coverage": "📰", "economic_downturn": "📉", "key_employee_left": "👋"}.get(event, "")
        event_str = f"{event_icon} {event}" if event_icon else event

        row = (
            f"| {obs.time.current_week - 1:02d} "
            f"| `{action.type}` "
            f"| {reward.total:.3f} "
            f"| ${obs.budget:,.0f} "
            f"| ${obs.metrics.revenue:,.0f} "
            f"| {obs.metrics.user_growth:.1f}% "
            f"| {obs.product.quality:.2f} "
            f"| {event_str} |\n"
        )
        yield row

    score = grader_fn(history)
    final = history[-1]["obs"] if history else obs
    reason = history[-1]["info"].get("done_reason", "unknown") if history else "unknown"

    reason_icon = {"bankruptcy": "💸", "max_weeks_reached": "🏁", "success_milestone": "🏆"}.get(reason, "❓")

    yield "\n---\n"
    yield f"### Results\n"
    yield f"- **Grader Score:** `{score:.4f}` / 1.0\n"
    yield f"- **Steps Run:** {step}\n"
    yield f"- **Done Reason:** {reason_icon} `{reason}`\n"
    yield f"- **Final Budget:** ${final.budget:,.0f}\n"
    yield f"- **Final Revenue:** ${final.metrics.revenue:,.0f}/week\n"
    yield f"- **Features Built:** {', '.join(f'`{f}`' for f in final.product.features_built) or 'none'}\n"
    yield f"- **Product Quality:** {final.product.quality:.2f}\n"


# ---------------------------------------------------------------------------
# Run all tasks at once
# ---------------------------------------------------------------------------

def run_all_tasks():
    all_output = ""
    scores = []

    for task_cfg, grader_fn in ALL_TASKS:
        config = task_cfg["config"]
        desc = task_cfg["description"]
        env = StartupEnv(config=config)
        obs = env.reset()
        history = []
        done = False
        step = 0
        log = []

        log.append(f"### 🏢 {task_cfg['display_name']} (`{task_cfg['difficulty']}`)\n")
        log.append(f"*{desc}*\n\n")
        log.append("| Week | Action | Reward | Budget | Revenue |\n")
        log.append("|------|--------|--------|--------|---------|\n")

        while not done and step < config["max_weeks"]:
            step += 1
            obs_dict = obs.model_dump()
            action = get_llm_action(obs_dict, desc, obs.time.current_week, obs.time.max_weeks)
            obs, reward, done, info = env.step(action)
            history.append({"obs": obs, "reward": reward, "action": action, "info": info})
            log.append(f"| {obs.time.current_week-1:02d} | `{action.type}` | {reward.total:.3f} | ${obs.budget:,.0f} | ${obs.metrics.revenue:,.0f} |\n")

        score = grader_fn(history)
        scores.append(score)
        reason = history[-1]["info"].get("done_reason", "unknown") if history else "unknown"
        final = history[-1]["obs"] if history else obs

        log.append(f"\n**Score:** `{score:.4f}` | **Steps:** {step} | **Done:** `{reason}`\n\n---\n")
        all_output += "".join(log)

    mean = sum(scores) / len(scores) if scores else 0
    all_output += f"\n## 📊 Final Summary\n| Task | Score |\n|------|-------|\n"
    for (tc, _), s in zip(ALL_TASKS, scores):
        all_output += f"| {tc['display_name']} | `{s:.4f}` |\n"
    all_output += f"\n**Mean Score: `{mean:.4f}`**\n"
    return all_output


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

TASK_CHOICES = [(f"{tc['display_name']} ({tc['difficulty']})", tc["name"]) for tc, _ in ALL_TASKS]

def build_ui():
    with gr.Blocks(
        title="🚀 Startup Decision Simulator",
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="cyan"),
        css="""
        .header-box { text-align: center; padding: 20px 0; }
        .metric-box { border-radius: 10px; padding: 10px; }
        footer { display: none !important; }
        """
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-box">
            <h1 style="font-size:2.2em; margin:0;">🚀 Startup Decision Simulator</h1>
            <p style="color:#888; font-size:1.1em; margin:6px 0 0;">
                OpenEnv-Compatible AI Agent Benchmark — Powered by <strong>Llama 3.3 70B via Groq</strong>
            </p>
        </div>
        """)

        # ── Info cards ───────────────────────────────────────────────────────
        with gr.Row():
            gr.HTML("""<div style="background:#1e1e2e;border-radius:10px;padding:16px;color:#cdd6f4;">
                <b>🎯 What this does</b><br>An LLM agent acts as a startup founder, making weekly decisions
                (hire, build, market, pivot) to maximise revenue and user growth across 3 tasks of increasing difficulty.
            </div>""")
            gr.HTML("""<div style="background:#1e1e2e;border-radius:10px;padding:16px;color:#cdd6f4;">
                <b>⚙️ How it works</b><br>Each step = 1 week. The agent receives the full startup state as JSON
                and returns a structured action. Stochastic events (viral growth, crashes) add realism.
            </div>""")
            gr.HTML("""<div style="background:#1e1e2e;border-radius:10px;padding:16px;color:#cdd6f4;">
                <b>📊 Scoring</b><br>Dense reward per step (revenue + user growth + quality + efficiency).
                Final grader score in [0, 1] per task. Mean score across all 3 tasks is the overall benchmark.
            </div>""")

        gr.Markdown("---")

        # ── Single task tab ──────────────────────────────────────────────────
        with gr.Tabs():
            with gr.TabItem("▶️ Run Single Task"):
                with gr.Row():
                    task_dropdown = gr.Dropdown(
                        choices=TASK_CHOICES,
                        value="mvp_launch",
                        label="Select Task",
                        scale=2,
                    )
                    run_btn = gr.Button("▶ Run Task", variant="primary", scale=1)

                task_output = gr.Markdown(label="Episode Log", value="*Select a task and click Run Task to start...*")

                run_btn.click(
                    fn=lambda t: "".join(run_task_streaming(t)),
                    inputs=task_dropdown,
                    outputs=task_output,
                )

            # ── All tasks tab ────────────────────────────────────────────────
            with gr.TabItem("🏁 Run All Tasks (Benchmark)"):
                run_all_btn = gr.Button("▶ Run All 3 Tasks", variant="primary")
                all_output = gr.Markdown(value="*Click to run the full benchmark across all 3 tasks...*")
                run_all_btn.click(fn=run_all_tasks, outputs=all_output)

            # ── Info tab ─────────────────────────────────────────────────────
            with gr.TabItem("📖 Environment Spec"):
                gr.Markdown("""
## Action Space

| Action | Payload | Effect |
|--------|---------|--------|
| `hire` | `{role: engineer\|designer\|marketer}` | Adds headcount |
| `fire` | `{role: engineer\|designer\|marketer}` | Reduces headcount |
| `build_feature` | `{feature_name: str}` | Ships a feature, boosts quality |
| `marketing` | `{budget: float ≥ 500}` | Boosts demand & user growth |
| `pivot` | `{new_trend: str}` | Repositions to a new market |
| `wait` | `{}` | No action |

## Reward Function

```
reward = 0.30 × revenue_component
       + 0.25 × user_growth_component
       + 0.25 × quality_component
       + 0.20 × efficiency_component
       − 0.15 × (1 if invalid action)
```

All components normalised to [0, 1]. Total reward in [0, 1].

## Tasks

| Task | Difficulty | Budget | Competition | Goal |
|------|-----------|--------|-------------|------|
| MVP Launch | 🟢 Easy | $120k | 15% | Ship product + revenue > $3k/wk |
| Growth Phase | 🟡 Medium | $350k | 40% | Scale users + revenue |
| Survival Mode | 🔴 Hard | $80k | 70% | Survive + maintain growth |

## Stochastic Events

`viral_growth` 🚀 · `server_crash` 💥 · `competitor_launch` ⚔️ · `press_coverage` 📰 · `economic_downturn` 📉 · `key_employee_left` 👋
                """)

        gr.Markdown(
            "<center style='color:#555;font-size:0.85em;'>OpenEnv-Compatible · Docker Space · "
            "<a href='https://github.com/Myst-12/AI-Startup-Decision-Simulator' target='_blank'>GitHub</a></center>"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
