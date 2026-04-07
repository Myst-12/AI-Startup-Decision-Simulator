"""
app.py — Refactored Gradio web interface for the Startup Decision Simulator.
Features: KPI Dashboard, Live Charts, Strategic Feedback, and Agent Reasoning.
"""

from __future__ import annotations

import json
import os
import time
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

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
# LLM client (lazy init)
# ---------------------------------------------------------------------------
_client = None

def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=GROQ_API_KEY or "sk-placeholder", base_url=API_BASE_URL)
    return _client


SYSTEM_PROMPT = """You are an experienced startup founder making strategic weekly decisions.
Return ONLY a valid JSON object with keys "type", "payload", and "reasoning".
"reasoning" should be a short (1-sentence) explanation of your strategy for this step.

Actions:
- {"type": "hire", "payload": {"role": "engineer"|"designer"|"marketer"}, "reasoning": "..."}
- {"type": "fire", "payload": {"role": "engineer"|"designer"|"marketer"}, "reasoning": "..."}
- {"type": "build_feature", "payload": {"feature_name": "<unique name>"}, "reasoning": "..."}
- {"type": "marketing", "payload": {"budget": <float >= 500>}, "reasoning": "..."}
- {"type": "pivot", "payload": {"new_trend": "AI/ML"|"sustainability"|"enterprise_saas"|"..." }, "reasoning": "..."}
- {"type": "wait", "payload": {}, "reasoning": "..."}

Strategy Tips: Build features before marketing. Watch burn rate. Pivot if competition is too high."""


def get_llm_action_with_reasoning(obs_dict: Dict, task_desc: str, week: int, max_weeks: int) -> Tuple[Action, str]:
    try:
        client = get_client()
        prompt = f"TASK: {task_desc}\nWEEK {week}/{max_weeks}\nSTATE:\n{json.dumps(obs_dict, indent=2)}\n\nReturn JSON action with reasoning:"
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
            raw = raw.strip()
        
        data = json.loads(raw)
        return Action.from_dict(data), data.get("reasoning", "No reasoning provided.")
    except Exception as e:
        print(f"LLM Error: {e}")
        return Action.from_dict({"type": "wait", "payload": {}}), "Fallback to wait due to error."


# ---------------------------------------------------------------------------
# Strategic Feedback Heuristics
# ---------------------------------------------------------------------------

def generate_strategic_feedback(history: List[Dict[str, Any]], final_score: float) -> str:
    suggestions = []
    last_obs = history[-1]["obs"]
    last_info = history[-1]["info"]
    
    if last_info.get("done_reason") == "bankruptcy":
        if last_obs.team.engineers > 3:
            suggestions.append("⚠️ **Over-hiring Engineering**: You scaled your team faster than your budget could support early on.")
        if last_obs.metrics.revenue < 1000:
            suggestions.append("⚠️ **Slow to Market**: You ran out of cash before generating significant revenue. Try shipping features faster.")
            
    if last_obs.product.quality < 0.3 and any(h["action"].type == "marketing" for h in history):
        suggestions.append("⚠️ **Premature Marketing**: You spent heavily on marketing while product quality was low. Focus on R&D (Build Feature) first.")

    if last_obs.metrics.revenue > 10000 and last_obs.metrics.user_growth < 5.0:
        suggestions.append("💡 **Scale Opportunity**: You have a strong product but slow growth. Consider more aggressive marketing or hiring a Marketer.")
        
    if not suggestions:
        if final_score > 0.8:
            suggestions.append("🌟 **Excellent Execution**: You balanced growth and survival perfectly.")
        else:
            suggestions.append("💡 **Incremental Gains**: Try to balance feature development with consistent marketing to avoid growth plateaus.")
            
    return "\n\n".join(suggestions)


# ---------------------------------------------------------------------------
# Task Runner
# ---------------------------------------------------------------------------

def run_task_streaming(task_name: str):
    task_cfg, grader_fn = get_task_by_name(task_name)
    env = StartupEnv(config=task_cfg["config"])
    obs = env.reset()
    
    history = []
    done = False
    
    # State tracking for charts
    df_metrics = pd.DataFrame(columns=["Week", "Revenue", "Users"])
    
    yield (
        obs.budget, obs.metrics.revenue, obs.metrics.user_growth, obs.product.quality,
        df_metrics, "", "", "", gr.update(visible=False)
    )

    while not done:
        obs_dict = obs.model_dump()
        action, reasoning = get_llm_action_with_reasoning(
            obs_dict, task_cfg["description"], obs.time.current_week, obs.time.max_weeks
        )
        
        obs, reward, done, info = env.step(action)
        history.append({"obs": obs, "reward": reward, "action": action, "info": info})
        
        # Update metric tracking
        new_row = {"Week": obs.time.current_week - 1, "Revenue": obs.metrics.revenue, "Users": obs.metrics.user_growth}
        df_metrics = pd.concat([df_metrics, pd.DataFrame([new_row])], ignore_index=True)
        
        # Event Highlight
        event = info.get("event", "none")
        event_banner = ""
        if event != "none":
            color = "#ef4444" if event in ["server_crash", "economic_downturn"] else "#10b981"
            event_banner = f"<div style='background:{color};color:white;padding:8px;border-radius:5px;text-align:center;'>🚀 **EVENT**: {event.replace('_', ' ').upper()}</div>"

        # Insight content
        why_str = "\n".join([f"• {w}" for w in info.get("why", [])])
        insights = f"**Agent Reasoning:**\n{reasoning}\n\n**Mechanical Breakdown:**\n{why_str}"
        
        # Yield mid-step updates
        yield (
            obs.budget, obs.metrics.revenue, obs.metrics.user_growth, obs.product.quality,
            df_metrics, event_banner, insights, "", gr.update(visible=False)
        )
        time.sleep(0.5)

    # Final breakdown
    score = grader_fn(history)
    feedback = generate_strategic_feedback(history, score)
    
    # Generate result markdown
    surv = 1 if history[-1]["obs"].budget > 0 else 0
    rev_max = max(h["obs"].metrics.revenue for h in history)
    ug_max = max(h["obs"].metrics.user_growth for h in history)
    
    breakdown_md = f"""
### 🏁 Run Analysis
| Component | Score | Weight |
|---|---|---|
| **Survival** | {surv}.0 | 40% |
| **Revenue Peak** | ${rev_max:,.0f} | 30% |
| **Growth Peak** | {ug_max:.1f}% | 30% |

**Final Balanced Score: `{score:.4f}`**

### 💡 Strategic Feedback
{feedback}
"""
    yield (
        obs.budget, obs.metrics.revenue, obs.metrics.user_growth, obs.product.quality,
        df_metrics, "", insights, breakdown_md, gr.update(visible=True)
    )

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

# Initial data for charts
df_metrics_init = pd.DataFrame(columns=["Week", "Revenue", "Users"])

def build_ui():
    with gr.Blocks(
        title="🚀 Startup Simulator v2",
        theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate"),
        css=".metric-card { background: #f8fafc; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; text-align: center; }"
    ) as demo:
        
        gr.HTML("<h1 style='text-align:center; margin:20px 0;'>🚀 Startup Decision Simulator <span style='font-size:0.5em; vertical-align:middle; background:#6366f1; color:white; padding:4px 8px; border-radius:12px;'>v2 PRO</span></h1>")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Mission Control")
                task_select = gr.Dropdown(
                    choices=[(f"{t['display_name']} ({t['difficulty'].upper()})", t["name"]) for t, _ in ALL_TASKS],
                    value="mvp_launch",
                    label="Choose Challenge"
                )
                mission_desc = gr.Markdown("Goal: Build an MVP and generate revenue.")
                run_btn = gr.Button("🚀 LAUNCH SIMULATION", variant="primary")
            
            with gr.Column(scale=2):
                with gr.Row():
                    m_budget = gr.Number(label="Budget ($)", value=0, precision=0, interactive=False)
                    m_revenue = gr.Number(label="Revenue ($/wk)", value=0, precision=0, interactive=False)
                    m_growth = gr.Number(label="User Growth (%)", value=0, precision=1, interactive=False)
                    m_quality = gr.Number(label="Product Quality", value=0, precision=2, interactive=False)

        event_alert = gr.HTML("")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Performance Visuals")
                chart = gr.LinePlot(
                    df_metrics_init,
                    x="Week",
                    y=["Revenue", "Users"],
                    title="Revenue & Growth Over Time",
                    width=600,
                    height=350,
                    overlay_point=True,
                    tooltip=["Week", "Revenue", "Users"]
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 Step Insights")
                insight_panel = gr.Markdown("Simulation not started...")
        
        results_panel = gr.Markdown("", visible=False)
        reset_btn = gr.Button("♻️ Reset & Clear", visible=False)

        # ── Logic ──────────────────────────────────────────────────────────
        
        def update_desc(name):
            task, _ = get_task_by_name(name)
            return f"**{task['display_name']}**: {task['description']}"

        task_select.change(fn=update_desc, inputs=task_select, outputs=mission_desc)
        
        run_btn.click(
            fn=run_task_streaming,
            inputs=task_select,
            outputs=[m_budget, m_revenue, m_growth, m_quality, chart, event_alert, insight_panel, results_panel, reset_btn]
        )
        
        reset_btn.click(lambda: (0, 0, 0, 0, pd.DataFrame(), "", "", "", gr.update(visible=False)), 
                        outputs=[m_budget, m_revenue, m_growth, m_quality, chart, event_alert, insight_panel, results_panel, reset_btn])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
