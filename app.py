"""
app.py — Gradio UI plus FastAPI OpenEnv endpoints for the Startup Decision Simulator.
"""

from __future__ import annotations

import copy
import json
import os
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from environment.models import Action, Observation
from environment.startup_env import StartupEnv
from environment.tasks import ALL_TASKS, get_task_by_name


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or ""
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
PORT = int(os.environ.get("PORT", "7860"))

_client = None
_SESSION_LOCK = Lock()
_SESSIONS: Dict[str, Dict[str, Any]] = {}

DIFFICULTY_TO_TASK = {
    "easy": "mvp_launch",
    "medium": "growth_phase",
    "hard": "survival_mode",
}


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=API_KEY or "sk-placeholder", base_url=API_BASE_URL)
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


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: Optional[str] = None
    task_name: Optional[str] = None
    difficulty: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: Optional[str] = None
    action: Optional[Dict[str, Any]] = None
    type: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


def get_llm_action_with_reasoning(
    obs_dict: Dict[str, Any], task_desc: str, week: int, max_weeks: int
) -> Tuple[Action, str]:
    prompt = (
        f"TASK: {task_desc}\nWEEK {week}/{max_weeks}\nSTATE:\n"
        f"{json.dumps(obs_dict, indent=2)}\n\nReturn JSON action with reasoning:"
    )
    raw = ""

    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as api_error:
        error_msg = str(api_error)
        if (
            "api_key" in error_msg.lower()
            or "authentication" in error_msg.lower()
            or "sk-placeholder" in error_msg.lower()
            or "401" in error_msg
        ):
            return (
                Action.from_dict({"type": "wait", "payload": {}}),
                "API key error: the configured OpenAI-compatible provider rejected the request.",
            )
        return Action.from_dict({"type": "wait", "payload": {}}), f"LLM error: {error_msg}"

    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        return Action.from_dict(data), data.get("reasoning", "No reasoning provided.")
    except Exception as parse_error:
        return (
            Action.from_dict({"type": "wait", "payload": {}}),
            f"Parse error: could not decode JSON from LLM: {parse_error}",
        )


def generate_strategic_feedback(history: List[Dict[str, Any]], final_score: float) -> str:
    suggestions = []
    last_obs = history[-1]["obs"]
    last_info = history[-1]["info"]

    if last_info.get("done_reason") == "bankruptcy":
        if last_obs.team.engineers > 3:
            suggestions.append(
                "⚠️ **Over-hiring**: You scaled your team faster than your budget could support early on."
            )
        if last_obs.metrics.revenue < 1000:
            suggestions.append(
                "⚠️ **Slow to Market**: You ran out of cash before generating significant revenue. "
                "Try shipping features faster."
            )

    if last_obs.product.quality < 0.3 and any(h["action"].type == "marketing" for h in history):
        suggestions.append(
            "⚠️ **Premature Marketing**: You spent heavily on marketing while product quality was low. "
            "Focus on R&D (Build Feature) first."
        )

    if last_obs.metrics.revenue > 10000 and last_obs.metrics.user_growth < 5.0:
        suggestions.append(
            "💡 **Scale Opportunity**: You have a strong product but slow growth. Consider more aggressive "
            "marketing or hiring a marketer."
        )

    if not suggestions:
        if final_score > 0.8:
            suggestions.append("🌟 **Excellent Execution**: You balanced growth and survival well.")
        else:
            suggestions.append(
                "💡 **Incremental Gains**: Balance feature development with consistent marketing to avoid "
                "growth plateaus."
            )

    return "\n\n".join(suggestions)


def run_task_streaming(task_name: str):
    task_cfg, grader_fn = get_task_by_name(task_name)
    env = StartupEnv(config=task_cfg["config"])
    obs = env.reset()

    history = []
    done = False
    df_metrics = pd.DataFrame(columns=["Week", "Revenue", "Users"])

    yield (
        obs.budget,
        obs.metrics.revenue,
        obs.metrics.user_growth,
        obs.product.quality,
        df_metrics,
        df_metrics,
        "",
        "",
        "",
        gr.update(visible=False),
    )

    while not done:
        obs_dict = obs.model_dump()
        action, reasoning = get_llm_action_with_reasoning(
            obs_dict, task_cfg["description"], obs.time.current_week, obs.time.max_weeks
        )

        obs, reward, done, info = env.step(action)
        history.append({"obs": obs, "reward": reward, "action": action, "info": info})

        new_row = {
            "Week": obs.time.current_week - 1,
            "Revenue": obs.metrics.revenue,
            "Users": obs.metrics.user_growth,
        }
        df_metrics = pd.concat([df_metrics, pd.DataFrame([new_row])], ignore_index=True)

        event = info.get("event", "none")
        event_banner = ""
        if event != "none":
            color = "#ef4444" if event in ["server_crash", "economic_downturn"] else "#10b981"
            event_banner = (
                "<div style='background:"
                f"{color};color:white;padding:8px;border-radius:5px;text-align:center;'>"
                f"🚀 <strong>EVENT</strong>: {event.replace('_', ' ').upper()}</div>"
            )

        why_str = "\n".join([f"• {reason}" for reason in info.get("why", [])])
        insights = f"**Agent Reasoning:** {reasoning}\n\n**Mechanical Breakdown:**\n{why_str}"

        yield (
            obs.budget,
            obs.metrics.revenue,
            obs.metrics.user_growth,
            obs.product.quality,
            df_metrics,
            df_metrics,
            event_banner,
            insights,
            "",
            gr.update(visible=False),
        )
        time.sleep(0.5)

    score = grader_fn(history)
    feedback = generate_strategic_feedback(history, score)

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
        obs.budget,
        obs.metrics.revenue,
        obs.metrics.user_growth,
        obs.product.quality,
        df_metrics,
        df_metrics,
        "",
        insights,
        breakdown_md,
        gr.update(visible=True),
    )


df_metrics_init = pd.DataFrame(columns=["Week", "Revenue", "Users"])


def build_ui():
    with gr.Blocks(title="Startup Simulator v2") as demo:
        gr.HTML(
            "<h1 style='text-align:center; margin:20px 0;'>🚀 Startup Decision Simulator "
            "<span style='font-size:0.5em; vertical-align:middle; background:#6366f1; color:white; "
            "padding:4px 8px; border-radius:12px;'>v2 PRO</span></h1>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Mission Control")
                task_select = gr.Dropdown(
                    choices=[
                        (f"{task['display_name']} ({task['difficulty'].upper()})", task["name"])
                        for task, _ in ALL_TASKS
                    ],
                    value="mvp_launch",
                    label="Choose Challenge",
                )
                mission_desc = gr.Markdown("Goal: Build an MVP and generate revenue.")
                run_btn = gr.Button("🚀 LAUNCH SIMULATION", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    m_budget = gr.Number(label="Budget ($)", value=0, precision=0, interactive=False)
                    m_revenue = gr.Number(
                        label="Revenue ($/wk)", value=0, precision=0, interactive=False
                    )
                    m_growth = gr.Number(
                        label="User Growth (%)", value=0, precision=1, interactive=False
                    )
                    m_quality = gr.Number(
                        label="Product Quality", value=0, precision=2, interactive=False
                    )

        event_alert = gr.HTML("")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Performance Visuals")
                with gr.Row():
                    chart_revenue = gr.LinePlot(
                        df_metrics_init,
                        x="Week",
                        y="Revenue",
                        title="Revenue Over Time",
                    )
                    chart_users = gr.LinePlot(
                        df_metrics_init,
                        x="Week",
                        y="Users",
                        title="User Growth Over Time",
                    )

            with gr.Column(scale=1):
                gr.Markdown("### 🧠 Step Insights")
                insight_panel = gr.Markdown("Simulation not started...")

        results_panel = gr.Markdown("", visible=False)
        reset_btn = gr.Button("♻️ Reset & Clear", visible=False)

        def update_desc(name: str):
            task, _ = get_task_by_name(name)
            return f"**{task['display_name']}**: {task['description']}"

        task_select.change(fn=update_desc, inputs=task_select, outputs=mission_desc)

        run_btn.click(
            fn=run_task_streaming,
            inputs=task_select,
            outputs=[
                m_budget,
                m_revenue,
                m_growth,
                m_quality,
                chart_revenue,
                chart_users,
                event_alert,
                insight_panel,
                results_panel,
                reset_btn,
            ],
        )

        reset_btn.click(
            lambda: (
                0,
                0,
                0,
                0,
                pd.DataFrame(),
                pd.DataFrame(),
                "",
                "",
                "",
                gr.update(visible=False),
            ),
            outputs=[
                m_budget,
                m_revenue,
                m_growth,
                m_quality,
                chart_revenue,
                chart_users,
                event_alert,
                insight_panel,
                results_panel,
                reset_btn,
            ],
        )

    return demo


def _resolve_task_name(task_name: Optional[str], difficulty: Optional[str]) -> str:
    if task_name:
        return task_name
    if difficulty:
        normalized = difficulty.strip().lower()
        return DIFFICULTY_TO_TASK.get(normalized, normalized)
    return "mvp_launch"


def _reset_session(
    session_id: str,
    task_name: Optional[str] = None,
    difficulty: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    resolved_task_name = _resolve_task_name(task_name, difficulty)

    try:
        task_cfg, _ = get_task_by_name(resolved_task_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    config = copy.deepcopy(task_cfg["config"])
    if seed is not None:
        config["seed"] = seed

    env = StartupEnv(config=config)
    observation = env.reset()

    session = {
        "env": env,
        "task": task_cfg,
        "session_id": session_id,
    }
    _SESSIONS[session_id] = session
    return {
        "session_id": session_id,
        "task_name": task_cfg["name"],
        "difficulty": task_cfg["difficulty"],
        "observation": observation.model_dump(),
        "done": False,
        "info": {"message": "Environment reset."},
    }


def _load_session(session_id: str) -> Dict[str, Any]:
    session = _SESSIONS.get(session_id)
    if session and "env" in session:
        return session
    return _SESSIONS.setdefault(
        session_id,
        {
            "env": StartupEnv(config=get_task_by_name("mvp_launch")[0]["config"]),
            "task": get_task_by_name("mvp_launch")[0],
            "session_id": session_id,
        },
    )


def _ensure_session_reset(session_id: str) -> Dict[str, Any]:
    session = _load_session(session_id)
    env = session["env"]
    if env.state().get("observation") is None:
        env.reset()
    return session


def _action_from_request(request: StepRequest) -> Action:
    if request.action is not None:
        action_data = dict(request.action)
    elif request.type is not None:
        action_data = {"type": request.type, "payload": request.payload}
    else:
        dumped = request.model_dump(exclude_none=True)
        if "type" in dumped:
            action_data = {"type": dumped["type"], "payload": dumped.get("payload", {})}
        else:
            action_data = {"type": "wait", "payload": {}}

    if "payload" not in action_data:
        action_data["payload"] = {}

    try:
        return Action.from_dict(action_data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action payload: {exc}") from exc


def create_app() -> FastAPI:
    api = FastAPI(
        title="Startup Decision Simulator",
        version="1.0.0",
        description="OpenEnv-compatible startup benchmark with a Gradio UI and HTTP API.",
    )

    state_schema = {
        "type": "object",
        "properties": {
            "observation": Observation.model_json_schema(),
            "step_count": {"type": "integer"},
            "invalid_actions": {"type": "integer"},
            "done": {"type": "boolean"},
            "history_length": {"type": "integer"},
        },
        "required": ["observation", "step_count", "invalid_actions", "done", "history_length"],
    }

    @api.get("/health")
    @api.get("/openenv/health")
    def healthcheck():
        return {"status": "healthy"}

    @api.get("/metadata")
    def metadata():
        return {
            "name": "startup-decision-simulator",
            "description": (
                "An OpenEnv-compatible startup benchmark where an agent acts as a founder making "
                "weekly resource allocation decisions."
            ),
            "version": "1.0.0",
        }

    @api.get("/schema")
    def schema():
        return {
            "action": Action.model_json_schema(),
            "observation": Observation.model_json_schema(),
            "state": state_schema,
        }

    @api.post("/mcp")
    def mcp(payload: Dict[str, Any] = Body(default_factory=dict)):
        return {
            "jsonrpc": "2.0",
            "id": payload.get("id"),
            "error": {
                "code": -32601,
                "message": "MCP methods are not implemented for this environment.",
            },
        }

    @api.post("/reset")
    @api.post("/openenv/reset")
    def reset_env(
        request: Optional[ResetRequest] = Body(default=None),
        session_id: Optional[str] = Query(default=None),
        task_name: Optional[str] = Query(default=None),
        difficulty: Optional[str] = Query(default=None),
        seed: Optional[int] = Query(default=None),
    ):
        payload = request or ResetRequest()
        resolved_session_id = payload.session_id or session_id or "default"
        resolved_task_name = payload.task_name or task_name
        resolved_difficulty = payload.difficulty or difficulty
        resolved_seed = payload.seed if payload.seed is not None else seed

        with _SESSION_LOCK:
            return _reset_session(
                session_id=resolved_session_id,
                task_name=resolved_task_name,
                difficulty=resolved_difficulty,
                seed=resolved_seed,
            )

    @api.post("/step")
    @api.post("/openenv/step")
    def step_env(request: Optional[StepRequest] = Body(default=None)):
        payload = request or StepRequest()
        session_id = payload.session_id or "default"
        action = _action_from_request(payload)

        with _SESSION_LOCK:
            session = _ensure_session_reset(session_id)
            env = session["env"]

            try:
                observation, reward, done, info = env.step(action)
            except RuntimeError:
                reset_response = _reset_session(session_id=session_id, task_name=session["task"]["name"])
                env = _SESSIONS[session_id]["env"]
                observation, reward, done, info = env.step(action)
                info = {
                    **info,
                    "auto_reset": True,
                    "reset_observation": reset_response["observation"],
                }

            return {
                "session_id": session_id,
                "task_name": session["task"]["name"],
                "difficulty": session["task"]["difficulty"],
                "observation": observation.model_dump(),
                "reward": reward.model_dump(),
                "done": done,
                "info": info,
            }

    @api.get("/state")
    @api.get("/openenv/state")
    def get_state(session_id: str = Query(default="default")):
        with _SESSION_LOCK:
            session = _ensure_session_reset(session_id)
            env = session["env"]
            return {
                "session_id": session_id,
                "task_name": session["task"]["name"],
                "difficulty": session["task"]["difficulty"],
                "state": env.state(),
            }

    demo = build_ui()
    return gr.mount_gradio_app(api, demo, path="/")


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
