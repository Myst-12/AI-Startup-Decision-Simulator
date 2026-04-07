"""
inference.py — Agent inference script for Startup Decision Simulator.

Reads environment variables (in priority order):
  GROQ_API_KEY  : Groq API key (preferred)
  HF_TOKEN      : Fallback API key
  API_BASE_URL  : LLM endpoint base URL (defaults to Groq)
  MODEL_NAME    : LLM model identifier (defaults to llama-3.3-70b-versatile)

Runs an LLM-powered agent against all three tasks and prints results
in the required logging format: [START], [STEP], [END].
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Load .env file if present (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can be set manually

from openai import OpenAI

from environment.startup_env import StartupEnv
from environment.models import Action
from environment.tasks import ALL_TASKS

# ---------------------------------------------------------------------------
# Configuration — Groq by default, OpenAI-compatible endpoint supported
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
# Key priority: GROQ_API_KEY → HF_TOKEN → OPENAI_API_KEY
GROQ_API_KEY: str = (
    os.environ.get("GROQ_API_KEY")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)
MAX_STEPS_PER_TASK: int = 50   # guard against runaway episodes
TEMPERATURE: float = 0.2       # low temperature for reproducibility

# ---------------------------------------------------------------------------
# OpenAI client initialisation
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=GROQ_API_KEY or "sk-placeholder",
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an experienced startup founder and CEO making strategic weekly decisions.

You receive the current state of your startup as a JSON object and must return exactly ONE action as JSON.

RULES:
1. Your response must be ONLY a valid JSON object with keys "type" and "payload".
2. Do NOT include explanations or markdown formatting — only the raw JSON object.
3. Choose the action that maximises long-term success, not just short-term gains.

AVAILABLE ACTIONS:
- hire: {"type": "hire", "payload": {"role": "engineer"|"designer"|"marketer"}}
- fire: {"type": "fire", "payload": {"role": "engineer"|"designer"|"marketer"}}
- build_feature: {"type": "build_feature", "payload": {"feature_name": "<unique name>"}}
- marketing: {"type": "marketing", "payload": {"budget": <float >= 500.0>}}
- pivot: {"type": "pivot", "payload": {"new_trend": "AI/ML"|"sustainability"|"consumer_health"|"fintech"|"enterprise_saas"|"creator_economy"|"web3"|"edtech"|"stable"}}
- wait: {"type": "wait", "payload": {}}

STRATEGY GUIDELINES:
- Build features before spending on marketing (product must exist first).
- Maintain at least 2 engineers to keep building.
- Watch your burn rate vs revenue — efficient spending is rewarded.
- Respond to events: viral_growth → invest in marketing; server_crash → improve quality.
- Pivot when the current trend is misaligned with your product's strengths.
"""


def build_user_prompt(obs_dict: Dict[str, Any], task_description: str, week: int, max_weeks: int) -> str:
    return f"""TASK: {task_description}

WEEK {week}/{max_weeks}

CURRENT STATE:
{json.dumps(obs_dict, indent=2)}

Based on this state, what is your action for this week? Return ONLY the JSON action object."""


# ---------------------------------------------------------------------------
# LLM action generation
# ---------------------------------------------------------------------------

def get_llm_action(
    obs_dict: Dict[str, Any],
    task_description: str,
    week: int,
    max_weeks: int,
    retries: int = 3,
) -> Optional[Action]:
    """
    Call the LLM to get an action. Returns None if all retries fail.
    Falls back to a safe DEFAULT action on parse failures.
    """
    user_prompt = build_user_prompt(obs_dict, task_description, week, max_weeks)
    last_error: Optional[Exception] = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            action_dict = json.loads(raw)
            return Action.from_dict(action_dict)

        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)  # exponential backoff

    print(f"  [WARN] LLM failed after {retries} attempts: {last_error}. Using fallback action.")
    return _fallback_action(obs_dict)


def _fallback_action(obs_dict: Dict[str, Any]) -> Action:
    """
    Deterministic fallback action when the LLM fails.
    Prefers building a feature if budget allows, else waits.
    """
    budget = obs_dict.get("budget", 0)
    features = obs_dict.get("product", {}).get("features_built", [])
    engineers = obs_dict.get("team", {}).get("engineers", 0)

    if engineers >= 1 and budget >= 8_000 and len(features) < 5:
        feature_name = f"feature_v{len(features) + 1}"
        return Action.from_dict({"type": "build_feature", "payload": {"feature_name": feature_name}})

    return Action.from_dict({"type": "wait", "payload": {}})


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------

def run_task(task_cfg: Dict[str, Any], grader_fn, step_limit: int = MAX_STEPS_PER_TASK) -> Dict[str, Any]:
    """
    Run one full episode for the given task. Logs in required format.
    Returns result dict with final score and stats.
    """
    task_name = task_cfg["display_name"]
    task_description = task_cfg["description"]
    config = task_cfg["config"]
    difficulty = task_cfg["difficulty"]

    print(f"\n[START] task={task_name!r} difficulty={difficulty!r} model={MODEL_NAME!r}")
    print(f"[START] config={json.dumps({k: v for k, v in config.items() if k != 'seed'}, separators=(',', ':'))}")

    env = StartupEnv(config=config)
    obs = env.reset()

    history: List[Dict[str, Any]] = []
    total_reward = 0.0
    done = False
    step = 0

    while not done and step < step_limit:
        step += 1
        obs_dict = obs.model_dump()
        action = get_llm_action(
            obs_dict=obs_dict,
            task_description=task_description,
            week=obs.time.current_week,
            max_weeks=obs.time.max_weeks,
        )

        if action is None:
            action = Action.from_dict({"type": "wait", "payload": {}})

        obs, reward, done, info = env.step(action)
        total_reward += reward.total
        history.append({"obs": obs, "reward": reward, "action": action, "info": info})

        events_str = ", ".join(obs.pending_events) if obs.pending_events else "none"
        print(
            f"[STEP] week={obs.time.current_week - 1:02d} "
            f"action={action.type!r} "
            f"reward={reward.total:.4f} "
            f"budget={obs.budget:.0f} "
            f"revenue={obs.metrics.revenue:.0f} "
            f"user_growth={obs.metrics.user_growth:.1f}% "
            f"quality={obs.product.quality:.2f} "
            f"event={events_str!r} "
            f"done={done}"
        )

    # Compute grader score from history
    grader_score = grader_fn(history)
    avg_reward = total_reward / max(step, 1)
    final_obs = history[-1]["obs"] if history else obs

    result = {
        "task": task_name,
        "difficulty": difficulty,
        "grader_score": grader_score,
        "avg_reward_per_step": round(avg_reward, 4),
        "total_steps": step,
        "done_reason": history[-1]["info"].get("done_reason") if history else "unknown",
        "final_budget": round(final_obs.budget, 2),
        "final_revenue": round(final_obs.metrics.revenue, 2),
        "final_user_growth": round(final_obs.metrics.user_growth, 2),
        "final_quality": round(final_obs.product.quality, 4),
        "features_built": final_obs.product.features_built,
    }

    print(
        f"[END] task={task_name!r} "
        f"grader_score={grader_score:.4f} "
        f"avg_reward={avg_reward:.4f} "
        f"steps={step} "
        f"done_reason={result['done_reason']!r}"
    )
    print(f"[END] final_state={json.dumps({k: result[k] for k in ('final_budget', 'final_revenue', 'final_user_growth', 'final_quality', 'features_built')}, separators=(',', ':'))}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  Startup Decision Simulator — Inference Run")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Endpoint: {API_BASE_URL}")
    print("=" * 70)

    if not GROQ_API_KEY:
        print("[WARN] GROQ_API_KEY not set. Requests will fail.")
        print("[WARN] Set it via: export GROQ_API_KEY=gsk_...")

    all_results: List[Dict[str, Any]] = []
    start_time = time.time()

    for task_cfg, grader_fn in ALL_TASKS:
        try:
            result = run_task(task_cfg, grader_fn)
            all_results.append(result)
        except Exception as exc:
            print(f"[ERROR] Task {task_cfg['display_name']} failed: {exc}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time

    # Summary table
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Task':<22} {'Difficulty':<10} {'Score':>8} {'Avg Reward':>12} {'Steps':>7}")
    print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*12} {'-'*7}")
    for r in all_results:
        print(
            f"  {r['task']:<22} {r['difficulty']:<10} "
            f"{r['grader_score']:>8.4f} {r['avg_reward_per_step']:>12.4f} {r['total_steps']:>7}"
        )
    if all_results:
        mean_score = sum(r["grader_score"] for r in all_results) / len(all_results)
        print(f"\n  Mean Grader Score : {mean_score:.4f}")
    print(f"  Total Runtime     : {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
