"""
Strict baseline inference runner for the Startup Decision Simulator.

The script emits only the required [START], [STEP], and [END] log lines.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from environment.models import Action
from environment.startup_env import StartupEnv
from environment.tasks import ALL_TASKS


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK_NAME = "startup-decision-simulator"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "160"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
MODEL_RETRIES = int(os.getenv("MODEL_RETRIES", "2"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))
MAX_STEPS_OVERRIDE = int(os.getenv("MAX_STEPS_PER_TASK", "0"))

SYSTEM_PROMPT = """You are controlling a startup simulation.
Return exactly one JSON object with keys "type" and "payload".

Valid actions:
- {"type":"hire","payload":{"role":"engineer"|"designer"|"marketer"}}
- {"type":"fire","payload":{"role":"engineer"|"designer"|"marketer"}}
- {"type":"build_feature","payload":{"feature_name":"short_name"}}
- {"type":"marketing","payload":{"budget":500_or_more}}
- {"type":"pivot","payload":{"new_trend":"AI/ML"|"sustainability"|"consumer_health"|"fintech"|"enterprise_saas"|"creator_economy"|"web3"|"edtech"|"stable"}}
- {"type":"wait","payload":{}}

Choose legal, budget-aware actions only."""


def get_client() -> Optional[OpenAI]:
    if not API_KEY:
        return None
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def serialize_action(action: Action) -> str:
    return json.dumps(
        {"type": action.type, "payload": action.payload.model_dump(exclude_none=True)},
        separators=(",", ":"),
        ensure_ascii=True,
    )


def build_user_prompt(
    task_description: str,
    obs_dict: Dict[str, Any],
    step: int,
    max_steps: int,
) -> str:
    return (
        f"Task: {task_description}\n"
        f"Step: {step}/{max_steps}\n"
        f"Observation: {json.dumps(obs_dict, separators=(',', ':'))}\n"
        "Return the next action as JSON."
    )


def heuristic_action(task_name: str, obs_dict: Dict[str, Any]) -> Action:
    budget = float(obs_dict["budget"])
    team = obs_dict["team"]
    product = obs_dict["product"]
    market = obs_dict["market"]
    metrics = obs_dict["metrics"]
    week = int(obs_dict["time"]["current_week"])
    features = list(product["features_built"])
    quality = float(product["quality"])

    if not features and budget >= 8_000 and team["engineers"] >= 1:
        return Action.from_dict(
            {"type": "build_feature", "payload": {"feature_name": f"mvp_week_{week}"}}
        )

    if task_name == "survival_mode" and budget < 20_000 and team["marketers"] > 0:
        return Action.from_dict({"type": "fire", "payload": {"role": "marketer"}})

    if quality < 0.35 and budget >= 8_000 and team["engineers"] >= 1 and len(features) < 3:
        return Action.from_dict(
            {"type": "build_feature", "payload": {"feature_name": f"feature_{week}"}}
        )

    if market["competition"] > 0.75 and market["trend"] != "enterprise_saas":
        return Action.from_dict(
            {"type": "pivot", "payload": {"new_trend": "enterprise_saas"}}
        )

    if (
        team["marketers"] == 0
        and len(features) >= 2
        and budget >= 7_200
        and task_name in {"growth_phase", "survival_mode"}
    ):
        return Action.from_dict({"type": "hire", "payload": {"role": "marketer"}})

    if team["engineers"] < 3 and budget >= 8_000 and quality < 0.45 and task_name != "survival_mode":
        return Action.from_dict({"type": "hire", "payload": {"role": "engineer"}})

    if features and budget >= 500:
        if task_name == "mvp_launch" and metrics["revenue"] < 5_000:
            return Action.from_dict({"type": "marketing", "payload": {"budget": 1_000}})
        if task_name == "growth_phase" and metrics["user_growth"] < 15.0 and budget >= 2_500:
            return Action.from_dict({"type": "marketing", "payload": {"budget": 2_500}})
        if task_name == "survival_mode" and metrics["revenue"] < 6_000 and budget >= 750:
            return Action.from_dict({"type": "marketing", "payload": {"budget": 750}})

    return Action.from_dict({"type": "wait", "payload": {}})


def parse_model_action(raw_text: str) -> Action:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return Action.from_dict(json.loads(cleaned))


def get_model_action(
    client: Optional[OpenAI],
    task_name: str,
    task_description: str,
    obs_dict: Dict[str, Any],
    step: int,
    max_steps: int,
) -> Action:
    if client is None:
        return heuristic_action(task_name, obs_dict)

    user_prompt = build_user_prompt(task_description, obs_dict, step, max_steps)

    for _ in range(MODEL_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return parse_model_action(text)
        except Exception:
            continue

    return heuristic_action(task_name, obs_dict)


def run_task(task_cfg: Dict[str, Any], grader_fn, client: Optional[OpenAI]) -> None:
    task_name = task_cfg["name"]
    task_description = task_cfg["description"]
    config = copy.deepcopy(task_cfg["config"])
    max_steps = MAX_STEPS_OVERRIDE if MAX_STEPS_OVERRIDE > 0 else int(config["max_weeks"])

    env = StartupEnv(config=config)
    history: List[Dict[str, Any]] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, max_steps + 1):
            obs_dict = obs.model_dump()
            action = get_model_action(
                client=client,
                task_name=task_name,
                task_description=task_description,
                obs_dict=obs_dict,
                step=step,
                max_steps=max_steps,
            )

            obs, reward, done, info = env.step(action)
            reward_value = float(reward.total)
            error = info.get("action_message") if not info.get("action_valid", True) else None

            history.append({"obs": obs, "reward": reward, "action": action, "info": info})
            rewards.append(reward_value)
            steps_taken = step

            log_step(
                step=step,
                action=serialize_action(action),
                reward=reward_value,
                done=done,
                error=error,
            )

            if done:
                break

        score = clamp_score(float(grader_fn(history)))
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = get_client()
    for task_cfg, grader_fn in ALL_TASKS:
        run_task(task_cfg=task_cfg, grader_fn=grader_fn, client=client)


if __name__ == "__main__":
    main()
