"""
Task definitions and graders for the Startup Decision Simulator.

Each task returns:
  - A config dict passed to StartupEnv()
  - A grader function: (history: List[Dict]) -> float in [0.0, 1.0]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math


# ---------------------------------------------------------------------------
# Helper utilities for graders
# ---------------------------------------------------------------------------

def _normalize(value: float, lo: float, hi: float) -> float:
    """Linearly normalise value to [0, 1] given expected [lo, hi]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _last_obs(history: List[Dict[str, Any]]) -> Optional[Any]:
    """Return the observation from the last step."""
    if not history:
        return None
    return history[-1]["obs"]


def _max_revenue(history: List[Dict[str, Any]]) -> float:
    return max((h["obs"].metrics.revenue for h in history), default=0.0)


def _max_user_growth(history: List[Dict[str, Any]]) -> float:
    return max((h["obs"].metrics.user_growth for h in history), default=0.0)


def _survived(history: List[Dict[str, Any]]) -> bool:
    obs = _last_obs(history)
    if obs is None:
        return False
    return obs.budget > 0.0


# ---------------------------------------------------------------------------
# Task 1 — EASY: "MVP Launch"
# ---------------------------------------------------------------------------

TASK_MVP_LAUNCH: Dict[str, Any] = {
    "name": "mvp_launch",
    "display_name": "MVP Launch",
    "difficulty": "easy",
    "description": (
        "You are a solo founder with a modest seed round. Your goal is to build a "
        "minimum viable product, ship at least one feature, and generate initial revenue "
        "before your budget runs out. Competition is low and the market is receptive."
    ),
    "config": {
        "initial_budget": 120_000.0,
        "initial_engineers": 2,
        "initial_designers": 1,
        "initial_marketers": 0,
        "initial_demand": 0.65,
        "initial_competition": 0.15,
        "initial_trend": "stable",
        "initial_quality": 0.0,
        "initial_revenue": 0.0,
        "initial_user_growth": 0.0,
        "initial_burn_rate": 0.0,
        "initial_features": [],
        "max_weeks": 20,
        "seed": 42,
        "task_name": "mvp_launch",
    },
}


def grade_mvp_launch(history: List[Dict[str, Any]]) -> float:
    """
    Grader for MVP Launch.

    score = 1.0 if:
      - Product has at least 1 feature shipped
      - Revenue > $3,000/week at any point in the episode

    Partial credit for partial completion.
    """
    if not history:
        return 0.0

    obs = _last_obs(history)
    max_rev = _max_revenue(history)

    product_score = 1.0 if obs.product.features_built else 0.0
    revenue_threshold = 3_000.0
    revenue_score = _normalize(max_rev, 0.0, revenue_threshold)

    # Binary bonus if both conditions simultaneously held at some point
    joint_bonus = 0.0
    for step in history:
        o = step["obs"]
        if o.product.features_built and o.metrics.revenue >= revenue_threshold:
            joint_bonus = 0.2
            break

    raw = 0.4 * product_score + 0.4 * revenue_score + joint_bonus
    return round(min(1.0, raw), 4)


# ---------------------------------------------------------------------------
# Task 2 — MEDIUM: "Growth Phase"
# ---------------------------------------------------------------------------

TASK_GROWTH_PHASE: Dict[str, Any] = {
    "name": "growth_phase",
    "display_name": "Growth Phase",
    "difficulty": "medium",
    "description": (
        "You have a launched product with a small but active user base. Your goal is to "
        "aggressively grow users and revenue over 36 weeks. Competition is moderate and "
        "increasing. Hiring, marketing, and smart feature development are key."
    ),
    "config": {
        "initial_budget": 350_000.0,
        "initial_engineers": 3,
        "initial_designers": 2,
        "initial_marketers": 1,
        "initial_demand": 0.55,
        "initial_competition": 0.40,
        "initial_trend": "enterprise_saas",
        "initial_quality": 0.30,
        "initial_revenue": 5_000.0,
        "initial_user_growth": 5.0,
        "initial_burn_rate": 0.0,
        "initial_features": ["core_dashboard"],
        "max_weeks": 36,
        "seed": 123,
        "task_name": "growth_phase",
    },
}


def grade_growth_phase(history: List[Dict[str, Any]]) -> float:
    """
    Grader for Growth Phase.

    score = normalised(user_growth_peak + revenue_peak)

    Targets: user_growth_peak >= 20%/week, revenue_peak >= $20,000/week
    """
    if not history:
        return 0.0

    max_rev = _max_revenue(history)
    max_ug = _max_user_growth(history)

    # Number of features built (measures product investment)
    obs = _last_obs(history)
    feature_count = len(obs.product.features_built)

    rev_score = _normalize(max_rev, 5_000.0, 20_000.0)          # 0 at $5k, 1 at $20k
    ug_score = _normalize(max_ug, 5.0, 20.0)                    # 0 at 5%, 1 at 20%
    feat_score = _normalize(feature_count, 1.0, 6.0)             # 0 at 1, 1 at 6
    survival = 1.0 if _survived(history) else 0.5

    raw = 0.35 * rev_score + 0.35 * ug_score + 0.20 * feat_score + 0.10 * survival
    return round(min(1.0, raw), 4)


# ---------------------------------------------------------------------------
# Task 3 — HARD: "Survival Mode"
# ---------------------------------------------------------------------------

TASK_SURVIVAL_MODE: Dict[str, Any] = {
    "name": "survival_mode",
    "display_name": "Survival Mode",
    "difficulty": "hard",
    "description": (
        "Your startup is in crisis. Budget is critically low, competition is fierce, "
        "and disruptive market events happen frequently. You must survive, maintain "
        "revenue, and grow users under extreme adversity over 48 weeks. Every decision counts."
    ),
    "config": {
        "initial_budget": 80_000.0,
        "initial_engineers": 2,
        "initial_designers": 1,
        "initial_marketers": 0,
        "initial_demand": 0.40,
        "initial_competition": 0.70,
        "initial_trend": "AI/ML",
        "initial_quality": 0.20,
        "initial_revenue": 2_000.0,
        "initial_user_growth": 2.0,
        "initial_burn_rate": 0.0,
        "initial_features": ["v1_core"],
        "max_weeks": 48,
        "seed": 999,
        "task_name": "survival_mode",
    },
}


def grade_survival_mode(history: List[Dict[str, Any]]) -> float:
    """
    Grader for Survival Mode (weighted multi-objective).

    Components:
      - survival (not bankrupt at end): 30%
      - revenue maintained / grown: 30%
      - user growth maintained: 25%
      - weeks survived: 15%

    All normalised to [0, 1].
    """
    if not history:
        return 0.0

    obs = _last_obs(history)
    weeks_survived = len(history)
    max_weeks = obs.time.max_weeks

    survival_score = 1.0 if _survived(history) else 0.0
    rev_score = _normalize(_max_revenue(history), 2_000.0, 15_000.0)
    ug_score = _normalize(_max_user_growth(history), 2.0, 15.0)
    duration_score = _normalize(weeks_survived, 10, max_weeks)

    raw = (
        0.30 * survival_score
        + 0.30 * rev_score
        + 0.25 * ug_score
        + 0.15 * duration_score
    )
    return round(min(1.0, raw), 4)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

ALL_TASKS = [
    (TASK_MVP_LAUNCH, grade_mvp_launch),
    (TASK_GROWTH_PHASE, grade_growth_phase),
    (TASK_SURVIVAL_MODE, grade_survival_mode),
]


def get_task_by_name(name: str):
    """Retrieve (task_dict, grader_fn) by task name. Raises KeyError if not found."""
    for task, grader in ALL_TASKS:
        if task["name"] == name:
            return task, grader
    raise KeyError(f"Unknown task: '{name}'. Available: {[t['name'] for t, _ in ALL_TASKS]}")
