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

# ---------------------------------------------------------------------------
# Task 1 — EASY: "MVP Launch"
# ---------------------------------------------------------------------------

TASK_MVP_LAUNCH: Dict[str, Any] = {
    "name": "mvp_launch",
    "display_name": "MVP Launch",
    "difficulty": "easy",
    "description": (
        "You are a solo founder with a modest seed round. Goal: Build an MVP, ship features, "
        "and hit $5k revenue without going bankrupt. Competition is low."
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
    if not history: return 0.0
    
    # Components
    survival_val = 1.0 if _survived(history) else 0.0
    
    # Target: $5,000 revenue
    rev_score = _normalize(_max_revenue(history), 0.0, 5_000.0)
    
    # Target: 8% natural user growth (easy)
    ug_score = _normalize(_max_user_growth(history), 0.0, 8.0)
    
    # Formula: 0.4*Surv + 0.3*Rev + 0.3*Growth
    raw = 0.4 * survival_val + 0.3 * rev_score + 0.3 * ug_score
    return round(min(1.0, raw), 4)


# ---------------------------------------------------------------------------
# Task 2 — MEDIUM: "Growth Phase"
# ---------------------------------------------------------------------------

TASK_GROWTH_PHASE: Dict[str, Any] = {
    "name": "growth_phase",
    "display_name": "Growth Phase",
    "difficulty": "medium",
    "description": (
        "Launched product with traction. Goal: Scale users to 15% growth and $20k revenue. "
        "Competition is moderate. Don't run out of cash!"
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
        "initial_user_growth": 4.0,
        "initial_burn_rate": 0.0,
        "initial_features": ["core_dashboard"],
        "max_weeks": 36,
        "seed": 123,
        "task_name": "growth_phase",
    },
}


def grade_growth_phase(history: List[Dict[str, Any]]) -> float:
    if not history: return 0.0

    survival_val = 1.0 if _survived(history) else 0.0
    
    # Target: $20,000 revenue
    rev_score = _normalize(_max_revenue(history), 5_000.0, 20_000.0)
    
    # Target: 15% user growth
    ug_score = _normalize(_max_user_growth(history), 4.0, 15.0)

    raw = 0.4 * survival_val + 0.3 * rev_score + 0.3 * ug_score
    return round(min(1.0, raw), 4)


# ---------------------------------------------------------------------------
# Task 3 — HARD: "Survival Mode"
# ---------------------------------------------------------------------------

TASK_SURVIVAL_MODE: Dict[str, Any] = {
    "name": "survival_mode",
    "display_name": "Survival Mode",
    "difficulty": "hard",
    "description": (
        "Crisis mode. Budget low, competition high. Goal: Survive 48 weeks and "
        "rebuild revenue to $10k. Frequent disruptive events."
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
    if not history: return 0.0

    survival_val = 1.0 if _survived(history) else 0.0
    
    # Target: $10,000 revenue (from $2k)
    rev_score = _normalize(_max_revenue(history), 2_000.0, 10_000.0)
    
    # Target: 10% user growth
    ug_score = _normalize(_max_user_growth(history), 2.0, 10.0)

    raw = 0.4 * survival_val + 0.3 * rev_score + 0.3 * ug_score
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
