"""
Core Startup Decision Simulator environment.
Implements the full OpenEnv interface: reset(), step(), state().
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action,
    BusinessMetrics,
    BuildFeaturePayload,
    FirePayload,
    HirePayload,
    Market,
    MarketingPayload,
    Observation,
    PivotPayload,
    Product,
    Reward,
    Team,
    TimeInfo,
    WaitPayload,
)

# ---------------------------------------------------------------------------
# Salary / cost constants (USD per week)
# ---------------------------------------------------------------------------
SALARY_ENGINEER: float = 3_000.0
SALARY_DESIGNER: float = 2_500.0
SALARY_MARKETER: float = 2_200.0
HIRING_COST: float = 5_000.0          # one-time cost per new hire
FEATURE_BUILD_COST: float = 8_000.0   # engineering sprint cost
FEATURE_QUALITY_GAIN: float = 0.08    # quality bump per feature shipped

# ---------------------------------------------------------------------------
# Market trend catalogue
# ---------------------------------------------------------------------------
MARKET_TRENDS: List[str] = [
    "AI/ML",
    "sustainability",
    "consumer_health",
    "fintech",
    "enterprise_saas",
    "creator_economy",
    "web3",
    "edtech",
    "stable",
]

# ---------------------------------------------------------------------------
# Stochastic events with probability weights
# ---------------------------------------------------------------------------
EVENTS: List[Tuple[str, float]] = [
    ("none", 0.55),
    ("competitor_launch", 0.10),
    ("viral_growth", 0.08),
    ("server_crash", 0.07),
    ("economic_downturn", 0.05),
    ("press_coverage", 0.08),
    ("key_employee_left", 0.07),
]

# Event effect callbacks are applied *after* normal step logic
EVENT_DEMAND_DELTA: Dict[str, float] = {
    "none": 0.0,
    "competitor_launch": -0.07,
    "viral_growth": +0.12,
    "server_crash": -0.03,
    "economic_downturn": -0.05,
    "press_coverage": +0.08,
    "key_employee_left": -0.02,
}

EVENT_REVENUE_FACTOR: Dict[str, float] = {
    "none": 1.0,
    "competitor_launch": 0.85,
    "viral_growth": 1.40,
    "server_crash": 0.70,
    "economic_downturn": 0.80,
    "press_coverage": 1.20,
    "key_employee_left": 0.95,
}

EVENT_USER_GROWTH_DELTA: Dict[str, float] = {
    "none": 0.0,
    "competitor_launch": -2.0,
    "viral_growth": +15.0,
    "server_crash": -5.0,
    "economic_downturn": -2.0,
    "press_coverage": +10.0,
    "key_employee_left": -1.0,
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _pick_event(rng: random.Random) -> str:
    events, weights = zip(*EVENTS)
    return rng.choices(events, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# StartupEnv
# ---------------------------------------------------------------------------

class StartupEnv:
    """
    OpenEnv-compatible startup simulation environment.

    Usage
    -----
    env = StartupEnv(config)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    full_state = env.state()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.max_weeks: int = cfg.get("max_weeks", 52)
        self.initial_budget: float = cfg.get("initial_budget", 500_000.0)
        self.seed: Optional[int] = cfg.get("seed", None)
        self.task_name: str = cfg.get("task_name", "default")

        # Market starting conditions
        self._init_demand: float = cfg.get("initial_demand", 0.5)
        self._init_competition: float = cfg.get("initial_competition", 0.3)
        self._init_trend: str = cfg.get("initial_trend", "stable")

        # Team starting conditions
        self._init_engineers: int = cfg.get("initial_engineers", 2)
        self._init_designers: int = cfg.get("initial_designers", 1)
        self._init_marketers: int = cfg.get("initial_marketers", 0)

        # Pre-built features (for MEDIUM/HARD tasks)
        self._init_features: List[str] = cfg.get("initial_features", [])
        self._init_quality: float = cfg.get("initial_quality", 0.0)
        self._init_revenue: float = cfg.get("initial_revenue", 0.0)
        self._init_user_growth: float = cfg.get("initial_user_growth", 0.0)
        self._init_burn_rate: float = cfg.get("initial_burn_rate", 0.0)

        # Internal mutable state (initialised in reset)
        self._obs: Observation = None  # type: ignore[assignment]
        self._rng: random.Random = random.Random(self.seed)
        self._history: List[Dict[str, Any]] = []
        self._done: bool = False
        self._step_count: int = 0
        self._invalid_actions: int = 0

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment to its initial state and return the first observation."""
        self._rng = random.Random(self.seed)
        self._history = []
        self._done = False
        self._step_count = 0
        self._invalid_actions = 0

        self._obs = Observation(
            budget=self.initial_budget,
            team=Team(
                engineers=self._init_engineers,
                designers=self._init_designers,
                marketers=self._init_marketers,
            ),
            product=Product(
                features_built=list(self._init_features),
                quality=self._init_quality,
            ),
            market=Market(
                demand=self._init_demand,
                competition=self._init_competition,
                trend=self._init_trend,
            ),
            metrics=BusinessMetrics(
                revenue=self._init_revenue,
                burn_rate=self._init_burn_rate,
                user_growth=self._init_user_growth,
            ),
            time=TimeInfo(current_week=1, max_weeks=self.max_weeks),
            pending_events=[],
        )
        return copy.deepcopy(self._obs)

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Advance the simulation by one week.

        Parameters
        ----------
        action : Action
            The agent's chosen action for this week.

        Returns
        -------
        observation : Observation
        reward      : Reward
        done        : bool
        info        : dict
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() before stepping.")

        self._step_count += 1
        info: Dict[str, Any] = {"week": self._obs.time.current_week, "action": action.dict()}

        # 1. Validate action → apply effects
        action_valid, action_msg = self._validate_action(action)
        info["action_valid"] = action_valid
        info["action_message"] = action_msg

        penalty = 0.0
        if not action_valid:
            self._invalid_actions += 1
            penalty = 0.15  # penalise invalid choices

        if action_valid:
            self._apply_action(action)

        # 2. Compute weekly burn rate
        burn = self._compute_burn_rate()
        self._obs.metrics.burn_rate = burn

        # 3. Simulate business dynamics
        self._simulate_dynamics()

        # 4. Inject stochastic event
        event = _pick_event(self._rng)
        self._apply_event(event)
        self._obs.pending_events = [event] if event != "none" else []
        info["event"] = event

        # 5. Deduct burn from budget
        self._obs.budget = max(0.0, self._obs.budget - burn)

        # 6. Advance time
        self._obs.time.current_week += 1

        # 7. Compute reward
        reward = self._compute_reward(penalty)

        # 8. Check termination
        done = self._check_done()
        self._done = done
        info["done_reason"] = self._done_reason() if done else None

        obs_snapshot = copy.deepcopy(self._obs)
        self._history.append({"obs": obs_snapshot, "reward": reward, "action": action, "info": info})

        return obs_snapshot, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return the full internal state as a dict (for debugging / logging)."""
        return {
            "observation": self._obs.dict() if self._obs else None,
            "step_count": self._step_count,
            "invalid_actions": self._invalid_actions,
            "done": self._done,
            "history_length": len(self._history),
        }

    # ------------------------------------------------------------------
    # Action validation
    # ------------------------------------------------------------------

    def _validate_action(self, action: Action) -> Tuple[bool, str]:
        obs = self._obs
        pld = action.payload

        if action.type == "hire":
            cost = HIRING_COST + self._weekly_salary_for_role(pld.role)
            if obs.budget < cost:
                return False, f"Insufficient budget to hire {pld.role} (need ${cost:.0f}, have ${obs.budget:.0f})"
            return True, "ok"

        if action.type == "fire":
            count = getattr(obs.team, f"{pld.role}s")
            if count <= 0:
                return False, f"No {pld.role}s to fire"
            return True, "ok"

        if action.type == "build_feature":
            if obs.team.engineers < 1:
                return False, "Need at least 1 engineer to build a feature"
            if obs.budget < FEATURE_BUILD_COST:
                return False, f"Insufficient budget to build feature (need ${FEATURE_BUILD_COST:.0f})"
            if pld.feature_name in obs.product.features_built:
                return False, f"Feature '{pld.feature_name}' already built"
            return True, "ok"

        if action.type == "marketing":
            if obs.budget < pld.budget:
                return False, f"Insufficient budget for marketing spend (need ${pld.budget:.0f})"
            if obs.product.features_built == []:
                return False, "Cannot run marketing campaigns before launching a product"
            return True, "ok"

        if action.type == "pivot":
            if pld.new_trend not in MARKET_TRENDS:
                return False, f"Unknown market trend '{pld.new_trend}'. Valid: {MARKET_TRENDS}"
            if pld.new_trend == obs.market.trend:
                return False, "Already positioned in this trend"
            return True, "ok"

        if action.type == "wait":
            return True, "ok"

        return False, f"Unknown action type '{action.type}'"

    # ------------------------------------------------------------------
    # Action effects
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action) -> None:
        obs = self._obs
        pld = action.payload

        if action.type == "hire":
            role: str = pld.role
            setattr(obs.team, f"{role}s", getattr(obs.team, f"{role}s") + 1)
            obs.budget -= HIRING_COST  # one-time hiring cost

        elif action.type == "fire":
            role = pld.role
            current = getattr(obs.team, f"{role}s")
            setattr(obs.team, f"{role}s", max(0, current - 1))

        elif action.type == "build_feature":
            obs.product.features_built.append(pld.feature_name)
            obs.product.quality = _clamp(obs.product.quality + FEATURE_QUALITY_GAIN, 0.0, 1.0)
            obs.budget -= FEATURE_BUILD_COST
            # Designer bonus
            if obs.team.designers > 0:
                obs.product.quality = _clamp(obs.product.quality + 0.02 * obs.team.designers, 0.0, 1.0)

        elif action.type == "marketing":
            spend: float = pld.budget
            obs.budget -= spend
            # Marketing lifts demand and user growth proportional to spend
            demand_lift = _clamp(spend / 50_000.0 * 0.15, 0.0, 0.15)
            obs.market.demand = _clamp(obs.market.demand + demand_lift, 0.0, 1.0)
            growth_lift = (spend / 10_000.0) * 3.0
            if obs.team.marketers > 0:
                growth_lift *= (1.0 + 0.2 * obs.team.marketers)
            obs.metrics.user_growth = max(0.0, obs.metrics.user_growth + growth_lift)

        elif action.type == "pivot":
            obs.market.trend = pld.new_trend
            # Pivot costs some quality (distraction) but can unlock higher demand
            obs.product.quality = _clamp(obs.product.quality - 0.05, 0.0, 1.0)
            obs.market.demand = _clamp(obs.market.demand + 0.10, 0.0, 1.0)

        elif action.type == "wait":
            pass  # intentional no-op

    # ------------------------------------------------------------------
    # Business dynamics
    # ------------------------------------------------------------------

    def _simulate_dynamics(self) -> None:
        obs = self._obs

        # ----- Revenue model -----
        if obs.product.is_launched:
            base_rev = (
                obs.product.quality
                * obs.market.demand
                * (1.0 - 0.5 * obs.market.competition)
                * 30_000.0                          # market size scalar
                * (1.0 + 0.05 * obs.team.engineers) # eng productivity
            )
            # Revenue smoothly converges toward base_rev
            obs.metrics.revenue = obs.metrics.revenue * 0.6 + base_rev * 0.4
        else:
            obs.metrics.revenue = 0.0

        # ----- User growth model -----
        if obs.product.is_launched:
            natural_growth = (
                obs.market.demand
                * obs.product.quality
                * (1.0 + obs.team.marketers * 0.1)
                * 5.0
            )
            natural_growth -= obs.market.competition * 2.0
            obs.metrics.user_growth = max(0.0, obs.metrics.user_growth * 0.8 + natural_growth * 0.2)
        else:
            obs.metrics.user_growth = 0.0

        # ----- Competitive drift (rivals grow) -----
        drift = 0.005 * (1.0 + 0.5 * self._rng.random())
        obs.market.competition = _clamp(obs.market.competition + drift, 0.0, 0.95)

        # ----- Demand mean-reversion -----
        mean_demand = 0.5
        obs.market.demand = _clamp(obs.market.demand + 0.02 * (mean_demand - obs.market.demand), 0.0, 1.0)

    def _apply_event(self, event: str) -> None:
        obs = self._obs
        obs.market.demand = _clamp(
            obs.market.demand + EVENT_DEMAND_DELTA.get(event, 0.0), 0.0, 1.0
        )
        obs.metrics.revenue *= EVENT_REVENUE_FACTOR.get(event, 1.0)
        obs.metrics.user_growth = max(
            0.0, obs.metrics.user_growth + EVENT_USER_GROWTH_DELTA.get(event, 0.0)
        )

        # key_employee_left removes a random team member
        if event == "key_employee_left":
            roles_present = [
                r for r in ("engineers", "designers", "marketers")
                if getattr(obs.team, r) > 0
            ]
            if roles_present:
                role = self._rng.choice(roles_present)
                setattr(obs.team, role, max(0, getattr(obs.team, role) - 1))

    def _compute_burn_rate(self) -> float:
        obs = self._obs
        return (
            obs.team.engineers * SALARY_ENGINEER
            + obs.team.designers * SALARY_DESIGNER
            + obs.team.marketers * SALARY_MARKETER
        )

    @staticmethod
    def _weekly_salary_for_role(role: str) -> float:
        return {"engineer": SALARY_ENGINEER, "designer": SALARY_DESIGNER, "marketer": SALARY_MARKETER}[role]

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(self, penalty: float) -> Reward:
        obs = self._obs

        # Revenue component (normalised against theoretical max ~30k/week)
        rev_max = 30_000.0
        rev_comp = _clamp(obs.metrics.revenue / rev_max, 0.0, 1.0)

        # User growth component (normalised against ~20% growth/week)
        ug_max = 20.0
        ug_comp = _clamp(obs.metrics.user_growth / ug_max, 0.0, 1.0)

        # Quality component (already 0-1)
        qual_comp = obs.product.quality

        # Efficiency: reward low burn relative to revenue
        if obs.metrics.burn_rate > 0:
            efficiency = _clamp(obs.metrics.revenue / (obs.metrics.burn_rate + 1e-9), 0.0, 3.0) / 3.0
        else:
            efficiency = 0.5  # neutral

        # Weighted combination
        raw = (
            0.30 * rev_comp
            + 0.25 * ug_comp
            + 0.25 * qual_comp
            + 0.20 * efficiency
        )
        total = _clamp(raw - penalty, 0.0, 1.0)

        return Reward(
            total=round(total, 4),
            revenue_component=round(rev_comp, 4),
            user_growth_component=round(ug_comp, 4),
            quality_component=round(qual_comp, 4),
            efficiency_component=round(efficiency, 4),
            penalty=round(penalty, 4),
            info={
                "raw_before_penalty": round(raw, 4),
                "budget_remaining": round(obs.budget, 2),
                "week": obs.time.current_week,
            },
        )

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_done(self) -> bool:
        obs = self._obs
        if obs.budget <= 0.0:
            return True
        if obs.time.current_week > obs.time.max_weeks:
            return True
        # Optional: huge success milestone
        if obs.metrics.revenue >= 100_000.0 and obs.product.quality >= 0.9:
            return True
        return False

    def _done_reason(self) -> str:
        obs = self._obs
        if obs.budget <= 0.0:
            return "bankruptcy"
        if obs.time.current_week > obs.time.max_weeks:
            return "max_weeks_reached"
        return "success_milestone"
