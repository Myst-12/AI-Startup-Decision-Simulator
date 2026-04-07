"""
Pydantic models for the Startup Decision Simulator OpenEnv environment.
Defines Observation, Action, and Reward schemas following the OpenEnv spec.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-models for Observation
# ---------------------------------------------------------------------------

class Team(BaseModel):
    """Represents the current team composition of the startup."""
    engineers: int = Field(0, ge=0, description="Number of software engineers")
    designers: int = Field(0, ge=0, description="Number of designers")
    marketers: int = Field(0, ge=0, description="Number of marketers")

    @property
    def total_headcount(self) -> int:
        return self.engineers + self.designers + self.marketers


class Product(BaseModel):
    """Tracks what has been built and the overall quality of the product."""
    features_built: List[str] = Field(default_factory=list, description="Names of released features")
    quality: float = Field(0.0, ge=0.0, le=1.0, description="Overall product quality (0–1)")

    @property
    def is_launched(self) -> bool:
        return len(self.features_built) > 0


class Market(BaseModel):
    """Represents current market conditions."""
    demand: float = Field(0.5, ge=0.0, le=1.0, description="Market demand for your product category (0–1)")
    competition: float = Field(0.3, ge=0.0, le=1.0, description="Competitive intensity (0–1)")
    trend: str = Field("stable", description="Current market trend label")


class BusinessMetrics(BaseModel):
    """Core financial and growth metrics."""
    revenue: float = Field(0.0, ge=0.0, description="Weekly revenue (USD)")
    burn_rate: float = Field(0.0, ge=0.0, description="Weekly cash burn (USD)")
    user_growth: float = Field(0.0, ge=0.0, description="Percentage weekly user growth rate")


class TimeInfo(BaseModel):
    """Simulation time tracking."""
    current_week: int = Field(1, ge=1, description="Current simulation week")
    max_weeks: int = Field(52, ge=1, description="Maximum weeks before forced termination")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Full observable state returned to the agent at each step.
    Complies with OpenEnv Observation schema.
    """
    budget: float = Field(..., description="Remaining cash budget (USD)")
    team: Team = Field(default_factory=Team)
    product: Product = Field(default_factory=Product)
    market: Market = Field(default_factory=Market)
    metrics: BusinessMetrics = Field(default_factory=BusinessMetrics)
    time: TimeInfo = Field(default_factory=TimeInfo)
    pending_events: List[str] = Field(default_factory=list, description="Active environment events this week")

    class Config:
        json_encoders = {float: lambda v: round(v, 4)}


# ---------------------------------------------------------------------------
# Action payloads
# ---------------------------------------------------------------------------

class HirePayload(BaseModel):
    role: Literal["engineer", "designer", "marketer"] = Field(..., description="Role to hire")


class FirePayload(BaseModel):
    role: Literal["engineer", "designer", "marketer"] = Field(..., description="Role to fire")


class BuildFeaturePayload(BaseModel):
    feature_name: str = Field(..., min_length=2, max_length=64, description="Name of the feature to build")


class MarketingPayload(BaseModel):
    budget: float = Field(..., ge=500.0, description="Marketing spend in USD (minimum $500)")


class PivotPayload(BaseModel):
    new_trend: str = Field(..., min_length=2, max_length=64, description="New market trend to pivot towards")


class WaitPayload(BaseModel):
    """No payload required — agent chooses to do nothing this week."""
    reason: Optional[str] = Field(None, description="Optional reason for waiting")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

ActionType = Literal["hire", "fire", "build_feature", "marketing", "pivot", "wait"]

_PAYLOAD_MAP: Dict[str, type] = {
    "hire": HirePayload,
    "fire": FirePayload,
    "build_feature": BuildFeaturePayload,
    "marketing": MarketingPayload,
    "pivot": PivotPayload,
    "wait": WaitPayload,
}


class Action(BaseModel):
    """
    Structured action submitted by the agent each step.
    Complies with OpenEnv Action schema.
    """
    type: ActionType = Field(..., description="Action type")
    payload: Union[
        HirePayload,
        FirePayload,
        BuildFeaturePayload,
        MarketingPayload,
        PivotPayload,
        WaitPayload,
    ] = Field(..., description="Action-specific parameters")

    @model_validator(mode="before")
    @classmethod
    def coerce_payload(cls, data: Any) -> Any:  # noqa: N805
        if not isinstance(data, dict):
            return data
        action_type = data.get("type")
        payload = data.get("payload", {})
        if action_type is None or not isinstance(payload, dict):
            return data
        target_cls = _PAYLOAD_MAP.get(action_type)
        if target_cls is None:
            return data
        data = dict(data)
        data["payload"] = target_cls(**payload)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Convenience constructor from a raw dict."""
        action_type = data.get("type")
        payload_raw = data.get("payload", {})
        target_cls = _PAYLOAD_MAP.get(action_type, WaitPayload)
        payload = target_cls(**payload_raw) if isinstance(payload_raw, dict) else payload_raw
        return cls(type=action_type, payload=payload)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Dense, multi-component reward returned each step.
    All components are normalised; total is in [0.0, 1.0].
    """
    total: float = Field(..., ge=0.0, le=1.0, description="Aggregate normalised reward")
    revenue_component: float = Field(0.0, description="Reward from revenue growth")
    user_growth_component: float = Field(0.0, description="Reward from user growth")
    quality_component: float = Field(0.0, description="Reward from product quality")
    efficiency_component: float = Field(0.0, description="Reward from burn-rate efficiency")
    penalty: float = Field(0.0, description="Penalty for bad or invalid actions")
    info: Dict[str, Any] = Field(default_factory=dict, description="Breakdown metadata")
