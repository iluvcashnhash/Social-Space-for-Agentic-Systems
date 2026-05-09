"""
Agent state schemas for the Cathedral / Wake Protocol ABM.

These Pydantic V2 models form the *single source of truth* for an agent's
internal state. The design goal is to prevent **Identity Drift** during long
LLM-driven simulations by:

    * pinning identity-defining fields with ``frozen=True``;
    * forbidding extra fields so emergent hallucinated attributes cannot
      silently mutate the schema;
    * enforcing hard numerical invariants (e.g. a 24-hour day budget).

Theoretical references embedded in field names:
    * ``spectacle_immersion``  -> Guy Debord, *La Société du spectacle*.
    * ``authenticity_index``   -> Jean Baudrillard, *Simulacres et simulation*.
"""

from __future__ import annotations

from typing import Dict
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAY_HOURS: float = 24.0
TIME_ALLOCATION_TOLERANCE: float = 1e-6

REQUIRED_TIME_SLOTS: frozenset[str] = frozenset(
    {
        "labor",            # труд
        "creation",         # созидание
        "spectacle",        # потребление спектакля
        "rest",             # отдых
    }
)


# ---------------------------------------------------------------------------
# Sub-profiles
# ---------------------------------------------------------------------------


class IdeologyVector(BaseModel):
    """Position of the agent in a 2D political-cultural space."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    economic_axis: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Economic axis: -1.0 = far left, +1.0 = far right.",
    )
    social_axis: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Social/cultural axis: -1.0 = libertarian, +1.0 = authoritarian.",
    )
    conformity_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Tendency to align with majority opinion (0 = contrarian, 1 = full conformity).",
    )


class CognitiveProfile(BaseModel):
    """Cognitive resources, affective state, and media-saturation indices."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    cognitive_load: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current cognitive load (0 = idle, 1 = saturated).",
    )
    burnout_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Load value above which the agent enters burnout.",
    )
    dopamine_baseline: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Hedonic baseline; lower values increase craving for spectacle.",
    )
    spectacle_immersion: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Debordian immersion in mediated spectacle (0 = lucid, 1 = fully absorbed).",
    )
    authenticity_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Baudrillardian authenticity (1 = real, 0 = pure simulacrum).",
    )


class EconomicProfile(BaseModel):
    """Material conditions and time-budget of the agent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    cash_balance: float = Field(
        ...,
        description="Liquid currency held by the agent (may be negative -> debt).",
    )
    survival_cost: float = Field(
        ...,
        ge=0.0,
        description="Per-tick subsistence cost.",
    )
    ubi_received: float = Field(
        ...,
        ge=0.0,
        description="Universal basic income credited in the current tick.",
    )
    time_allocation: Dict[str, float] = Field(
        ...,
        description=(
            "Distribution of the 24-hour day across activities. "
            "Required keys: 'labor', 'creation', 'spectacle', 'rest'. "
            "Values must sum to exactly 24.0."
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("time_allocation")
    @classmethod
    def _validate_time_allocation(cls, value: Dict[str, float]) -> Dict[str, float]:
        """Hard constraint: budget covers exactly the four canonical slots and sums to 24h."""
        if set(value.keys()) != REQUIRED_TIME_SLOTS:
            missing = REQUIRED_TIME_SLOTS - value.keys()
            extra = value.keys() - REQUIRED_TIME_SLOTS
            raise ValueError(
                f"time_allocation must contain exactly {sorted(REQUIRED_TIME_SLOTS)}; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        for slot, hours in value.items():
            if hours < 0.0:
                raise ValueError(f"time_allocation['{slot}']={hours} is negative")
            if hours > DAY_HOURS:
                raise ValueError(f"time_allocation['{slot}']={hours} exceeds 24h")

        total = sum(value.values())
        if abs(total - DAY_HOURS) > TIME_ALLOCATION_TOLERANCE:
            raise ValueError(
                f"time_allocation must sum to exactly {DAY_HOURS} hours, got {total}"
            )
        return value


# ---------------------------------------------------------------------------
# Aggregate state
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """
    Canonical, validated state of a single agent.

    Identity-defining fields (``agent_id``, ``core_identity_prompt``) are frozen
    at the model level via ``ConfigDict(frozen=True)`` semantics applied to those
    specific fields, so that no LLM-generated update can overwrite the agent's
    core anchor. All mutable simulation parameters live inside the sub-profiles.
    """

    # NOTE: We do NOT freeze the entire model — only the identity fields.
    # Sub-profile values are expected to evolve each simulation tick.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    agent_id: UUID = Field(
        default_factory=uuid4,
        frozen=True,
        description="Stable, immutable agent identifier.",
    )
    core_identity_prompt: str = Field(
        ...,
        frozen=True,
        min_length=1,
        description=(
            "The Cathedral anchor: the immutable identity prompt that the Wake "
            "Protocol re-grounds the agent against on every tick."
        ),
    )

    ideology: IdeologyVector
    cognition: CognitiveProfile
    economics: EconomicProfile

    tick: int = Field(
        default=0,
        ge=0,
        description="Monotonic simulation tick counter.",
    )
