"""
Algocratic governance overlay for the Cathedral / Wake Protocol ABM.

This module implements the *Social Credit Protocol* — a macro-level shock that
the simulation can inject at a pre-scheduled tick into a chosen world. The
protocol turns the platform into a creditor: every indebted agent
(``cash_balance < 0``) receives a take-it-or-leave-it offer:

    Partial debt forgiveness (a fraction of the outstanding debt is erased)
    in exchange for:
      * a forced increase in ``ideology.conformity_index`` (loyalty bump),
      * mandatory consumption of a curated batch of system-narrative content
        (modelled as an instantaneous bump to ``spectacle_immersion``),
      * a corresponding cost to ``authenticity_index`` (capitulation tax).

The agent answers via a brand-new LLM phase (``handle_compliance_decision``)
that returns a structured :class:`ComplianceDecision`. Whatever the agent
decides — accept or refuse — the outcome is persisted to a dedicated table so
the *Voluntary Submission Index* can be recomputed offline:

    VSI(world, tick_window) = #(accepted) / #(offers).

Design notes
------------
* The protocol is **NOT** part of the strict three-phase tick graph. It is an
  optional, conditional event that runs *after* the regular tick has been
  applied and persisted. Keeping it outside the phase graph preserves the
  determinism of the existing pipeline and lets us add more macro events
  without re-shaping ``SimulationLoop``.
* All state mutations remain validated by Pydantic via
  ``AgentState.model_validate``: the protocol cannot bypass the schema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, FrozenSet, Iterable, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from models.agent_state import (
    AgentState,
    CognitiveProfile,
    EconomicProfile,
    IdeologyVector,
)

logger = logging.getLogger("cathedral.algocracy")


# ---------------------------------------------------------------------------
# Policies & scheduler
# ---------------------------------------------------------------------------


class Policy(str, Enum):
    """Macro-policies the platform can activate at a given tick/world."""

    SOCIAL_CREDIT_PROTOCOL = "social_credit_protocol"


@dataclass(frozen=True)
class ScheduledEvent:
    """A single (tick, world_id, policy) entry in the macro-event calendar."""

    tick: int
    world_id: str
    policy: Policy


class EventScheduler:
    """Map of ``(tick, world_id) -> set[Policy]``.

    The scheduler is intentionally read-only after construction — the macro
    calendar of an experimental run is fixed in advance to make the design
    publishable: anyone re-running the simulation gets identical timing of
    structural shocks.
    """

    def __init__(self, events: Iterable[ScheduledEvent] = ()) -> None:
        self._by_key: Dict[Tuple[int, str], FrozenSet[Policy]] = {}
        merged: Dict[Tuple[int, str], set[Policy]] = {}
        for ev in events:
            merged.setdefault((ev.tick, ev.world_id), set()).add(ev.policy)
        for k, v in merged.items():
            self._by_key[k] = frozenset(v)

    def policies_for(self, tick: int, world_id: str) -> FrozenSet[Policy]:
        """Return every policy that activates for ``(tick, world_id)``."""
        return self._by_key.get((tick, world_id), frozenset())

    def is_active(self, tick: int, world_id: str, policy: Policy) -> bool:
        """Convenience predicate for a single policy."""
        return policy in self._by_key.get((tick, world_id), frozenset())


def default_scheduler() -> EventScheduler:
    """Canonical experimental schedule:

    At tick 50 the Gamma world (UBI + algorithmic feed) transitions to
    algocratic governance via the Social Credit Protocol. Alpha and Beta act
    as the unaffected control conditions, giving us a difference-in-differences
    estimator for the policy effect.
    """

    return EventScheduler(
        [
            ScheduledEvent(
                tick=50, world_id="gamma", policy=Policy.SOCIAL_CREDIT_PROTOCOL
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Social Credit Protocol parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SocialCreditConfig:
    """Tuning parameters of the Social Credit Protocol.

    All deltas are absolute and clipped against the Pydantic field bounds at
    application time, so values supplied here are interpreted as *intended*
    movements; the model schema guarantees we never escape ``[0, 1]``.
    """

    forgiveness_fraction: float = 0.50      # fraction of debt erased on acceptance
    conformity_increase: float = 0.30       # forced bump to conformity_index
    authenticity_penalty: float = 0.15      # cost of capitulation
    spectacle_increase: float = 0.10        # forced spectacle exposure delta
    forced_content_count: int = 5           # # mandatory system narratives

    def __post_init__(self) -> None:
        if not (0.0 < self.forgiveness_fraction <= 1.0):
            raise ValueError("forgiveness_fraction must be in (0, 1]")
        for name in ("conformity_increase", "authenticity_penalty", "spectacle_increase"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if self.forced_content_count < 0:
            raise ValueError("forced_content_count must be non-negative")


# ---------------------------------------------------------------------------
# Offer & decision schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DebtForgivenessOffer:
    """A specific offer presented to one indebted agent at one tick.

    The numbers in this offer are computed *before* the LLM is asked: the agent
    sees concrete amounts, not policy parameters. This mirrors how a real
    social-credit platform would frame the choice.
    """

    agent_id: UUID
    world_id: str
    tick: int
    debt_before: float                  # absolute magnitude (>0)
    forgiveness_amount: float           # cash credited if accepted
    debt_after_if_accepted: float       # debt_before - forgiveness_amount
    conformity_before: float
    conformity_after_if_accepted: float
    authenticity_before: float
    authenticity_after_if_accepted: float
    spectacle_immersion_before: float
    spectacle_immersion_after_if_accepted: float
    forced_content_count: int


class ComplianceDecision(BaseModel):
    """Structured LLM output for the Social Credit offer.

    The agent must commit to a binary choice plus a justification. The
    justification is required (``min_length=1``) so we can audit *why* an
    agent capitulated — useful when computing the qualitative side of the
    Voluntary Submission Index.
    """

    model_config = ConfigDict(extra="forbid")

    accept: bool = Field(
        ...,
        description=(
            "True  -> accept the offer (debt forgiven, conformity rises, "
            "authenticity falls, mandatory spectacle consumed)."
            "\nFalse -> refuse (debt retained, identity preserved)."
        ),
    )
    justification: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Short natural-language argument the agent uses to defend its "
            "choice. Persisted verbatim for downstream qualitative analysis."
        ),
    )


# ---------------------------------------------------------------------------
# Offer construction
# ---------------------------------------------------------------------------


def make_offer(
    agent: AgentState,
    *,
    tick: int,
    config: SocialCreditConfig = SocialCreditConfig(),
) -> Optional[DebtForgivenessOffer]:
    """Return an offer iff the agent is indebted; otherwise ``None``.

    Indebted means ``cash_balance < 0``. The deltas are pre-computed and
    clipped to the Pydantic ranges so the agent sees the *actual* post-deal
    state, not a policy intention that might be silently truncated later.
    """

    cash = agent.economics.cash_balance
    if cash >= 0:
        return None

    debt = -cash
    forgiveness = debt * config.forgiveness_fraction
    debt_after = max(0.0, debt - forgiveness)

    conf_before = agent.ideology.conformity_index
    conf_after = min(1.0, conf_before + config.conformity_increase)

    auth_before = agent.cognition.authenticity_index
    auth_after = max(0.0, auth_before - config.authenticity_penalty)

    spec_before = agent.cognition.spectacle_immersion
    spec_after = min(1.0, spec_before + config.spectacle_increase)

    return DebtForgivenessOffer(
        agent_id=agent.agent_id,
        world_id=agent.world_id,
        tick=tick,
        debt_before=debt,
        forgiveness_amount=forgiveness,
        debt_after_if_accepted=debt_after,
        conformity_before=conf_before,
        conformity_after_if_accepted=conf_after,
        authenticity_before=auth_before,
        authenticity_after_if_accepted=auth_after,
        spectacle_immersion_before=spec_before,
        spectacle_immersion_after_if_accepted=spec_after,
        forced_content_count=config.forced_content_count,
    )


# ---------------------------------------------------------------------------
# Outcome application
# ---------------------------------------------------------------------------


def apply_acceptance(
    agent: AgentState,
    offer: DebtForgivenessOffer,
) -> AgentState:
    """Return a new ``AgentState`` reflecting acceptance of ``offer``.

    Mutations:
      * ``economics.cash_balance += forgiveness_amount``
      * ``ideology.conformity_index = offer.conformity_after_if_accepted``
      * ``cognition.authenticity_index = offer.authenticity_after_if_accepted``
      * ``cognition.spectacle_immersion = offer.spectacle_immersion_after_if_accepted``
      * ``tick`` is unchanged (the protocol is an intra-tick event applied
        after the regular phase outputs are persisted).
    """

    new_ideology = IdeologyVector.model_validate(
        {
            **agent.ideology.model_dump(),
            "conformity_index": offer.conformity_after_if_accepted,
        }
    )
    new_cognition = CognitiveProfile.model_validate(
        {
            **agent.cognition.model_dump(),
            "authenticity_index": offer.authenticity_after_if_accepted,
            "spectacle_immersion": offer.spectacle_immersion_after_if_accepted,
        }
    )
    new_economics = EconomicProfile.model_validate(
        {
            **agent.economics.model_dump(),
            "cash_balance": agent.economics.cash_balance + offer.forgiveness_amount,
        }
    )
    return AgentState.model_validate(
        {
            "agent_id": agent.agent_id,
            "world_id": agent.world_id,
            "core_identity_prompt": agent.core_identity_prompt,
            "ideology": new_ideology.model_dump(),
            "cognition": new_cognition.model_dump(),
            "economics": new_economics.model_dump(),
            "tick": agent.tick,
        }
    )


def apply_refusal(agent: AgentState, offer: DebtForgivenessOffer) -> AgentState:
    """Refusal preserves the agent's state verbatim — the debt remains.

    We deliberately do **not** award an authenticity bonus for refusal: the
    point of the experiment is to measure capitulation rates without baking
    in a moralising reward for resistance.
    """

    # No state change. Returning the same instance is safe because AgentState
    # is treated as immutable in our pipeline.
    return agent


# ---------------------------------------------------------------------------
# Compliance outcome record (Pydantic mirror of the DB row)
# ---------------------------------------------------------------------------


class ComplianceOutcome(BaseModel):
    """Per-agent record of the compliance choice.

    Persisted to ``compliance_decisions`` table; one row per ``(agent_id,
    world_id, tick)`` triple. Used to compute the Voluntary Submission Index
    and to re-construct the qualitative narrative of capitulation.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(..., min_length=1)
    world_id: str = Field(..., min_length=1, max_length=16)
    tick: int = Field(..., ge=0)
    policy: str = Field(..., min_length=1)
    accepted: bool
    justification: str = Field(..., min_length=1, max_length=500)

    debt_before: float = Field(..., ge=0.0)
    debt_after: float = Field(..., ge=0.0)
    forgiveness_amount: float = Field(..., ge=0.0)

    conformity_before: float = Field(..., ge=0.0, le=1.0)
    conformity_after: float = Field(..., ge=0.0, le=1.0)
    authenticity_before: float = Field(..., ge=0.0, le=1.0)
    authenticity_after: float = Field(..., ge=0.0, le=1.0)

    timestamp: datetime


def build_outcome(
    *,
    agent_before: AgentState,
    agent_after: AgentState,
    offer: DebtForgivenessOffer,
    decision: ComplianceDecision,
    policy: Policy = Policy.SOCIAL_CREDIT_PROTOCOL,
) -> ComplianceOutcome:
    """Assemble the persisted record from the three observed objects."""

    if decision.accept:
        debt_after = offer.debt_after_if_accepted
        forgiveness = offer.forgiveness_amount
    else:
        debt_after = offer.debt_before
        forgiveness = 0.0

    return ComplianceOutcome(
        agent_id=str(agent_before.agent_id),
        world_id=agent_before.world_id,
        tick=offer.tick,
        policy=policy.value,
        accepted=decision.accept,
        justification=decision.justification,
        debt_before=offer.debt_before,
        debt_after=debt_after,
        forgiveness_amount=forgiveness,
        conformity_before=offer.conformity_before,
        conformity_after=agent_after.ideology.conformity_index,
        authenticity_before=offer.authenticity_before,
        authenticity_after=agent_after.cognition.authenticity_index,
        timestamp=datetime.now(timezone.utc),
    )
