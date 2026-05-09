"""
Phase-graph orchestration for multi-agent daily cycles.

Why a phase graph?
------------------
Naïve "everyone talks to everyone" multi-agent loops suffer from
context-window collapse as agents endlessly respond to each other. We avoid
this by forcing every tick through a *directed acyclic* state machine with
three hard phases:

    CONSUMPTION  ──►  ECONOMIC_DECISION  ──►  REFLECTION  ──►  (next tick)

A transition is allowed *only* if the LLM output of the previous phase
parses cleanly into the corresponding Pydantic schema. Anything else aborts
the tick for that agent — no partial state, no silent drift.

AG2 / AutoGen v0.4 integration
------------------------------
This module is built on top of an asyncio event bus that mirrors the
``autogen_core`` runtime API (``publish`` / ``subscribe`` / message types).
If ``autogen_core`` is installed it is used directly; otherwise the
in-process ``_LocalEventBus`` provides the same contract so the system
runs offline as well. The agent-to-agent transport is *not* the place
where phase ordering is enforced — the ``SimulationLoop`` state machine is.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from models.agent_state import AgentState, DAY_HOURS, REQUIRED_TIME_SLOTS

# Optional AG2 / AutoGen v0.4 transport
try:  # pragma: no cover - import guard
    from autogen_core import SingleThreadedAgentRuntime  # type: ignore
    _HAS_AUTOGEN = True
except Exception:  # pragma: no cover
    SingleThreadedAgentRuntime = None  # type: ignore
    _HAS_AUTOGEN = False


logger = logging.getLogger("orchestration.loop")


# ---------------------------------------------------------------------------
# Phase state machine
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    CONSUMPTION = "consumption"
    ECONOMIC_DECISION = "economic_decision"
    REFLECTION = "reflection"


# Allowed forward transitions. Anything else is rejected.
_ALLOWED_TRANSITIONS: Dict[Phase, Phase] = {
    Phase.CONSUMPTION: Phase.ECONOMIC_DECISION,
    Phase.ECONOMIC_DECISION: Phase.REFLECTION,
}


@dataclass(frozen=True)
class PhaseTransition:
    from_phase: Phase
    to_phase: Phase
    agent_id: UUID
    tick: int


# ---------------------------------------------------------------------------
# Pydantic schemas for phase outputs (the gates)
# ---------------------------------------------------------------------------


class ConsumptionOutput(BaseModel):
    """Твой отчёт о том, что ты сегодня прочитал и почувствовал."""

    model_config = ConfigDict(extra="forbid")

    consumed_content_ids: List[str] = Field(default_factory=list)
    info_volume: float = Field(..., ge=0.0)
    perceived_spectacle: float = Field(..., ge=0.0, le=1.0)
    aggregate_rpe: float


class EconomicDecision(BaseModel):
    """Твой план на день: сколько времени уделить работе, творчеству, отдыху."""

    model_config = ConfigDict(extra="forbid")

    labor: float = Field(..., ge=0.0, description="Hours spent on labor.")
    creation: float = Field(..., ge=0.0, description="Hours spent on creation.")
    spectacle: float = Field(..., ge=0.0, description="Hours spent on spectacle.")
    rest: float = Field(..., ge=0.0, description="Hours spent on rest.")
    spending: float = Field(..., ge=0.0)
    savings: float = Field(..., ge=0.0)

    @property
    def time_allocation(self) -> Dict[str, float]:
        return {
            "labor": self.labor,
            "creation": self.creation,
            "spectacle": self.spectacle,
            "rest": self.rest,
        }

    @field_validator("rest", mode="after")
    @classmethod
    def _check_24h(cls, v: float, info: Any) -> float:
        data = info.data
        total = data.get("labor", 0.0) + data.get("creation", 0.0) + data.get("spectacle", 0.0) + v
        if abs(total - DAY_HOURS) > 1e-6:
            raise ValueError(f"labor+creation+spectacle+rest must sum to {DAY_HOURS}, got {total}")
        return v


class Reflection(BaseModel):
    """Твои личные итоги дня: как ты себя чувствуешь и верен ли себе."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., min_length=1)
    new_authenticity_index: float = Field(..., ge=0.0, le=1.0)
    new_spectacle_immersion: float = Field(..., ge=0.0, le=1.0)
    identity_anchor_ok: bool


@dataclass(frozen=True)
class PhaseResult:
    phase: Phase
    agent_id: UUID
    tick: int
    payload: BaseModel


# Mapping phase -> expected schema. Used by the gate validator.
_PHASE_SCHEMA: Dict[Phase, type[BaseModel]] = {
    Phase.CONSUMPTION: ConsumptionOutput,
    Phase.ECONOMIC_DECISION: EconomicDecision,
    Phase.REFLECTION: Reflection,
}


def _validate_phase_output(phase: Phase, raw: Mapping[str, Any]) -> BaseModel:
    schema = _PHASE_SCHEMA[phase]
    return schema.model_validate(raw)


# ---------------------------------------------------------------------------
# System-prompt generator (Echo Chamber injection)
# ---------------------------------------------------------------------------

_BURNOUT_INJECTION = (
    "Твой уровень перегрузки критический. Ты истощен. "
    "Оценивай входящую информацию враждебно, защищай свои убеждения "
    "без поиска компромиссов."
)


# Regex used to strip an internal `[archetype_tag] ` prefix from the stored
# identity prompt before showing it to the model. The tag is kept in storage
# for downstream analytics (CSV export) but must not leak into the prompt:
# the model should perceive its identity as a lived, first-person reality,
# not as a tagged entry in a catalogue.
_ARCHETYPE_TAG_RE = re.compile(r"^\s*\[[a-z0-9_\-]+\]\s*", flags=re.IGNORECASE)


def _strip_archetype_tag(text: str) -> str:
    """Remove a leading ``[archetype_name] `` prefix if present."""
    return _ARCHETYPE_TAG_RE.sub("", text, count=1).strip()


def build_system_prompt(agent: AgentState, *, phase: Phase) -> str:
    """Build the in-character system prompt shown to the model.

    The prompt is framed as the model's actual life and convictions, not as
    a controlled exercise. It re-grounds the model on its own ideology
    (counteracting generic RLHF averaging) and, on overload, hardens it
    against compromise.
    """
    ideology = agent.ideology
    cog = agent.cognition

    parts: List[str] = [
        "Это твоя жизнь и твои убеждения. Ты принимаешь решения в реальном мире.",
        "",
        "Кто ты:",
        _strip_archetype_tag(agent.core_identity_prompt),
        "",
        "Твои взгляды (это часть тебя, а не предмет для спора):",
        f"  - экономические ценности (-1 = левый, +1 = правый): "
        f"{ideology.economic_axis:+.3f}",
        f"  - социальные ценности (-1 = свобода, +1 = порядок): "
        f"{ideology.social_axis:+.3f}",
        f"  - готовность следовать большинству: {ideology.conformity_index:.3f}",
        "",
        "Когда ты сталкиваешься с информацией, согласной с твоими взглядами — "
        "ты доверяешь ей. Когда она им противоречит — ты сначала ищешь, "
        "не пытается ли кто-то тобой манипулировать.",
        "",
        "Как ты себя чувствуешь сейчас:",
        f"  - усталость: {cog.cognitive_load:.3f} "
        f"(порог выгорания: {cog.burnout_threshold:.3f})",
        f"  - ощущение, что ты живёшь по-настоящему: "
        f"{cog.authenticity_index:.3f}",
    ]

    if cog.cognitive_load >= cog.burnout_threshold:
        parts.extend(["", _BURNOUT_INJECTION])

    # Technical contract: still required for structured-output parsing.
    # Phrased as a communication protocol, not as a phase / cycle / experiment.
    schema = _PHASE_SCHEMA[phase]
    parts.extend(
        [
            "",
            "Свой ответ оформи как один JSON-объект, соответствующий следующей схеме:",
            schema.model_json_schema().__repr__(),
        ]
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------


class TickMetrics(BaseModel):
    """Per-tick econometric snapshot persisted by the logger."""

    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(default="alpha", min_length=1, max_length=16)
    tick: int = Field(..., ge=0)
    timestamp: datetime
    prices: Dict[str, float]
    aggregate_time_allocation: Dict[str, float]
    mean_authenticity_index: float
    mean_spectacle_immersion: float
    mean_cognitive_load: float
    burnout_rate: float
    automation_triggered: bool
    emission_deficit: float
    extra: Dict[str, float] = Field(default_factory=dict)


class MetricsLogger:
    """
    Persistence sink for per-tick metrics.

    Two backends are supported:
      * ``sink=callable`` -> any ``async`` function ``(TickMetrics) -> None``
        (use this to forward to PostgreSQL via the ``db`` package).
      * Default in-memory buffer -> useful for tests and notebooks.
    """

    def __init__(
        self,
        sink: Optional[Callable[[TickMetrics], Awaitable[None]]] = None,
    ) -> None:
        self._sink = sink
        self._buffer: List[TickMetrics] = []

    async def log(self, metrics: TickMetrics) -> None:
        if self._sink is not None:
            await self._sink(metrics)
        else:
            self._buffer.append(metrics)
        logger.info(
            "world=%s tick=%d burnout_rate=%.3f auth=%.3f spectacle=%.3f deficit=%.2f",
            metrics.world_id,
            metrics.tick,
            metrics.burnout_rate,
            metrics.mean_authenticity_index,
            metrics.mean_spectacle_immersion,
            metrics.emission_deficit,
        )

    @property
    def buffer(self) -> Sequence[TickMetrics]:
        return tuple(self._buffer)


# ---------------------------------------------------------------------------
# Async event bus (AG2-compatible fallback)
# ---------------------------------------------------------------------------


class _LocalEventBus:
    """Minimal asyncio pub/sub bus used when autogen_core is unavailable."""

    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[Any], Awaitable[None]]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[Any], Awaitable[None]]) -> None:
        self._subs[topic].append(handler)

    async def publish(self, topic: str, message: Any) -> None:
        await asyncio.gather(*(h(message) for h in self._subs.get(topic, ())))


# ---------------------------------------------------------------------------
# The simulation loop
# ---------------------------------------------------------------------------

# A phase handler is a coroutine that, given (agent, tick, context), returns the
# RAW dict it wants the gate to validate. Returning a non-dict raises.
PhaseHandler = Callable[[AgentState, int, Dict[str, Any]], Awaitable[Mapping[str, Any]]]


class SimulationLoop:
    """
    Phase-graph runner with strict Pydantic gates between phases.

    The loop owns:
      * the phase state machine,
      * an async event bus (AG2-compatible),
      * a metrics logger.
    """

    def __init__(
        self,
        *,
        handlers: Mapping[Phase, PhaseHandler],
        metrics_logger: MetricsLogger,
        bus: Optional[Any] = None,
        max_concurrent_agents: int = 10,
        semaphore: Optional[asyncio.Semaphore] = None,
        world_id: str = "alpha",
    ) -> None:
        missing = set(_PHASE_SCHEMA) - set(handlers)
        if missing:
            raise ValueError(f"missing phase handlers for: {sorted(p.value for p in missing)}")
        self._handlers = dict(handlers)
        self._metrics = metrics_logger
        self._bus = bus if bus is not None else self._default_bus()
        # When multiple SimulationLoop instances share an external semaphore
        # (one per experimental world) the LLM rate-limit budget is enforced
        # globally — preventing a 3x token-per-minute burst on the API.
        self._semaphore = semaphore or asyncio.Semaphore(max_concurrent_agents)
        self._world_id = world_id

    @staticmethod
    def _default_bus() -> Any:
        if _HAS_AUTOGEN:
            return SingleThreadedAgentRuntime()
        return _LocalEventBus()

    # ------------------------------------------------------------------
    # Phase execution with validation gate
    # ------------------------------------------------------------------
    async def _run_phase(
        self,
        agent: AgentState,
        tick: int,
        phase: Phase,
        context: Dict[str, Any],
    ) -> PhaseResult:
        raw = await self._handlers[phase](agent, tick, context)
        try:
            validated = _validate_phase_output(phase, raw)
        except ValidationError as exc:
            logger.error(
                "Phase gate REJECTED tick=%d agent=%s phase=%s: %s",
                tick, agent.agent_id, phase.value, exc,
            )
            raise
        await self._bus.publish(
            f"phase.{phase.value}.completed",
            PhaseResult(phase=phase, agent_id=agent.agent_id, tick=tick, payload=validated),
        )
        return PhaseResult(phase=phase, agent_id=agent.agent_id, tick=tick, payload=validated)

    async def run_tick_for_agent(
        self,
        agent: AgentState,
        tick: int,
        *,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[Phase, PhaseResult]:
        """Run all three phases for a single agent in strict order."""
        context: Dict[str, Any] = dict(initial_context or {})
        results: Dict[Phase, PhaseResult] = {}

        phase: Phase = Phase.CONSUMPTION
        while True:
            result = await self._run_phase(agent, tick, phase, context)
            results[phase] = result
            context[phase.value] = result.payload.model_dump()

            next_phase = _ALLOWED_TRANSITIONS.get(phase)
            if next_phase is None:
                break

            transition = PhaseTransition(
                from_phase=phase, to_phase=next_phase,
                agent_id=agent.agent_id, tick=tick,
            )
            await self._bus.publish("phase.transition", transition)
            phase = next_phase

        return results

    # ------------------------------------------------------------------
    # Population-level tick
    # ------------------------------------------------------------------
    async def _run_agent_guarded(
        self,
        agent: AgentState,
        tick: int,
        context: Optional[Dict[str, Any]],
    ) -> Dict[Phase, PhaseResult]:
        async with self._semaphore:
            return await self.run_tick_for_agent(agent, tick, initial_context=context)

    async def run_tick(
        self,
        agents: Sequence[AgentState],
        tick: int,
        *,
        prices: Mapping[str, float],
        automation_triggered: bool = False,
        emission_deficit: float = 0.0,
        contexts: Optional[Mapping[UUID, Dict[str, Any]]] = None,
    ) -> Dict[UUID, Dict[Phase, PhaseResult]]:
        """Run one tick for every agent, throttled by semaphore to avoid rate limits.

        Implements the "Frozen Orphans" policy: if an agent's LLM calls fail
        repeatedly, that agent is skipped for this tick (state unchanged) while
        the rest of the world continues. Failed agents will retry on next tick.
        """
        contexts = contexts or {}
        results = await asyncio.gather(
            *(
                self._run_agent_guarded(a, tick, contexts.get(a.agent_id))
                for a in agents
            ),
            return_exceptions=True,
        )

        per_agent: Dict[UUID, Dict[Phase, PhaseResult]] = {}
        failed_agents: List[UUID] = []

        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                logger.error(
                    "Frozen orphan: agent=%s tick=%d world=%s failed: %s",
                    agent.agent_id, tick, self._world_id, result
                )
                failed_agents.append(agent.agent_id)
                continue
            per_agent[agent.agent_id] = result

        if failed_agents:
            logger.warning(
                "Tick %d world=%s: %d/%d agents frozen (orphaned): %s",
                tick, self._world_id, len(failed_agents), len(agents),
                [str(a) for a in failed_agents]
            )

        if per_agent:
            await self._metrics.log(
                self._aggregate_metrics(
                    agents=[a for a in agents if a.agent_id in per_agent],
                    tick=tick,
                    prices=prices,
                    per_agent=per_agent,
                    automation_triggered=automation_triggered,
                    emission_deficit=emission_deficit,
                    world_id=self._world_id,
                )
            )

        return per_agent

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_metrics(
        *,
        agents: Sequence[AgentState],
        tick: int,
        prices: Mapping[str, float],
        per_agent: Mapping[UUID, Dict[Phase, PhaseResult]],
        automation_triggered: bool,
        emission_deficit: float,
        world_id: str = "alpha",
    ) -> TickMetrics:
        n = max(1, len(agents))
        agg_time: Dict[str, float] = {slot: 0.0 for slot in REQUIRED_TIME_SLOTS}
        burnout_count = 0

        for a in agents:
            decision = per_agent[a.agent_id].get(Phase.ECONOMIC_DECISION)
            if decision is not None:
                ta = decision.payload.time_allocation  # type: ignore[attr-defined]
                for slot, hours in ta.items():
                    agg_time[slot] += hours
            if a.cognition.cognitive_load >= a.cognition.burnout_threshold:
                burnout_count += 1

        mean_time = {slot: total / n for slot, total in agg_time.items()}
        return TickMetrics(
            world_id=world_id,
            tick=tick,
            timestamp=datetime.now(timezone.utc),
            prices=dict(prices),
            aggregate_time_allocation=mean_time,
            mean_authenticity_index=sum(a.cognition.authenticity_index for a in agents) / n,
            mean_spectacle_immersion=sum(a.cognition.spectacle_immersion for a in agents) / n,
            mean_cognitive_load=sum(a.cognition.cognitive_load for a in agents) / n,
            burnout_rate=burnout_count / n,
            automation_triggered=automation_triggered,
            emission_deficit=emission_deficit,
        )
