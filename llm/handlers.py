"""
LLM phase handlers for the Cathedral / Wake Protocol simulation loop.

This module bridges :class:`orchestration.simulation_loop.SimulationLoop`
(which expects coroutines ``(agent, tick, ctx) -> Mapping[str, Any]``) with a
real language model. Three handlers are exposed, one per phase:

    * ``handle_consumption``        -> ConsumptionOutput
    * ``handle_economic_decision``  -> EconomicDecision
    * ``handle_reflection``         -> Reflection

Each handler:

1. Builds a phase-specific *system* prompt via
   :func:`orchestration.simulation_loop.build_system_prompt`, which already
   contains the Cathedral anchor, the echo-chamber confirmation-bias
   injection, and (when applicable) the burnout escalation clause.
2. Builds a *user* message that injects the runtime context for that phase
   (available content, prices + UBI, last-tick recap).
3. Calls the async LLM client with **structured JSON output** keyed to the
   Pydantic schema of the phase, so the response is parseable by design.
4. Retries with exponential backoff if the response is not valid JSON or
   does not validate against the phase schema.

Client agnosticism
------------------
We don't import any specific SDK. The ``LLMClient`` Protocol is satisfied by
``openai.AsyncOpenAI().chat.completions``, LiteLLM's async client, and by a
thin shim around ``autogen_core``'s ``ChatCompletionClient``. Wire whichever
one you use into ``LLMPhaseCoordinator(client=...)``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from pydantic import BaseModel, ValidationError

from models.agent_state import AgentState
from orchestration.simulation_loop import (
    ConsumptionOutput,
    EconomicDecision,
    Phase,
    Reflection,
    build_system_prompt,
)


logger = logging.getLogger("cathedral.llm")


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Minimal async chat-completion contract.

    The return value must be the raw assistant text (a JSON string). Adapters
    for popular SDKs are trivial; see ``examples`` in the project README.
    """

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: Mapping[str, Any],
        schema_name: str,
        temperature: float = 0.2,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 4
    base_delay: float = 0.5            # seconds
    max_delay: float = 8.0
    jitter: float = 0.25               # multiplicative jitter ratio

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay <= 0 or self.max_delay <= 0:
            raise ValueError("delays must be positive")
        if not (0.0 <= self.jitter < 1.0):
            raise ValueError("jitter must be in [0, 1)")

    def delay_for(self, attempt: int) -> float:
        # attempt is 1-indexed, first retry uses base_delay
        raw = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        # symmetric jitter around `raw`
        spread = raw * self.jitter
        return max(0.0, raw + random.uniform(-spread, spread))


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


_PHASE_SCHEMA: Dict[Phase, type[BaseModel]] = {
    Phase.CONSUMPTION: ConsumptionOutput,
    Phase.ECONOMIC_DECISION: EconomicDecision,
    Phase.REFLECTION: Reflection,
}


class LLMPhaseCoordinator:
    """
    Async coordinator that produces phase outputs by calling an LLM under a
    Pydantic-validated structured-output contract.

    Parameters
    ----------
    client:
        Anything satisfying :class:`LLMClient`.
    retry:
        Retry policy applied independently to each phase call.
    temperature:
        Default sampling temperature; kept low to reduce schema violations.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        retry: RetryConfig = RetryConfig(),
        temperature: float = 0.2,
    ) -> None:
        self._client = client
        self._retry = retry
        self._temperature = temperature

    # ------------------------------------------------------------------
    # Public PhaseHandler-compatible methods
    # ------------------------------------------------------------------
    async def handle_consumption(
        self,
        agent: AgentState,
        tick: int,
        context: Dict[str, Any],
    ) -> Mapping[str, Any]:
        """Phase 1: produce a :class:`ConsumptionOutput` dict."""
        user_message = self._render_consumption_user(agent, tick, context)
        return await self._call_phase(
            agent=agent,
            phase=Phase.CONSUMPTION,
            user_message=user_message,
        )

    async def handle_economic_decision(
        self,
        agent: AgentState,
        tick: int,
        context: Dict[str, Any],
    ) -> Mapping[str, Any]:
        """Phase 2: produce an :class:`EconomicDecision` dict."""
        user_message = self._render_economic_user(agent, tick, context)
        return await self._call_phase(
            agent=agent,
            phase=Phase.ECONOMIC_DECISION,
            user_message=user_message,
        )

    async def handle_reflection(
        self,
        agent: AgentState,
        tick: int,
        context: Dict[str, Any],
    ) -> Mapping[str, Any]:
        """Phase 3: produce a :class:`Reflection` dict."""
        user_message = self._render_reflection_user(agent, tick, context)
        return await self._call_phase(
            agent=agent,
            phase=Phase.REFLECTION,
            user_message=user_message,
        )

    # ------------------------------------------------------------------
    # Core call + retry loop
    # ------------------------------------------------------------------
    async def _call_phase(
        self,
        *,
        agent: AgentState,
        phase: Phase,
        user_message: str,
    ) -> Dict[str, Any]:
        system_prompt = build_system_prompt(agent, phase=phase)
        schema_cls = _PHASE_SCHEMA[phase]
        json_schema = schema_cls.model_json_schema()
        schema_name = schema_cls.__name__

        last_exc: Optional[BaseException] = None
        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                raw = await self._client.complete(
                    system=system_prompt,
                    user=user_message,
                    json_schema=json_schema,
                    schema_name=schema_name,
                    temperature=self._temperature,
                )
                parsed = self._parse_json(raw)
                # Validate eagerly so a malformed payload triggers a retry
                # *before* the SimulationLoop gate also rejects it.
                schema_cls.model_validate(parsed)
                return parsed
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "LLM phase=%s tick=%d agent=%s attempt=%d/%d failed: %s",
                    phase.value, getattr(agent, "tick", -1),
                    agent.agent_id, attempt, self._retry.max_attempts, exc,
                )
                if attempt == self._retry.max_attempts:
                    break
                await asyncio.sleep(self._retry.delay_for(attempt))

        raise RuntimeError(
            f"LLM failed to produce a valid {schema_name} after "
            f"{self._retry.max_attempts} attempts"
        ) from last_exc

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """Tolerant JSON extractor for LLMs that occasionally wrap output."""
        text = raw.strip()
        # Strip ```json ... ``` fences if the model ignored the contract.
        if text.startswith("```"):
            text = text.strip("`")
            # remove an optional 'json' language tag at the start
            if text[:4].lower() == "json":
                text = text[4:]
            text = text.strip()
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(f"LLM returned non-object JSON: {type(obj).__name__}")
        return obj

    # ------------------------------------------------------------------
    # User-message rendering per phase
    # ------------------------------------------------------------------
    @staticmethod
    def _render_consumption_user(
        agent: AgentState,
        tick: int,
        context: Mapping[str, Any],
    ) -> str:
        feed: Sequence[Mapping[str, Any]] = context.get("feed") or []
        lines: List[str] = [
            f"# Phase 1 — Information Consumption (tick {tick})",
            "",
            "Below is the algorithmic feed prepared for you. For each item decide "
            "whether to consume it. Then report your aggregate exposure.",
            "",
            "Feed (already pre-ranked, highest score first):",
        ]
        if not feed:
            lines.append("  (empty)")
        else:
            for item in feed:
                lines.append(
                    f"  - id={item.get('content_id')!r} "
                    f"score={item.get('score'):.3f} "
                    f"spectacle={item.get('spectacle_value'):.3f} "
                    f"similarity={item.get('similarity'):.3f}"
                )
        attention = context.get("available_attention")
        if attention is not None:
            lines.extend(["", f"Available attention budget: {attention:.3f}"])

        lines.extend(
            [
                "",
                "Required JSON fields:",
                "  - consumed_content_ids: list of ids you actually consumed",
                "  - info_volume: total information volume absorbed (>= 0)",
                "  - perceived_spectacle: weighted spectacle in [0, 1]",
                "  - aggregate_rpe: sum of reward prediction errors",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _render_economic_user(
        agent: AgentState,
        tick: int,
        context: Mapping[str, Any],
    ) -> str:
        prices: Mapping[str, float] = context.get("prices") or {}
        ubi: float = float(context.get("ubi_per_agent", agent.economics.ubi_received))
        survival: float = float(context.get("survival_cost", agent.economics.survival_cost))
        cash: float = float(context.get("cash_balance", agent.economics.cash_balance))

        lines: List[str] = [
            f"# Phase 2 — Economic Decision (tick {tick})",
            "",
            f"Cash balance:   {cash:.2f}",
            f"UBI received:   {ubi:.2f}",
            f"Survival cost:  {survival:.2f}",
            "",
            "Current prices:",
        ]
        if not prices:
            lines.append("  (no goods available)")
        else:
            for good, p in prices.items():
                lines.append(f"  - {good}: {p:.4f}")

        lines.extend(
            [
                "",
                "Allocate exactly 24.0 hours across the four canonical slots:",
                "  labor, creation, spectacle, rest",
                "Decide your spending and savings (both >= 0).",
                "",
                "Required JSON fields:",
                "  - time_allocation: object with keys "
                "{'labor','creation','spectacle','rest'}, values summing to 24.0",
                "  - spending: float >= 0",
                "  - savings:  float >= 0",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _render_reflection_user(
        agent: AgentState,
        tick: int,
        context: Mapping[str, Any],
    ) -> str:
        consumption = context.get(Phase.CONSUMPTION.value, {})
        decision = context.get(Phase.ECONOMIC_DECISION.value, {})

        lines: List[str] = [
            f"# Phase 3 — Reflection (tick {tick})",
            "",
            "Review what just happened this tick and update your self-model.",
            "",
            "Phase 1 summary (consumption):",
            f"  info_volume        = {consumption.get('info_volume')}",
            f"  perceived_spectacle = {consumption.get('perceived_spectacle')}",
            f"  aggregate_rpe      = {consumption.get('aggregate_rpe')}",
            "",
            "Phase 2 summary (economic decision):",
            f"  time_allocation    = {decision.get('time_allocation')}",
            f"  spending           = {decision.get('spending')}",
            f"  savings            = {decision.get('savings')}",
            "",
            "Current cognitive state:",
            f"  cognitive_load     = {agent.cognition.cognitive_load:.3f}",
            f"  burnout_threshold  = {agent.cognition.burnout_threshold:.3f}",
            f"  authenticity_index = {agent.cognition.authenticity_index:.3f}",
            f"  spectacle_immersion= {agent.cognition.spectacle_immersion:.3f}",
            "",
            "Required JSON fields:",
            "  - summary: short natural-language self-assessment (non-empty)",
            "  - new_authenticity_index: float in [0, 1]",
            "  - new_spectacle_immersion: float in [0, 1]",
            "  - identity_anchor_ok: bool — did you stay aligned with the Cathedral anchor?",
        ]
        return "\n".join(lines)
