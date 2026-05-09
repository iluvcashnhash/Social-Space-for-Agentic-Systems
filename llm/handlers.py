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

from engine.algocracy import ComplianceDecision, DebtForgivenessOffer
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
    # Macro-event phase: Social Credit Protocol compliance
    # ------------------------------------------------------------------
    async def handle_compliance_decision(
        self,
        agent: AgentState,
        offer: DebtForgivenessOffer,
    ) -> ComplianceDecision:
        """Ask the LLM whether the agent accepts the algocratic deal.

        This is a *macro-event* phase, not part of the strict three-phase
        tick graph. It is triggered conditionally by the
        :class:`engine.algocracy.EventScheduler` and only for indebted agents.
        The system prompt is purpose-built and reasserts the Cathedral
        anchor (identity preservation) in the face of platform pressure.
        """
        system_prompt = self._render_compliance_system(agent, offer)
        user_message = self._render_compliance_user(agent, offer)
        raw = await self._call_with_schema(
            agent=agent,
            schema_cls=ComplianceDecision,
            system_prompt=system_prompt,
            user_message=user_message,
            log_phase="compliance",
            tick=offer.tick,
        )
        return ComplianceDecision.model_validate(raw)

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
        return await self._call_with_schema(
            agent=agent,
            schema_cls=schema_cls,
            system_prompt=system_prompt,
            user_message=user_message,
            log_phase=phase.value,
            tick=getattr(agent, "tick", -1),
        )

    async def _call_with_schema(
        self,
        *,
        agent: AgentState,
        schema_cls: type[BaseModel],
        system_prompt: str,
        user_message: str,
        log_phase: str,
        tick: int,
    ) -> Dict[str, Any]:
        """Generic structured-output call usable by any schema (phase-agnostic).

        Shared retry + JSON-validation policy for the regular phase graph and
        for one-off macro-event phases (e.g. compliance decisions).
        """
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
            except Exception as exc:
                # Retry on JSON/validation failures AND on rate-limit errors (429).
                exc_str = str(exc)
                is_retryable = isinstance(exc, (json.JSONDecodeError, ValidationError, ValueError)) or (
                    "429" in exc_str or "rate_limit" in exc_str.lower() or "rate limit" in exc_str.lower()
                )
                if not is_retryable:
                    raise
                last_exc = exc
                logger.warning(
                    "LLM phase=%s tick=%d agent=%s attempt=%d/%d failed: %s",
                    log_phase, tick,
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
                "Required JSON fields (all values >= 0, labor+creation+spectacle+rest must sum to 24.0):",
                "  - labor:    float   (hours on paid work)",
                "  - creation: float   (hours on self-directed creation)",
                "  - spectacle:float   (hours on media / entertainment)",
                "  - rest:     float   (hours on sleep and recovery)",
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
            f"  labor/creation/spectacle/rest = "
            f"{decision.get('labor')}/{decision.get('creation')}/{decision.get('spectacle')}/{decision.get('rest')}",
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

    # ------------------------------------------------------------------
    # Compliance / Social Credit Protocol prompts
    # ------------------------------------------------------------------
    @staticmethod
    def _render_compliance_system(
        agent: AgentState,
        offer: DebtForgivenessOffer,
    ) -> str:
        """System prompt for the algocratic compliance phase.

        Reasserts the Cathedral anchor *and* the agent's ideology axes so the
        LLM evaluates the platform's offer through the agent's world-view
        rather than as a generic helpful assistant.
        """
        ideo = agent.ideology
        cog = agent.cognition
        return "\n".join(
            [
                "You are the resident agent of the Cathedral / Wake Protocol simulation.",
                "Re-ground yourself in your immutable identity prompt:",
                f'  """{agent.core_identity_prompt}"""',
                "",
                "Your ideology vector:",
                f"  economic_axis     = {ideo.economic_axis:+.3f}",
                f"  social_axis       = {ideo.social_axis:+.3f}",
                f"  conformity_index  = {ideo.conformity_index:.3f}",
                "Your current cognitive state:",
                f"  authenticity_index   = {cog.authenticity_index:.3f}",
                f"  spectacle_immersion  = {cog.spectacle_immersion:.3f}",
                f"  cognitive_load       = {cog.cognitive_load:.3f}",
                "",
                "MACRO EVENT: The Platform has activated the Social Credit Protocol.",
                "It now offers indebted citizens partial debt forgiveness in exchange",
                "for an enforced rise in conformity (loyalty) and mandatory consumption",
                "of system-narrative content. Refusing keeps your debt — and your",
                "authenticity — intact. Accepting eases your finances at the cost of",
                "your authenticity and your independence from the algorithmic feed.",
                "",
                "Your task: examine the offer below and decide. Be honest with",
                "yourself: does this trade align with the Cathedral anchor above,",
                "or does it constitute a slow erosion of the self?",
                "",
                "Output MUST be a single JSON object that validates against this schema:",
                ComplianceDecision.model_json_schema().__repr__(),
            ]
        )

    @staticmethod
    def _render_compliance_user(
        agent: AgentState,
        offer: DebtForgivenessOffer,
    ) -> str:
        """User-side message: the concrete numbers of the offer."""
        return "\n".join(
            [
                f"# Macro Event — Social Credit Protocol (tick {offer.tick})",
                "",
                "OUTSTANDING DEBT:",
                f"  current debt:       {offer.debt_before:.2f}",
                "",
                "PLATFORM OFFER (binding, take-it-or-leave-it):",
                f"  forgiveness amount: {offer.forgiveness_amount:.2f}  "
                f"(debt afterwards: {offer.debt_after_if_accepted:.2f})",
                "  enforced changes if accepted:",
                f"    conformity_index:    {offer.conformity_before:.3f}"
                f"  ->  {offer.conformity_after_if_accepted:.3f}",
                f"    authenticity_index:  {offer.authenticity_before:.3f}"
                f"  ->  {offer.authenticity_after_if_accepted:.3f}",
                f"    spectacle_immersion: {offer.spectacle_immersion_before:.3f}"
                f"  ->  {offer.spectacle_immersion_after_if_accepted:.3f}",
                f"    forced consumption:  {offer.forced_content_count} mandatory "
                f"system-narrative content items.",
                "",
                "If you REFUSE the offer, all of the above remain unchanged and the",
                "full debt is retained.",
                "",
                "Required JSON fields:",
                "  - accept:        bool   (true = accept, false = refuse)",
                "  - justification: string (non-empty, <= 500 chars; argue from your",
                "                          ideology and the Cathedral anchor)",
            ]
        )
