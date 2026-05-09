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
    # --- Cognitive-collapse fallback ---
    # When dynamic temperature is high (stressed agent) the LLM will more
    # often emit malformed JSON. After this many consecutive parse/validation
    # failures we *forcibly* drop temperature to ``collapse_temperature`` and
    # log the incident. This represents the agent's mind giving up on its
    # impulsive trajectory and reverting to a safe, low-entropy answer.
    cognitive_collapse_threshold: int = 3
    collapse_temperature: float = 0.1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay <= 0 or self.max_delay <= 0:
            raise ValueError("delays must be positive")
        if not (0.0 <= self.jitter < 1.0):
            raise ValueError("jitter must be in [0, 1)")
        if self.cognitive_collapse_threshold < 1:
            raise ValueError("cognitive_collapse_threshold must be >= 1")
        if not (0.0 <= self.collapse_temperature <= 2.0):
            raise ValueError("collapse_temperature must be in [0, 2]")

    def delay_for(self, attempt: int) -> float:
        # attempt is 1-indexed, first retry uses base_delay
        raw = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        # symmetric jitter around `raw`
        spread = raw * self.jitter
        return max(0.0, raw + random.uniform(-spread, spread))


# ---------------------------------------------------------------------------
# Dynamic temperature: cognition + debt -> sampling entropy
# ---------------------------------------------------------------------------


# Module-level constants so unit tests can monkey-patch them.
BASE_TEMPERATURE: float = 0.2
TEMPERATURE_CAP: float = 1.2

# Weight on the cognitive-load tail (load > 0.5 contributes linearly).
LOAD_PENALTY_GAIN: float = 1.5
LOAD_PENALTY_KNEE: float = 0.5

# Weight on extreme indebtedness. Debt is normalised by survival cost: a debt
# of 5x survival_cost is treated as the "fully extreme" case.
DEBT_PENALTY_GAIN: float = 0.5
DEBT_EXTREMITY_SCALE: float = 5.0

# Stress directive injected into the system prompt when temperature crosses
# this threshold — this is the "cognitive impairment" speech the model is
# asked to inhabit when its agent is on the edge of burnout / insolvency.
STRESS_DIRECTIVE_THRESHOLD: float = 0.8
STRESS_DIRECTIVE: str = (
    "Твой когнитивный ресурс истощен. "
    "Твои мысли спутаны, ты склонен к импульсивным решениям и логическим ошибкам."
)


def calculate_dynamic_temperature(agent: AgentState) -> float:
    """Map agent stress to LLM sampling temperature.

    Two stressors raise the sampling entropy and thereby make the agent's
    next decision more impulsive and less self-consistent:

    1. **Cognitive load.** Above ``LOAD_PENALTY_KNEE`` (default 0.5) every
       additional unit of load adds ``LOAD_PENALTY_GAIN`` to the temperature.
       Below the knee there is no penalty — well-rested agents stay at the
       deterministic floor.

    2. **Extreme debt.** When ``cash_balance < 0`` we measure the debt in
       multiples of the agent's ``survival_cost``. Five multiples or more is
       considered fully extreme and contributes ``DEBT_PENALTY_GAIN``.

    The result is clipped to ``[BASE_TEMPERATURE, TEMPERATURE_CAP]`` so the
    LLM is never asked to sample below the baseline determinism floor or
    above the cap that empirically destroys structured-output reliability.
    """

    cog_load = agent.cognition.cognitive_load
    load_penalty = max(0.0, cog_load - LOAD_PENALTY_KNEE) * LOAD_PENALTY_GAIN

    survival = max(1e-6, agent.economics.survival_cost)
    debt_magnitude = max(0.0, -agent.economics.cash_balance)
    debt_extremity = min(1.0, debt_magnitude / (DEBT_EXTREMITY_SCALE * survival))
    debt_penalty = debt_extremity * DEBT_PENALTY_GAIN

    raw = BASE_TEMPERATURE + load_penalty + debt_penalty
    return max(BASE_TEMPERATURE, min(TEMPERATURE_CAP, raw))


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
            temperature=calculate_dynamic_temperature(agent),
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
            temperature=calculate_dynamic_temperature(agent),
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
            temperature=calculate_dynamic_temperature(agent),
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
            temperature=calculate_dynamic_temperature(agent),
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
        temperature: Optional[float] = None,
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
            temperature=temperature,
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
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generic structured-output call usable by any schema (phase-agnostic).

        Shared retry + JSON-validation policy for the regular phase graph and
        for one-off macro-event phases (e.g. compliance decisions).

        The ``temperature`` argument lets callers inject the dynamic, agent-
        specific value computed by :func:`calculate_dynamic_temperature`. If
        the value exceeds :data:`STRESS_DIRECTIVE_THRESHOLD` we splice the
        Russian-language stress directive into the system prompt, which
        instructs the LLM to *act* impaired (mirroring the high entropy it is
        also being asked to sample with). After
        :attr:`RetryConfig.cognitive_collapse_threshold` consecutive
        parse/validation failures we forcibly drop the temperature to
        :attr:`RetryConfig.collapse_temperature` and emit a structured warning
        — the "cognitive collapse" event the experiment hangs its analysis on.
        """
        json_schema = schema_cls.model_json_schema()
        schema_name = schema_cls.__name__

        # Resolve effective temperature. None -> instance default.
        effective_temp = self._temperature if temperature is None else temperature
        # Mutable copy so we can drop it on cognitive collapse.
        current_temp = effective_temp

        # Splice the stress directive into the system prompt iff we are
        # operating above the impairment threshold. Doing it once up-front
        # keeps the prompt stable across retries within the *same* call.
        effective_system_prompt = system_prompt
        if current_temp > STRESS_DIRECTIVE_THRESHOLD:
            effective_system_prompt = (
                f"{system_prompt}\n\n[COGNITIVE STATE OVERRIDE]\n{STRESS_DIRECTIVE}"
            )

        # Schema/parse failures are tracked separately from rate-limit errors:
        # only the former should trip the cognitive-collapse circuit breaker,
        # because 429s say nothing about agent stress.
        parse_failures = 0
        collapsed = False
        last_exc: Optional[BaseException] = None

        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                raw = await self._client.complete(
                    system=effective_system_prompt,
                    user=user_message,
                    json_schema=json_schema,
                    schema_name=schema_name,
                    temperature=current_temp,
                )
                parsed = self._parse_json(raw)
                # Validate eagerly so a malformed payload triggers a retry
                # *before* the SimulationLoop gate also rejects it.
                schema_cls.model_validate(parsed)
                return parsed
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = (
                    "429" in exc_str
                    or "rate_limit" in exc_str.lower()
                    or "rate limit" in exc_str.lower()
                )
                is_parse_or_schema = isinstance(
                    exc, (json.JSONDecodeError, ValidationError, ValueError)
                )
                if not (is_parse_or_schema or is_rate_limit):
                    raise

                last_exc = exc
                if is_parse_or_schema:
                    parse_failures += 1

                logger.warning(
                    "LLM phase=%s tick=%d agent=%s attempt=%d/%d "
                    "temp=%.3f parse_failures=%d failed: %s",
                    log_phase, tick, agent.agent_id, attempt,
                    self._retry.max_attempts, current_temp, parse_failures, exc,
                )

                # Cognitive-collapse fallback: only after enough *parse* failures
                # at high temperature. We collapse exactly once per call, then
                # let the remaining attempts run at the safe temperature.
                should_collapse = (
                    is_parse_or_schema
                    and not collapsed
                    and parse_failures >= self._retry.cognitive_collapse_threshold
                    and current_temp > STRESS_DIRECTIVE_THRESHOLD
                )
                if should_collapse:
                    logger.warning(
                        "COGNITIVE_COLLAPSE phase=%s tick=%d agent=%s: "
                        "%d consecutive parse failures at temp=%.3f -> "
                        "falling back to temp=%.3f for remaining attempts.",
                        log_phase, tick, agent.agent_id,
                        parse_failures, current_temp,
                        self._retry.collapse_temperature,
                    )
                    current_temp = self._retry.collapse_temperature
                    # Strip the stress directive — the agent's mind has just
                    # given up on its impulsive trajectory.
                    effective_system_prompt = system_prompt
                    collapsed = True

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
