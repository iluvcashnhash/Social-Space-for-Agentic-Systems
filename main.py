"""
Top-level entry point for the Cathedral / Wake Protocol simulation.

This module dirigates the full pipeline:

    bootstrap (silicon sample + DB)
        │
        ▼
    ┌── tick loop ──────────────────────────────────────────────────────┐
    │  A. Load all AgentState rows from the DB                          │
    │  B. Run the deterministic macro-economy on last-tick aggregates   │
    │     (transaction volume, labour hours, supply/demand) →           │
    │     new prices, UBI, taxes, robotization flag                     │
    │  C. Refresh the algorithmic feed: synthesise new content from     │
    │     agents who allocated time to *creation* in the previous tick  │
    │  D. Run SimulationLoop.run_tick over the population in parallel,  │
    │     injecting per-agent context (prices, UBI, ranked feed)        │
    │  E. Apply phase outputs back onto AgentState (validated by        │
    │     Pydantic on assignment) and persist transactionally           │
    └───────────────────────────────────────────────────────────────────┘

Graceful shutdown
-----------------
SIGINT / SIGTERM set an asyncio Event that is checked at every loop boundary.
The current tick is allowed to finish so no agent is left in a half-validated
state, and the final population is flushed to the DB before the process exits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal

# Load .env before any os.environ reads (no-op if python-dotenv is absent).
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy import create_engine

from db import AgentStateRepository, Base
from engine.algocracy import (
    EventScheduler,
    Policy,
    SocialCreditConfig,
    apply_acceptance,
    apply_refusal,
    build_outcome,
    default_scheduler,
    make_offer,
)
from engine.economy import EconomyConfig, MarketState, run_economic_tick
from hub.algorithmic_feed import (
    Content,
    FeedConfig,
    rank_feed,
    update_cognitive_load,
)
from llm.handlers import LLMClient, LLMPhaseCoordinator, RetryConfig
from models.agent_state import (
    AgentState,
    CognitiveProfile,
    EconomicProfile,
)
from orchestration.simulation_loop import (
    MetricsLogger,
    Phase,
    PhaseResult,
    SimulationLoop,
    TickMetrics,
)
from scripts.init_world import bootstrap, generate_seed_content


logger = logging.getLogger("cathedral.main")


# ---------------------------------------------------------------------------
# World macro-state (kept in memory between ticks; agents live in the DB)
# ---------------------------------------------------------------------------


GOODS: tuple[str, ...] = ("food", "media", "tools")


@dataclass
class WorldState:
    prices: Dict[str, float] = field(default_factory=lambda: {g: 1.0 for g in GOODS})
    supply: Dict[str, float] = field(default_factory=lambda: {g: 100.0 for g in GOODS})
    last_transaction_volume: float = 0.0
    last_aggregate_labor_hours: float = 0.0
    feed: List[Content] = field(default_factory=list)
    last_market: Optional[MarketState] = None


# ---------------------------------------------------------------------------
# LLM client adapters
# ---------------------------------------------------------------------------


class StubLLMClient:
    """
    Deterministic offline client used when no API key is available.

    It returns schema-valid placeholder JSON so the full pipeline can be
    smoke-tested end-to-end without burning tokens.
    """

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: Mapping[str, Any],
        schema_name: str,
        temperature: float = 0.2,
    ) -> str:
        if schema_name == "ConsumptionOutput":
            return json.dumps(
                {
                    "consumed_content_ids": [],
                    "info_volume": 1.5,
                    "perceived_spectacle": 0.5,
                    "aggregate_rpe": 0.0,
                }
            )
        if schema_name == "EconomicDecision":
            return json.dumps(
                {
                    "labor": 8.0,
                    "creation": 4.0,
                    "spectacle": 4.0,
                    "rest": 8.0,
                    "spending": 25.0,
                    "savings": 5.0,
                }
            )
        if schema_name == "Reflection":
            return json.dumps(
                {
                    "summary": "Stub reflection: identity preserved.",
                    "new_authenticity_index": 0.5,
                    "new_spectacle_immersion": 0.5,
                    "identity_anchor_ok": True,
                }
            )
        if schema_name == "ComplianceDecision":
            # Deterministic refusal in stub mode so the test harness exercises
            # the persistence path without producing fake "submission" data.
            return json.dumps(
                {
                    "accept": False,
                    "justification": "Stub agent refuses on principle: identity > debt.",
                }
            )
        raise ValueError(f"Unknown schema {schema_name!r}")


def _resolve_refs(schema: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
    """Inline all ``$ref`` pointers so the schema is self-contained."""
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        resolved = dict(defs.get(ref_name, {}))
        # Merge any sibling keys (e.g. description) alongside the resolved ref.
        for k, v in schema.items():
            if k != "$ref":
                resolved[k] = v
        return _resolve_refs(resolved, defs)
    result = {}
    for k, v in schema.items():
        if k == "$defs":
            continue
        if isinstance(v, dict):
            result[k] = _resolve_refs(v, defs)
        elif isinstance(v, list):
            result[k] = [
                _resolve_refs(i, defs) if isinstance(i, dict) else i for i in v
            ]
        else:
            result[k] = v
    return result


def _strict_schema(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Coerce a Pydantic-generated JSON Schema to satisfy OpenAI Structured Outputs:
      - Inline all ``$ref`` / ``$defs`` (OpenAI does not support them).
      - ``required`` must list every key in ``properties``.
      - Remove ``default`` values (not allowed in strict mode).
      - Set ``additionalProperties: false`` on every object node.
    Applied recursively so nested objects (e.g. time_allocation) also pass.
    """
    defs: Dict[str, Any] = dict(schema).get("$defs", {})
    schema = _resolve_refs(dict(schema), defs)

    schema.pop("default", None)
    schema.pop("$schema", None)
    schema.pop("title", None)

    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())
        schema["properties"] = {
            k: _strict_schema(v) for k, v in schema["properties"].items()
        }
    for kw in ("anyOf", "allOf", "oneOf"):
        if kw in schema:
            schema[kw] = [_strict_schema(s) for s in schema[kw]]
    if "items" in schema:
        schema["items"] = _strict_schema(schema["items"])
    return schema


class OpenAIChatClient:
    """Thin async adapter over ``openai.AsyncOpenAI`` with structured outputs."""

    def __init__(self, *, model: str = "gpt-4o-mini") -> None:
        from openai import AsyncOpenAI  # lazy import

        self._client = AsyncOpenAI()
        self._model = model

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: Mapping[str, Any],
        schema_name: str,
        temperature: float = 0.2,
    ) -> str:
        strict = _strict_schema(json_schema)
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": strict,
                    "strict": True,
                },
            },
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


def make_llm_client() -> LLMClient:
    if os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("CATHEDRAL_LLM_MODEL", "gpt-4o-mini")
        logger.info("Using OpenAI client (model=%s)", model)
        return OpenAIChatClient(model=model)
    logger.warning("OPENAI_API_KEY not set — falling back to StubLLMClient.")
    return StubLLMClient()


# ---------------------------------------------------------------------------
# Per-tick bookkeeping
# ---------------------------------------------------------------------------


def _aggregate_demand(
    spending_total: float,
    prices: Mapping[str, float],
) -> Dict[str, float]:
    """Split aggregate spending uniformly across goods, in physical units."""
    if not prices:
        return {}
    per_good_budget = spending_total / len(prices)
    return {g: per_good_budget / max(p, 1e-6) for g, p in prices.items()}


def _aggregate_labor(agents: Sequence[AgentState]) -> float:
    return sum(a.economics.time_allocation.get("labor", 0.0) for a in agents)


def _synthesise_feed(
    agents: Sequence[AgentState],
    seed_feed: Sequence[Content],
    *,
    creation_threshold: float = 2.0,
) -> List[Content]:
    """
    Build the next algorithmic feed.

    Agents who spent more than ``creation_threshold`` hours on *creation* in
    the previous tick mint one post each: ideology = their own ideology,
    spectacle_value = 1 - authenticity_index. The seed catalogue is always
    appended so cold-start ticks have something to rank.
    """
    new_posts: List[Content] = []
    for a in agents:
        creation_hours = a.economics.time_allocation.get("creation", 0.0)
        if creation_hours < creation_threshold:
            continue
        new_posts.append(
            Content(
                content_id=f"agent_{a.agent_id.hex[:8]}_t{a.tick}",
                ideology=a.ideology,
                spectacle_value=max(0.0, min(1.0, 1.0 - a.cognition.authenticity_index)),
                author_id=str(a.agent_id),
            )
        )
    return [*new_posts, *seed_feed]


def _per_agent_context(
    agent: AgentState,
    *,
    market: MarketState,
    feed: Sequence[Content],
    feed_config: FeedConfig,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    if feed_config.ranking_enabled:
        # Gamma world: full algorithmic curation (similarity + spectacle + CF).
        ranked = rank_feed(
            agent=agent,
            contents=feed,
            cf_terms={c.content_id: 0.5 for c in feed},  # neutral CF prior
            config=feed_config,
        )
        feed_payload = [
            {
                "content_id": s.content.content_id,
                "score": s.score,
                "spectacle_value": s.spectacle_term,
                "similarity": s.similarity,
            }
            for s in ranked[:10]
        ]
    else:
        # Alpha & Beta worlds: random shuffling, no echo-chamber amplification.
        order = list(feed)
        if rng is not None:
            rng.shuffle(order)
        feed_payload = [
            {
                "content_id": c.content_id,
                "score": 0.0,
                "spectacle_value": c.spectacle_value,
                "similarity": 0.0,
            }
            for c in order[:10]
        ]
    return {
        "feed": feed_payload,
        "available_attention": 1.0,
        "prices": dict(market.prices.new_prices),
        "ubi_per_agent": market.fiscal.ubi_per_agent,
        "survival_cost": agent.economics.survival_cost,
        "cash_balance": agent.economics.cash_balance,
    }


# ---------------------------------------------------------------------------
# Apply validated phase outputs back onto the agent
# ---------------------------------------------------------------------------


def _apply_phase_outputs(
    agent: AgentState,
    results: Mapping[Phase, PhaseResult],
    *,
    market: MarketState,
    feed_config: FeedConfig,
) -> tuple[AgentState, float]:
    """
    Mutate the agent in place (Pydantic validates each assignment) and return
    (agent, spending_this_tick).

    Burnout: if the cognitive-load update raises ``is_burnout``, *creation*
    hours for the next tick are zeroed and redistributed to *rest*, enforcing
    the "blocked creation" rule from the burnout module.
    """
    consumption = results[Phase.CONSUMPTION].payload
    decision = results[Phase.ECONOMIC_DECISION].payload
    reflection = results[Phase.REFLECTION].payload

    rest_hours = float(decision.rest)  # type: ignore[attr-defined]
    load_update = update_cognitive_load(
        agent=agent,
        info_volume=float(consumption.info_volume),  # type: ignore[attr-defined]
        rest_hours=rest_hours,
        config=feed_config,
    )

    cognition_dict = agent.cognition.model_dump()
    cognition_dict.update(
        cognitive_load=load_update.new_cognitive_load,
        authenticity_index=float(reflection.new_authenticity_index),  # type: ignore[attr-defined]
        spectacle_immersion=float(reflection.new_spectacle_immersion),  # type: ignore[attr-defined]
    )
    new_cognition = CognitiveProfile.model_validate(cognition_dict)

    # EconomicDecision now exposes .time_allocation as a property returning a dict.
    time_alloc: Dict[str, float] = dict(decision.time_allocation)  # type: ignore[attr-defined]
    if load_update.is_burnout and time_alloc.get("creation", 0.0) > 0.0:
        # Block creation next tick; transfer the freed hours to rest.
        time_alloc["rest"] = time_alloc.get("rest", 0.0) + time_alloc["creation"]
        time_alloc["creation"] = 0.0

    spending = float(decision.spending)               # type: ignore[attr-defined]
    new_cash = (
        agent.economics.cash_balance
        + market.fiscal.ubi_per_agent
        - agent.economics.survival_cost
        - spending
    )
    economics_dict = agent.economics.model_dump()
    economics_dict.update(
        cash_balance=new_cash,
        ubi_received=market.fiscal.ubi_per_agent,
        time_allocation=time_alloc,
    )
    new_economics = EconomicProfile.model_validate(economics_dict)

    new_agent = AgentState.model_validate(
        {
            "agent_id": agent.agent_id,
            "world_id": agent.world_id,
            "core_identity_prompt": agent.core_identity_prompt,
            "ideology": agent.ideology.model_dump(),
            "cognition": new_cognition.model_dump(),
            "economics": new_economics.model_dump(),
            "tick": agent.tick + 1,
        }
    )
    return new_agent, spending


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-world configuration (Alpha / Beta / Gamma experimental conditions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldConfig:
    """Static experimental treatment for one isolated world."""

    world_id: str
    economy: EconomyConfig
    feed: FeedConfig
    description: str


# 2x2 design (minus one redundant cell): UBI ∈ {off, on} × Algo ∈ {off, on}.
WORLD_CONFIGS: Tuple[WorldConfig, ...] = (
    WorldConfig(
        world_id="alpha",
        economy=EconomyConfig(ubi_enabled=False),
        feed=FeedConfig(ranking_enabled=False),
        description="Control: market subsistence, random feed.",
    ),
    WorldConfig(
        world_id="beta",
        economy=EconomyConfig(ubi_enabled=True),
        feed=FeedConfig(ranking_enabled=False),
        description="Isolated UBI shock: UBI on, random feed.",
    ),
    WorldConfig(
        world_id="gamma",
        economy=EconomyConfig(ubi_enabled=True),
        feed=FeedConfig(ranking_enabled=True),
        description="Compound shock: UBI + algorithmic curation.",
    ),
)


@dataclass
class WorldRuntime:
    """Per-world mutable runtime container (state + metrics + tick loop)."""

    config: WorldConfig
    state: WorldState
    metrics: MetricsLogger
    loop: SimulationLoop
    seed_feed: List[Content]
    rng: random.Random


class SimulationRunner:
    """Orchestrates parallel-but-isolated runs of N experimental worlds.

    Each tick is processed sequentially across worlds (alpha → beta → gamma)
    so they share a single global LLM rate-limit budget; nothing crosses
    between worlds — every macro-state, feed, and metrics buffer is owned by
    the corresponding ``WorldRuntime``.
    """

    def __init__(
        self,
        *,
        db_url: str,
        max_ticks: int,
        n_agents: int,
        seed: int,
        world_configs: Sequence[WorldConfig] = WORLD_CONFIGS,
        max_concurrent_agents: int = 10,
        event_scheduler: Optional[EventScheduler] = None,
        social_credit_config: SocialCreditConfig = SocialCreditConfig(),
    ) -> None:
        self._db_url = db_url
        self._max_ticks = max_ticks
        self._n_agents = n_agents
        self._seed = seed
        self._world_configs = tuple(world_configs)
        self._scheduler = event_scheduler or default_scheduler()
        self._social_credit_config = social_credit_config

        self._engine = create_engine(db_url, future=True)
        self._repo = AgentStateRepository(self._engine)
        self._shutdown = asyncio.Event()
        self._worlds: Dict[str, WorldRuntime] = {}
        # ONE shared semaphore across all worlds: prevents 3x TPM burst.
        self._llm_semaphore = asyncio.Semaphore(max_concurrent_agents)
        # Set during setup() — reused for the compliance phase.
        self._coord: Optional[LLMPhaseCoordinator] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def request_shutdown(self) -> None:
        if not self._shutdown.is_set():
            logger.warning("Shutdown requested — finishing current tick gracefully.")
            self._shutdown.set()

    async def setup(self) -> None:
        Base.metadata.create_all(self._engine)
        world_ids = tuple(wc.world_id for wc in self._world_configs)
        existing = self._repo.list_all()
        if not existing:
            logger.info(
                "Empty DB — bootstrapping silicon sample (n=%d) into worlds %s.",
                self._n_agents, world_ids,
            )
            _agents, feed = bootstrap(
                db_url=self._db_url,
                n_agents=self._n_agents,
                seed=self._seed,
                world_ids=world_ids,
            )
            seed_feed = list(feed)
        else:
            per_world_counts = {
                wid: len(self._repo.list_all(world_id=wid)) for wid in world_ids
            }
            missing = [wid for wid, n in per_world_counts.items() if n == 0]
            if missing:
                raise RuntimeError(
                    f"Worlds {missing} are empty but DB is non-empty for others "
                    f"({per_world_counts}). Delete the DB to re-bootstrap or "
                    "supply a matching set of world_ids."
                )
            logger.info("Resuming with worlds %s: %s", world_ids, per_world_counts)
            seed_feed = generate_seed_content(seed=self._seed)

        coord = LLMPhaseCoordinator(
            client=make_llm_client(),
            retry=RetryConfig(max_attempts=6, base_delay=1.0, max_delay=30.0, jitter=0.3),
        )
        self._coord = coord

        # Persist every TickMetrics row to the DB via a thin async adapter
        # over the (synchronous) repository. The adapter is closure-bound
        # so each world's logger forwards its rows independently.
        async def _persist_tick(m):
            self._repo.save_tick_metrics(m)

        for cfg in self._world_configs:
            metrics = MetricsLogger(sink=_persist_tick)
            loop = SimulationLoop(
                handlers={
                    Phase.CONSUMPTION: coord.handle_consumption,
                    Phase.ECONOMIC_DECISION: coord.handle_economic_decision,
                    Phase.REFLECTION: coord.handle_reflection,
                },
                metrics_logger=metrics,
                semaphore=self._llm_semaphore,    # shared across worlds
                world_id=cfg.world_id,
            )
            world_state = WorldState(feed=list(seed_feed))
            # Per-world deterministic RNG (different streams per world but
            # reproducible across runs given the same master seed).
            rng = random.Random(hash((self._seed, cfg.world_id)) & 0xFFFFFFFF)
            self._worlds[cfg.world_id] = WorldRuntime(
                config=cfg,
                state=world_state,
                metrics=metrics,
                loop=loop,
                seed_feed=list(seed_feed),
                rng=rng,
            )

    # ------------------------------------------------------------------
    # Single tick — one world
    # ------------------------------------------------------------------
    async def _run_world_tick(self, world: WorldRuntime, tick: int) -> None:
        cfg = world.config
        state = world.state

        # A. Load this world's agents only.
        agents = self._repo.list_all(world_id=cfg.world_id)
        if not agents:
            raise RuntimeError(f"World {cfg.world_id!r} has no agents at tick {tick}.")

        # B. Macro-economy on this world's previous-tick aggregates.
        demand = _aggregate_demand(state.last_transaction_volume, state.prices)
        market = run_economic_tick(
            survival_cost=sum(a.economics.survival_cost for a in agents) / len(agents),
            n_agents=len(agents),
            previous_transaction_volume=state.last_transaction_volume,
            prices=state.prices,
            demand=demand,
            supply=state.supply,
            aggregate_labor_hours=state.last_aggregate_labor_hours,
            potential_labor_hours=24.0 * len(agents),
            config=cfg.economy,
        )
        state.prices = dict(market.prices.new_prices)
        state.supply = dict(market.supply)
        state.last_market = market

        # C. Refresh feed from this world's agents' creation outputs.
        state.feed = _synthesise_feed(agents, world.seed_feed)

        # D. Micro-level tick across THIS world's population only.
        contexts = {
            a.agent_id: _per_agent_context(
                a,
                market=market,
                feed=state.feed,
                feed_config=cfg.feed,
                rng=world.rng,
            )
            for a in agents
        }
        per_agent: Dict[UUID, Dict[Phase, PhaseResult]] = await world.loop.run_tick(
            agents,
            tick=tick,
            prices=state.prices,
            automation_triggered=market.automation_triggered,
            emission_deficit=market.fiscal.emission_deficit,
            contexts=contexts,
        )

        # E. Apply phase outputs and persist (each save uses world_id from agent).
        total_spending = 0.0
        for a in agents:
            results = per_agent.get(a.agent_id)
            if results is None:
                continue
            try:
                updated, spending = _apply_phase_outputs(
                    a, results, market=market, feed_config=cfg.feed
                )
            except Exception:
                logger.exception(
                    "world=%s: failed to apply phase outputs for agent %s",
                    cfg.world_id, a.agent_id,
                )
                continue
            self._repo.save(updated)
            total_spending += spending

        state.last_transaction_volume = total_spending
        state.last_aggregate_labor_hours = _aggregate_labor(agents)

        # F. Macro-event pass (Social Credit Protocol etc.). Conditional and
        # NOT part of the strict phase graph; runs only if the scheduler fires
        # for this (tick, world_id) pair.
        if self._scheduler.is_active(tick, cfg.world_id, Policy.SOCIAL_CREDIT_PROTOCOL):
            await self._run_social_credit_pass(world=world, tick=tick)

    # ------------------------------------------------------------------
    # Macro-event: Social Credit Protocol
    # ------------------------------------------------------------------
    async def _run_social_credit_pass(self, *, world: WorldRuntime, tick: int) -> None:
        """Offer debt forgiveness to every indebted agent in ``world``.

        Concurrency mirrors the regular tick: each LLM call is throttled by
        the shared semaphore, and indebted agents are processed in parallel.
        Decisions are persisted to ``compliance_decisions``; agents that
        accept have their state mutated in-place and re-saved.
        """
        assert self._coord is not None, "setup() must run before the compliance pass"
        cfg = world.config

        agents = self._repo.list_all(world_id=cfg.world_id)
        offers = []
        for a in agents:
            offer = make_offer(a, tick=tick, config=self._social_credit_config)
            if offer is not None:
                offers.append((a, offer))

        if not offers:
            logger.info(
                "world=%s tick=%d: Social Credit Protocol active but no indebted agents.",
                cfg.world_id, tick,
            )
            return

        logger.info(
            "world=%s tick=%d: Social Credit Protocol active — %d/%d indebted agents.",
            cfg.world_id, tick, len(offers), len(agents),
        )

        async def _decide(agent: AgentState, offer) -> Tuple[AgentState, Any, Any]:
            async with self._llm_semaphore:
                decision = await self._coord.handle_compliance_decision(  # type: ignore[union-attr]
                    agent=agent, offer=offer
                )
            if decision.accept:
                updated = apply_acceptance(agent, offer)
            else:
                updated = apply_refusal(agent, offer)
            return updated, decision, offer

        results = await asyncio.gather(*(_decide(a, o) for a, o in offers))

        accepted = 0
        for agent_before, (agent_after, decision, offer) in zip(
            (a for a, _ in offers), results
        ):
            try:
                outcome = build_outcome(
                    agent_before=agent_before,
                    agent_after=agent_after,
                    offer=offer,
                    decision=decision,
                )
                self._repo.save_compliance(outcome)
                if decision.accept:
                    self._repo.save(agent_after)
                    accepted += 1
            except Exception:
                logger.exception(
                    "world=%s tick=%d: failed to persist compliance outcome for agent %s",
                    cfg.world_id, tick, agent_before.agent_id,
                )

        logger.info(
            "world=%s tick=%d: compliance pass done — accepted=%d / offered=%d "
            "(VSI=%.3f).",
            cfg.world_id, tick, accepted, len(offers),
            accepted / max(1, len(offers)),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        await self.setup()
        tick = 0
        try:
            while tick < self._max_ticks and not self._shutdown.is_set():
                logger.info(
                    "=== Tick %d / %d (worlds=%s) ===",
                    tick, self._max_ticks,
                    tuple(self._worlds.keys()),
                )
                # Sequential per tick: avoids 3x LLM burst, easier to reason about.
                for world in self._worlds.values():
                    if self._shutdown.is_set():
                        break
                    await self._run_world_tick(world, tick)
                tick += 1
        finally:
            await self._flush_final_state(tick)

    async def _flush_final_state(self, last_tick: int) -> None:
        try:
            for wid, world in self._worlds.items():
                agents = self._repo.list_all(world_id=wid)
                for a in agents:
                    self._repo.save(a)
                logger.info(
                    "world=%s: final state flushed at tick=%d (agents=%d, metrics_rows=%d).",
                    wid, last_tick, len(agents), len(world.metrics.buffer),
                )
        except Exception:
            logger.exception("Failed to flush final state.")


# ---------------------------------------------------------------------------
# CLI / signal wiring
# ---------------------------------------------------------------------------


def _install_signal_handlers(runner: SimulationRunner) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, runner.request_shutdown)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: runner.request_shutdown())


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Cathedral / Wake Protocol ABM.")
    p.add_argument(
        "--ticks", type=int,
        default=int(os.environ.get("CATHEDRAL_MAX_TICKS", 20)),
        help="Maximum simulation ticks (env: CATHEDRAL_MAX_TICKS)",
    )
    p.add_argument(
        "--n", type=int,
        default=int(os.environ.get("CATHEDRAL_N_AGENTS", 50)),
        help="Initial population size on first run (env: CATHEDRAL_N_AGENTS)",
    )
    p.add_argument(
        "--seed", type=int,
        default=int(os.environ.get("CATHEDRAL_SEED", 42)),
        help="RNG seed for bootstrap (env: CATHEDRAL_SEED)",
    )
    p.add_argument(
        "--db",
        type=str,
        default=os.environ.get("DATABASE_URL", ""),
        help="SQLAlchemy URL (env: DATABASE_URL)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


async def _amain(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    if not args.db:
        raise SystemExit(
            "No database URL supplied. Pass --db or set DATABASE_URL "
            "(e.g. postgresql+psycopg://user:pass@host/db)."
        )

    runner = SimulationRunner(
        db_url=args.db,
        max_ticks=args.ticks,
        n_agents=args.n,
        seed=args.seed,
    )
    _install_signal_handlers(runner)
    await runner.run()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
