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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

from sqlalchemy import create_engine

from db import AgentStateRepository, Base
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
                    "time_allocation": {
                        "labor": 8.0,
                        "creation": 4.0,
                        "spectacle": 4.0,
                        "rest": 8.0,
                    },
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
) -> Dict[str, Any]:
    ranked = rank_feed(
        agent=agent,
        contents=feed,
        cf_terms={c.content_id: 0.5 for c in feed},  # neutral CF prior
        config=feed_config,
    )
    return {
        "feed": [
            {
                "content_id": s.content.content_id,
                "score": s.score,
                "spectacle_value": s.spectacle_term,
                "similarity": s.similarity,
            }
            for s in ranked[:10]
        ],
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


class SimulationRunner:
    def __init__(
        self,
        *,
        db_url: str,
        max_ticks: int,
        n_agents: int,
        seed: int,
        economy_config: EconomyConfig = EconomyConfig(),
        feed_config: FeedConfig = FeedConfig(),
    ) -> None:
        self._db_url = db_url
        self._max_ticks = max_ticks
        self._n_agents = n_agents
        self._seed = seed
        self._economy = economy_config
        self._feed_config = feed_config

        self._engine = create_engine(db_url, future=True)
        self._repo = AgentStateRepository(self._engine)
        self._metrics = MetricsLogger()
        self._world = WorldState()
        self._seed_feed: List[Content] = []
        self._shutdown = asyncio.Event()
        self._loop: Optional[SimulationLoop] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def request_shutdown(self) -> None:
        if not self._shutdown.is_set():
            logger.warning("Shutdown requested — finishing current tick gracefully.")
            self._shutdown.set()

    async def setup(self) -> None:
        Base.metadata.create_all(self._engine)
        existing = self._repo.list_all()
        if not existing:
            logger.info("Empty DB — bootstrapping silicon sample (n=%d).", self._n_agents)
            _agents, feed = bootstrap(
                db_url=self._db_url,
                n_agents=self._n_agents,
                seed=self._seed,
            )
            self._seed_feed = list(feed)
        else:
            logger.info("Found %d existing agents — resuming.", len(existing))
            self._seed_feed = generate_seed_content(seed=self._seed)

        self._world.feed = list(self._seed_feed)

        coord = LLMPhaseCoordinator(
            client=make_llm_client(),
            retry=RetryConfig(max_attempts=6, base_delay=1.0, max_delay=30.0, jitter=0.3),
        )
        self._loop = SimulationLoop(
            handlers={
                Phase.CONSUMPTION: coord.handle_consumption,
                Phase.ECONOMIC_DECISION: coord.handle_economic_decision,
                Phase.REFLECTION: coord.handle_reflection,
            },
            metrics_logger=self._metrics,
        )

    # ------------------------------------------------------------------
    # Single tick
    # ------------------------------------------------------------------
    async def _run_one_tick(self, tick: int) -> None:
        assert self._loop is not None

        # A. Load all agents from the DB (re-validated by Pydantic on the way in).
        agents = self._repo.list_all()
        if not agents:
            raise RuntimeError("No agents in DB; cannot run a tick.")

        # B. Macro-economy on previous-tick aggregates.
        demand = _aggregate_demand(self._world.last_transaction_volume, self._world.prices)
        market = run_economic_tick(
            survival_cost=sum(a.economics.survival_cost for a in agents) / len(agents),
            n_agents=len(agents),
            previous_transaction_volume=self._world.last_transaction_volume,
            prices=self._world.prices,
            demand=demand,
            supply=self._world.supply,
            aggregate_labor_hours=self._world.last_aggregate_labor_hours,
            potential_labor_hours=24.0 * len(agents),
            config=self._economy,
        )
        self._world.prices = dict(market.prices.new_prices)
        self._world.supply = dict(market.supply)
        self._world.last_market = market

        # C. Refresh the algorithmic feed from agents' creation outputs.
        self._world.feed = _synthesise_feed(agents, self._seed_feed)

        # D. Parallel micro-level tick across the population.
        contexts = {
            a.agent_id: _per_agent_context(
                a, market=market, feed=self._world.feed, feed_config=self._feed_config
            )
            for a in agents
        }
        per_agent: Dict[UUID, Dict[Phase, PhaseResult]] = await self._loop.run_tick(
            agents,
            tick=tick,
            prices=self._world.prices,
            automation_triggered=market.automation_triggered,
            emission_deficit=market.fiscal.emission_deficit,
            contexts=contexts,
        )

        # E. Apply phase outputs and persist transactionally.
        total_spending = 0.0
        for a in agents:
            results = per_agent.get(a.agent_id)
            if results is None:
                continue
            try:
                updated, spending = _apply_phase_outputs(
                    a, results, market=market, feed_config=self._feed_config
                )
            except Exception:  # validation failure -> keep previous state
                logger.exception("Failed to apply phase outputs for agent %s", a.agent_id)
                continue
            self._repo.save(updated)
            total_spending += spending

        self._world.last_transaction_volume = total_spending
        self._world.last_aggregate_labor_hours = _aggregate_labor(agents)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        await self.setup()
        tick = 0
        try:
            while tick < self._max_ticks and not self._shutdown.is_set():
                logger.info("=== Tick %d / %d ===", tick, self._max_ticks)
                await self._run_one_tick(tick)
                tick += 1
        finally:
            await self._flush_final_state(tick)

    async def _flush_final_state(self, last_tick: int) -> None:
        try:
            agents = self._repo.list_all()
            for a in agents:
                self._repo.save(a)
            logger.info(
                "Final state flushed at tick=%d (agents=%d, metrics_rows=%d).",
                last_tick, len(agents), len(self._metrics.buffer),
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
