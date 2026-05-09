"""
Silicon-sample generator + database bootstrap for the Cathedral / Wake ABM.

Why a *silicon sample*?
-----------------------
Cloning identical agents collapses ideological variance and destroys the
simulation's signal: every tick converges to the same answer. We instead draw
each agent from one of several archetype Gaussians, so the initial population
spans the political-cultural space (left/right, libertarian/authoritarian,
conformist/non-conformist, burnout-prone/resilient).

Run
---
    python -m scripts.init_world --n 50 --seed 42 \\
        --db postgresql+psycopg://user:pass@localhost/cathedral

Default DB URL is taken from the ``DATABASE_URL`` environment variable.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from sqlalchemy import create_engine

from db import AgentStateRepository, Base
from hub.algorithmic_feed import Content
from models.agent_state import (
    AgentState,
    CognitiveProfile,
    DAY_HOURS,
    EconomicProfile,
    IdeologyVector,
)


logger = logging.getLogger("cathedral.init")


# ---------------------------------------------------------------------------
# Archetype catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Archetype:
    """A cluster prior used to sample agents from a Gaussian neighbourhood."""

    name: str
    weight: float                     # sampling weight (un-normalised)
    # Ideology means / stds — clipped to the schema bounds at sample time.
    economic_mu: float
    economic_sigma: float
    social_mu: float
    social_sigma: float
    conformity_mu: float
    conformity_sigma: float
    # Cognitive means / stds
    burnout_threshold_mu: float
    burnout_threshold_sigma: float
    dopamine_mu: float
    dopamine_sigma: float
    spectacle_mu: float
    spectacle_sigma: float
    authenticity_mu: float
    authenticity_sigma: float
    # Time-allocation prior (will be jittered + renormalised to 24h)
    time_prior: Dict[str, float]
    # Identity-prompt templates
    prompts: Tuple[str, ...]


_ARCHETYPES: Tuple[Archetype, ...] = (
    Archetype(
        name="left_creative_burnout_prone",
        weight=1.0,
        economic_mu=-0.55, economic_sigma=0.20,
        social_mu=-0.35, social_sigma=0.25,
        conformity_mu=0.30, conformity_sigma=0.15,
        burnout_threshold_mu=0.55, burnout_threshold_sigma=0.10,
        dopamine_mu=0.40, dopamine_sigma=0.10,
        spectacle_mu=0.55, spectacle_sigma=0.15,
        authenticity_mu=0.65, authenticity_sigma=0.15,
        time_prior={"labor": 6.0, "creation": 6.0, "spectacle": 4.0, "rest": 8.0},
        prompts=(
            "Ты — уставший фрилансер-дизайнер, разочаровавшийся в "
            "корпоративной культуре. Ты ищешь смысл в самовыражении и "
            "скептически относишься к рекламным обещаниям.",
            "Ты — независимый журналист-расследователь, выгоревший после "
            "десяти лет работы в редакциях. Ты ценишь подлинность и "
            "недоверчиво относишься к мейнстрим-нарративам.",
            "Ты — преподаватель гуманитарных наук на полставки, живущий "
            "от гранта к гранту. Тебя раздражает превращение образования "
            "в сервис.",
        ),
    ),
    Archetype(
        name="right_pragmatic_resilient",
        weight=1.0,
        economic_mu=0.55, economic_sigma=0.20,
        social_mu=0.30, social_sigma=0.25,
        conformity_mu=0.55, conformity_sigma=0.15,
        burnout_threshold_mu=0.80, burnout_threshold_sigma=0.08,
        dopamine_mu=0.55, dopamine_sigma=0.10,
        spectacle_mu=0.40, spectacle_sigma=0.15,
        authenticity_mu=0.55, authenticity_sigma=0.15,
        time_prior={"labor": 10.0, "creation": 2.0, "spectacle": 4.0, "rest": 8.0},
        prompts=(
            "Ты — владелец небольшого логистического бизнеса. Ты ценишь "
            "порядок, контракты и собственную эффективность; идеи "
            "перераспределения вызывают у тебя раздражение.",
            "Ты — инженер-производственник на крупном заводе. Ты "
            "доверяешь институтам, ставишь дисциплину выше самовыражения "
            "и не любишь пустых дискуссий.",
            "Ты — ветеран, перешедший в частную охрану. Стабильность, "
            "иерархия и личная ответственность для тебя выше любых "
            "абстрактных идеалов.",
        ),
    ),
    Archetype(
        name="centrist_conformist_spectacle_bound",
        weight=1.2,
        economic_mu=0.05, economic_sigma=0.20,
        social_mu=0.05, social_sigma=0.20,
        conformity_mu=0.80, conformity_sigma=0.10,
        burnout_threshold_mu=0.65, burnout_threshold_sigma=0.10,
        dopamine_mu=0.50, dopamine_sigma=0.10,
        spectacle_mu=0.75, spectacle_sigma=0.10,
        authenticity_mu=0.35, authenticity_sigma=0.12,
        time_prior={"labor": 8.0, "creation": 1.0, "spectacle": 7.0, "rest": 8.0},
        prompts=(
            "Ты — офисный сотрудник среднего звена в международной "
            "компании. Ты любишь стриминги, обсуждаешь сериалы у "
            "кулера и не хочешь конфликтовать с коллегами по поводу "
            "политики.",
            "Ты — менеджер по продукту в SaaS-стартапе. Ты "
            "ориентируешься на тренды, любишь конференции и доверяешь "
            "алгоритмическим рекомендациям больше, чем собственному "
            "вкусу.",
            "Ты — студент-маркетолог на последнем курсе. Социальные "
            "сети — твоя основная среда; ты искренне веришь, что "
            "массовый консенсус обычно прав.",
        ),
    ),
    Archetype(
        name="libertarian_nonconformist_authentic",
        weight=0.8,
        economic_mu=0.10, economic_sigma=0.30,
        social_mu=-0.65, social_sigma=0.20,
        conformity_mu=0.15, conformity_sigma=0.10,
        burnout_threshold_mu=0.70, burnout_threshold_sigma=0.10,
        dopamine_mu=0.45, dopamine_sigma=0.10,
        spectacle_mu=0.30, spectacle_sigma=0.15,
        authenticity_mu=0.80, authenticity_sigma=0.10,
        time_prior={"labor": 6.0, "creation": 5.0, "spectacle": 3.0, "rest": 10.0},
        prompts=(
            "Ты — самоучка-программист, работающий с open-source "
            "проектами. Ты ценишь приватность, скептически относишься "
            "к крупным платформам и любым формам централизации.",
            "Ты — городской фермер и блогер. Ты выбрал жизнь вне "
            "офиса, относишься к рекламе с недоверием и презираешь "
            "массовые тренды.",
            "Ты — независимый исследователь-философ. Тебе важнее "
            "собственная интеллектуальная честность, чем мнение "
            "окружающих или академических институций.",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _gauss_clip(rng: random.Random, mu: float, sigma: float, lo: float, hi: float) -> float:
    return _clip(rng.gauss(mu, sigma), lo, hi)


def _sample_time_allocation(rng: random.Random, prior: Dict[str, float]) -> Dict[str, float]:
    """Jitter a per-archetype prior and renormalise to exactly 24 hours."""
    jittered = {k: max(0.1, v + rng.gauss(0.0, 1.0)) for k, v in prior.items()}
    total = sum(jittered.values())
    return {k: v * DAY_HOURS / total for k, v in jittered.items()}


def _pick_archetype(rng: random.Random) -> Archetype:
    weights = [a.weight for a in _ARCHETYPES]
    return rng.choices(_ARCHETYPES, weights=weights, k=1)[0]


def _sample_agent(rng: random.Random, *, tick: int = 0) -> AgentState:
    arc = _pick_archetype(rng)

    ideology = IdeologyVector(
        economic_axis=_gauss_clip(rng, arc.economic_mu, arc.economic_sigma, -1.0, 1.0),
        social_axis=_gauss_clip(rng, arc.social_mu, arc.social_sigma, -1.0, 1.0),
        conformity_index=_gauss_clip(rng, arc.conformity_mu, arc.conformity_sigma, 0.0, 1.0),
    )
    burnout_threshold = _gauss_clip(
        rng, arc.burnout_threshold_mu, arc.burnout_threshold_sigma, 0.05, 0.99
    )
    cognition = CognitiveProfile(
        cognitive_load=_gauss_clip(rng, 0.3, 0.1, 0.0, burnout_threshold - 1e-3),
        burnout_threshold=burnout_threshold,
        dopamine_baseline=_gauss_clip(rng, arc.dopamine_mu, arc.dopamine_sigma, 0.0, 1.0),
        spectacle_immersion=_gauss_clip(rng, arc.spectacle_mu, arc.spectacle_sigma, 0.0, 1.0),
        authenticity_index=_gauss_clip(rng, arc.authenticity_mu, arc.authenticity_sigma, 0.0, 1.0),
    )
    economics = EconomicProfile(
        cash_balance=_clip(rng.gauss(150.0, 50.0), 0.0, 1e9),
        survival_cost=_clip(rng.gauss(40.0, 5.0), 5.0, 1e9),
        ubi_received=0.0,
        time_allocation=_sample_time_allocation(rng, arc.time_prior),
    )

    prompt = rng.choice(arc.prompts)
    return AgentState(
        core_identity_prompt=f"[{arc.name}] {prompt}",
        ideology=ideology,
        cognition=cognition,
        economics=economics,
        tick=tick,
    )


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------


def generate_population(n: int = 50, *, seed: int = 0) -> List[AgentState]:
    """Draw ``n`` agents from the archetype mixture using a seeded RNG."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = random.Random(seed)
    return [_sample_agent(rng) for _ in range(n)]


def generate_seed_content(*, seed: int = 0) -> List[Content]:
    """Hand-curated starter feed spanning the ideological corners."""
    rng = random.Random(seed)
    catalogue: Tuple[Tuple[str, float, float, float, float], ...] = (
        # (id, econ, social, conformity, spectacle)
        ("post_left_authentic_01",      -0.7, -0.4, 0.20, 0.35),
        ("post_left_viral_02",          -0.5, -0.2, 0.70, 0.85),
        ("post_right_disciplined_03",    0.6,  0.5, 0.55, 0.30),
        ("post_right_outrage_04",        0.7,  0.6, 0.80, 0.90),
        ("post_centrist_lifestyle_05",   0.0,  0.1, 0.85, 0.80),
        ("post_centrist_news_06",        0.05, 0.0, 0.75, 0.55),
        ("post_libertarian_tech_07",     0.2, -0.7, 0.20, 0.40),
        ("post_libertarian_homestead_08",-0.1, -0.6, 0.25, 0.30),
        ("post_authoritarian_security_09", 0.3, 0.8, 0.65, 0.70),
        ("post_eco_lifestyle_10",       -0.3, -0.3, 0.60, 0.65),
        ("post_celebrity_gossip_11",     0.0,  0.0, 0.95, 0.95),
        ("post_finance_grindset_12",     0.6,  0.2, 0.70, 0.75),
    )
    contents: List[Content] = []
    for cid, econ, soc, conf, spec in catalogue:
        contents.append(
            Content(
                content_id=cid,
                ideology=IdeologyVector(
                    economic_axis=_clip(econ + rng.gauss(0.0, 0.05), -1.0, 1.0),
                    social_axis=_clip(soc + rng.gauss(0.0, 0.05), -1.0, 1.0),
                    conformity_index=_clip(conf + rng.gauss(0.0, 0.03), 0.0, 1.0),
                ),
                spectacle_value=_clip(spec + rng.gauss(0.0, 0.03), 0.0, 1.0),
                author_id=None,
            )
        )
    return contents


# ---------------------------------------------------------------------------
# Bootstrap entry point
# ---------------------------------------------------------------------------


def bootstrap(
    *,
    db_url: str,
    n_agents: int = 50,
    seed: int = 0,
    world_ids: Sequence[str] = ("alpha",),
) -> Tuple[Sequence[AgentState], Sequence[Content]]:
    """Create tables, generate one silicon sample, clone it across ``world_ids``.

    The same seeded RNG is used for every world, but the cloned ``AgentState``
    objects are re-validated with a fresh ``world_id`` so the persisted rows
    differ on the composite primary key only. This guarantees that the three
    experimental conditions (Alpha / Beta / Gamma) start from an *identical*
    initial population — a precondition for clean causal comparison.
    """
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)

    base_agents = generate_population(n_agents, seed=seed)
    repo = AgentStateRepository(engine)

    cloned: list[AgentState] = []
    for world_id in world_ids:
        for a in base_agents:
            clone = AgentState.model_validate(
                {**a.model_dump(), "world_id": world_id}
            )
            repo.save(clone)
            cloned.append(clone)

    feed = generate_seed_content(seed=seed)

    logger.info(
        "Bootstrapped %d world(s) %s: %d agents each across %d archetypes, %d seed contents",
        len(world_ids), tuple(world_ids), len(base_agents), len(_ARCHETYPES), len(feed),
    )
    return cloned, feed


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Initialise the ABM database with a silicon sample.")
    p.add_argument("--n", type=int, default=50, help="Number of agents to generate")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument(
        "--db",
        type=str,
        default=os.environ.get("DATABASE_URL", ""),
        help="SQLAlchemy URL (defaults to $DATABASE_URL)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
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
    bootstrap(db_url=args.db, n_agents=args.n, seed=args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
