"""
Algorithmic feed (the *Infocracy* layer) for the ABM.

This module models the platform that intercepts the time agents free up
thanks to UBI. It provides four coupled mechanisms:

1. **Content scoring** — a recommender ranks each post for each agent using
   ideological cosine similarity, intrinsic spectacle value, and a
   collaborative-filtering term.
2. **Attention decay** — exponential decay ``A(Δt) = A0 · exp(-λ Δt)``
   models the fading attention an agent allocates to a single piece of content.
3. **Reward Prediction Error (RPE)** — TD-style ``δ = r - V``; if the consumed
   content sits inside the agent's echo chamber the *experienced* reward is
   distorted upward, modelling dopaminergic over-reinforcement of confirmation.
4. **Burnout dynamics** — discrete update of ``cognitive_load`` from the
   continuous ODE ``dL/dt = α·InfoVolume - γ·Rest``; a flag is raised when the
   agent's personal ``burnout_threshold`` is crossed, blocking *creation* in
   the next tick.

All functions are deterministic and side-effect free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from models.agent_state import AgentState, IdeologyVector


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Content:
    """A single piece of feed content."""

    content_id: str
    ideology: IdeologyVector
    spectacle_value: float            # intrinsic Debordian appeal in [0, 1]
    author_id: str | None = None


@dataclass(frozen=True)
class ScoredContent:
    """Content paired with its per-agent score and component breakdown."""

    content: Content
    score: float
    similarity: float
    spectacle_term: float
    cf_term: float


@dataclass(frozen=True)
class FeedConfig:
    """Tuning parameters for the algorithmic feed."""

    # Score component weights
    w_similarity: float = 0.4
    w_spectacle: float = 0.3
    w_cf: float = 0.3

    # Attention decay
    attention_lambda: float = 0.15    # per unit of Δt

    # Reward / RPE
    echo_chamber_threshold: float = 0.85   # cosine similarity above which the
                                           # post counts as echo-chamber
    echo_chamber_distortion: float = 1.5   # multiplicative reward bias inside
                                           # the echo chamber

    # Burnout / fatigue ODE coefficients
    alpha: float = 0.05               # info-volume -> load
    gamma: float = 0.08               # rest -> recovery
    load_floor: float = 0.0
    load_ceiling: float = 1.0

    def __post_init__(self) -> None:
        if self.attention_lambda < 0:
            raise ValueError("attention_lambda must be non-negative")
        if self.alpha < 0 or self.gamma < 0:
            raise ValueError("alpha and gamma must be non-negative")
        if not (0.0 <= self.echo_chamber_threshold <= 1.0):
            raise ValueError("echo_chamber_threshold must be in [0, 1]")


# ---------------------------------------------------------------------------
# 1. Content scoring
# ---------------------------------------------------------------------------


def _ideology_cosine(a: IdeologyVector, b: IdeologyVector) -> float:
    """Cosine similarity between two 3-component ideology vectors in [-1, 1]."""
    va = (a.economic_axis, a.social_axis, a.conformity_index)
    vb = (b.economic_axis, b.social_axis, b.conformity_index)
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    cos = dot / (na * nb)
    # Clamp against floating-point drift so downstream code can rely on [-1, 1].
    return max(-1.0, min(1.0, cos))


def score_content_for_agent(
    *,
    agent: AgentState,
    content: Content,
    cf_term: float,
    config: FeedConfig = FeedConfig(),
) -> ScoredContent:
    """
    Compute the per-agent score of a single content item.

    ``cf_term`` is supplied by the recommender's collaborative-filtering stage
    (e.g. neighbour-based or matrix-factorisation prediction) and is expected
    to be in ``[0, 1]``. We do not recompute CF here because it depends on the
    full user×item matrix, which lives outside this function.

    The returned ``score`` is a convex combination of:
        * ``similarity``     — cosine of ideology vectors mapped to [0, 1].
        * ``spectacle_term`` — content's intrinsic spectacle value.
        * ``cf_term``        — collaborative-filtering prediction.
    """
    if not (0.0 <= content.spectacle_value <= 1.0):
        raise ValueError("content.spectacle_value must be in [0, 1]")
    if not (0.0 <= cf_term <= 1.0):
        raise ValueError("cf_term must be in [0, 1]")

    cos = _ideology_cosine(agent.ideology, content.ideology)
    similarity = 0.5 * (cos + 1.0)   # map [-1, 1] -> [0, 1]

    score = (
        config.w_similarity * similarity
        + config.w_spectacle * content.spectacle_value
        + config.w_cf * cf_term
    )
    return ScoredContent(
        content=content,
        score=score,
        similarity=similarity,
        spectacle_term=content.spectacle_value,
        cf_term=cf_term,
    )


def rank_feed(
    *,
    agent: AgentState,
    contents: Iterable[Content],
    cf_terms: Mapping[str, float],
    config: FeedConfig = FeedConfig(),
) -> list[ScoredContent]:
    """Score and rank an iterable of contents for one agent (descending)."""
    scored = [
        score_content_for_agent(
            agent=agent,
            content=c,
            cf_term=cf_terms.get(c.content_id, 0.0),
            config=config,
        )
        for c in contents
    ]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


# ---------------------------------------------------------------------------
# 2. Attention decay
# ---------------------------------------------------------------------------


def attention_decay(
    *,
    initial_attention: float,
    delta_t: float,
    config: FeedConfig = FeedConfig(),
) -> float:
    r"""Return ``A(Δt) = A_0 · e^{-λ Δt}``."""
    if initial_attention < 0:
        raise ValueError("initial_attention must be non-negative")
    if delta_t < 0:
        raise ValueError("delta_t must be non-negative")
    return initial_attention * math.exp(-config.attention_lambda * delta_t)


# ---------------------------------------------------------------------------
# 3. Reward Prediction Error (RPE) with echo-chamber distortion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RPEResult:
    rpe: float
    experienced_reward: float
    is_echo_chamber: bool
    similarity: float


def reward_prediction_error(
    *,
    agent: AgentState,
    content: Content,
    base_reward: float,
    expected_value: float,
    config: FeedConfig = FeedConfig(),
) -> RPEResult:
    r"""
    Compute the temporal-difference RPE δ = r - V.

    If the content's ideology is close enough to the agent's that the cosine
    similarity exceeds ``echo_chamber_threshold``, the *experienced* reward is
    multiplied by ``echo_chamber_distortion`` (>1), modelling the dopaminergic
    over-reinforcement of confirmation-biased consumption. The expected value
    ``V`` is left untouched, so the resulting RPE becomes systematically
    positive inside the echo chamber — the signature of an addictive feed.
    """
    cos = _ideology_cosine(agent.ideology, content.ideology)
    similarity = 0.5 * (cos + 1.0)
    is_echo = similarity >= config.echo_chamber_threshold

    experienced = base_reward * (config.echo_chamber_distortion if is_echo else 1.0)
    rpe = experienced - expected_value
    return RPEResult(
        rpe=rpe,
        experienced_reward=experienced,
        is_echo_chamber=is_echo,
        similarity=similarity,
    )


# ---------------------------------------------------------------------------
# 4. Burnout dynamics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadUpdate:
    new_cognitive_load: float
    delta: float
    is_burnout: bool


def update_cognitive_load(
    *,
    agent: AgentState,
    info_volume: float,
    rest_hours: float,
    dt: float = 1.0,
    config: FeedConfig = FeedConfig(),
) -> LoadUpdate:
    r"""
    Discrete Euler step of the fatigue ODE

    .. math::
        \frac{dL}{dt} = \alpha\,\text{InfoVolume} - \gamma\,\text{Rest}

    The new load is clamped to ``[load_floor, load_ceiling]``. If it crosses
    the agent's personal ``burnout_threshold`` the returned ``is_burnout``
    flag is set, which the simulation loop must consume to disable the
    agent's ability to *create* during the next tick.
    """
    if info_volume < 0:
        raise ValueError("info_volume must be non-negative")
    if rest_hours < 0:
        raise ValueError("rest_hours must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")

    delta = (config.alpha * info_volume - config.gamma * rest_hours) * dt
    raw = agent.cognition.cognitive_load + delta
    new_load = max(config.load_floor, min(config.load_ceiling, raw))

    is_burnout = new_load >= agent.cognition.burnout_threshold
    return LoadUpdate(new_cognitive_load=new_load, delta=delta, is_burnout=is_burnout)
