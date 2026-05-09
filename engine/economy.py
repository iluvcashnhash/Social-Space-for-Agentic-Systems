"""
Deterministic macro-economic engine for the ABM.

Design principles
-----------------
* **Deterministic.** No randomness, no LLM calls. Given identical inputs the
  engine returns identical outputs — agents are never allowed to *hallucinate*
  prices, taxes, or supply.
* **Bounded dynamics.** All update rules are saturating (tanh / clamps), so the
  simulation cannot diverge to infinity in a single tick.
* **Explicit invariants.** Every function returns a typed dataclass; ad-hoc
  dicts are forbidden so downstream code cannot silently corrupt state.

Components
----------
1. ``calculate_ubi_and_taxes`` — UBI sized at 110 % of survival cost, financed
   by a flat transaction tax bounded by ``max_tax_rate``; the residual gap is
   reported as a *monetary-emission deficit*.
2. ``walrasian_price_update`` — bounded tâtonnement using ``tanh`` to prevent
   exponential price explosions.
3. ``endogenous_robotization`` — when aggregate labour collapses below a
   critical floor (e.g. because UBI removes the need to work), the system
   auto-generates a baseline supply, modelling automation of labour.
4. ``run_economic_tick`` — orchestrator combining the three above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping


# ---------------------------------------------------------------------------
# Configuration & data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EconomyConfig:
    """Static, tick-invariant parameters of the economy."""

    # Fiscal
    ubi_enabled: bool = True                  # if False -> no transfers (Alpha world)
    ubi_coverage_ratio: float = 1.10          # UBI = 110% of survival cost
    max_tax_rate: float = 0.60                # hard cap on flat transaction tax

    # Walrasian price update
    beta: float = 1.0                         # tanh slope
    default_eta: float = 0.10                 # firm-specific responsiveness fallback
    min_price: float = 1e-3                   # numerical floor on prices

    # Robotization
    critical_labor_ratio: float = 0.20        # T_w / T_w_potential floor
    automation_fill_factor: float = 1.0       # fraction of demand auto-supplied
    automation_min_supply: float = 1.0        # absolute floor per good when triggered

    def __post_init__(self) -> None:
        if self.ubi_coverage_ratio < 1.0:
            raise ValueError("ubi_coverage_ratio must be >= 1.0 to cover survival")
        if not (0.0 <= self.max_tax_rate <= 1.0):
            raise ValueError("max_tax_rate must be in [0, 1]")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if not (0.0 < self.critical_labor_ratio < 1.0):
            raise ValueError("critical_labor_ratio must be in (0, 1)")


@dataclass(frozen=True)
class FiscalReport:
    """Output of the UBI/tax computation for a single tick."""

    ubi_per_agent: float
    required_budget: float
    applied_tax_rate: float
    tax_revenue: float
    emission_deficit: float          # > 0 => central bank must mint this much
    n_agents: int


@dataclass(frozen=True)
class PriceUpdate:
    """New prices and the excess-demand signal that produced them."""

    new_prices: Dict[str, float]
    excess_demand: Dict[str, float]   # (D - S) / S, raw signal before tanh


@dataclass(frozen=True)
class MarketState:
    """Full snapshot of the market after one economic tick."""

    fiscal: FiscalReport
    prices: PriceUpdate
    supply: Dict[str, float]
    automation_triggered: bool
    automation_goods: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. UBI + flat transaction tax
# ---------------------------------------------------------------------------


def calculate_ubi_and_taxes(
    *,
    survival_cost: float,
    n_agents: int,
    previous_transaction_volume: float,
    config: EconomyConfig = EconomyConfig(),
) -> FiscalReport:
    """
    Compute the UBI per agent and the flat tax rate needed to fund it.

    Logic
    -----
    * UBI per agent  = ``ubi_coverage_ratio * survival_cost`` (≥ 110 % of needs).
    * Required budget = ``UBI * n_agents``.
    * Ideal tax rate  = ``required_budget / previous_transaction_volume``.
      If it exceeds ``max_tax_rate`` we cap it and report the funding gap as
      ``emission_deficit`` (to be covered by central-bank money creation).
    * If there were no transactions in the previous cycle, the entire required
      budget falls into ``emission_deficit``.
    """
    if survival_cost < 0:
        raise ValueError("survival_cost must be non-negative")
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    if previous_transaction_volume < 0:
        raise ValueError("previous_transaction_volume must be non-negative")

    # Control world (Alpha): UBI disabled -> pure market subsistence economy.
    if not config.ubi_enabled:
        return FiscalReport(
            ubi_per_agent=0.0,
            required_budget=0.0,
            applied_tax_rate=0.0,
            tax_revenue=0.0,
            emission_deficit=0.0,
            n_agents=n_agents,
        )

    ubi_per_agent = config.ubi_coverage_ratio * survival_cost
    required_budget = ubi_per_agent * n_agents

    if previous_transaction_volume == 0.0:
        return FiscalReport(
            ubi_per_agent=ubi_per_agent,
            required_budget=required_budget,
            applied_tax_rate=0.0,
            tax_revenue=0.0,
            emission_deficit=required_budget,
            n_agents=n_agents,
        )

    ideal_rate = required_budget / previous_transaction_volume
    applied_rate = min(ideal_rate, config.max_tax_rate)
    tax_revenue = applied_rate * previous_transaction_volume
    deficit = max(0.0, required_budget - tax_revenue)

    return FiscalReport(
        ubi_per_agent=ubi_per_agent,
        required_budget=required_budget,
        applied_tax_rate=applied_rate,
        tax_revenue=tax_revenue,
        emission_deficit=deficit,
        n_agents=n_agents,
    )


# ---------------------------------------------------------------------------
# 2. Walrasian (tâtonnement) price update with tanh saturation
# ---------------------------------------------------------------------------


def walrasian_price_update(
    *,
    prices: Mapping[str, float],
    demand: Mapping[str, float],
    supply: Mapping[str, float],
    eta: Mapping[str, float] | None = None,
    config: EconomyConfig = EconomyConfig(),
) -> PriceUpdate:
    r"""
    Apply the bounded tâtonnement update:

    .. math::
        P_{j,t} = P_{j,t-1}\,\Bigl(1 + \eta_j\,\tanh\!\bigl(\beta\,
                  \tfrac{D_{j,t-1} - S_{j,t-1}}{S_{j,t-1}}\bigr)\Bigr)

    The ``tanh`` envelope keeps the multiplicative factor inside
    ``[1 - eta_j, 1 + eta_j]`` regardless of how large the excess demand grows,
    which prevents exponential blow-ups under thin supply.

    A vanishing supply is treated as "extreme scarcity": the price is multiplied
    by ``(1 + eta_j)``. Prices are floored at ``config.min_price``.
    """
    if set(demand.keys()) != set(prices.keys()) or set(supply.keys()) != set(prices.keys()):
        raise ValueError("prices, demand and supply must share the same goods")

    eta = eta or {}
    new_prices: Dict[str, float] = {}
    excess: Dict[str, float] = {}

    for good, p_prev in prices.items():
        if p_prev <= 0:
            raise ValueError(f"price for {good!r} must be positive, got {p_prev}")

        d = demand[good]
        s = supply[good]
        eta_j = eta.get(good, config.default_eta)
        if eta_j < 0:
            raise ValueError(f"eta[{good!r}] must be non-negative")

        if s <= 0.0:
            # Extreme scarcity: saturate to +eta_j (tanh(+inf) = 1).
            z = 1.0
            excess[good] = float("inf") if d > 0 else 0.0
        else:
            raw = (d - s) / s
            excess[good] = raw
            z = math.tanh(config.beta * raw)

        p_new = p_prev * (1.0 + eta_j * z)
        new_prices[good] = max(p_new, config.min_price)

    return PriceUpdate(new_prices=new_prices, excess_demand=excess)


# ---------------------------------------------------------------------------
# 3. Endogenous robotization (automation safety net)
# ---------------------------------------------------------------------------


def endogenous_robotization(
    *,
    aggregate_labor_hours: float,
    potential_labor_hours: float,
    supply: Mapping[str, float],
    demand: Mapping[str, float],
    config: EconomyConfig = EconomyConfig(),
) -> tuple[Dict[str, float], bool, Dict[str, float]]:
    """
    Inject automated supply when human labour collapses.

    Returns
    -------
    (augmented_supply, triggered, automation_contribution)

    Logic
    -----
    * If ``T_w / T_w_potential >= critical_labor_ratio`` -> no intervention.
    * Otherwise the deficit is filled proportionally to the labour shortfall:
      let ``g = (critical_ratio - actual_ratio) / critical_ratio`` ∈ (0, 1].
      For each good ``j`` we add
      ``max(automation_min_supply, automation_fill_factor * g * D_j)``
      to ``S_j``, simulating robots/AI taking over production.
    """
    if potential_labor_hours <= 0:
        raise ValueError("potential_labor_hours must be positive")
    if aggregate_labor_hours < 0:
        raise ValueError("aggregate_labor_hours must be non-negative")
    if set(supply.keys()) != set(demand.keys()):
        raise ValueError("supply and demand must share the same goods")

    ratio = aggregate_labor_hours / potential_labor_hours
    if ratio >= config.critical_labor_ratio:
        return dict(supply), False, {}

    gap = (config.critical_labor_ratio - ratio) / config.critical_labor_ratio
    augmented: Dict[str, float] = {}
    contribution: Dict[str, float] = {}

    for good, s in supply.items():
        d = demand[good]
        auto = max(
            config.automation_min_supply,
            config.automation_fill_factor * gap * max(d, 0.0),
        )
        contribution[good] = auto
        augmented[good] = s + auto

    return augmented, True, contribution


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------


def run_economic_tick(
    *,
    survival_cost: float,
    n_agents: int,
    previous_transaction_volume: float,
    prices: Mapping[str, float],
    demand: Mapping[str, float],
    supply: Mapping[str, float],
    aggregate_labor_hours: float,
    potential_labor_hours: float,
    eta: Mapping[str, float] | None = None,
    config: EconomyConfig = EconomyConfig(),
) -> MarketState:
    """
    Execute one full deterministic economic tick.

    Order of operations matters:

    1. Robotization is evaluated *first* on the previous-tick supply, so that
       the price update sees the supply that will actually be available.
    2. Walrasian price update is computed against the (possibly augmented)
       supply.
    3. Fiscal block is independent and runs on the previous-tick transaction
       volume.
    """
    augmented_supply, triggered, contribution = endogenous_robotization(
        aggregate_labor_hours=aggregate_labor_hours,
        potential_labor_hours=potential_labor_hours,
        supply=supply,
        demand=demand,
        config=config,
    )

    price_update = walrasian_price_update(
        prices=prices,
        demand=demand,
        supply=augmented_supply,
        eta=eta,
        config=config,
    )

    fiscal = calculate_ubi_and_taxes(
        survival_cost=survival_cost,
        n_agents=n_agents,
        previous_transaction_volume=previous_transaction_volume,
        config=config,
    )

    return MarketState(
        fiscal=fiscal,
        prices=price_update,
        supply=augmented_supply,
        automation_triggered=triggered,
        automation_goods=contribution,
    )
