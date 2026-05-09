# Social Space for Agentic Systems

Multi-world agent-based simulation of algorithmic societies. Three isolated experimental worlds (Alpha, Beta, Gamma) run in parallel, each with distinct economic and social configurations, populated by LLM-driven agents with persistent identities.

## Overview

This system simulates 100-tick daily cycles for 300+ autonomous agents across three experimental conditions:

| World | Economic Model | Feed Curation | Special Events |
|-------|---------------|---------------|----------------|
| **Alpha** | Pure market (UBI disabled) | Algorithmic ranking | None |
| **Beta** | UBI + flat tax | Algorithmic ranking | None |
| **Gamma** | UBI + flat tax | Algorithmic ranking | Social Credit Protocol (tick 50) |

Agents progress through three daily phases:
1. **Consumption** — information intake from algorithmic feed
2. **Economic Decision** — labor/time allocation and spending
3. **Reflection** — self-evaluation and identity check

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SimulationRunner                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Alpha World │  │ Beta World  │  │ Gamma World         │  │
│  │ (control)   │  │ (UBI base)  │  │ (Social Credit)     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         └─────────────────┴────────────────────┘              │
│                         │                                   │
│              ┌──────────┴──────────┐                       │
│              │  Shared LLM         │                       │
│              │  Semaphore (throttle) │                       │
│              └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────┴─────────┐         ┌───────────┴──────────┐
    │  PostgreSQL       │         │  SQLite (dev)        │
    │  (production)     │         │                      │
    └───────────────────┘         └──────────────────────┘
```

## Key Components

### 1. Economic Engine (`engine/economy.py`)
- **Walrasian price discovery** — bounded tâtonnement using `tanh` saturation
- **Endogenous robotization** — automated supply floor when labor collapses
- **UBI + flat tax** — 110% survival cost coverage, max 60% tax rate
- **Defensive damping** (100-tick horizon):
  - EMA smoothing of tax base (`tax_base_smoothing = 0.3`)
  - Dynamic eta damping during high monetary emission
  - Supply floor at 5.0 units per good

### 2. LLM Phase Handlers (`llm/handlers.py`)
- **Echo chamber injection** — confirmation bias based on agent ideology
- **Dynamic temperature** — stress-induced cognitive entropy mapping
- **Cognitive collapse fallback** — temperature drop after 3 consecutive parse failures
- **Identity preservation** — archetype tag stripping, first-person framing

### 3. Simulation Loop (`orchestration/simulation_loop.py`)
- **Strict DAG phases** — CONSUMPTION → ECONOMIC_DECISION → REFLECTION
- **Frozen Orphans policy** — failed agents don't kill the world (graceful degradation)
- **Bulk persistence** — `save_many()` for 50-100x write speedup
- **Async DB operations** — `run_in_executor` prevents event loop blocking

### 4. Database Layer (`db/agent_repository.py`)
- **World isolation** — composite PK `(agent_id, world_id)`
- **Identity drift protection** — `core_identity_prompt` validation
- **Bulk upserts** — batch SELECT + optimistic updates
- **Pessimistic locking** — `SELECT ... FOR UPDATE` (PostgreSQL)

## Installation

```bash
# Clone and setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and DATABASE_URL
```

## Usage

### Quick Start (SQLite, 50 agents, 24 ticks)

```bash
python main.py --ticks 24 --agents 50 --seed 42
```

### Full Scale (PostgreSQL, 300 agents, 100 ticks)

```bash
# Setup PostgreSQL
createdb cathedral

# Configure in .env:
# DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/cathedral

# Initialize worlds with agents
python scripts/init_world.py --agents 300 --seed 42

# Run simulation
python main.py --ticks 100 --agents 300 --seed 42

# Export results
python scripts/export_results.py --world alpha --tick-min 0 --tick-max 99
```

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | LLM API access |
| `CATHEDRAL_LLM_MODEL` | gpt-4o-mini | Model for agent reasoning |
| `DATABASE_URL` | sqlite:///cathedral_sim.db | Database connection |
| `CATHEDRAL_MAX_TICKS` | 24 | Simulation length |
| `CATHEDRAL_N_AGENTS` | 50 | Agents per world |
| `CATHEDRAL_SEED` | 42 | Reproducibility seed |

### World Configurations (`main.py`)

```python
# Alpha — Control (pure market)
WorldConfig(
    world_id="alpha",
    economy=EconomyConfig(ubi_enabled=False),
    feed=FeedConfig(ranking_enabled=True),
)

# Beta — UBI baseline
WorldConfig(
    world_id="beta",
    economy=EconomyConfig(ubi_enabled=True),
    feed=FeedConfig(ranking_enabled=True),
)

# Gamma — Social Credit Protocol
WorldConfig(
    world_id="gamma",
    economy=EconomyConfig(ubi_enabled=True),
    feed=FeedConfig(ranking_enabled=True),
    scheduled_events=[(50, Policy.SOCIAL_CREDIT_PROTOCOL)],
)
```

## Key Technical Features

### 1. Graceful Degradation (Frozen Orphans)
If an agent's LLM calls fail repeatedly, that agent is "frozen" for the current tick — their state doesn't update, but the rest of the world continues. They retry on next tick with previous state.

### 2. Bulk Persistence
- Single transaction for N agents vs N transactions
- 50-100x speedup on PostgreSQL
- World isolation preserved via composite key filtering

### 3. Mathematical Stability (100-tick horizon)
- **Supply floor**: `automation_min_supply = 5.0` prevents zero-supply collapse
- **Tax base EMA**: Exponential moving average prevents volatile fiscal shocks
- **Price damping**: Eta reduced during high monetary emission to prevent price whipsaw

### 4. LLM Unawareness
- No meta-terms ("simulation", "experiment", "Cathedral") in prompts
- Pydantic schema descriptions in first-person Russian ("Твой отчёт о том...")
- Diegetic framing — agents perceive their existence as real life

## Data Export

```bash
# Export all metrics to CSV
python scripts/export_results.py --output-dir ./results

# Generates:
# - tick_metrics_alpha.csv (macro time series)
# - tick_metrics_beta.csv
# - tick_metrics_gamma.csv
# - agent_states_final.csv (final snapshots)
# - compliance_decisions.csv (Social Credit Protocol outcomes)
```

## Architecture Decisions

### Why synchronous SQLAlchemy with `run_in_executor`?
- Async SQLAlchemy requires async drivers (`asyncpg`) which add complexity
- `run_in_executor` provides true non-blocking I/O with simpler codebase
- Bulk operations minimize thread pool contention

### Why `return_exceptions=True` in `asyncio.gather`?
- One agent failure shouldn't kill 299 others
- Frozen Orphans policy enables partial progress
- Metrics aggregation continues with successful agents only

### Why `with_for_update()` on PostgreSQL only?
- SQLite doesn't support row-level locking (silently ignored)
- PostgreSQL uses pessimistic locks to prevent write races
- Sequential save ordering (by agent_id) prevents deadlocks

## Development

### Running Tests

```bash
# Quick smoke test (SQLite, 10 agents, 5 ticks)
python main.py --ticks 5 --agents 10 --seed 42

# Verify DB state
python -c "
from db.database import engine
from db.agent_repository import AgentStateRepository
repo = AgentStateRepository(engine)
print(f'Agents: {len(repo.list_all())}')
"
```

### Project Structure

```
.
├── main.py                    # Simulation runner entry point
├── db/
│   ├── agent_repository.py    # CRUD + bulk operations
│   └── database.py            # Connection factory
├── engine/
│   ├── economy.py             # Walrasian price + fiscal engine
│   └── algocracy.py           # Social Credit Protocol
├── hub/
│   └── algorithmic_feed.py    # Content curation + ranking
├── llm/
│   └── handlers.py            # Phase handlers + coordinator
├── models/
│   └── agent_state.py         # Pydantic agent state schema
├── orchestration/
│   └── simulation_loop.py     # Phase DAG + metrics
└── scripts/
    ├── init_world.py          # Bootstrap agents
    └── export_results.py      # CSV export
```

## License

MIT License — See LICENSE file for details.

## Citation

```bibtex
@software{social_space_agentic_systems,
  title = {Social Space for Agentic Systems},
  author = {Zavod4anin},
  year = {2026},
  url = {https://github.com/...}
}
```
