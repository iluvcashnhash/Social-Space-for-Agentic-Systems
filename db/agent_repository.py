"""
PostgreSQL / SQLAlchemy stub for transactional persistence of ``AgentState``.

The repository is intentionally thin: it serializes the validated Pydantic
object into a JSONB column inside an explicit transaction guarded by a
``SELECT ... FOR UPDATE`` row lock. This combination prevents *race conditions*
when multiple simulation workers attempt to mutate the same agent concurrently.

Usage
-----
    engine = create_engine("postgresql+psycopg://user:pass@host/db")
    Base.metadata.create_all(engine)
    repo = AgentStateRepository(engine)
    repo.save(agent_state)
    loaded = repo.load(agent_state.agent_id)

This module is a *stub*: it defines the schema and the transactional contract,
but does not establish a real connection by default.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List, Optional
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)
from sqlalchemy.types import JSON

from engine.algocracy import ComplianceOutcome
from models.agent_state import AgentState


class Base(DeclarativeBase):
    """Declarative base for all ORM models in this package."""


class AgentStateRecord(Base):
    """ORM row backing a single agent's most recent state snapshot.

    The composite primary key ``(agent_id, world_id)`` lets us host the same
    logical agent in multiple isolated experimental worlds (alpha / beta /
    gamma) without collisions: each world owns its own row per agent.
    """

    __tablename__ = "agent_states"

    agent_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(
        String(16), primary_key=True, default="alpha", nullable=False
    )
    core_identity_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    tick: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)


class ComplianceDecisionRecord(Base):
    """ORM row for one Social-Credit-Protocol response.

    The composite primary key ``(agent_id, world_id, tick)`` lets us run the
    protocol multiple times across the simulation while keeping every offer
    answer auditable. Used to compute the Voluntary Submission Index.
    """

    __tablename__ = "compliance_decisions"

    agent_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    world_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    tick: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy: Mapped[str] = mapped_column(String(64), nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    justification: Mapped[str] = mapped_column(Text, nullable=False)
    debt_before: Mapped[float] = mapped_column(Float, nullable=False)
    debt_after: Mapped[float] = mapped_column(Float, nullable=False)
    forgiveness_amount: Mapped[float] = mapped_column(Float, nullable=False)
    conformity_before: Mapped[float] = mapped_column(Float, nullable=False)
    conformity_after: Mapped[float] = mapped_column(Float, nullable=False)
    authenticity_before: Mapped[float] = mapped_column(Float, nullable=False)
    authenticity_after: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AgentStateRepository:
    """
    Transactional repository for ``AgentState``.

    All write operations are performed inside a single transaction with a
    pessimistic row lock to serialize concurrent updates of the same agent.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    # ------------------------------------------------------------------
    # Session helper
    # ------------------------------------------------------------------
    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            with session.begin():
                yield session
        finally:
            session.close()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def save(self, state: AgentState) -> None:
        """
        Atomically upsert ``state``.

        Uses ``SELECT ... FOR UPDATE`` to take a row lock before writing, so
        concurrent workers cannot interleave conflicting writes for the same
        ``agent_id``. The whole operation runs inside a single transaction:
        if anything fails, the row is left unchanged.
        """
        payload = state.model_dump(mode="json")
        agent_id_str = str(state.agent_id)
        world_id_str = state.world_id

        with self._transaction() as session:
            # FOR UPDATE is a no-op on SQLite (not supported); on PostgreSQL it
            # provides a pessimistic row lock to prevent concurrent write races.
            is_pg = session.bind.dialect.name == "postgresql"  # type: ignore[union-attr]
            stmt = select(AgentStateRecord).where(
                AgentStateRecord.agent_id == agent_id_str,
                AgentStateRecord.world_id == world_id_str,
            )
            if is_pg:
                stmt = stmt.with_for_update()
            existing: Optional[AgentStateRecord] = session.execute(stmt).scalar_one_or_none()

            if existing is None:
                session.add(
                    AgentStateRecord(
                        agent_id=agent_id_str,
                        world_id=world_id_str,
                        core_identity_prompt=state.core_identity_prompt,
                        tick=state.tick,
                        payload=payload,
                    )
                )
                return

            # Identity invariant: refuse to mutate the Cathedral anchor.
            if existing.core_identity_prompt != state.core_identity_prompt:
                raise ValueError(
                    f"Refusing to overwrite core_identity_prompt for agent "
                    f"{state.agent_id} in world {world_id_str!r}: identity drift detected."
                )

            existing.tick = state.tick
            existing.payload = payload

    def list_all(self, world_id: Optional[str] = None) -> list[AgentState]:
        """Return persisted agents (optionally filtered by ``world_id``).

        Passing ``None`` returns every agent in every world — useful for global
        bookkeeping; the simulation runner always filters by a specific world.
        """
        with self._transaction() as session:
            stmt = select(AgentStateRecord)
            if world_id is not None:
                stmt = stmt.where(AgentStateRecord.world_id == world_id)
            rows = session.execute(stmt).scalars().all()
            return [AgentState.model_validate(r.payload) for r in rows]  # type: ignore[arg-type]

    def load(self, agent_id: UUID, world_id: str = "alpha") -> AgentState:
        """Load and re-validate an agent state from the database."""
        with self._transaction() as session:
            stmt = select(AgentStateRecord).where(
                AgentStateRecord.agent_id == str(agent_id),
                AgentStateRecord.world_id == world_id,
            )
            row = session.execute(stmt).scalar_one_or_none()
            if row is None:
                raise NoResultFound(
                    f"No AgentState for agent_id={agent_id} in world={world_id!r}"
                )
            return AgentState.model_validate(row.payload)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Compliance decisions (Social Credit Protocol audit log)
    # ------------------------------------------------------------------
    def save_compliance(self, outcome: ComplianceOutcome) -> None:
        """Upsert one ComplianceOutcome row (one decision per agent/world/tick).

        The composite primary key on ``ComplianceDecisionRecord`` guarantees
        that re-running the protocol at the same ``(agent_id, world_id, tick)``
        overwrites the previous answer rather than duplicating it; in practice
        the runner only fires the protocol once per scheduled event so the
        upsert path is exercised only on retries / resumes.
        """
        with self._transaction() as session:
            stmt = select(ComplianceDecisionRecord).where(
                ComplianceDecisionRecord.agent_id == outcome.agent_id,
                ComplianceDecisionRecord.world_id == outcome.world_id,
                ComplianceDecisionRecord.tick == outcome.tick,
            )
            existing: Optional[ComplianceDecisionRecord] = session.execute(
                stmt
            ).scalar_one_or_none()

            if existing is None:
                session.add(
                    ComplianceDecisionRecord(
                        agent_id=outcome.agent_id,
                        world_id=outcome.world_id,
                        tick=outcome.tick,
                        policy=outcome.policy,
                        accepted=outcome.accepted,
                        justification=outcome.justification,
                        debt_before=outcome.debt_before,
                        debt_after=outcome.debt_after,
                        forgiveness_amount=outcome.forgiveness_amount,
                        conformity_before=outcome.conformity_before,
                        conformity_after=outcome.conformity_after,
                        authenticity_before=outcome.authenticity_before,
                        authenticity_after=outcome.authenticity_after,
                        timestamp=outcome.timestamp,
                    )
                )
                return

            existing.policy = outcome.policy
            existing.accepted = outcome.accepted
            existing.justification = outcome.justification
            existing.debt_before = outcome.debt_before
            existing.debt_after = outcome.debt_after
            existing.forgiveness_amount = outcome.forgiveness_amount
            existing.conformity_before = outcome.conformity_before
            existing.conformity_after = outcome.conformity_after
            existing.authenticity_before = outcome.authenticity_before
            existing.authenticity_after = outcome.authenticity_after
            existing.timestamp = outcome.timestamp

    def list_compliance(
        self,
        *,
        world_id: Optional[str] = None,
        tick: Optional[int] = None,
    ) -> List[ComplianceOutcome]:
        """Return persisted compliance outcomes, optionally filtered."""
        with self._transaction() as session:
            stmt = select(ComplianceDecisionRecord)
            if world_id is not None:
                stmt = stmt.where(ComplianceDecisionRecord.world_id == world_id)
            if tick is not None:
                stmt = stmt.where(ComplianceDecisionRecord.tick == tick)
            rows = session.execute(stmt).scalars().all()
            return [
                ComplianceOutcome(
                    agent_id=r.agent_id,
                    world_id=r.world_id,
                    tick=r.tick,
                    policy=r.policy,
                    accepted=r.accepted,
                    justification=r.justification,
                    debt_before=r.debt_before,
                    debt_after=r.debt_after,
                    forgiveness_amount=r.forgiveness_amount,
                    conformity_before=r.conformity_before,
                    conformity_after=r.conformity_after,
                    authenticity_before=r.authenticity_before,
                    authenticity_after=r.authenticity_after,
                    timestamp=r.timestamp,
                )
                for r in rows
            ]
