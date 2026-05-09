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
from typing import Iterator, Optional
from uuid import UUID

from sqlalchemy import Integer, String, Text, select
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

from models.agent_state import AgentState


class Base(DeclarativeBase):
    """Declarative base for all ORM models in this package."""


class AgentStateRecord(Base):
    """ORM row backing a single agent's most recent state snapshot."""

    __tablename__ = "agent_states"

    agent_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    core_identity_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    tick: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)


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

        with self._transaction() as session:
            # FOR UPDATE is a no-op on SQLite (not supported); on PostgreSQL it
            # provides a pessimistic row lock to prevent concurrent write races.
            is_pg = session.bind.dialect.name == "postgresql"  # type: ignore[union-attr]
            stmt = select(AgentStateRecord).where(
                AgentStateRecord.agent_id == agent_id_str
            )
            if is_pg:
                stmt = stmt.with_for_update()
            existing: Optional[AgentStateRecord] = session.execute(stmt).scalar_one_or_none()

            if existing is None:
                session.add(
                    AgentStateRecord(
                        agent_id=agent_id_str,
                        core_identity_prompt=state.core_identity_prompt,
                        tick=state.tick,
                        payload=payload,
                    )
                )
                return

            # Identity invariant: refuse to mutate the Cathedral anchor.
            if existing.core_identity_prompt != state.core_identity_prompt:
                raise ValueError(
                    f"Refusing to overwrite core_identity_prompt for agent {state.agent_id}: "
                    "identity drift detected."
                )

            existing.tick = state.tick
            existing.payload = payload

    def list_all(self) -> list[AgentState]:
        """Return every persisted agent, re-validated through the Pydantic schema."""
        with self._transaction() as session:
            rows = session.execute(select(AgentStateRecord)).scalars().all()
            return [AgentState.model_validate(r.payload) for r in rows]  # type: ignore[arg-type]

    def load(self, agent_id: UUID) -> AgentState:
        """Load and re-validate an agent state from the database."""
        with self._transaction() as session:
            stmt = select(AgentStateRecord).where(
                AgentStateRecord.agent_id == str(agent_id)
            )
            row = session.execute(stmt).scalar_one_or_none()
            if row is None:
                raise NoResultFound(f"No AgentState for agent_id={agent_id}")
            return AgentState.model_validate(row.payload)  # type: ignore[arg-type]
