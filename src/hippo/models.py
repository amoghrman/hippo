"""SQLAlchemy ORM models and Pydantic public types."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class MemoryRow(Base):
    """Internal ORM row for the memories table."""

    __tablename__ = "memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(Text, nullable=False)
    user_id = Column(Text, nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(), nullable=True)
    importance = Column(Float, nullable=False, default=0.5)
    # "metadata" clashes with SQLAlchemy's Base.metadata; use metadata_ as attr name
    metadata_ = Column("metadata", JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    superseded_by = Column(UUID(as_uuid=True), ForeignKey("memories.id"), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)

    __table_args__ = (
        Index("idx_memories_agent", "agent_id", "user_id", "is_active", "created_at"),
    )


class ConflictLogRow(Base):
    """Internal ORM row for conflict resolution audit log."""

    __tablename__ = "conflict_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id_old = Column(UUID(as_uuid=True), ForeignKey("memories.id"), nullable=False)
    memory_id_new = Column(UUID(as_uuid=True), ForeignKey("memories.id"), nullable=False)
    decision = Column(Text, nullable=False)  # supersede | merge | coexist
    reason = Column(Text, nullable=True)
    ts = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


# ── Public Pydantic types ──────────────────────────────────────────────────────


class Memory(BaseModel):
    """A memory returned by recall().

    Example::

        results = await hippo.recall("dark mode", agent_id="agent-1")
        for mem in results:
            print(mem.content, mem.score)
    """

    id: uuid.UUID
    agent_id: str
    user_id: Optional[str] = None
    content: str
    importance: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    superseded_by: Optional[uuid.UUID] = None
    is_active: bool = True
    score: float = 0.0  # filled in by recall(), not stored in DB

    model_config = {"from_attributes": True}


class ConflictLog(BaseModel):
    """An entry in the conflict resolution audit log.

    Example::

        log = await hippo.get_conflict_log(agent_id="agent-1")
    """

    id: uuid.UUID
    memory_id_old: uuid.UUID
    memory_id_new: uuid.UUID
    decision: str
    reason: Optional[str] = None
    ts: datetime

    model_config = {"from_attributes": True}
