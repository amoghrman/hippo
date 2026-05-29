"""Conflict resolution — the Hippo differentiator.

On every remember(), we:
  1. Find semantically similar active memories (cosine >= threshold).
  2. Ask an LLM whether the new memory contradicts each candidate.
  3. Apply the resolution: supersede, merge, or coexist.
  4. Log every conflict decision to conflict_log.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession

from .llm.base import LLM
from .models import ConflictLogRow, MemoryRow

logger = logging.getLogger(__name__)


@dataclass
class _Candidate:
    memory_id: UUID
    content: str
    similarity: float


async def _find_similar(
    session: AsyncSession,
    embedding: list[float],
    agent_id: str,
    user_id: str | None,
    exclude_id: UUID,
    threshold: float,
    limit: int,
) -> list[_Candidate]:
    """Return active memories whose cosine similarity >= threshold, excluding exclude_id."""
    embedding_literal = "[" + ",".join(str(v) for v in embedding) + "]"

    sql = text("""
        SELECT
            id,
            content,
            1 - (embedding <=> :embedding ::vector) AS similarity
        FROM memories
        WHERE agent_id    = :agent_id
          AND is_active   = TRUE
          AND id          != :exclude_id
          AND (:user_id ::text IS NULL OR user_id = :user_id)
          AND embedding   IS NOT NULL
          AND 1 - (embedding <=> :embedding ::vector) >= :threshold
        ORDER BY similarity DESC
        LIMIT :limit
    """)

    result = await session.execute(
        sql,
        {
            "embedding": embedding_literal,
            "agent_id": agent_id,
            "user_id": user_id,
            "exclude_id": exclude_id,
            "threshold": threshold,
            "limit": limit,
        },
    )
    return [
        _Candidate(
            memory_id=row["id"],
            content=row["content"],
            similarity=float(row["similarity"]),
        )
        for row in result.mappings().all()
    ]


async def _supersede(session: AsyncSession, old_id: UUID, new_id: UUID) -> None:
    await session.execute(
        update(MemoryRow)
        .where(MemoryRow.id == old_id)
        .values(is_active=False, superseded_by=new_id)
    )


async def _log(
    session: AsyncSession,
    old_id: UUID,
    new_id: UUID,
    decision: str,
    reason: str,
) -> None:
    session.add(
        ConflictLogRow(
            memory_id_old=old_id,
            memory_id_new=new_id,
            decision=decision,
            reason=reason,
        )
    )


async def resolve_conflicts(
    session: AsyncSession,
    new_content: str,
    new_embedding: list[float],
    new_id: UUID,
    agent_id: str,
    user_id: str | None,
    llm: LLM,
    similarity_threshold: float = 0.85,
) -> str | None:
    """Detect conflicts between new_id and existing memories, then resolve them.

    Called after the new memory is flushed (but before commit). Returns merged
    content if a merge decision was made — the caller must update the new
    memory's content and re-embed.

    Example::

        merged = await resolve_conflicts(
            session, "User likes Rust", embedding, new_id,
            "agent-1", "user-42", llm,
        )
        if merged:
            await session.execute(update(MemoryRow).where(...).values(content=merged))
    """
    candidates = await _find_similar(
        session,
        new_embedding,
        agent_id,
        user_id,
        exclude_id=new_id,
        threshold=similarity_threshold,
        limit=5,
    )

    merged_content: str | None = None

    for candidate in candidates:
        result = await llm.check_conflict(candidate.content, new_content)

        if not result.contradicts:
            logger.debug(
                "No conflict: old=%s, new=%s (sim=%.3f)",
                candidate.memory_id,
                new_id,
                candidate.similarity,
            )
            continue

        logger.info(
            "Conflict [%s]: old=%s → new=%s | %s",
            result.resolution,
            candidate.memory_id,
            new_id,
            result.reason,
        )

        if result.resolution == "supersede":
            await _supersede(session, candidate.memory_id, new_id)
            await _log(session, candidate.memory_id, new_id, "supersede", result.reason)

        elif result.resolution == "merge":
            if merged_content is None:  # apply first merge only in MVP
                merged_content = await llm.synthesize_merge(candidate.content, new_content)
            await _supersede(session, candidate.memory_id, new_id)
            await _log(session, candidate.memory_id, new_id, "merge", result.reason)

        else:  # coexist despite contradiction flag
            await _log(session, candidate.memory_id, new_id, "coexist", result.reason)

    return merged_content
