"""Conflict resolution — the Hippo differentiator.

On every remember(), we:
  1. Find semantically similar active memories (cosine >= threshold).
  2. Ask the LLM whether the new memory contradicts each candidate — in parallel,
     rate-limited to MAX_CONCURRENT_LLM_CALLS concurrent calls.
  3. Categorise: supersede, merge, or coexist.
  4. If any merges: synthesise all merge candidates into a new merged memory row,
     then deactivate both the originals and the temporary new memory.
     This eliminates the in-place-update race condition that could occur when two
     concurrent writers both decide to merge with the same old memory.
  5. Apply supersede decisions, log everything.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession

from .llm.base import LLM
from .models import ConflictLogRow, MemoryRow

if TYPE_CHECKING:
    from .embedders.base import Embedder

logger = logging.getLogger(__name__)

MAX_CONCURRENT_LLM_CALLS = 10


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
    embedder: Embedder,
    importance: float,
    metadata_: dict[str, Any],
    similarity_threshold: float = 0.85,
) -> UUID | None:
    """Detect conflicts between new_id and existing memories, then resolve them.

    Returns the UUID of the canonical memory the caller should surface:
    - ``None``  — no merges occurred; caller should use their original ``new_id``.
    - ``UUID``  — a new merged memory was created; caller should return this ID.
      The temporary ``new_id`` is marked inactive (superseded by the merged row).

    All LLM ``check_conflict`` calls for this invocation run in parallel under a
    shared semaphore (``MAX_CONCURRENT_LLM_CALLS``).  If multiple candidates
    resolve to "merge", their contents are all synthesised into a single new row
    via pairwise ``synthesize_merge`` calls — multi-conflict chaining.
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

    if not candidates:
        return None

    # ── Parallel LLM conflict checks ──────────────────────────────────────────
    sem = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

    async def _check(candidate: _Candidate):
        async with sem:
            return await llm.check_conflict(candidate.content, new_content)

    check_results = await asyncio.gather(*[_check(c) for c in candidates])

    # ── Categorise decisions ──────────────────────────────────────────────────
    to_merge: list[tuple[_Candidate, str]] = []
    to_supersede: list[tuple[_Candidate, str]] = []
    to_coexist: list[tuple[_Candidate, str]] = []

    for candidate, result in zip(candidates, check_results, strict=True):
        if not result.contradicts:
            logger.debug(
                "No conflict: old=%s new=%s (sim=%.3f)",
                candidate.memory_id,
                new_id,
                candidate.similarity,
            )
            continue

        logger.info(
            "Conflict [%s]: old=%s new=%s | %s",
            result.resolution,
            candidate.memory_id,
            new_id,
            result.reason,
        )

        if result.resolution == "merge":
            to_merge.append((candidate, result.reason))
        elif result.resolution == "supersede":
            to_supersede.append((candidate, result.reason))
        else:
            to_coexist.append((candidate, result.reason))

    if not to_merge and not to_supersede and not to_coexist:
        return None

    # ── Determine winner ──────────────────────────────────────────────────────
    winner_id = new_id
    merged_id: UUID | None = None

    if to_merge:
        # Multi-conflict merge: pairwise-accumulate all merge candidates.
        merged_text = new_content
        for candidate, _ in to_merge:
            merged_text = await llm.synthesize_merge(candidate.content, merged_text)

        merged_embedding = await embedder.embed(merged_text)
        merged_id = uuid.uuid4()
        session.add(
            MemoryRow(
                id=merged_id,
                agent_id=agent_id,
                user_id=user_id,
                content=merged_text,
                embedding=merged_embedding,
                importance=importance,
                metadata_=metadata_,
            )
        )
        await session.flush()

        # Deactivate the temporary new memory — the merged row supersedes it.
        await _supersede(session, new_id, merged_id)
        winner_id = merged_id

    # ── Apply supersede decisions ─────────────────────────────────────────────
    for candidate, reason in to_supersede:
        await _supersede(session, candidate.memory_id, winner_id)
        await _log(session, candidate.memory_id, winner_id, "supersede", reason)

    # ── Apply merge decisions ─────────────────────────────────────────────────
    for candidate, reason in to_merge:
        await _supersede(session, candidate.memory_id, winner_id)
        await _log(session, candidate.memory_id, winner_id, "merge", reason)

    # ── Log coexist (contradiction acknowledged but kept) ─────────────────────
    for candidate, reason in to_coexist:
        await _log(session, candidate.memory_id, winner_id, "coexist", reason)

    return merged_id  # None if no merges (caller keeps new_id)
