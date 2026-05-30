"""Hybrid retrieval: vector similarity + BM25 + recency decay + importance."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Memory


async def hybrid_recall(
    session: AsyncSession,
    query_embedding: list[float],
    query_text: str,
    agent_id: str,
    user_id: str | None,
    limit: int,
    weights: dict[str, float],
) -> list[Memory]:
    """Run a hybrid-scored retrieval and return ranked Memory objects.

    Score formula (weights configurable)::

        score = w_vec * cosine_sim
              + w_bm25 * ts_rank (normalised)
              + w_recency * exp(-age_days / 30)
              + w_importance * importance

    Example::

        results = await hybrid_recall(
            session, embedding, "dark mode", "agent-1", None, 5,
            {"vector": 0.5, "bm25": 0.2, "recency": 0.15, "importance": 0.15},
        )
    """
    w_vec = weights.get("vector", 0.5)
    w_bm25 = weights.get("bm25", 0.2)
    w_recency = weights.get("recency", 0.15)
    w_importance = weights.get("importance", 0.15)

    embedding_literal = "[" + ",".join(str(v) for v in query_embedding) + "]"

    sql = text("""
        WITH base AS (
            SELECT
                id,
                agent_id,
                user_id,
                content,
                importance,
                metadata,
                created_at,
                updated_at,
                superseded_by,
                is_active,
                1 - (embedding <=> :embedding ::vector)                          AS vector_sim,
                ts_rank(
                    to_tsvector('english', content),
                    plainto_tsquery('english', :query),
                    1
                )                                                                AS bm25_raw,
                EXP(
                    -EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 / 30.0
                )                                                                AS recency
            FROM memories
            WHERE agent_id = :agent_id
              AND is_active = TRUE
              AND (:user_id ::text IS NULL OR user_id = :user_id)
              AND embedding IS NOT NULL
        )
        SELECT *,
            (
                :w_vec       * vector_sim
              + :w_bm25      * LEAST(bm25_raw, 1.0)
              + :w_recency   * recency
              + :w_importance * importance
            ) AS score
        FROM base
        ORDER BY score DESC
        LIMIT :limit
    """)

    result = await session.execute(
        sql,
        {
            "embedding": embedding_literal,
            "query": query_text,
            "agent_id": agent_id,
            "user_id": user_id,
            "w_vec": w_vec,
            "w_bm25": w_bm25,
            "w_recency": w_recency,
            "w_importance": w_importance,
            "limit": limit,
        },
    )

    memories: list[Memory] = []
    for row in result.mappings().all():
        mem = Memory(
            id=row["id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            content=row["content"],
            importance=row["importance"],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            superseded_by=row["superseded_by"],
            is_active=row["is_active"],
            score=float(row["score"]),
        )
        memories.append(mem)

    return memories
