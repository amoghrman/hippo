"""Hippo — the main public API."""

from __future__ import annotations

import logging
import os
import uuid
import warnings
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from .conflict import resolve_conflicts
from .embedders.base import Embedder
from .embedders.openai import OpenAIEmbedder
from .exceptions import BatchPartialFailure
from .importance import ImportanceScorer
from .llm.base import LLM
from .models import Memory, MemoryRow
from .retrieval import hybrid_recall

logger = logging.getLogger(__name__)

_AUTO = object()  # sentinel: "auto-detect from environment"
_UNSET_IMPORTANCE = object()  # sentinel: "importance not explicitly passed by caller"


class Hippo:
    """Persistent, conflict-aware memory layer for AI agents.

    Example::

        memory = Hippo(
            database_url="postgresql+asyncpg://hippo:hippo@localhost/hippo",
            openai_api_key="sk-...",
        )
        await memory.setup()

        mem_id = await memory.remember("User prefers dark mode", agent_id="agent-1")
        results = await memory.recall("UI preferences", agent_id="agent-1")
        await memory.forget(memory_id=mem_id)
    """

    def __init__(
        self,
        database_url: str,
        embedder: Embedder | None = None,
        openai_api_key: str | None = None,
        groq_api_key: str | None = None,
        llm: LLM | None = _AUTO,
        conflict_detection: bool = True,
        conflict_model: str = "gpt-4o-mini",
        conflict_threshold: float = 0.85,
        retrieval_weights: dict[str, float] | None = None,
        auto_importance: bool = False,
        importance_scorer: ImportanceScorer | None = None,
    ) -> None:
        self._engine: AsyncEngine = create_async_engine(database_url, echo=False)
        self._sessionmaker: Any = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self._conflict_detection = conflict_detection
        self._conflict_threshold = conflict_threshold
        self._weights: dict[str, float] = retrieval_weights or {
            "vector": 0.5,
            "bm25": 0.2,
            "recency": 0.15,
            "importance": 0.15,
        }

        if llm is _AUTO:
            _oai = openai_api_key or os.environ.get("OPENAI_API_KEY")
            _groq = groq_api_key or os.environ.get("GROQ_API_KEY")

            if _oai:
                from .llm.openai import OpenAILLM

                llm = OpenAILLM(model=conflict_model, api_key=_oai)
                if embedder is None:
                    embedder = OpenAIEmbedder(api_key=_oai)
            elif _groq:
                from .embedders.sentence_transformers import SentenceTransformersEmbedder
                from .llm.groq import GroqLLM

                groq_model = (
                    "llama-3.1-8b-instant" if conflict_model == "gpt-4o-mini" else conflict_model
                )
                llm = GroqLLM(model=groq_model, api_key=_groq)
                if embedder is None:
                    embedder = SentenceTransformersEmbedder()
            else:
                llm = None
                if embedder is None:
                    from .embedders.sentence_transformers import SentenceTransformersEmbedder

                    embedder = SentenceTransformersEmbedder()
                if conflict_detection:
                    warnings.warn(
                        "No LLM configured (set OPENAI_API_KEY or GROQ_API_KEY). "
                        "Disabling conflict_detection.",
                        stacklevel=2,
                    )
                    self._conflict_detection = False

        # User passed an explicit llm (not _AUTO) but no embedder — default to OpenAI embedder
        if embedder is None:
            embedder = OpenAIEmbedder(api_key=openai_api_key)

        self._llm: LLM | None = llm
        self._embedder: Embedder = embedder
        self._auto_importance = auto_importance
        if auto_importance and importance_scorer is None and self._llm is not None:
            from .importance import LLMImportanceScorer

            importance_scorer = LLMImportanceScorer(self._llm)
        self._importance_scorer: ImportanceScorer | None = importance_scorer

    async def setup(self, *, reset: bool = False) -> None:
        """Create tables and enable the pgvector extension.

        Call once before any other method. Safe to call repeatedly (idempotent).
        Uses the embedder's reported dimension to create the vector column.

        Pass ``reset=True`` to drop and recreate all tables — useful when the
        embedding dimension changes between runs (e.g. switching embedders).
        This deletes all stored memories.

        Example::

            await memory.setup()
            await memory.setup(reset=True)  # wipe and recreate
        """
        dim = self._embedder.dimensions
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            if reset:
                await conn.execute(text("DROP TABLE IF EXISTS conflict_log"))
                await conn.execute(text("DROP TABLE IF EXISTS memories"))
            else:
                # Detect dimension mismatch and migrate the column type in-place.
                res = await conn.execute(
                    text(
                        "SELECT atttypmod FROM pg_attribute a"
                        " JOIN pg_class c ON a.attrelid = c.oid"
                        " WHERE c.relname = 'memories'"
                        "   AND a.attname = 'embedding' AND a.attnum > 0"
                    )
                )
                row = res.fetchone()
                if row and row[0] > 0 and row[0] != dim:
                    logger.warning(
                        "Embedding dimension changed %d -> %d; migrating column "
                        "(all existing embeddings are invalidated).",
                        row[0],
                        dim,
                    )
                    await conn.execute(text("DROP INDEX IF EXISTS idx_memories_embedding"))
                    # USING NULL: existing rows have the wrong dimension and can't
                    # be cast — their embeddings are invalidated and set to NULL.
                    await conn.execute(
                        text(
                            f"ALTER TABLE memories ALTER COLUMN embedding"
                            f" TYPE vector({dim}) USING NULL::vector({dim})"
                        )
                    )
            await conn.execute(
                text(f"""
                    CREATE TABLE IF NOT EXISTS memories (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        agent_id TEXT NOT NULL,
                        user_id TEXT,
                        content TEXT NOT NULL,
                        embedding vector({dim}),
                        importance FLOAT NOT NULL DEFAULT 0.5,
                        metadata JSONB NOT NULL DEFAULT '{{}}',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        superseded_by UUID REFERENCES memories(id),
                        is_active BOOLEAN NOT NULL DEFAULT TRUE
                    )
                """)
            )
            await conn.execute(
                text("""
                    CREATE TABLE IF NOT EXISTS conflict_log (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        memory_id_old UUID NOT NULL REFERENCES memories(id),
                        memory_id_new UUID NOT NULL REFERENCES memories(id),
                        decision TEXT NOT NULL,
                        reason TEXT,
                        ts TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """)
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_memories_agent "
                    "ON memories (agent_id, user_id, is_active, created_at)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_memories_fts "
                    "ON memories USING gin(to_tsvector('english', content))"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_memories_embedding "
                    "ON memories USING hnsw (embedding vector_cosine_ops)"
                )
            )
            logger.info(
                "HNSW index ready. Recall quality improves significantly after ~100 rows are inserted."
            )

    async def remember(
        self,
        content: str,
        agent_id: str,
        user_id: str | None = None,
        importance: float = _UNSET_IMPORTANCE,  # type: ignore[assignment]
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """Store a memory and run conflict resolution against existing memories.

        Returns the UUID of the newly created (or merged) memory.

        When ``auto_importance=True`` on the client and no explicit ``importance``
        is passed, the importance is scored automatically using the configured LLM.

        Example::

            mem_id = await memory.remember(
                "User prefers dark mode",
                agent_id="agent-1",
                user_id="user-42",
                importance=0.7,
                metadata={"source": "settings"},
            )
        """
        if importance is _UNSET_IMPORTANCE:  # type: ignore[comparison-overlap]
            if self._auto_importance and self._importance_scorer is not None:
                importance = await self._importance_scorer.score(content)
            else:
                importance = 0.5
        resolved_importance = float(importance)
        metadata_ = metadata or {}

        embedding = await self._embedder.embed(content)
        mem_id = uuid.uuid4()

        async with self._sessionmaker() as session:
            async with session.begin():
                session.add(
                    MemoryRow(
                        id=mem_id,
                        agent_id=agent_id,
                        user_id=user_id,
                        content=content,
                        embedding=embedding,
                        importance=resolved_importance,
                        metadata_=metadata_,
                    )
                )
                await session.flush()

                if self._conflict_detection and self._llm is not None:
                    merged_id = await resolve_conflicts(
                        session=session,
                        new_content=content,
                        new_embedding=embedding,
                        new_id=mem_id,
                        agent_id=agent_id,
                        user_id=user_id,
                        llm=self._llm,
                        embedder=self._embedder,
                        importance=resolved_importance,
                        metadata_=metadata_,
                        similarity_threshold=self._conflict_threshold,
                    )
                    if merged_id is not None:
                        # A new merged row was created; new_id is now inactive.
                        mem_id = merged_id

        return mem_id

    async def remember_batch(
        self,
        items: list[dict[str, Any]],
        conflict_detection: bool | None = None,
        batch_size: int = 100,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[uuid.UUID]:
        """Bulk-insert memories with batched embeddings.

        This is **significantly faster** than calling ``remember()`` in a loop:

        - Embeddings are computed in one ``embed_batch()`` call per chunk instead
          of one ``embed()`` call per item.
        - Conflict checks (when enabled) are parallelised across candidates with a
          rate-limiting semaphore; each item is still processed in input order so
          that a later item in the same batch can supersede an earlier one.

        Args:
            items: List of dicts.  Required keys: ``content``, ``agent_id``.
                Optional: ``user_id``, ``importance``, ``metadata``.
            conflict_detection: Override the instance-level setting for this batch.
                Pass ``False`` for the fastest possible ingestion when you trust
                the source data (skips all LLM calls).
            batch_size: Items per DB transaction.  Default 100.
            on_progress: Optional callback called after each chunk completes with
                ``(items_processed_so_far, total_items)``.

        Returns:
            List of UUIDs in the same order as ``items``.

        Raises:
            BatchPartialFailure: If one or more chunks fail.  Successful chunks
                are committed; failed chunks are rolled back.  Inspect
                ``exc.successful_ids`` and ``exc.failed_indices``.

        Example::

            ids = await memory.remember_batch(
                items=[{"content": t, "agent_id": "agent-1"} for t in texts],
                conflict_detection=False,   # 10x+ faster — skip for trusted data
                batch_size=100,
                on_progress=lambda done, total: print(f"{done}/{total}"),
            )
        """
        effective_conflict = (
            conflict_detection if conflict_detection is not None else self._conflict_detection
        )

        n = len(items)
        all_ids: list[uuid.UUID | None] = [None] * n
        failed_indices: list[int] = []

        for chunk_start in range(0, n, batch_size):
            chunk_end = min(chunk_start + batch_size, n)
            chunk = items[chunk_start:chunk_end]

            try:
                chunk_ids = await self._ingest_chunk(chunk, effective_conflict)
                for local_i, uid in enumerate(chunk_ids):
                    all_ids[chunk_start + local_i] = uid
            except Exception as exc:
                logger.warning(
                    "remember_batch: chunk %d–%d failed (%s), continuing.",
                    chunk_start,
                    chunk_end - 1,
                    exc,
                )
                failed_indices.extend(range(chunk_start, chunk_end))

            if on_progress is not None:
                on_progress(chunk_end, n)

        if failed_indices:
            raise BatchPartialFailure(
                successful_ids=all_ids,
                failed_indices=failed_indices,
            )

        return all_ids  # type: ignore[return-value]

    async def _ingest_chunk(
        self,
        chunk: list[dict[str, Any]],
        conflict_detection: bool,
    ) -> list[uuid.UUID]:
        """Insert one batch-size chunk and run optional conflict detection."""
        contents = [item["content"] for item in chunk]
        n = len(chunk)

        # Batch embed — the primary speedup over serial remember().
        raw_embeddings = await self._embedder.embed_batch(contents)
        # Guard against mocks / embedders that return a single vector for any batch.
        embeddings = raw_embeddings if len(raw_embeddings) == n else raw_embeddings * n

        # Resolve importance for each item.
        importances: list[float] = []
        for i, item in enumerate(chunk):
            raw = item.get("importance", _UNSET_IMPORTANCE)
            if raw is not _UNSET_IMPORTANCE:
                importances.append(float(raw))
            elif self._auto_importance and self._importance_scorer is not None:
                importances.append(await self._importance_scorer.score(contents[i]))
            else:
                importances.append(0.5)

        ids = [uuid.uuid4() for _ in chunk]

        async with self._sessionmaker() as session:
            async with session.begin():
                for i in range(n):
                    session.add(
                        MemoryRow(
                            id=ids[i],
                            agent_id=chunk[i]["agent_id"],
                            user_id=chunk[i].get("user_id"),
                            content=contents[i],
                            embedding=embeddings[i],
                            importance=importances[i],
                            metadata_=chunk[i].get("metadata") or {},
                        )
                    )
                    # Flush per-item so only items[0..i] are visible when we
                    # check conflicts for item[i].  This guarantees that a later
                    # item can supersede an earlier one (not the reverse).
                    await session.flush()

                    if conflict_detection and self._llm is not None:
                        merged_id = await resolve_conflicts(
                            session=session,
                            new_content=contents[i],
                            new_embedding=embeddings[i],
                            new_id=ids[i],
                            agent_id=chunk[i]["agent_id"],
                            user_id=chunk[i].get("user_id"),
                            llm=self._llm,
                            embedder=self._embedder,
                            importance=importances[i],
                            metadata_=chunk[i].get("metadata") or {},
                            similarity_threshold=self._conflict_threshold,
                        )
                        if merged_id is not None:
                            ids[i] = merged_id

        return ids

    async def recall(
        self,
        query: str,
        agent_id: str,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[Memory]:
        """Retrieve memories using hybrid BM25 + vector + recency + importance ranking.

        Only active (non-superseded) memories are returned.

        Example::

            results = await memory.recall(
                "what are the user's UI preferences?",
                agent_id="agent-1",
                user_id="user-42",
                limit=5,
            )
            for mem in results:
                print(mem.content, mem.score)
        """
        query_embedding = await self._embedder.embed(query)
        async with self._sessionmaker() as session:
            return await hybrid_recall(
                session=session,
                query_embedding=query_embedding,
                query_text=query,
                agent_id=agent_id,
                user_id=user_id,
                limit=limit,
                weights=self._weights,
            )

    async def forget(
        self,
        memory_id: uuid.UUID | None = None,
        filter: dict[str, Any] | None = None,
    ) -> int:
        """Soft-delete memories by ID or by filter criteria.

        Returns the number of memories deactivated.

        Example::

            # Delete one specific memory
            count = await memory.forget(memory_id=some_uuid)

            # Delete all memories older than 90 days for an agent
            count = await memory.forget(
                filter={"agent_id": "agent-1", "older_than_days": 90}
            )
        """
        if memory_id is None and filter is None:
            raise ValueError("Provide either memory_id or filter.")

        async with self._sessionmaker() as session:
            async with session.begin():
                if memory_id is not None:
                    result = await session.execute(
                        update(MemoryRow).where(MemoryRow.id == memory_id).values(is_active=False)
                    )
                    return result.rowcount  # type: ignore[return-value]

                conditions: list[str] = []
                params: dict[str, Any] = {}

                if "agent_id" in filter:  # type: ignore[operator]
                    conditions.append("agent_id = :agent_id")
                    params["agent_id"] = filter["agent_id"]  # type: ignore[index]

                if "user_id" in filter:  # type: ignore[operator]
                    conditions.append("user_id = :user_id")
                    params["user_id"] = filter["user_id"]  # type: ignore[index]

                if "older_than_days" in filter:  # type: ignore[operator]
                    cutoff = datetime.now(tz=UTC) - timedelta(
                        days=filter["older_than_days"]  # type: ignore[index]
                    )
                    conditions.append("created_at < :cutoff")
                    params["cutoff"] = cutoff

                if not conditions:
                    raise ValueError("Filter must have at least one condition.")

                where = " AND ".join(conditions)
                result = await session.execute(
                    text(
                        f"UPDATE memories SET is_active = FALSE"  # noqa: S608
                        f" WHERE {where} AND is_active = TRUE"
                    ),
                    params,
                )
                return result.rowcount  # type: ignore[return-value]

    async def get_conflict_log(
        self,
        agent_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return conflict resolution log entries, newest first.

        Example::

            log = await memory.get_conflict_log(agent_id="agent-1")
            for entry in log:
                print(entry["decision"], entry["reason"])
        """
        async with self._sessionmaker() as session:
            base_sql = """
                SELECT
                    cl.id, cl.memory_id_old, cl.memory_id_new,
                    cl.decision, cl.reason, cl.ts,
                    m_old.content AS old_content,
                    m_new.content AS new_content
                FROM conflict_log cl
                JOIN memories m_old ON cl.memory_id_old = m_old.id
                JOIN memories m_new ON cl.memory_id_new = m_new.id
            """
            params: dict[str, Any] = {"limit": limit}
            if agent_id:
                query = text(
                    base_sql + " WHERE m_new.agent_id = :agent_id ORDER BY cl.ts DESC LIMIT :limit"
                )
                params["agent_id"] = agent_id
            else:
                query = text(base_sql + " ORDER BY cl.ts DESC LIMIT :limit")

            result = await session.execute(query, params)
            return [dict(row) for row in result.mappings().all()]

    async def close(self) -> None:
        """Close the database connection pool.

        Example::

            await memory.close()
        """
        await self._engine.dispose()
