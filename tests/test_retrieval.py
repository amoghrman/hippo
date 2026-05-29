"""Tests for the hybrid retrieval algorithm."""
import uuid

import pytest

from hippo import Hippo


@pytest.mark.asyncio
async def test_hybrid_score_uses_importance(memory_client: Hippo) -> None:
    """Higher importance pushes a memory up even when BM25 scores are equal."""
    agent_id = f"importance-{uuid.uuid4().hex[:8]}"

    # Same content, different importance
    await memory_client.remember("User works remotely", agent_id=agent_id, importance=0.1)
    await memory_client.remember("User works remotely", agent_id=agent_id, importance=0.9)

    results = await memory_client.recall("remote work", agent_id=agent_id, limit=5)

    assert len(results) == 2
    assert results[0].importance >= results[1].importance


@pytest.mark.asyncio
async def test_recall_empty_returns_empty_list(memory_client: Hippo) -> None:
    """Recall on an agent with no memories returns an empty list."""
    results = await memory_client.recall(
        "anything", agent_id=f"empty-agent-{uuid.uuid4().hex}"
    )
    assert results == []


@pytest.mark.asyncio
async def test_bm25_keyword_boost(memory_client: Hippo) -> None:
    """A memory that shares keywords with the query outranks an unrelated one."""
    agent_id = f"bm25-{uuid.uuid4().hex[:8]}"

    await memory_client.remember(
        "User plays chess and enjoys strategy games",
        agent_id=agent_id,
        importance=0.5,
    )
    await memory_client.remember(
        "The sky turns orange during sunset",
        agent_id=agent_id,
        importance=0.5,
    )

    results = await memory_client.recall(
        "Does the user enjoy chess?", agent_id=agent_id, limit=2
    )

    assert "chess" in results[0].content.lower()


@pytest.mark.asyncio
async def test_scores_are_non_negative(memory_client: Hippo) -> None:
    """All scores returned by recall() are non-negative."""
    agent_id = f"nonneg-{uuid.uuid4().hex[:8]}"
    for i in range(5):
        await memory_client.remember(f"Fact {i}", agent_id=agent_id)

    results = await memory_client.recall("fact", agent_id=agent_id, limit=10)
    assert all(r.score >= 0 for r in results)
