"""Tests for the core Hippo API: remember, recall, forget."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from hippo import Hippo, Memory


@pytest.mark.asyncio
async def test_remember_returns_uuid(memory_client: Hippo) -> None:
    """remember() returns a valid UUID."""
    mem_id = await memory_client.remember(
        "User prefers dark mode",
        agent_id="test-agent",
        user_id="user-1",
    )
    assert isinstance(mem_id, uuid.UUID)


@pytest.mark.asyncio
async def test_remember_and_recall(memory_client: Hippo) -> None:
    """Basic roundtrip: stored content is retrievable via recall."""
    agent_id = f"roundtrip-{uuid.uuid4().hex[:8]}"
    await memory_client.remember(
        "User prefers dark mode",
        agent_id=agent_id,
        user_id="user-1",
    )

    results = await memory_client.recall(
        "dark mode preference",
        agent_id=agent_id,
        user_id="user-1",
    )

    assert len(results) >= 1
    assert isinstance(results[0], Memory)
    assert "dark mode" in results[0].content


@pytest.mark.asyncio
async def test_recall_respects_agent_isolation(memory_client: Hippo) -> None:
    """Memories are isolated by agent_id — agent-B cannot see agent-A's memories."""
    agent_a = f"agent-a-{uuid.uuid4().hex[:8]}"
    agent_b = f"agent-b-{uuid.uuid4().hex[:8]}"

    await memory_client.remember("Secret fact for A", agent_id=agent_a)

    results = await memory_client.recall("Secret fact", agent_id=agent_b)
    assert all(r.agent_id == agent_b for r in results)
    assert not any("Secret fact for A" in r.content for r in results)


@pytest.mark.asyncio
async def test_recall_ranking(memory_client: Hippo) -> None:
    """The most relevant memory (high importance + keyword match) ranks first."""
    agent_id = f"rank-{uuid.uuid4().hex[:8]}"

    # High-importance target with query-matching keywords
    await memory_client.remember(
        "User prefers Python as their primary programming language",
        agent_id=agent_id,
        importance=0.95,
    )

    # Low-importance noise memories with no query keyword overlap
    for i in range(9):
        await memory_client.remember(
            f"The colour of object number {i} is a shade of blue",
            agent_id=agent_id,
            importance=0.05,
        )

    results = await memory_client.recall(
        "What programming language does the user prefer?",
        agent_id=agent_id,
        limit=5,
    )

    assert len(results) >= 1
    assert "Python" in results[0].content, (
        f"Expected Python memory first, got: {results[0].content!r}"
    )


@pytest.mark.asyncio
async def test_recall_only_returns_active(memory_client: Hippo) -> None:
    """Recall never returns memories where is_active=False."""
    agent_id = f"active-{uuid.uuid4().hex[:8]}"

    mem_id = await memory_client.remember("Temporary fact", agent_id=agent_id)
    await memory_client.forget(memory_id=mem_id)

    results = await memory_client.recall("fact", agent_id=agent_id)
    assert not any(r.id == mem_id for r in results)


@pytest.mark.asyncio
async def test_recall_returns_score(memory_client: Hippo) -> None:
    """Each returned Memory has a non-negative score."""
    agent_id = f"score-{uuid.uuid4().hex[:8]}"
    await memory_client.remember("User likes coffee", agent_id=agent_id)

    results = await memory_client.recall("coffee", agent_id=agent_id)
    assert all(r.score >= 0 for r in results)


@pytest.mark.asyncio
async def test_recall_respects_limit(memory_client: Hippo) -> None:
    """recall() returns at most limit results."""
    agent_id = f"limit-{uuid.uuid4().hex[:8]}"
    for i in range(10):
        await memory_client.remember(f"Fact number {i}", agent_id=agent_id)

    results = await memory_client.recall("fact", agent_id=agent_id, limit=3)
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_forget_by_id(memory_client: Hippo) -> None:
    """forget(memory_id=) deactivates exactly that memory."""
    agent_id = f"forget-id-{uuid.uuid4().hex[:8]}"
    mem_id = await memory_client.remember("Fact to delete", agent_id=agent_id)
    await memory_client.remember("Fact to keep", agent_id=agent_id)

    count = await memory_client.forget(memory_id=mem_id)
    assert count == 1

    results = await memory_client.recall("fact", agent_id=agent_id)
    assert not any(r.id == mem_id for r in results)
    assert any("keep" in r.content for r in results)


@pytest.mark.asyncio
async def test_forget_by_filter(memory_client: Hippo) -> None:
    """forget(filter=older_than_days) removes old memories but keeps recent ones."""
    agent_id = f"forget-filter-{uuid.uuid4().hex[:8]}"

    old_id = await memory_client.remember("Old stale memory", agent_id=agent_id)
    recent_id = await memory_client.remember("Fresh recent memory", agent_id=agent_id)

    # Backdate the old memory to 100 days ago
    cutoff_ts = datetime.now(tz=UTC) - timedelta(days=100)
    async with memory_client._sessionmaker() as session:
        async with session.begin():
            await session.execute(
                text("UPDATE memories SET created_at = :ts WHERE id = :id"),
                {"ts": cutoff_ts, "id": old_id},
            )

    count = await memory_client.forget(filter={"agent_id": agent_id, "older_than_days": 90})
    assert count == 1

    results = await memory_client.recall("memory", agent_id=agent_id)
    ids = {r.id for r in results}
    assert recent_id in ids
    assert old_id not in ids


@pytest.mark.asyncio
async def test_forget_no_args_raises(memory_client: Hippo) -> None:
    """forget() with no arguments raises ValueError."""
    with pytest.raises(ValueError):
        await memory_client.forget()


@pytest.mark.asyncio
async def test_metadata_round_trips(memory_client: Hippo) -> None:
    """Metadata stored with remember() is returned in recall()."""
    agent_id = f"meta-{uuid.uuid4().hex[:8]}"
    await memory_client.remember(
        "User is a senior engineer",
        agent_id=agent_id,
        metadata={"source": "onboarding", "confidence": 0.9},
    )

    results = await memory_client.recall("engineer", agent_id=agent_id)
    assert results[0].metadata["source"] == "onboarding"
