"""Tests for remember_batch() — bulk ingestion."""

import uuid
from unittest.mock import AsyncMock

import pytest

from hippo import BatchPartialFailure, Hippo
from hippo.llm.base import ConflictResult

from .conftest import FIXED_VEC

# ── Basic ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remember_batch_basic(memory_client: Hippo) -> None:
    """50 items inserted, IDs returned in input order."""
    agent_id = f"batch-basic-{uuid.uuid4().hex[:8]}"
    items = [{"content": f"Memory {i}", "agent_id": agent_id} for i in range(50)]

    ids = await memory_client.remember_batch(items, conflict_detection=False)

    assert len(ids) == 50
    assert all(isinstance(uid, uuid.UUID) for uid in ids)
    # IDs must be distinct.
    assert len(set(ids)) == 50

    # All memories must be retrievable.
    results = await memory_client.recall("Memory", agent_id=agent_id, limit=50)
    stored_ids = {r.id for r in results}
    assert set(ids).issubset(stored_ids)


# ── No conflict detection ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remember_batch_no_conflict_detection(memory_client_conflict: Hippo) -> None:
    """With conflict_detection=False, the LLM is never called."""
    agent_id = f"batch-nocd-{uuid.uuid4().hex[:8]}"
    items = [{"content": f"Fact {i}", "agent_id": agent_id} for i in range(10)]

    await memory_client_conflict.remember_batch(items, conflict_detection=False)

    memory_client_conflict._llm.check_conflict.assert_not_awaited()


# ── Intra-batch conflict ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remember_batch_intra_batch_conflict(memory_client_conflict: Hippo) -> None:
    """A later item in the batch supersedes an earlier item in the same batch."""
    agent_id = f"batch-intra-{uuid.uuid4().hex[:8]}"

    # Mock: first call (item[1] checks against item[0]) → supersede.
    # All other calls (if any) → coexist.
    call_count = 0

    async def check_conflict_side_effect(old, new):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ConflictResult(True, "supersede", "Rust supersedes Python")
        return ConflictResult(False, "coexist", "no conflict")

    memory_client_conflict._llm.check_conflict = AsyncMock(
        side_effect=check_conflict_side_effect
    )

    items = [
        {"content": "User prefers Python", "agent_id": agent_id},
        {"content": "User has switched to Rust", "agent_id": agent_id},
    ]
    await memory_client_conflict.remember_batch(items, conflict_detection=True)

    results = await memory_client_conflict.recall("programming language", agent_id=agent_id, limit=10)
    active_contents = [r.content for r in results]

    assert not any("Python" in c and "Rust" not in c for c in active_contents), (
        "Python-only memory should be superseded"
    )
    assert any("Rust" in c for c in active_contents), "Rust memory should be active"

    log = await memory_client_conflict.get_conflict_log(agent_id=agent_id)
    assert any(e["decision"] == "supersede" for e in log)


# ── Existing-memory conflict ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remember_batch_existing_conflict(memory_client_conflict: Hippo) -> None:
    """A batch item supersedes a memory that was inserted before the batch."""
    agent_id = f"batch-exist-{uuid.uuid4().hex[:8]}"

    # Pre-insert a memory.
    memory_client_conflict._llm.check_conflict = AsyncMock(
        return_value=ConflictResult(False, "coexist", "no conflict")
    )
    old_id = await memory_client_conflict.remember("User likes cats", agent_id=agent_id)

    # Now batch-insert a contradicting memory.
    memory_client_conflict._llm.check_conflict = AsyncMock(
        return_value=ConflictResult(True, "supersede", "User is now allergic")
    )
    items = [{"content": "User is allergic to cats", "agent_id": agent_id}]
    await memory_client_conflict.remember_batch(items, conflict_detection=True)

    results = await memory_client_conflict.recall("cats", agent_id=agent_id, limit=10)
    result_ids = {r.id for r in results}
    assert old_id not in result_ids, "Pre-existing memory should be superseded"


# ── Progress callback ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remember_batch_progress_callback(memory_client: Hippo) -> None:
    """on_progress is called once per chunk with (done, total)."""
    agent_id = f"batch-prog-{uuid.uuid4().hex[:8]}"
    items = [{"content": f"Item {i}", "agent_id": agent_id} for i in range(9)]

    calls: list[tuple[int, int]] = []
    await memory_client.remember_batch(
        items,
        conflict_detection=False,
        batch_size=3,
        on_progress=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(3, 9), (6, 9), (9, 9)]


# ── Partial failure ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remember_batch_partial_failure(memory_client: Hippo) -> None:
    """When chunk 2 fails, chunks 1 and 3 still commit; BatchPartialFailure is raised."""
    agent_id = f"batch-fail-{uuid.uuid4().hex[:8]}"
    items = [{"content": f"Item {i}", "agent_id": agent_id} for i in range(9)]

    call_num = 0

    async def embed_batch_side_effect(texts):
        nonlocal call_num
        call_num += 1
        if call_num == 2:
            raise RuntimeError("Simulated embed error on chunk 2")
        return [FIXED_VEC] * len(texts)

    memory_client._embedder.embed_batch = AsyncMock(side_effect=embed_batch_side_effect)

    with pytest.raises(BatchPartialFailure) as exc_info:
        await memory_client.remember_batch(items, conflict_detection=False, batch_size=3)

    err = exc_info.value
    assert err.failed_indices == [3, 4, 5], "Chunk 2 items (indices 3-5) should have failed"

    # Chunk 1 and 3 should have succeeded.
    succeeded = [uid for uid in err.successful_ids if uid is not None]
    assert len(succeeded) == 6, "6 items from chunks 1 and 3 should have been inserted"

    # Verify chunk 1 is actually in DB.
    results = await memory_client.recall("Item", agent_id=agent_id, limit=10)
    stored_ids = {r.id for r in results}
    chunk1_ids = [err.successful_ids[i] for i in range(3)]
    assert all(uid in stored_ids for uid in chunk1_ids)


# ── embed_batch usage ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remember_batch_uses_embed_batch(memory_client: Hippo) -> None:
    """remember_batch() calls embed_batch(), not embed() in a loop."""
    agent_id = f"batch-emb-{uuid.uuid4().hex[:8]}"
    items = [{"content": f"Memory {i}", "agent_id": agent_id} for i in range(30)]

    await memory_client.remember_batch(items, conflict_detection=False, batch_size=100)

    # embed_batch called once (30 items fit in one chunk of 100).
    memory_client._embedder.embed_batch.assert_awaited_once()
    # embed (the serial version) must NOT have been called.
    memory_client._embedder.embed.assert_not_awaited()
