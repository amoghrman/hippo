"""Tests for conflict resolution — Hippo's core differentiator.

All LLM calls are mocked via ConflictResult. Postgres is real.
"""
import uuid
from unittest.mock import AsyncMock

import pytest

from hippo import Hippo
from hippo.llm.base import ConflictResult


def _patch_llm(client: Hippo, contradicts: bool, resolution: str, reason: str) -> None:
    """Override the mock LLM's check_conflict to return a fixed ConflictResult."""
    client._llm.check_conflict = AsyncMock(
        return_value=ConflictResult(contradicts=contradicts, resolution=resolution, reason=reason)
    )


# ── Supersede ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conflict_supersede(memory_client_conflict: Hippo) -> None:
    """When new contradicts old with resolution=supersede, old becomes inactive."""
    agent_id = f"supersede-{uuid.uuid4().hex[:8]}"
    _patch_llm(
        memory_client_conflict,
        contradicts=True,
        resolution="supersede",
        reason="User explicitly switched from dark to light mode",
    )

    old_id = await memory_client_conflict.remember(
        "User prefers dark mode for all interfaces",
        agent_id=agent_id,
    )
    new_id = await memory_client_conflict.remember(
        "User has switched to light mode as default theme",
        agent_id=agent_id,
    )

    results = await memory_client_conflict.recall(
        "What is the user's theme preference?", agent_id=agent_id, limit=10
    )
    result_ids = {r.id for r in results}
    contents = [r.content for r in results]

    assert new_id in result_ids, "New memory should be active"
    assert old_id not in result_ids, "Old memory should be superseded (inactive)"
    assert not any("dark mode" in c for c in contents), "Superseded content must not appear"


@pytest.mark.asyncio
async def test_supersede_logs_conflict(memory_client_conflict: Hippo) -> None:
    """A supersede decision is recorded in conflict_log."""
    agent_id = f"log-{uuid.uuid4().hex[:8]}"
    _patch_llm(
        memory_client_conflict,
        contradicts=True,
        resolution="supersede",
        reason="Direct contradiction",
    )

    await memory_client_conflict.remember("User likes cats", agent_id=agent_id)
    await memory_client_conflict.remember("User is allergic to cats", agent_id=agent_id)

    log = await memory_client_conflict.get_conflict_log(agent_id=agent_id)
    assert len(log) >= 1
    assert log[0]["decision"] == "supersede"
    assert log[0]["reason"] == "Direct contradiction"


# ── Coexist ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conflict_coexist(memory_client_conflict: Hippo) -> None:
    """Non-contradicting memories both stay active."""
    agent_id = f"coexist-{uuid.uuid4().hex[:8]}"
    _patch_llm(
        memory_client_conflict,
        contradicts=False,
        resolution="coexist",
        reason="Different topics — location and career are complementary",
    )

    id1 = await memory_client_conflict.remember("User lives in Tokyo", agent_id=agent_id)
    id2 = await memory_client_conflict.remember(
        "User works in the finance industry", agent_id=agent_id
    )

    results = await memory_client_conflict.recall(
        "Tell me about the user", agent_id=agent_id, limit=10
    )
    result_ids = {r.id for r in results}

    assert id1 in result_ids, "Tokyo memory should stay active"
    assert id2 in result_ids, "Finance memory should stay active"


@pytest.mark.asyncio
async def test_coexist_does_not_log(memory_client_conflict: Hippo) -> None:
    """Coexist (no contradiction) produces no conflict_log entry."""
    agent_id = f"coexist-log-{uuid.uuid4().hex[:8]}"
    _patch_llm(
        memory_client_conflict,
        contradicts=False,
        resolution="coexist",
        reason="Complementary facts",
    )

    await memory_client_conflict.remember("User lives in Berlin", agent_id=agent_id)
    await memory_client_conflict.remember("User enjoys hiking", agent_id=agent_id)

    log = await memory_client_conflict.get_conflict_log(agent_id=agent_id)
    assert log == [], f"Expected empty log, got: {log}"


# ── Merge ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conflict_merge(memory_client_conflict: Hippo) -> None:
    """Merge decision supersedes old, updates new with synthesised content."""
    agent_id = f"merge-{uuid.uuid4().hex[:8]}"

    merged_text = "User programs in both Python and Rust depending on the project"
    memory_client_conflict._llm.check_conflict = AsyncMock(
        return_value=ConflictResult(True, "merge", "Partial overlap — both partially true")
    )
    memory_client_conflict._llm.synthesize_merge = AsyncMock(return_value=merged_text)

    old_id = await memory_client_conflict.remember(
        "User programs primarily in Python", agent_id=agent_id
    )
    new_id = await memory_client_conflict.remember(
        "User has started using Rust for systems work", agent_id=agent_id
    )

    results = await memory_client_conflict.recall(
        "programming languages", agent_id=agent_id, limit=10
    )
    result_ids = {r.id for r in results}
    contents = [r.content for r in results]

    assert old_id not in result_ids, "Old memory should be superseded after merge"
    assert new_id in result_ids, "Merged (new) memory should be active"
    assert any("Rust" in c for c in contents), "Merged content should mention Rust"
    assert any("Python" in c for c in contents), "Merged content should mention Python"

    log = await memory_client_conflict.get_conflict_log(agent_id=agent_id)
    assert any(entry["decision"] == "merge" for entry in log)


# ── Conflict detection disabled ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conflict_detection_disabled(memory_client: Hippo) -> None:
    """With conflict_detection=False, contradicting memories both stay active."""
    agent_id = f"disabled-{uuid.uuid4().hex[:8]}"

    id1 = await memory_client.remember("User prefers dark mode", agent_id=agent_id)
    id2 = await memory_client.remember("User prefers light mode", agent_id=agent_id)

    results = await memory_client.recall("theme", agent_id=agent_id, limit=10)
    result_ids = {r.id for r in results}

    assert id1 in result_ids
    assert id2 in result_ids
