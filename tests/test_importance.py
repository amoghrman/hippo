"""Tests for importance auto-scoring."""

import uuid
from unittest.mock import AsyncMock

import pytest

from hippo import Hippo
from hippo.importance import LLMImportanceScorer


@pytest.mark.asyncio
async def test_auto_importance_disabled_uses_default(memory_client: Hippo) -> None:
    """With auto_importance=False (default), importance defaults to 0.5."""
    agent_id = f"imp-off-{uuid.uuid4().hex[:8]}"
    await memory_client.remember("User likes jazz music", agent_id=agent_id)
    results = await memory_client.recall("music", agent_id=agent_id)
    assert results[0].importance == 0.5


@pytest.mark.asyncio
async def test_auto_importance_scores_via_llm(memory_client_conflict: Hippo) -> None:
    """With auto_importance=True, importance is set from the LLM scorer."""
    agent_id = f"imp-on-{uuid.uuid4().hex[:8]}"
    memory_client_conflict._auto_importance = True
    memory_client_conflict._llm.complete = AsyncMock(return_value="0.8")
    memory_client_conflict._importance_scorer = LLMImportanceScorer(memory_client_conflict._llm)

    await memory_client_conflict.remember("User is severely allergic to peanuts", agent_id=agent_id)
    results = await memory_client_conflict.recall("allergy", agent_id=agent_id)
    assert abs(results[0].importance - 0.8) < 1e-6
    memory_client_conflict._llm.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_importance_parse_failure_falls_back_to_default(
    memory_client_conflict: Hippo,
) -> None:
    """When the LLM returns unparseable text, importance defaults to 0.5."""
    agent_id = f"imp-fail-{uuid.uuid4().hex[:8]}"
    memory_client_conflict._auto_importance = True
    memory_client_conflict._llm.complete = AsyncMock(return_value="not a number")
    memory_client_conflict._importance_scorer = LLMImportanceScorer(memory_client_conflict._llm)

    await memory_client_conflict.remember("User mentioned something briefly", agent_id=agent_id)
    results = await memory_client_conflict.recall("something", agent_id=agent_id)
    assert results[0].importance == 0.5


@pytest.mark.asyncio
async def test_explicit_importance_overrides_auto(memory_client_conflict: Hippo) -> None:
    """Explicitly passing importance= bypasses auto-scoring entirely."""
    agent_id = f"imp-explicit-{uuid.uuid4().hex[:8]}"
    memory_client_conflict._auto_importance = True
    memory_client_conflict._llm.complete = AsyncMock(return_value="0.9")
    memory_client_conflict._importance_scorer = LLMImportanceScorer(memory_client_conflict._llm)

    await memory_client_conflict.remember(
        "User likes coffee", agent_id=agent_id, importance=0.3
    )
    results = await memory_client_conflict.recall("coffee", agent_id=agent_id)
    assert abs(results[0].importance - 0.3) < 1e-6
    memory_client_conflict._llm.complete.assert_not_awaited()
