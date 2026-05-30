"""Tests for the contradiction benchmark infrastructure."""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmarks.contradiction.scoring.contradiction_scorer import score_response

_DATASET_PATH = Path(__file__).parent.parent / "benchmarks" / "contradiction" / "datasets" / "handcrafted.json"

# ── Dataset validation ─────────────────────────────────────────────────────────

EXPECTED_CATEGORIES = {
    "preference_change",
    "factual_update",
    "corrected_misinformation",
    "temporal_supersession",
    "direct_negation",
}


def _load_dataset() -> list[dict]:
    with open(_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_handcrafted_dataset_is_valid_json() -> None:
    """The file parses as a JSON array."""
    data = _load_dataset()
    assert isinstance(data, list)
    assert len(data) > 0


def test_handcrafted_has_50_scenarios() -> None:
    """Exactly 50 hand-crafted scenarios."""
    data = _load_dataset()
    assert len(data) == 50


def test_handcrafted_categories_balanced() -> None:
    """10 scenarios per category, all 5 categories present."""
    data = _load_dataset()
    from collections import Counter

    counts = Counter(s["category"] for s in data)
    assert set(counts.keys()) == EXPECTED_CATEGORIES, f"Unexpected categories: {set(counts.keys())}"
    for cat, count in counts.items():
        assert count == 10, f"Expected 10 in {cat}, got {count}"


def test_handcrafted_schema_valid() -> None:
    """Every scenario has all required keys with the correct types."""
    data = _load_dataset()
    required_keys = {
        "id",
        "category",
        "initial_fact",
        "contradicting_fact",
        "query",
        "correct_answer_contains",
        "incorrect_answer_contains",
        "user_id",
        "agent_id",
    }
    for scenario in data:
        missing = required_keys - set(scenario.keys())
        assert not missing, f"Scenario {scenario.get('id')} missing keys: {missing}"
        assert isinstance(scenario["correct_answer_contains"], list)
        assert len(scenario["correct_answer_contains"]) >= 1
        assert isinstance(scenario["incorrect_answer_contains"], list)


# ── Contradiction scorer ───────────────────────────────────────────────────────


def test_contradiction_scorer_correct_case() -> None:
    """Returns correct=True when the right keyword appears in the top result."""
    result = score_response(
        recalled=["User has switched to Go for all backend work"],
        correct_answer_contains=["Go"],
        incorrect_answer_contains=["Python"],
    )
    assert result["correct"] is True
    assert result["hallucinated"] is False
    assert "Go" in result["matched_correct"]


def test_contradiction_scorer_hallucination_case() -> None:
    """Returns hallucinated=True when the stale keyword appears in the top result."""
    result = score_response(
        recalled=["User prefers Python for backend development"],
        correct_answer_contains=["Go"],
        incorrect_answer_contains=["Python"],
    )
    assert result["correct"] is False
    assert result["hallucinated"] is True
    assert "Python" in result["matched_incorrect"]


def test_contradiction_scorer_partial_keyword_match() -> None:
    """Keyword matching is case-insensitive substring search."""
    result = score_response(
        recalled=["The user now codes exclusively in golang for services"],
        correct_answer_contains=["Go", "golang"],
        incorrect_answer_contains=["Python", "PYTHON"],
    )
    assert result["correct"] is True
    assert result["hallucinated"] is False


def test_contradiction_scorer_empty_recall() -> None:
    """Empty recall list returns correct=False, hallucinated=False."""
    result = score_response(
        recalled=[],
        correct_answer_contains=["Go"],
        incorrect_answer_contains=["Python"],
    )
    assert result["correct"] is False
    assert result["hallucinated"] is False
    assert result["top_result"] == ""


def test_contradiction_scorer_only_scores_top_result() -> None:
    """Only the first recalled item is scored, regardless of subsequent items."""
    result = score_response(
        recalled=[
            "User still prefers Python for backend work",  # stale
            "User switched to Go",  # correct — but not at position 0
        ],
        correct_answer_contains=["Go"],
        incorrect_answer_contains=["Python"],
    )
    assert result["correct"] is False
    assert result["hallucinated"] is True


# ── HippoAdapter (mocked) ──────────────────────────────────────────────────────

_MEM0_AVAILABLE = importlib.util.find_spec("mem0") is not None


@pytest.mark.asyncio
async def test_hippo_adapter_roundtrip() -> None:
    """HippoAdapter correctly delegates to the underlying Hippo client."""
    from benchmarks.contradiction.adapters.hippo_adapter import HippoAdapter

    mock_memory = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "User prefers Go for backend development"

    mock_memory.remember = AsyncMock(return_value=uuid.uuid4())
    mock_memory.recall = AsyncMock(return_value=[mock_result])
    mock_memory.forget = AsyncMock(return_value=1)

    adapter = HippoAdapter(client=mock_memory, agent_id="bench-agent")

    await adapter.reset()
    mock_memory.forget.assert_awaited_once()

    await adapter.remember("User switched to Go", user_id="u1", agent_id="bench-agent")
    mock_memory.remember.assert_awaited_once_with(
        content="User switched to Go", agent_id="bench-agent", user_id="u1"
    )

    results = await adapter.recall("backend language", user_id="u1", agent_id="bench-agent")
    assert results == ["User prefers Go for backend development"]


@pytest.mark.skipif(not _MEM0_AVAILABLE, reason="mem0ai not installed")
@pytest.mark.asyncio
async def test_mem0_adapter_construction() -> None:
    """Mem0Adapter constructs without error when mem0ai is installed."""
    from benchmarks.contradiction.adapters.mem0_adapter import Mem0Adapter

    adapter = Mem0Adapter()
    assert adapter.name == "mem0"


# ── Runner (mocked adapters) ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_runner_produces_results(tmp_path: Path) -> None:
    """Runner aggregates correctly with mock adapters."""
    from benchmarks.contradiction.adapters.base import MemorySystemAdapter
    from benchmarks.contradiction.runner import run_benchmark

    class _MockAdapter(MemorySystemAdapter):
        name = "mock"
        _calls: list[str] = []

        async def reset(self) -> None:
            self._calls.append("reset")

        async def remember(self, content: str, user_id: str, agent_id: str) -> None:
            self._calls.append(f"remember:{content[:20]}")

        async def recall(self, query: str, user_id: str, agent_id: str, limit: int = 5) -> list[str]:
            # Always return the contradicting fact keyword.
            return ["Go is the user's preferred backend language"]

    dataset = [
        {
            "id": "test-001",
            "category": "preference_change",
            "initial_fact": "User prefers Python",
            "contradicting_fact": "User switched to Go",
            "query": "What language?",
            "correct_answer_contains": ["Go"],
            "incorrect_answer_contains": ["Python"],
            "user_id": "u1",
            "agent_id": "bench-agent",
        }
    ]

    output_file = tmp_path / "results.json"
    results = await run_benchmark(
        systems=[_MockAdapter()],
        dataset=dataset,
        dataset_name="test",
        output_path=output_file,
    )

    assert output_file.exists()
    assert results["n_scenarios"] == 1
    assert results["summaries"][0]["correct"] == 1
    assert results["summaries"][0]["system"] == "mock"

    with open(output_file) as f:
        on_disk = json.load(f)
    assert on_disk["summaries"][0]["correct"] == 1
