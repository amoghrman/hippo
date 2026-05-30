"""Shared fixtures for the Hippo test suite.

Tests require a running Postgres with pgvector. Start one with:
    docker compose up -d

The database URL defaults to the docker-compose values. Override via:
    TEST_DATABASE_URL=postgresql+asyncpg://... pytest
"""

import os

# Provide a dummy key so OpenAILLM can be instantiated in tests.
# All LLM calls are mocked; this key is never sent to a real API.
os.environ.setdefault("OPENAI_API_KEY", "test-placeholder-no-real-api-calls")

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import text

from hippo import Hippo
from hippo.embedders.base import Embedder
from hippo.llm.base import ConflictResult

TEST_DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://hippo:hippo@localhost:5432/hippo",
)

# ── Embedder mock ──────────────────────────────────────────────────────────────

FIXED_VEC: list[float] = [0.1] * 1536


def make_mock_embedder() -> MagicMock:
    """Embedder that always returns [0.1]*1536 — no OpenAI calls."""
    emb = MagicMock(spec=Embedder)
    emb.embed = AsyncMock(return_value=FIXED_VEC)
    # embed_batch must return one vector per input text.
    emb.embed_batch = AsyncMock(side_effect=lambda texts: [FIXED_VEC] * len(texts))
    emb.dimensions = 1536
    return emb


@pytest.fixture
def mock_embedder() -> MagicMock:
    return make_mock_embedder()


# ── LLM mock ───────────────────────────────────────────────────────────────────


def make_mock_llm() -> MagicMock:
    """LLM that returns coexist by default — override per test as needed."""
    llm = MagicMock()
    llm.check_conflict = AsyncMock(
        return_value=ConflictResult(contradicts=False, resolution="coexist", reason="default mock")
    )
    llm.synthesize_merge = AsyncMock(return_value="merged content")
    return llm


@pytest.fixture
def mock_llm() -> MagicMock:
    return make_mock_llm()


# ── DB helpers ─────────────────────────────────────────────────────────────────


async def _truncate(client: Hippo) -> None:
    async with client._sessionmaker() as session:
        async with session.begin():
            # TRUNCATE CASCADE handles FK ordering (conflict_log → memories,
            # and the self-referential memories.superseded_by) atomically.
            await session.execute(text("TRUNCATE TABLE conflict_log, memories CASCADE"))


# ── Client fixtures ────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def memory_client(mock_embedder: MagicMock) -> Hippo:
    """Hippo client with conflict detection OFF (fast, no LLM calls)."""
    client = Hippo(
        database_url=TEST_DB_URL,
        embedder=mock_embedder,
        conflict_detection=False,
    )
    await client.setup()
    yield client
    await _truncate(client)
    await client.close()


@pytest_asyncio.fixture
async def memory_client_conflict(mock_embedder: MagicMock, mock_llm: MagicMock) -> Hippo:
    """Hippo client with conflict detection ON, using a controllable mock LLM.

    Individual tests override mock_llm.check_conflict / synthesize_merge as needed.
    """
    client = Hippo(
        database_url=TEST_DB_URL,
        embedder=mock_embedder,
        llm=mock_llm,
        conflict_detection=True,
        conflict_threshold=0.80,
    )
    await client.setup()
    yield client
    await _truncate(client)
    await client.close()
