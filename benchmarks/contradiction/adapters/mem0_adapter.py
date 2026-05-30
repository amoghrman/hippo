"""Mem0Adapter — wraps mem0ai for benchmarking.

mem0 is an optional dependency.  Install it with:
    pip install "hippo-memory[bench]"
or directly:
    pip install mem0ai

Differences from Hippo that affect benchmark validity:
- mem0 manages its own LLM/embedder config; we pass the same model names
  to minimise systematic differences, but internal chunking and prompting
  will differ.
- mem0's ``search()`` returns dicts with a "memory" key; we extract that field.
- mem0 does not guarantee FIFO ordering of search results; the top result
  is the highest-relevance hit according to mem0's internal scoring.
- Conflict resolution in mem0 is probabilistic (LLM-driven); it may or may
  not supersede the initial fact depending on prompt behaviour.

These differences are documented in benchmarks/contradiction/README.md.
"""

from __future__ import annotations

from typing import Any

from .base import MemorySystemAdapter


def _require_mem0() -> Any:
    try:
        from mem0 import Memory  # type: ignore[import]

        return Memory
    except ImportError as exc:
        raise ImportError(
            "mem0ai is required for Mem0Adapter. "
            "Install it with: pip install 'hippo-memory[bench]'"
        ) from exc


class Mem0Adapter(MemorySystemAdapter):
    """Drives mem0 (mem0ai package) for apples-to-apples contradiction benchmarking.

    Args:
        config: mem0 configuration dict.  Keys vary by version; see mem0ai docs.
            At minimum, pass ``{"llm": {"provider": ..., "model": ...},
            "embedder": {"provider": ..., "model": ...}}``.

    Example::

        adapter = Mem0Adapter(config={
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "embedder": {"provider": "openai", "model": "text-embedding-3-small"},
        })
    """

    name = "mem0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        Memory = _require_mem0()
        self._client = Memory.from_config(config) if config else Memory()
        self._known_user_ids: set[str] = set()

    async def reset(self) -> None:
        for user_id in list(self._known_user_ids):
            try:
                self._client.delete_all(user_id=user_id)
            except Exception:
                pass
        self._known_user_ids.clear()

    async def remember(self, content: str, user_id: str, agent_id: str) -> None:
        # mem0's add() is synchronous in the open-source library.
        self._client.add(content, user_id=user_id)
        self._known_user_ids.add(user_id)

    async def recall(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        limit: int = 5,
    ) -> list[str]:
        results = self._client.search(query, user_id=user_id, limit=limit)
        # mem0 returns a list of dicts; the memory text is under the "memory" key.
        return [r.get("memory", str(r)) for r in results]
