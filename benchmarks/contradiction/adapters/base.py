"""Abstract adapter interface for memory systems under benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod


class MemorySystemAdapter(ABC):
    """Common interface so the runner can drive any memory system identically."""

    name: str  # "hippo", "mem0", etc.

    @abstractmethod
    async def reset(self) -> None:
        """Wipe all stored memories — called between scenarios for isolation."""
        ...

    @abstractmethod
    async def remember(self, content: str, user_id: str, agent_id: str) -> None:
        """Store a single piece of information."""
        ...

    @abstractmethod
    async def recall(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        limit: int = 5,
    ) -> list[str]:
        """Return textual content of recalled memories, highest-relevance first."""
        ...
