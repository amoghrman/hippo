"""HippoAdapter — wraps the Hippo memory library for benchmarking."""

from __future__ import annotations

from hippo import Hippo

from .base import MemorySystemAdapter


class HippoAdapter(MemorySystemAdapter):
    """Drives a pre-configured Hippo instance.

    The caller is responsible for calling ``await client.setup()`` before
    constructing this adapter.  ``reset()`` wipes the agent's memories by
    calling ``forget(filter={"agent_id": ...})``.

    Args:
        client: A fully initialised Hippo instance.
        agent_id: The agent scope used for all operations in this benchmark run.

    Example::

        hippo = Hippo(database_url="...", conflict_detection=True, ...)
        await hippo.setup()
        adapter = HippoAdapter(client=hippo, agent_id="bench-agent")
    """

    name = "hippo"

    def __init__(self, client: Hippo, agent_id: str = "bench-agent") -> None:
        self._client = client
        self._agent_id = agent_id

    async def reset(self) -> None:
        try:
            await self._client.forget(filter={"agent_id": self._agent_id})
        except Exception:
            pass  # table may be empty — not an error

    async def remember(self, content: str, user_id: str, agent_id: str) -> None:
        await self._client.remember(
            content=content,
            agent_id=agent_id,
            user_id=user_id,
        )

    async def recall(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        limit: int = 5,
    ) -> list[str]:
        results = await self._client.recall(
            query=query,
            agent_id=agent_id,
            user_id=user_id,
            limit=limit,
        )
        return [r.content for r in results]
