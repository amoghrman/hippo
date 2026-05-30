"""Memory consolidation — stub interface for future implementation.

Planned features:
- Periodic merging of related memories into denser summaries
- Ebbinghaus forgetting curves: decay importance over time
- Pruning memories below a minimum importance threshold

Track progress: https://github.com/<YOUR_USERNAME>/hippo/issues
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import Hippo


class Consolidator:
    """Periodic memory consolidation engine.

    Not yet implemented — raises NotImplementedError on all methods.

    Example::

        consolidator = Consolidator(hippo)
        await consolidator.run(agent_id="agent-1")  # NotImplementedError
    """

    def __init__(self, hippo: Hippo) -> None:
        self._hippo = hippo

    async def run(self, agent_id: str, user_id: str | None = None) -> dict:
        """Run a full consolidation pass (cluster → summarise → prune).

        Example::

            stats = await consolidator.run(agent_id="agent-1")
        """
        raise NotImplementedError("Consolidation is on the roadmap but not yet implemented.")

    async def apply_forgetting_curve(
        self,
        agent_id: str,
        user_id: str | None = None,
        min_importance: float = 0.1,
    ) -> int:
        """Decay importance scores using the Ebbinghaus forgetting curve.

        Example::

            deactivated = await consolidator.apply_forgetting_curve("agent-1")
        """
        raise NotImplementedError("Forgetting curves are on the roadmap but not yet implemented.")
