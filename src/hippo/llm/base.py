"""Abstract LLM interface for conflict resolution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

_CONFLICT_PROMPT = """\
You are a memory conflict detector for an AI agent system.

OLD MEMORY: {old}

NEW MEMORY: {new}

Does the new memory contradict or supersede the old memory?

Reply with valid JSON only — no markdown, no explanation outside the JSON:
{{"contradicts": true|false, "resolution": "supersede"|"merge"|"coexist", "reason": "brief explanation"}}

Guidelines:
- "supersede": new info clearly replaces old (preference changed, fact corrected, explicit update)
- "merge": both carry partial truths that should be synthesised into one memory
- "coexist": both can be true simultaneously (different topics, complementary facts)
- Set contradicts=false when the memories are simply about different things.
- When in doubt, choose "coexist".
"""


@dataclass
class ConflictResult:
    contradicts: bool
    resolution: str  # "supersede" | "merge" | "coexist"
    reason: str


class LLM(ABC):
    """Abstract base class for LLMs used in conflict resolution."""

    @abstractmethod
    async def check_conflict(self, old_content: str, new_content: str) -> ConflictResult:
        """Ask the LLM whether new_content contradicts old_content."""
        ...

    @abstractmethod
    async def synthesize_merge(self, old_content: str, new_content: str) -> str:
        """Synthesise a single merged memory from two partially overlapping ones."""
        ...
