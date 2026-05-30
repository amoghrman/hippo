"""Importance auto-scoring for stored memories."""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod

from .llm.base import LLM

logger = logging.getLogger(__name__)

_IMPORTANCE_PROMPT = """\
On a scale of 0.0 to 1.0, how important is this memory likely to be for future agent decisions?

Return ONLY a decimal number between 0.0 and 1.0. No explanation, no units, just the number.

Guidelines:
- Important (0.7-1.0): stable preferences, identity facts, critical constraints, ongoing commitments
- Moderate (0.4-0.6): useful recurring context, project details, stated goals
- Unimportant (0.0-0.3): one-off events, small talk, transient state, minor casual remarks

Memory: {content}
"""


class ImportanceScorer(ABC):
    """Abstract scorer that assigns an importance value in [0, 1] to a memory."""

    @abstractmethod
    async def score(self, content: str) -> float:
        """Return an importance score between 0.0 and 1.0."""
        ...


class LLMImportanceScorer(ImportanceScorer):
    """Scores importance by asking the configured LLM.

    Results are cached by content hash so identical text is never scored twice.

    Example::

        scorer = LLMImportanceScorer(llm)
        importance = await scorer.score("User is severely allergic to shellfish")
        # 0.95
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self._cache: dict[str, float] = {}

    async def score(self, content: str) -> float:
        key = hashlib.sha256(content.encode()).hexdigest()[:16]
        if key in self._cache:
            return self._cache[key]

        prompt = _IMPORTANCE_PROMPT.format(content=content)
        raw = ""
        try:
            raw = await self._llm.complete(prompt)
            value = float(raw.strip())
            value = max(0.0, min(1.0, value))
        except (ValueError, TypeError, Exception):
            logger.warning(
                "ImportanceScorer: could not parse %r as float, defaulting to 0.5", raw
            )
            value = 0.5

        self._cache[key] = value
        return value
