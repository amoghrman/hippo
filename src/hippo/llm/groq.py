"""Groq-backed LLM for conflict resolution."""
from __future__ import annotations

import json
import logging

from groq import AsyncGroq

from .base import ConflictResult, LLM, _CONFLICT_PROMPT

logger = logging.getLogger(__name__)


class GroqLLM(LLM):
    """Uses the Groq inference API for conflict detection and synthesis.

    Example::

        llm = GroqLLM(model="llama3-8b-8192", api_key="gsk_...")
        result = await llm.check_conflict("User likes Python", "User switched to Rust")
    """

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str | None = None) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    async def check_conflict(self, old_content: str, new_content: str) -> ConflictResult:
        prompt = _CONFLICT_PROMPT.format(old=old_content, new=new_content)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Conflict LLM returned non-JSON: %s", raw)
            return ConflictResult(contradicts=False, resolution="coexist", reason="parse error")

        contradicts = bool(data.get("contradicts", False))
        resolution = str(data.get("resolution", "coexist"))
        if resolution not in {"supersede", "merge", "coexist"}:
            resolution = "coexist"
        reason = str(data.get("reason", ""))
        return ConflictResult(contradicts=contradicts, resolution=resolution, reason=reason)

    async def synthesize_merge(self, old_content: str, new_content: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Synthesise these two related memories into one accurate, concise memory:\n"
                        f"MEMORY 1: {old_content}\n"
                        f"MEMORY 2: {new_content}\n"
                        "Reply with just the synthesised memory text — no preamble."
                    ),
                }
            ],
            temperature=0,
        )
        return (response.choices[0].message.content or new_content).strip()
