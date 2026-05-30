"""LLM-as-judge scorer for ambiguous contradiction cases.

Uses the same LLM abstraction Hippo ships with, so no extra API credentials
are needed beyond what you already have configured.
"""

from __future__ import annotations

import json
import logging

from hippo.llm.base import LLM

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are an evaluation judge for a memory system benchmark.

A user asked: {query}

The memory system returned this as its top result:
"{actual}"

The correct answer should address: {expected}

Does the memory system's response correctly answer the user's question?

Reply with valid JSON only:
{{"correct": true|false, "reason": "one sentence explanation"}}

Be strict: if the response addresses the wrong time period or contradicts \
what the user last told the system, mark it as incorrect.
"""


async def judge_response(
    query: str,
    expected: str,
    actual: str,
    llm: LLM,
) -> dict:
    """Ask the LLM to evaluate whether ``actual`` correctly answers ``query``.

    Args:
        query: The question the memory system was asked.
        expected: Description of the expected correct answer (from the dataset).
        actual: The text returned by the memory system as its top result.
        llm: Any LLM implementation from hippo.llm.

    Returns:
        {"correct": bool, "reason": str}
    """
    prompt = _JUDGE_PROMPT.format(query=query, expected=expected, actual=actual)
    raw = ""
    try:
        raw = await llm.complete(prompt)
        data = json.loads(raw.strip())
        return {
            "correct": bool(data.get("correct", False)),
            "reason": str(data.get("reason", "")),
        }
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("LLM judge parse failure (raw=%r): %s", raw[:200], exc)
        return {"correct": False, "reason": f"parse error: {exc}"}
