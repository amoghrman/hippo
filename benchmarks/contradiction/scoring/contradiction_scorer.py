"""Rule-based contradiction scoring.

Scores only the TOP recalled result (index 0) against keyword lists.
This mirrors how an agent would actually use the memory — take the top hit and
act on it.  Keyword matching is case-insensitive substring search.
"""

from __future__ import annotations


def score_response(
    recalled: list[str],
    correct_answer_contains: list[str],
    incorrect_answer_contains: list[str],
) -> dict:
    """Score the top recalled memory against keyword expectations.

    Args:
        recalled: List of memory contents returned by the system, best-first.
        correct_answer_contains: Keywords that SHOULD appear in the top result.
        incorrect_answer_contains: Keywords that should NOT appear in the top result.

    Returns:
        {
            "correct": bool,        # at least one correct keyword found in top result
            "hallucinated": bool,   # at least one incorrect keyword found in top result
            "top_result": str,      # the text of the first recalled memory (or "")
            "matched_correct": list[str],
            "matched_incorrect": list[str],
        }
    """
    top = recalled[0].lower() if recalled else ""

    matched_correct = [kw for kw in correct_answer_contains if kw.lower() in top]
    matched_incorrect = [kw for kw in incorrect_answer_contains if kw.lower() in top]

    return {
        "correct": len(matched_correct) > 0,
        "hallucinated": len(matched_incorrect) > 0,
        "top_result": recalled[0] if recalled else "",
        "matched_correct": matched_correct,
        "matched_incorrect": matched_incorrect,
    }
