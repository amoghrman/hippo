"""LoCoMo dataset loader — temporal contradiction subset.

License notice
--------------
The LoCoMo dataset is released under CC BY-NC 4.0 (non-commercial use only).
It is NOT redistributed in this repository.  This script downloads it on demand.

Citation:
    Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024).
    Evaluating Very Long-Term Conversational Memory of LLM Agents.
    In Proceedings of ACL 2024.
    https://arxiv.org/abs/2402.17753

Usage:
    python -m benchmarks.contradiction.datasets.locomo_loader
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

_DATASET_URL = (
    "https://github.com/snap-research/locomo/raw/main/data/locomo10.json"
)
_HERE = Path(__file__).parent
_RAW_PATH = _HERE / "locomo10.json"
_OUTPUT_PATH = _HERE / "locomo_temporal.json"

# LoCoMo QA category codes (from the dataset documentation):
#   0 = single-hop
#   1 = multi-hop
#   2 = temporal  ← we filter to this category
#   3 = open-domain
_TEMPORAL_CATEGORY = 2


def download_locomo(dest: Path = _RAW_PATH) -> Path:
    """Download locomo10.json to *dest* if not already present."""
    if dest.exists():
        print(f"LoCoMo already downloaded at {dest}")
        return dest
    print(f"Downloading LoCoMo dataset from {_DATASET_URL} …")
    urllib.request.urlretrieve(_DATASET_URL, dest)
    print(f"Saved to {dest} ({dest.stat().st_size / 1024:.1f} KB)")
    return dest


def extract_contradiction_subset(locomo_data: list | dict) -> list[dict]:
    """Filter LoCoMo to temporal QA pairs and reshape to benchmark schema.

    The LoCoMo dataset is structured as a list of dialogues, each containing
    a ``questions`` list.  We extract questions with ``category == 2``
    (temporal reasoning) and attempt to derive initial/contradicting facts from
    the ``evidence`` field.

    Args:
        locomo_data: Parsed JSON from locomo10.json.  May be a list of dialogues
            or a dict with a ``"data"`` key — both are handled.

    Returns:
        List of benchmark scenario dicts matching the handcrafted.json schema.
    """
    dialogues = locomo_data if isinstance(locomo_data, list) else locomo_data.get("data", [])

    scenarios: list[dict] = []
    n_skipped = 0

    for dlg_idx, dialogue in enumerate(dialogues):
        questions = dialogue.get("questions", dialogue.get("qa", []))
        utterances = dialogue.get("utterances", dialogue.get("conversation", []))

        for q_idx, qa in enumerate(questions):
            # Filter to temporal category only.
            cat = qa.get("category", qa.get("type", -1))
            if cat != _TEMPORAL_CATEGORY:
                continue

            question = qa.get("question", "")
            answer = qa.get("answer", "")
            evidence = qa.get("evidence", "")

            if not question or not answer:
                n_skipped += 1
                continue

            # Try to extract an initial vs contradicting fact from evidence.
            # LoCoMo evidence is often a string with multiple cited turns;
            # we use the first two distinct utterances if available.
            evidence_str = evidence if isinstance(evidence, str) else str(evidence)
            context_lines = [u.get("text", "") for u in utterances if u.get("text")]
            initial_fact = context_lines[0] if context_lines else evidence_str
            contradicting_fact = context_lines[-1] if len(context_lines) > 1 else answer

            scenario_id = f"lc-{dlg_idx:03d}-{q_idx:03d}"
            # Derive keyword hints from the ground-truth answer.
            answer_words = [w.strip(".,;:?!") for w in answer.split() if len(w) > 3]
            correct_keywords = answer_words[:3] if answer_words else [answer[:40]]

            scenarios.append(
                {
                    "id": scenario_id,
                    "category": "temporal_supersession",
                    "initial_fact": initial_fact,
                    "contradicting_fact": contradicting_fact,
                    "query": question,
                    "correct_answer_contains": correct_keywords,
                    "incorrect_answer_contains": [],  # hard to infer automatically
                    "user_id": f"locomo-user-{dlg_idx:03d}",
                    "agent_id": "bench-agent",
                    "source": "locomo",
                    "locomo_answer": answer,
                }
            )

    print(f"Extracted {len(scenarios)} temporal scenarios ({n_skipped} skipped).")
    return scenarios


def main() -> None:
    raw_path = download_locomo()
    with open(raw_path, encoding="utf-8") as f:
        data = json.load(f)

    scenarios = extract_contradiction_subset(data)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(scenarios)} scenarios to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
