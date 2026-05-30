# Contradiction Handling Benchmark

Measures how accurately a memory system retrieves the **most recent** fact
when a user's information has changed.  This is Hippo's core differentiator:
other tools may accumulate both the old and new fact, causing agents to give
outdated or contradictory answers.

---

## Methodology

For each scenario, we:

1. Store an **initial fact** (e.g. "User prefers Python for backend development")
2. Store a **contradicting fact** (e.g. "User has switched to Go")
3. Ask the **query** (e.g. "What language does the user use for backend?")
4. Score the **top-1 recalled result** against expected keywords

Scoring is intentionally strict: only the top result matters, because that is
what an agent typically injects into its context.  A system that returns the
correct fact at position 3 but the stale fact at position 1 is considered
incorrect for this benchmark.

---

## What we measure / what we don't

**We measure:** contradiction handling — does the system surface the latest
version of a fact after an update?

**We do not measure:** general QA accuracy, recall latency at scale, faithfulness
on open-ended questions, or multi-hop reasoning.  mem0, Letta, Zep, and other
tools may outperform Hippo on benchmarks that measure those dimensions.  This
benchmark targets the specific failure mode Hippo is designed to address.

---

## Datasets

### Hand-crafted (50 scenarios, MIT-licensed, included in repo)

Located at `datasets/handcrafted.json`.  50 scenarios across 5 categories:

| Category | Count | Example |
|---|---|---|
| `preference_change` | 10 | "User switched from Python to Go for backend" |
| `factual_update` | 10 | "User moved from NYC to San Francisco" |
| `corrected_misinformation` | 10 | "I said 3 kids but actually 2" |
| `temporal_supersession` | 10 | "Q3 launch pushed to Q4" |
| `direct_negation` | 10 | "I love TypeScript" → "TypeScript frustrates me" |

Each scenario has `correct_answer_contains` (keywords that SHOULD appear in the
top result) and `incorrect_answer_contains` (keywords from the stale fact that
should NOT appear).

### LoCoMo temporal subset (CC BY-NC 4.0, downloaded separately)

LoCoMo (Maharana et al., ACL 2024) is a long-term conversational memory dataset.
We extract QA pairs with `category == 2` (temporal reasoning).  The raw dataset
is **not redistributed** in this repo.  Download it with:

```bash
python -m benchmarks.contradiction.datasets.locomo_loader
```

This creates `datasets/locomo_temporal.json`.  LoCoMo is licensed CC BY-NC 4.0
(non-commercial use only).

---

## Scoring

**Keyword scoring (default):** case-insensitive substring match on the top result.
Fast and reproducible, but can miss paraphrases.

**LLM-as-judge (optional, `--judge`):** passes ambiguous cases to the configured
LLM with a strict judge prompt.  Adds ~1 LLM call per incorrect scenario.

---

## How to run

**Prerequisites:** a running Postgres with pgvector, configured LLM backend.

```bash
# Hippo only (fastest, no mem0ai needed)
python -m benchmarks.contradiction.runner --dataset handcrafted --systems hippo

# Hippo vs mem0 (requires pip install mem0ai)
python -m benchmarks.contradiction.runner --dataset handcrafted --systems hippo,mem0

# With LLM-as-judge for ambiguous cases
python -m benchmarks.contradiction.runner --dataset handcrafted --systems hippo --judge

# LoCoMo temporal subset (download first)
python -m benchmarks.contradiction.datasets.locomo_loader
python -m benchmarks.contradiction.runner --dataset locomo --systems hippo
```

**Environment variables:**
```bash
DATABASE_URL=postgresql+asyncpg://hippo:hippo@localhost:5432/hippo
OPENAI_API_KEY=sk-...   # or GROQ_API_KEY for local-friendly runs
```

**Cost and time estimates (per system, per full run):**

| Dataset | LLM calls (Hippo remember) | LLM calls (judge) | Approximate cost (GPT-4o-mini) | Time |
|---|---|---|---|---|
| handcrafted (50) | ~100 conflict checks | 0–50 judge calls | ~$0.01 | 2–5 min |
| locomo temporal (~100) | ~200 conflict checks | 0–100 judge calls | ~$0.02 | 5–10 min |

Groq free tier is sufficient for full runs at 30 req/min throughput.

---

## Reproducibility

Results land in `results/{timestamp}_{dataset}.json` with full per-scenario
detail (inputs, recalled text, scores, latencies).  To reproduce:

```bash
git clone https://github.com/amoghrman/hippo && cd hippo
uv sync --extra dev
docker compose up -d
export OPENAI_API_KEY=sk-...
python -m benchmarks.contradiction.runner --dataset handcrafted --systems hippo
```

---

## Citation

If you use the LoCoMo subset, please cite:

```bibtex
@inproceedings{maharana2024locomo,
    title     = {Evaluating Very Long-Term Conversational Memory of {LLM} Agents},
    author    = {Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and
                 Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
    booktitle = {Proceedings of the 62nd Annual Meeting of the Association
                 for Computational Linguistics (ACL 2024)},
    year      = {2024},
    url       = {https://arxiv.org/abs/2402.17753}
}
```
