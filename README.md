# 🦛 Hippo

**A memory layer for AI agents that handles conflict resolution, so your agent stops believing two contradictory things at once.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/amoghrman/hippo/actions/workflows/ci.yml/badge.svg)](https://github.com/amoghrman/hippo/actions/workflows/ci.yml)

<!-- TODO: insert demo.gif here after recording -->

---

## The problem with agent memory today

Most agent frameworks bolt on a vector database and call it "memory." The result:

- **Agents forget** — retrieval misses older facts as the index grows.
- **Agents drown** — too many half-relevant memories dilute the context window.
- **Agents lie to themselves** — the user says "I switched to Rust" but the old "prefers Python" memory stays active, and the agent confidently gives Python advice anyway.

Existing tools (mem0, Letta, Zep) are vector-search duct tape. None of them detect when a new memory contradicts an old one.

## The fix: conflict-aware memory

```python
from hippo import Hippo

memory = Hippo(database_url="postgresql+asyncpg://hippo:hippo@localhost/hippo")
await memory.setup()

await memory.remember("User prefers Python", agent_id="agent-1")
await memory.remember("User switched to Rust", agent_id="agent-1")  # conflict detected

results = await memory.recall("What language does the user prefer?", agent_id="agent-1")
print(results[0].content)  # "User switched to Rust ..."  — Python memory is superseded
```

Every `remember()` call checks existing memories for semantic contradictions and resolves them before they can silently poison future `recall()` results.

---

## Try it in 60 seconds

**Option A — Free, no credit card (local embeddings + Groq)**

```bash
pip install "hippo-memory[local,groq]"
docker compose up -d          # needs Docker, or use local Postgres 15+
export GROQ_API_KEY=gsk-...   # free at console.groq.com
python examples/demo.py
```

**Option B — OpenAI**

```bash
pip install hippo-memory
docker compose up -d
export OPENAI_API_KEY=sk-...
python examples/demo.py
```

Hippo auto-detects which backend to use: `OPENAI_API_KEY` → OpenAI for both embeddings and conflict LLM; `GROQ_API_KEY` → Groq + local sentence-transformers; neither → local embeddings only (conflict detection disabled with a warning).

---

## Why Hippo vs. the alternatives?

| Feature | Hippo | mem0 | Letta | Zep |
|---|:---:|:---:|:---:|:---:|
| Conflict resolution | **Yes** | No | No¹ | No |
| Conflict audit log | **Yes** | No | No | No |
| Hybrid retrieval (BM25 + vector) | Yes | Yes² | ? | ? |
| Pluggable embedder | Yes | Yes | Yes | Yes |
| Self-hosted | Yes | Yes | Yes | Yes |

¹ Letta uses block-based memory that overwrites rather than detecting semantic contradictions between stored facts.
² mem0 hybrid search requires `pip install mem0ai[nlp]`; standard install is vector-only.

*Comparison reflects upstream behavior as of May 2026; corrections welcome via [PR](https://github.com/amoghrman/hippo/pulls).*

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │           Hippo             │
                    │                             │
        remember()  │  ┌──────────────────────┐  │  recall()
      ─────────────►│  │  Conflict Resolver   │  │◄─────────────
                    │  │                      │  │
        forget()    │  │  1. vector search    │  │  ┌──────────────────┐
      ─────────────►│  │  2. LLM check        │  │  │ Hybrid Retriever │
                    │  │  3. supersede/merge  │  │  │                  │
                    │  └──────────┬───────────┘  │  │  0.5 × cosine    │
                    │             │               │  │  0.2 × BM25      │
                    └─────────────┼───────────────┘  │  0.15 × recency  │
                                  │                  │  0.15 × import.  │
                    ┌─────────────▼──────────────────▼──────────┐
                    │           PostgreSQL 15 + pgvector         │
                    │                                            │
                    │   memories table  │  conflict_log table    │
                    │   HNSW index      │  GIN (FTS) index       │
                    └────────────────────────────────────────────┘
```

---

## Install

**With OpenAI (embeddings + conflict LLM)**

```bash
# uv (recommended)
uv add hippo-memory

# pip
pip install hippo-memory
```

**With Groq + local embeddings (no paid API tier required)**

```bash
uv add "hippo-memory[local,groq]"
```

Requires **Postgres 15+ with pgvector**:

```bash
docker compose up -d   # uses the included docker-compose.yml
```

---

## API reference

### `Hippo(database_url, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `database_url` | `str` | required | asyncpg connection string |
| `embedder` | `Embedder \| None` | auto | Plug in any embedder |
| `openai_api_key` | `str \| None` | env `OPENAI_API_KEY` | Used for embeddings + conflict LLM |
| `groq_api_key` | `str \| None` | env `GROQ_API_KEY` | Used for conflict LLM (pairs with local embeddings) |
| `llm` | `LLM \| None` | auto | Override the conflict LLM directly |
| `conflict_detection` | `bool` | `True` | Set `False` to skip LLM conflict checks |
| `conflict_model` | `str` | `"gpt-4o-mini"` | LLM model for conflict resolution |
| `conflict_threshold` | `float` | `0.85` | Cosine similarity threshold for candidates |
| `retrieval_weights` | `dict \| None` | see below | `{vector, bm25, recency, importance}` |

### `await memory.remember(content, agent_id, ...)`

Stores a memory and runs conflict resolution against existing memories. Returns the new memory's UUID.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `content` | `str` | required | The memory text |
| `agent_id` | `str` | required | Scopes the memory to an agent |
| `user_id` | `str \| None` | `None` | Further scopes to a user |
| `importance` | `float` | `0.5` | 0–1 weight in retrieval scoring |
| `metadata` | `dict \| None` | `{}` | Arbitrary JSON metadata |

### `await memory.recall(query, agent_id, ..., limit=5)`

Hybrid-ranked retrieval. Returns `List[Memory]` sorted by score (highest first). Only active (non-superseded) memories are returned.

### `await memory.forget(memory_id=None, filter=None)`

Soft-deletes memories. Supported filter keys: `agent_id`, `user_id`, `older_than_days`. Returns count of deactivated memories.

### `await memory.get_conflict_log(agent_id=None, limit=50)`

Returns the conflict resolution audit log, newest first.

### `await memory.setup(*, reset=False)`

Creates tables and enables the pgvector extension. Idempotent. Pass `reset=True` to drop and recreate all tables — useful when switching embedders with a different dimension.

---

## Pluggable embedders

```python
from hippo.embedders import Embedder

class MyEmbedder(Embedder):
    async def embed(self, text: str) -> list[float]:
        return my_model.encode(text).tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return my_model.encode(texts).tolist()

    @property
    def dimensions(self) -> int:
        return 768

memory = Hippo(database_url="...", embedder=MyEmbedder())
```

---

## Running tests

```bash
docker compose up -d
uv run pytest -m "not slow"   # fast suite, all LLM calls mocked, no API key needed
uv run pytest                 # full suite, includes sentence-transformers model download
```

---

## Roadmap

- **Consolidation** — periodic summarisation of related memories into denser representations
- **Forgetting curves** — Ebbinghaus-style importance decay over time
- **Batch ingestion** — `remember_batch()` for high-throughput pipelines
- **Benchmark suite** — MRR and latency at scale vs. mem0 and vanilla pgvector
- **TypeScript SDK** — `npm install hippo-memory`

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Open an issue before starting large changes.

```bash
git clone https://github.com/amoghrman/hippo
cd hippo
uv sync --extra dev
docker compose up -d
uv run pytest -m "not slow"
```

---

## License

MIT — see [LICENSE](LICENSE).
