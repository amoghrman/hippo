# Contributing to Hippo

Thanks for your interest. Hippo is MIT-licensed and welcomes pull requests.

**Open an issue before starting a large change** — this avoids wasted effort if the direction doesn't fit the roadmap.

---

## Dev environment

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker (or local Postgres 15+ with pgvector)

```bash
git clone https://github.com/<YOUR_USERNAME>/hippo
cd hippo

# Install all dependencies including dev extras
uv sync --extra dev

# Start Postgres + pgvector
docker compose up -d

# Verify the setup
uv run pytest -m "not slow"
```

---

## Running tests

```bash
# Fast suite — all mocked, no model downloads (~6 s)
uv run pytest -m "not slow"

# Full suite — includes sentence-transformers model download (~2–3 min first run)
uv run pytest

# Single file
uv run pytest tests/test_conflict.py -v
```

The `@slow` mark covers tests that load a local sentence-transformers model. They are
skipped in CI by default and are safe to run locally once the model is cached.

---

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting/formatting and
[mypy](https://mypy-lang.org/) for type checking.

```bash
# Lint + auto-fix
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

CI will reject PRs that fail either check. Run both before pushing.

---

## PR checklist

- [ ] One feature or fix per PR — keep diffs reviewable
- [ ] New behaviour is covered by tests
- [ ] `uv run pytest -m "not slow"` passes locally
- [ ] `uv run ruff check src/ tests/` passes (no new lint errors)
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`
- [ ] Docstrings updated if the public API changed

---

## Project layout

```
src/hippo/
  client.py          # Hippo class — public API
  conflict.py        # conflict detection + resolution logic
  retrieval.py       # hybrid BM25 + vector + recency + importance scoring
  models.py          # SQLAlchemy ORM + Pydantic public types
  embedders/         # Embedder ABC + OpenAI / SentenceTransformers impls
  llm/               # LLM ABC + OpenAI / Groq impls
  consolidation.py   # stub — not yet implemented
tests/
  conftest.py        # shared fixtures (mock embedder, mock LLM, DB client)
  test_client.py     # remember / recall / forget
  test_conflict.py   # conflict resolution scenarios
  test_retrieval.py  # hybrid scoring
  test_embedders.py  # embedder dimension + roundtrip
examples/
  demo.py            # end-to-end story: Python → Rust conflict supersession
```
