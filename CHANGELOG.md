# Changelog

All notable changes to Hippo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Batch ingestion (`remember_batch`) for high-throughput pipelines
- Consolidation with Ebbinghaus importance decay
- Benchmark suite (MRR, latency at scale vs. mem0 and vanilla pgvector)
- TypeScript SDK

## [0.1.1] — 2026-05-29

### Added
- GitHub Actions CI: Python 3.11 and 3.12 matrix, pgvector service container
- Optional LLM-based importance auto-scoring (`auto_importance=True` on `Hippo`)
- `ImportanceScorer` abstraction for custom scoring strategies
- `LLMImportanceScorer`: scores via the configured LLM, caches results by content hash
- `LLM.complete()` abstract method for free-form prompting (used by importance scorer)
- `examples/demo_auto_importance.py` demonstrating auto-scoring end-to-end

### Changed
- Comparison table in README corrected against current upstream docs (mem0, Letta)
  — mem0 "Vector only" → "Yes²" (hybrid with `[nlp]` extra); "No (cloud)" → "Yes" self-hosted
  — Letta "Partial" conflict resolution → "No" (block overwrite, no contradiction detection)
  — Added footnotes and "corrections welcome via PR" attribution
- `<YOUR_USERNAME>` placeholders replaced with `amoghrman` everywhere

### Fixed
- All ruff lint errors auto-fixed (import ordering, `datetime.UTC` alias, `Optional[X]` → `X | None`)
- `remember()` now uses a sentinel default so `auto_importance` can distinguish
  "caller passed nothing" from "caller explicitly passed 0.5"

## [0.1.0] — 2026-05-26

### Added
- Core API: `remember()`, `recall()`, `forget()`, `get_conflict_log()`
- Conflict resolution with supersede / merge / coexist decisions, decided by LLM
- Audit log of every conflict decision (`conflict_log` table)
- Hybrid retrieval: 0.5 × vector + 0.2 × BM25 + 0.15 × recency decay + 0.15 × importance
- Soft delete via `is_active` flag; `superseded_by` chain for provenance
- Pluggable embedder abstraction — ships with `OpenAIEmbedder` and `SentenceTransformersEmbedder`
- Pluggable LLM abstraction — ships with `OpenAILLM` and `GroqLLM`
- Auto-detection of available backends from environment variables
- Free local path: sentence-transformers + Groq, no OpenAI account required
- Dynamic embedding dimension: column width set from embedder at `setup()` time
- Dimension migration: `setup()` detects and migrates existing columns on dimension change
- `setup(reset=True)` to wipe and recreate tables (useful when switching embedders)
- Docker Compose setup for Postgres 15 + pgvector
- 22 tests covering remember / recall / forget / conflict / embedder scenarios

### Known limitations
- Merge re-embed race condition under high concurrency (single-process safe)
- Multi-conflict: only the first merge is synthesised; subsequent matches are superseded
- Consolidation is stubbed (`NotImplementedError`) — not yet implemented
- No benchmark suite yet
