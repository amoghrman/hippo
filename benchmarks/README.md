# Benchmarks

Benchmark suite coming soon. Planned measurements:

| Benchmark | Description |
|---|---|
| `recall_latency` | p50/p95/p99 recall latency at 1k / 10k / 100k memories |
| `conflict_detection_overhead` | Extra latency added by LLM conflict check vs. bare insert |
| `hnsw_vs_exact` | HNSW approximate recall accuracy vs. exact cosine search |
| `hybrid_vs_vector_only` | MRR improvement from hybrid scoring over pure vector retrieval |

## Running (once implemented)

```bash
cd benchmarks
python bench_recall.py --memories 10000 --queries 100
```

## Preliminary notes

- HNSW index (`m=16, ef_construction=64`) handles ~1M vectors on a single Postgres node.
- Conflict detection adds one LLM round-trip (~200–400 ms with gpt-4o-mini). Can be batched.
- Hybrid scoring adds ~2 ms vs. pure vector search at 100k rows.
