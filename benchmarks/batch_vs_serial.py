"""Benchmark: serial remember() vs remember_batch().

Simulates realistic API latencies using asyncio.sleep so results scale predictably
regardless of local machine speed.

Requires a running Postgres with pgvector:
    docker compose up -d
    python benchmarks/batch_vs_serial.py

Latency model
    MockEmbedder.embed()       : 5 ms  per call  (one item)
    MockEmbedder.embed_batch() : 10 ms per call  (flat, any batch size)
    MockLLM.check_conflict()   : 200 ms per call

Expected speedup (conflict off):
    N=100 serial  : 100 × 5 ms = 500 ms
    N=100 batch   : 1 × 10 ms  = 10 ms  (50× faster)
"""

from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv

from hippo import Hippo
from hippo.embedders.base import Embedder
from hippo.llm.base import LLM, ConflictResult

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://hippo:hippo@localhost:5432/hippo")

EMBED_LATENCY_S = 0.005   # 5 ms per embed() call
BATCH_LATENCY_S = 0.010   # 10 ms per embed_batch() call (flat)
LLM_LATENCY_S   = 0.200   # 200 ms per check_conflict() call


# ── Mock backends ──────────────────────────────────────────────────────────────


class _MockEmbedder(Embedder):
    _DIM = 128

    async def embed(self, text: str) -> list[float]:
        await asyncio.sleep(EMBED_LATENCY_S)
        return [0.1] * self._DIM

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(BATCH_LATENCY_S)
        return [[0.1] * self._DIM] * len(texts)

    @property
    def dimensions(self) -> int:
        return self._DIM


class _MockLLM(LLM):
    async def check_conflict(self, old: str, new: str) -> ConflictResult:
        await asyncio.sleep(LLM_LATENCY_S)
        return ConflictResult(contradicts=False, resolution="coexist", reason="bench mock")

    async def synthesize_merge(self, old: str, new: str) -> str:
        return f"{old} | {new}"

    async def complete(self, prompt: str) -> str:
        return "0.5"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _items(n: int, agent_id: str) -> list[dict]:
    return [{"content": f"Memory {i}: the user said something relevant.", "agent_id": agent_id}
            for i in range(n)]


async def _run_serial(client: Hippo, agent_id: str, n: int) -> float:
    t0 = time.perf_counter()
    for item in _items(n, agent_id):
        await client.remember(item["content"], agent_id=agent_id)
    return time.perf_counter() - t0


async def _run_batch(client: Hippo, agent_id: str, n: int, conflict: bool) -> float:
    t0 = time.perf_counter()
    await client.remember_batch(
        _items(n, agent_id),
        conflict_detection=conflict,
        batch_size=100,
    )
    return time.perf_counter() - t0


# ── Main ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    embedder = _MockEmbedder()
    llm = _MockLLM()

    client = Hippo(
        database_url=DATABASE_URL,
        embedder=embedder,
        llm=llm,
        conflict_detection=False,  # serial baseline also has it off; we compare separately
    )
    await client.setup(reset=True)

    col = "{:<6}  {:>14}  {:>14}  {:>14}  {:>14}"
    row = "{:<6}  {:>13.2f}s  {:>13.2f}s  {:>13.2f}s  {:>13.1f}x"

    print()
    print("Benchmark: serial remember() vs remember_batch()")
    print(f"  embed latency   : {EMBED_LATENCY_S*1000:.0f} ms/call  (serial)")
    print(f"  batch latency   : {BATCH_LATENCY_S*1000:.0f} ms/call  (flat per chunk)")
    print(f"  conflict check  : {LLM_LATENCY_S*1000:.0f} ms/call")
    print()
    print(col.format("N", "serial (s)", "batch no-CD (s)", "batch+CD (s)", "speedup (no-CD)"))
    print("-" * 70)

    for n in (100, 500, 1000):
        agent = f"bench-{n}"

        # Reset between runs
        await client.setup(reset=True)

        t_serial = await _run_serial(client, agent, n)
        await client.setup(reset=True)

        t_batch_nocd = await _run_batch(client, agent, n, conflict=False)
        await client.setup(reset=True)

        t_batch_cd = await _run_batch(client, agent, n, conflict=True)

        speedup = t_serial / t_batch_nocd if t_batch_nocd > 0 else float("inf")
        print(row.format(n, t_serial, t_batch_nocd, t_batch_cd, speedup))

    print()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
