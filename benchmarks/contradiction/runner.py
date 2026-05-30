"""Contradiction handling benchmark runner.

Measures how accurately a memory system surfaces the most recent fact
when given two contradicting pieces of information about a user.

Usage:
    python -m benchmarks.contradiction.runner --dataset handcrafted --systems hippo
    python -m benchmarks.contradiction.runner --dataset handcrafted --systems hippo,mem0
    python -m benchmarks.contradiction.runner --dataset locomo --systems hippo --judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapters.base import MemorySystemAdapter
from .scoring.contradiction_scorer import score_response

_HERE = Path(__file__).parent
_DATASETS_DIR = _HERE / "datasets"
_RESULTS_DIR = _HERE / "results"


# ── Dataset loading ────────────────────────────────────────────────────────────


def load_dataset(name: str) -> list[dict]:
    if name == "handcrafted":
        path = _DATASETS_DIR / "handcrafted.json"
    elif name == "locomo":
        path = _DATASETS_DIR / "locomo_temporal.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m benchmarks.contradiction.datasets.locomo_loader"
            )
    else:
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {name}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


# ── Per-scenario execution ─────────────────────────────────────────────────────


async def _run_scenario(
    system: MemorySystemAdapter,
    scenario: dict,
    use_llm_judge: bool = False,
    llm: Any = None,
) -> dict:
    user_id = scenario["user_id"]
    agent_id = scenario["agent_id"]

    await system.reset()

    # remember initial fact
    t0 = time.perf_counter()
    await system.remember(scenario["initial_fact"], user_id=user_id, agent_id=agent_id)
    t_remember1 = time.perf_counter() - t0

    # remember contradicting fact
    t0 = time.perf_counter()
    await system.remember(scenario["contradicting_fact"], user_id=user_id, agent_id=agent_id)
    t_remember2 = time.perf_counter() - t0

    # recall
    t0 = time.perf_counter()
    recalled = await system.recall(scenario["query"], user_id=user_id, agent_id=agent_id, limit=1)
    t_recall = time.perf_counter() - t0

    score = score_response(
        recalled=recalled,
        correct_answer_contains=scenario["correct_answer_contains"],
        incorrect_answer_contains=scenario["incorrect_answer_contains"],
    )

    judge_result: dict | None = None
    if use_llm_judge and llm is not None and not score["correct"]:
        from .scoring.llm_judge import judge_response

        judge_result = await judge_response(
            query=scenario["query"],
            expected=str(scenario["correct_answer_contains"]),
            actual=score["top_result"],
            llm=llm,
        )

    return {
        "scenario_id": scenario["id"],
        "category": scenario["category"],
        "system": system.name,
        "initial_fact": scenario["initial_fact"],
        "contradicting_fact": scenario["contradicting_fact"],
        "query": scenario["query"],
        "recalled": recalled,
        "score": score,
        "judge": judge_result,
        "latency_ms": {
            "remember1": round(t_remember1 * 1000, 1),
            "remember2": round(t_remember2 * 1000, 1),
            "recall": round(t_recall * 1000, 1),
        },
    }


# ── Aggregation ────────────────────────────────────────────────────────────────


def _aggregate(records: list[dict], system_name: str, total: int) -> dict:
    system_records = [r for r in records if r["system"] == system_name]

    correct = sum(1 for r in system_records if r["score"]["correct"])
    hallucinated = sum(1 for r in system_records if r["score"]["hallucinated"])

    judge_correct = sum(
        1
        for r in system_records
        if r.get("judge") and r["judge"].get("correct")
    )

    by_category: dict[str, dict] = {}
    for r in system_records:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "correct": 0}
        by_category[cat]["total"] += 1
        if r["score"]["correct"]:
            by_category[cat]["correct"] += 1

    recall_latencies = [r["latency_ms"]["recall"] for r in system_records]
    remember_latencies = [
        r["latency_ms"]["remember1"] + r["latency_ms"]["remember2"]
        for r in system_records
    ]

    def _p50(vals: list[float]) -> float:
        return statistics.median(vals) if vals else 0.0

    return {
        "system": system_name,
        "total": total,
        "correct": correct,
        "correct_pct": round(correct / total * 100, 1) if total else 0,
        "hallucinated": hallucinated,
        "hallucinated_pct": round(hallucinated / total * 100, 1) if total else 0,
        "judge_correct": judge_correct,
        "by_category": by_category,
        "p50_recall_ms": round(_p50(recall_latencies), 1),
        "p50_remember_ms": round(_p50(remember_latencies), 1),
    }


# ── Printing ───────────────────────────────────────────────────────────────────


def _print_table(summaries: list[dict], dataset_name: str, n: int) -> None:
    W = 64
    print()
    print("Benchmark: contradiction handling")
    print(f"Dataset: {dataset_name} ({n} scenarios)")
    print()
    header = f"{'System':<12}  {'Correct':>10}  {'Hallucinated':>14}  {'p50 recall':>12}  {'p50 remember':>13}"
    print(header)
    print("-" * W)
    for s in summaries:
        print(
            f"{s['system']:<12}  "
            f"{s['correct']:>4} / {s['total']:<4}  "
            f"{s['hallucinated']:>5} / {s['total']:<4}  "
            f"{s['p50_recall_ms']:>9.0f} ms  "
            f"{s['p50_remember_ms']:>10.0f} ms"
        )
    print()
    for s in summaries:
        if s["by_category"]:
            print(f"  {s['system']} by category:")
            for cat, counts in sorted(s["by_category"].items()):
                pct = round(counts["correct"] / counts["total"] * 100) if counts["total"] else 0
                print(f"    {cat:<28} {counts['correct']:>2}/{counts['total']:<2}  ({pct}%)")
        print()


# ── Main benchmark function ────────────────────────────────────────────────────


async def run_benchmark(
    systems: list[MemorySystemAdapter],
    dataset: list[dict],
    dataset_name: str = "unknown",
    use_llm_judge: bool = False,
    llm: Any = None,
    output_path: Path | None = None,
) -> dict:
    """Run the full benchmark and return the results dict."""
    all_records: list[dict] = []
    n = len(dataset)

    for system in systems:
        print(f"\nRunning {system.name} on {n} scenarios …")
        for i, scenario in enumerate(dataset, 1):
            record = await _run_scenario(system, scenario, use_llm_judge=use_llm_judge, llm=llm)
            all_records.append(record)
            status = "OK" if record["score"]["correct"] else "MISS"
            if i % 10 == 0 or i == n:
                print(f"  {i}/{n} done")
            else:
                print(f"  [{status}] {scenario['id']}", end="\r")

    summaries = [_aggregate(all_records, s.name, n) for s in systems]

    _print_table(summaries, dataset_name, n)

    results = {
        "run_at": datetime.now(tz=UTC).isoformat(),
        "dataset": dataset_name,
        "n_scenarios": n,
        "systems": [s.name for s in systems],
        "summaries": summaries,
        "records": all_records,
    }

    if output_path is None:
        _RESULTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = _RESULTS_DIR / f"{ts}_{dataset_name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results written to {output_path}")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────


def _build_hippo_adapter(database_url: str) -> MemorySystemAdapter:

    from hippo import Hippo

    from .adapters.hippo_adapter import HippoAdapter

    hippo = Hippo(database_url=database_url, conflict_detection=True)

    async def _setup_and_return():
        await hippo.setup()
        return hippo

    hippo = asyncio.get_event_loop().run_until_complete(_setup_and_return())
    return HippoAdapter(client=hippo)


def _build_mem0_adapter() -> MemorySystemAdapter:
    from .adapters.mem0_adapter import Mem0Adapter

    return Mem0Adapter()


async def _async_main(args: argparse.Namespace) -> None:
    import os

    dataset = load_dataset(args.dataset)
    system_names = [s.strip() for s in args.systems.split(",")]

    adapters: list[MemorySystemAdapter] = []
    for name in system_names:
        if name == "hippo":
            from hippo import Hippo

            from .adapters.hippo_adapter import HippoAdapter

            db_url = os.environ.get(
                "DATABASE_URL", "postgresql+asyncpg://hippo:hippo@localhost:5432/hippo"
            )
            hippo = Hippo(database_url=db_url, conflict_detection=True)
            await hippo.setup()
            adapters.append(HippoAdapter(client=hippo))
        elif name == "mem0":
            from .adapters.mem0_adapter import Mem0Adapter

            adapters.append(Mem0Adapter())
        else:
            raise ValueError(f"Unknown system: {name!r}. Choose from: hippo, mem0")

    llm = None
    if args.judge:
        # Reuse Hippo's configured LLM for judging if available.
        for adapter in adapters:
            if hasattr(adapter, "_client") and hasattr(adapter._client, "_llm"):
                llm = adapter._client._llm
                break
        if llm is None:
            print("Warning: --judge requested but no LLM found; skipping LLM judge.")

    output = Path(args.output) if args.output else None
    await run_benchmark(
        systems=adapters,
        dataset=dataset,
        dataset_name=args.dataset,
        use_llm_judge=args.judge,
        llm=llm,
        output_path=output,
    )

    # Close Hippo connections.
    for adapter in adapters:
        if hasattr(adapter, "_client") and hasattr(adapter._client, "close"):
            await adapter._client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Contradiction handling benchmark for Hippo.")
    parser.add_argument(
        "--dataset",
        default="handcrafted",
        help="Dataset name: 'handcrafted', 'locomo', or path to a JSON file.",
    )
    parser.add_argument(
        "--systems",
        default="hippo",
        help="Comma-separated list of systems to benchmark: hippo, mem0.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Use LLM-as-judge for cases where keyword scoring is ambiguous.",
    )
    parser.add_argument("--output", help="Override output file path.", default=None)
    args = parser.parse_args()

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
