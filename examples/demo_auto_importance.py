"""Demo: automatic importance scoring.

Hippo scores each memory's importance automatically using the configured LLM,
so callers don't have to guess a number. The score influences recall ranking.

Run:
    python examples/demo_auto_importance.py
"""

import asyncio
import os

from dotenv import load_dotenv

from hippo import Hippo

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://hippo:hippo@localhost:5432/hippo")
AGENT_ID = "demo-importance"
SEP = "-" * 60


async def main() -> None:
    print("\n" + "=" * 60)
    print("           Hippo Auto-Importance Demo")
    print("=" * 60)

    memory = Hippo(
        database_url=DATABASE_URL,
        conflict_detection=False,
        auto_importance=True,
    )
    print(f"\n  Backends: embedder={type(memory._embedder).__name__}, llm={type(memory._llm).__name__ if memory._llm else 'none'}")

    if memory._importance_scorer is None:
        print("\n  [!] No LLM configured -- auto_importance requires OPENAI_API_KEY or GROQ_API_KEY.")
        print("      Falling back to manual demo with explicit importance values.\n")
        await memory.close()
        return

    await memory.setup(reset=True)

    memories = [
        "User is severely allergic to shellfish",
        "User mentioned the weather looked nice today",
        "User's primary decision-maker for all engineering hires at their company",
        "User said 'thanks' after I helped them",
        "User prefers dark mode on all interfaces",
        "User is learning Spanish as a hobby",
        "The user works at Acme Corp as a staff engineer",
    ]

    print(f"\n  Storing {len(memories)} memories with auto-importance scoring...\n")

    scored = []
    for text in memories:
        await memory.remember(text, agent_id=AGENT_ID)
        results = await memory.recall(text[:40], agent_id=AGENT_ID, limit=1)
        if results:
            scored.append((results[0].importance, text))

    scored.sort(reverse=True)
    print(f"  {'Score':>6}  Memory")
    print(f"  {'-'*6}  {'-'*50}")
    for score, text in scored:
        print(f"  {score:.2f}   {text!r}")

    await memory.close()
    print(f"\n{SEP}")
    print("  Demo complete.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    asyncio.run(main())
