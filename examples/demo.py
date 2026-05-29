"""Hippo end-to-end demo -- conflict resolution story.

Shows Hippo's core value: an agent that stops believing two contradictory
things at once.

Backend priority (auto-detected from environment):
  1. OPENAI_API_KEY set  -> OpenAI embeddings + GPT-4o-mini for conflict LLM
  2. GROQ_API_KEY set    -> sentence-transformers (local) + Groq LLaMA for conflict LLM
  3. Neither set         -> sentence-transformers only, conflict detection disabled

Prerequisites:
    docker compose up -d   # or a local PostgreSQL 15+ with pgvector

Run:
    # With OpenAI:
    set OPENAI_API_KEY=sk-...  && python examples/demo.py

    # With Groq + local embeddings:
    set GROQ_API_KEY=gsk-...   && python examples/demo.py

    # Embeddings only (no conflict detection):
    python examples/demo.py
"""
import asyncio
import os

from dotenv import load_dotenv

from hippo import Hippo

load_dotenv()  # reads .env from the repo root; safe no-op if file is absent

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://hippo:hippo@localhost:5432/hippo",
)

AGENT_ID = "demo-agent"
USER_ID = "demo-user"
SEP = "-" * 60


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _active_backend_label(memory: Hippo) -> str:
    embedder_name = type(memory._embedder).__name__
    llm_name = type(memory._llm).__name__ if memory._llm else "none"
    return f"embedder={embedder_name}, llm={llm_name}, conflict={memory._conflict_detection}"


async def main() -> None:
    print("\n" + "=" * 60)
    print("           Hippo Memory Layer -- Demo")
    print("=" * 60)

    memory = Hippo(
        database_url=DATABASE_URL,
        conflict_detection=True,
        conflict_threshold=0.60,  # bge-small-en-v1.5 produces lower scores than OpenAI embeddings
    )

    print(f"\n  Active backends: {_active_backend_label(memory)}")

    # reset=True recreates tables with the correct embedding dimension,
    # handling the case where a previous run used a different embedder.
    await memory.setup(reset=True)

    # -- Step 1: Agent learns the user's language preference ------------------
    section("Step 1 -- Agent learns: 'User prefers Python'")

    await memory.remember(
        "User prefers Python as their primary programming language",
        agent_id=AGENT_ID,
        user_id=USER_ID,
        importance=0.8,
        metadata={"source": "onboarding"},
    )
    print("  [ok] Memory stored: 'User prefers Python ...'")

    # -- Step 2: Agent recalls language preference ----------------------------
    section("Step 2 -- Agent recalls: 'What language does the user prefer?'")

    results = await memory.recall(
        "What programming language does the user prefer?",
        agent_id=AGENT_ID,
        user_id=USER_ID,
    )
    top = results[0]
    print(f"  Top memory : {top.content!r}")
    print(f"  Score      : {top.score:.4f}")
    print(f"  Importance : {top.importance}")
    assert "Python" in top.content, "Expected Python memory"

    # -- Step 3: User says they switched to Rust ------------------------------
    section("Step 3 -- User update: 'Actually, I switched to Rust'")
    print("  Storing: 'User has switched from Python to Rust for all new projects'")
    if memory._conflict_detection:
        print("  Hippo is running conflict detection ... (LLM call)")

    await memory.remember(
        "User has switched from Python to Rust for all new projects",
        agent_id=AGENT_ID,
        user_id=USER_ID,
        importance=0.9,
        metadata={"source": "user-correction"},
    )
    print("  [ok] Memory stored")

    # -- Step 4: Agent recalls again -- should get Rust -----------------------
    section("Step 4 -- Agent recalls again: 'What language?'")

    results = await memory.recall(
        "What programming language does the user prefer?",
        agent_id=AGENT_ID,
        user_id=USER_ID,
    )
    top = results[0]
    print(f"  Top memory : {top.content!r}")
    print(f"  Score      : {top.score:.4f}")

    python_present = any("Python" in r.content and "Rust" not in r.content for r in results)
    rust_present = any("Rust" in r.content for r in results)
    print(f"  Rust memory present  : {rust_present}")
    print(f"  Stale Python present : {python_present}  (should be False when conflict=on)")

    # -- Step 5: Inspect the conflict log -------------------------------------
    section("Step 5 -- Conflict resolution log")

    log = await memory.get_conflict_log(agent_id=AGENT_ID)
    if not log:
        print("  (no conflict log entries)")
        if not memory._conflict_detection:
            print("  Tip: set OPENAI_API_KEY or GROQ_API_KEY to enable conflict detection.")
    else:
        for entry in log:
            print(f"  Decision  : {entry['decision'].upper()}")
            print(f"  Old memory: {entry['old_content']!r}")
            print(f"  New memory: {entry['new_content']!r}")
            print(f"  Reason    : {entry['reason']}")
            print(f"  Timestamp : {entry['ts']}")

    await memory.close()

    print(f"\n{SEP}")
    print("  Demo complete.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    asyncio.run(main())
