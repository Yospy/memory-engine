#!/usr/bin/env python3
"""Terminal Chat with Verbose Memory Logging.

Every memory operation (recall, observe, reinforce) is visibly logged
with confidence scores, tags, types, and storage status.

Usage:
    python terminalchat.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

from memory_engine.engine import MemoryEngine
from memory_engine.llm.openai import OpenAIProvider
from memory_engine.embeddings.openai import OpenAIEmbeddings
from memory_engine.models import Conversation, Message, Memory


# ── Colour helpers (ANSI) ───────────────────────────────────────────
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _confidence_bar(conf: float) -> str:
    filled = int(conf * 10)
    return f"[{'█' * filled}{'░' * (10 - filled)}] {conf:.2f}"


def _tags_str(tags: list[str]) -> str:
    return ", ".join(f"#{t}" for t in tags) if tags else "—"


# ── Print helpers ───────────────────────────────────────────────────
def print_header(text: str, colour: str = CYAN) -> None:
    print(f"\n{colour}{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}{RESET}")


def print_memory_detail(
    mem: Memory, score: float | None = None, prefix: str = ""
) -> None:
    label = prefix or f"  [{mem.memory_type.value.upper()}]"
    conf = _confidence_bar(mem.confidence)
    tags = _tags_str(mem.tags)
    parts = [
        f"{label} {mem.content}",
        f"        confidence: {conf}",
        f"        tags: {tags}  |  accesses: {mem.access_count}",
    ]
    if score is not None:
        parts.insert(1, f"        relevance:  {score:.4f}")
    print("\n".join(parts))


# ── Startup stats ──────────────────────────────────────────────────
def print_startup_stats(engine: MemoryEngine) -> None:
    memories = engine.get_all_memories(active_only=True)
    inactive = engine.get_all_memories(active_only=False)
    total = len(inactive)
    active = len(memories)

    print(f"\n{BOLD}Memory Engine — Terminal Chat (verbose){RESET}")
    print(f"{DIM}DB: terminalchat.db{RESET}")
    print(f"\n  Total memories : {total}")
    print(f"  Active         : {active}")
    print(f"  Inactive       : {total - active}")

    if memories:
        types = Counter(m.memory_type.value for m in memories)
        for t, c in types.most_common():
            print(f"    {t:12s} : {c}")
    print()


# ── RECALL phase ───────────────────────────────────────────────────
def recall_phase(engine: MemoryEngine, user_msg: str) -> list[tuple[Memory, float]]:
    recalled = engine.recall(user_msg, limit=5, use_reranking=False)

    print_header("RECALL", CYAN)
    if not recalled:
        print(f"  {DIM}(no relevant memories found){RESET}")
    else:
        print(f"  {len(recalled)} memories retrieved:\n")
        for mem, score in recalled:
            print_memory_detail(mem, score=score)
            print()
    return recalled


# ── RESPOND phase ──────────────────────────────────────────────────
def respond_phase(
    engine: MemoryEngine,
    conversation: Conversation,
    user_msg: str,
    recalled: list[tuple[Memory, float]],
) -> str:
    base_system = (
        "You are a helpful, concise assistant. "
        "Use any known memories about the user to personalise your responses."
    )
    system_prompt = engine.builder.build_full_system_prompt(
        base_system, recalled, user_msg
    )

    # Build full message history for the LLM
    prompt_lines: list[str] = []
    for msg in conversation.messages:
        prompt_lines.append(f"{msg.role}: {msg.content}")
    full_prompt = "\n".join(prompt_lines)

    print_header("RESPOND", GREEN)
    response = engine.llm.complete(full_prompt, system=system_prompt)
    print(f"  {BOLD}Assistant:{RESET} {response}")
    return response


# ── OBSERVE phase (verbose, with reinforcement detection) ──────────
def observe_phase(engine: MemoryEngine, conversation: Conversation) -> list[Memory]:
    print_header("OBSERVE", YELLOW)

    # Step 1: extract via LLM
    extracted = engine.observer.observe(conversation)

    if not extracted:
        print(f"  {DIM}(no memories extracted this turn){RESET}")
        return []

    print(f"  {len(extracted)} memories extracted from conversation:\n")

    stored: list[Memory] = []
    for mem in extracted:
        # Step 2: generate embedding
        mem.embedding = engine.embeddings.embed(mem.content)

        # Step 3: check for existing similar memory
        existing = engine.store.find_similar(mem.embedding, threshold=0.85)

        if existing:
            # Reinforcement
            boost = min(0.1, mem.confidence * 0.2)
            old_conf = existing.confidence
            engine.store.update_confidence(existing.id, boost)
            new_conf = min(1.0, old_conf + boost)
            print(
                f"  {MAGENTA}↑ REINFORCED{RESET}  {existing.content}\n"
                f"        type: {existing.memory_type.value}  |  "
                f"tags: {_tags_str(existing.tags)}\n"
                f"        confidence: {old_conf:.2f} → {new_conf:.2f}  "
                f"(+{boost:.3f})\n"
            )
        else:
            # New memory
            engine.store.add(mem)
            stored.append(mem)
            print(
                f"  {GREEN}★ NEW MEMORY{RESET}  {mem.content}\n"
                f"        type: {mem.memory_type.value}  |  "
                f"tags: {_tags_str(mem.tags)}\n"
                f"        confidence: {_confidence_bar(mem.confidence)}\n"
            )

    return stored


# ── Commands ───────────────────────────────────────────────────────
def cmd_memories(engine: MemoryEngine) -> None:
    memories = engine.get_all_memories(active_only=True)
    if not memories:
        print(f"\n  {DIM}No memories stored yet.{RESET}\n")
        return
    print_header(f"ALL MEMORIES ({len(memories)})", MAGENTA)
    for mem in memories:
        print_memory_detail(mem)
        print(f"        created: {mem.created_at}  |  last used: {mem.last_used}")
        print()


def cmd_decay(engine: MemoryEngine) -> None:
    print_header("DECAY", RED)
    before = engine.get_all_memories(active_only=True)
    deactivated = engine.decay()
    after = engine.get_all_memories(active_only=True)
    print(f"  Memories before : {len(before)}")
    print(f"  Deactivated     : {deactivated}")
    print(f"  Memories after  : {len(after)}")

    if deactivated > 0:
        # Show which ones were deactivated
        active_ids = {m.id for m in after}
        for m in before:
            if m.id not in active_ids:
                print(
                    f"  {RED}✗{RESET} {m.content} "
                    f"(was {m.confidence:.2f})"
                )
    print()


def cmd_stats(engine: MemoryEngine) -> None:
    memories = engine.get_all_memories(active_only=True)
    if not memories:
        print(f"\n  {DIM}No memories.{RESET}\n")
        return
    print_header("STATS", CYAN)
    types = Counter(m.memory_type.value for m in memories)
    avg_conf = sum(m.confidence for m in memories) / len(memories)
    total_access = sum(m.access_count for m in memories)

    print(f"  Total active    : {len(memories)}")
    print(f"  Avg confidence  : {avg_conf:.3f}")
    print(f"  Total accesses  : {total_access}")
    print(f"  By type:")
    for t, c in types.most_common():
        print(f"    {t:12s} : {c}")
    print()


# ── Main loop ──────────────────────────────────────────────────────
def main() -> None:
    engine = MemoryEngine(
        llm=OpenAIProvider(),
        embeddings=OpenAIEmbeddings(),
        db_path="terminalchat.db",
        use_reranking=False,
    )
    conversation = Conversation()

    print_startup_stats(engine)

    print(f"Commands: {BOLD}/memories  /decay  /stats  /clear  /quit{RESET}")
    print(f"{DIM}Every turn shows RECALL → RESPOND → OBSERVE phases{RESET}\n")

    try:
        while True:
            try:
                user_input = input(f"{BOLD}You:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Commands
            if user_input == "/quit":
                print("Goodbye!")
                break
            if user_input == "/memories":
                cmd_memories(engine)
                continue
            if user_input == "/decay":
                cmd_decay(engine)
                continue
            if user_input == "/stats":
                cmd_stats(engine)
                continue
            if user_input == "/clear":
                conversation = Conversation()
                print(f"  {DIM}Conversation cleared (memories preserved).{RESET}\n")
                continue

            # ── Turn pipeline ──────────────────────────────────────
            # 1. RECALL
            recalled = recall_phase(engine, user_input)

            # Add user message to conversation
            conversation.messages.append(Message(role="user", content=user_input))

            # 2. RESPOND
            response = respond_phase(engine, conversation, user_input, recalled)

            # Add assistant message
            conversation.messages.append(
                Message(role="assistant", content=response)
            )

            # 3. OBSERVE (every turn for verbose demo)
            observe_phase(engine, conversation)

    finally:
        engine.close()


if __name__ == "__main__":
    main()
