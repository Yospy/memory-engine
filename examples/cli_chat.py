#!/usr/bin/env python3
"""Simple CLI chat demo showcasing the Memory Engine.

Usage:
    python examples/cli_chat.py [--provider openai|anthropic] [--db memory.db]
"""
from __future__ import annotations

import argparse
import sys
import os

# Add parent dir to path so we can import memory_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_engine.engine import MemoryEngine
from memory_engine.models import Conversation, Message


def create_engine(provider: str, db_path: str) -> MemoryEngine:
    """Create a MemoryEngine with the specified provider."""
    if provider == "openai":
        from memory_engine.llm.openai import OpenAIProvider
        from memory_engine.embeddings.openai import OpenAIEmbeddings

        llm = OpenAIProvider()
        embeddings = OpenAIEmbeddings()
    elif provider == "anthropic":
        from memory_engine.llm.anthropic import AnthropicProvider
        from memory_engine.embeddings.openai import OpenAIEmbeddings

        llm = AnthropicProvider()
        # Anthropic doesn't have embeddings, use OpenAI for that
        embeddings = OpenAIEmbeddings()
    else:
        print(f"Unknown provider: {provider}")
        sys.exit(1)

    return MemoryEngine(llm=llm, embeddings=embeddings, db_path=db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory Engine CLI Chat Demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--db",
        default="memory.db",
        help="Path to SQLite database file (default: memory.db)",
    )
    args = parser.parse_args()

    engine = create_engine(args.provider, args.db)
    conversation = Conversation()

    base_system = "You are a helpful assistant. Be concise and friendly."

    print("Memory Engine CLI Chat")
    print("=" * 40)
    print(f"Provider: {args.provider}")
    print(f"Database: {args.db}")
    print()

    # Show existing memories
    existing = engine.get_all_memories()
    if existing:
        print(f"Loaded {len(existing)} existing memories:")
        for mem in existing[:5]:
            print(f"  - {mem.content} (confidence: {mem.confidence:.2f})")
        if len(existing) > 5:
            print(f"  ... and {len(existing) - 5} more")
        print()

    print("Type your messages below. Commands:")
    print("  /memories  - show all stored memories")
    print("  /decay     - apply memory decay")
    print("  /quit      - exit")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye!")
            break

        if user_input == "/memories":
            memories = engine.get_all_memories()
            if not memories:
                print("No memories stored yet.\n")
            else:
                print(f"\nStored memories ({len(memories)}):")
                for mem in memories:
                    print(
                        f"  [{mem.memory_type.value}] {mem.content} "
                        f"(confidence: {mem.confidence:.2f}, "
                        f"accessed: {mem.access_count}x)"
                    )
                print()
            continue

        if user_input == "/decay":
            deactivated = engine.decay()
            print(f"Decay applied. {deactivated} memories deactivated.\n")
            continue

        # Add user message to conversation
        conversation.messages.append(Message(role="user", content=user_input))

        # Build prompt with memories
        system_prompt = engine.build_prompt(
            base_system, user_input, limit=5, use_reranking=False
        )

        # Generate response
        if hasattr(engine.llm, "complete"):
            response = engine.llm.complete(user_input, system=system_prompt)
        else:
            response = "I understand."

        print(f"Assistant: {response}\n")

        # Add assistant message to conversation
        conversation.messages.append(Message(role="assistant", content=response))

        # Observe conversation for new memories (every 3 turns)
        if len(conversation.messages) >= 4 and len(conversation.messages) % 4 == 0:
            new_memories = engine.observe(conversation)
            if new_memories:
                print(f"  [Memory Engine: stored {len(new_memories)} new memories]")
                for mem in new_memories:
                    print(f"    + {mem.content} (confidence: {mem.confidence:.2f})")
                print()

    engine.close()


if __name__ == "__main__":
    main()
