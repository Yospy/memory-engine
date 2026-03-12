from __future__ import annotations

from typing import Optional

from memory_engine.builder import PromptBuilder
from memory_engine.embeddings.base import EmbeddingProvider
from memory_engine.llm.base import LLMProvider
from memory_engine.models import Conversation, Memory
from memory_engine.observer import MemoryObserver
from memory_engine.retriever import MemoryRetriever
from memory_engine.store import MemoryStore


class MemoryEngine:
    """Main orchestrator for the memory pipeline.

    Ties together observation, storage, retrieval, and prompt building
    into a simple API.
    """

    def __init__(
        self,
        llm: LLMProvider,
        embeddings: EmbeddingProvider,
        db_path: str = "memory.db",
        half_life_days: float = 30.0,
        max_memories: int = 10,
        use_reranking: bool = True,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.store = MemoryStore(db_path=db_path)
        self.observer = MemoryObserver(llm=llm)
        self.retriever = MemoryRetriever(
            store=self.store, embeddings=embeddings, llm=llm if use_reranking else None
        )
        self.builder = PromptBuilder(max_memories=max_memories)
        self.half_life_days = half_life_days

    def observe(self, conversation: Conversation) -> list[Memory]:
        """Extract and store memories from a conversation.

        Args:
            conversation: The conversation to analyze.

        Returns:
            List of newly stored memories.
        """
        # Extract memories via LLM
        new_memories = self.observer.observe(conversation)

        stored: list[Memory] = []
        for memory in new_memories:
            # Generate embedding
            memory.embedding = self.embeddings.embed(memory.content)

            # Check for existing similar memory
            existing = self.store.find_similar(memory.embedding, threshold=0.85)
            if existing:
                # Reinforce existing memory's confidence
                boost = min(0.1, memory.confidence * 0.2)
                self.store.update_confidence(existing.id, boost)
            else:
                # Store new memory
                self.store.add(memory)
                stored.append(memory)

        return stored

    def recall(
        self,
        query: str,
        limit: int = 5,
        use_reranking: bool = True,
    ) -> list[tuple[Memory, float]]:
        """Retrieve relevant memories for a query.

        Args:
            query: The current query or user message.
            limit: Maximum number of memories to return.
            use_reranking: Whether to use LLM re-ranking.

        Returns:
            List of (memory, relevance_score) tuples.
        """
        return self.retriever.retrieve(
            query=query, limit=limit, use_reranking=use_reranking
        )

    def build_prompt(
        self,
        base_system: str,
        query: str,
        limit: int = 5,
        use_reranking: bool = True,
    ) -> str:
        """Retrieve memories and build a system prompt.

        Convenience method that combines recall + prompt building.

        Args:
            base_system: The base system prompt.
            query: The user's current message.
            limit: Max memories to include.
            use_reranking: Whether to use LLM re-ranking.

        Returns:
            Complete system prompt with memory section.
        """
        memories = self.recall(query, limit=limit, use_reranking=use_reranking)
        return self.builder.build_full_system_prompt(base_system, memories, query)

    def process_turn(
        self,
        conversation: Conversation,
        query: str,
        base_system: str = "",
        limit: int = 5,
    ) -> tuple[list[Memory], str]:
        """Full pipeline: observe conversation + build prompt for next response.

        Args:
            conversation: The conversation so far (for memory extraction).
            query: The user's latest message (for retrieval).
            base_system: Base system prompt.
            limit: Max memories to inject.

        Returns:
            Tuple of (newly_stored_memories, system_prompt_with_memories).
        """
        new_memories = self.observe(conversation)
        system_prompt = self.build_prompt(
            base_system, query, limit=limit, use_reranking=True
        )
        return new_memories, system_prompt

    def decay(self) -> int:
        """Apply time-based confidence decay to all memories.

        Returns:
            Number of memories deactivated.
        """
        return self.store.decay_memories(half_life_days=self.half_life_days)

    def get_all_memories(self, active_only: bool = True) -> list[Memory]:
        """Get all stored memories."""
        return self.store.get_all(active_only=active_only)

    def close(self) -> None:
        """Close the database connection."""
        self.store.close()
