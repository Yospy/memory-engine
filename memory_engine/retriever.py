from __future__ import annotations

import json
import re
from typing import Optional

from memory_engine.embeddings.base import EmbeddingProvider
from memory_engine.llm.base import LLMProvider
from memory_engine.models import Memory
from memory_engine.store import MemoryStore


RERANK_SYSTEM = """You are a memory relevance judge. Given a user's current query and a list
of candidate memories, determine which memories are actually relevant.

For each memory, assign a relevance score from 0.0 to 1.0:
- 1.0 = directly relevant, should definitely be included
- 0.5 = somewhat relevant, might be useful
- 0.0 = not relevant to the current query

Respond ONLY with a JSON array of objects, each with:
- "id": the memory ID
- "relevance": float 0.0-1.0

Example:
[{"id": "abc-123", "relevance": 0.9}, {"id": "def-456", "relevance": 0.1}]"""


class MemoryRetriever:
    """Hybrid memory retrieval: embedding similarity + LLM re-ranking."""

    def __init__(
        self,
        store: MemoryStore,
        embeddings: EmbeddingProvider,
        llm: Optional[LLMProvider] = None,
    ):
        self.store = store
        self.embeddings = embeddings
        self.llm = llm  # If None, skip re-ranking

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        candidate_limit: int = 15,
        min_relevance: float = 0.3,
        use_reranking: bool = True,
    ) -> list[tuple[Memory, float]]:
        """Retrieve relevant memories for a query.

        Two-stage pipeline:
        1. Embedding similarity search (fast, broad)
        2. LLM re-ranking (slow, precise) — optional

        Args:
            query: The user's current message/query.
            limit: Max memories to return.
            candidate_limit: How many candidates to fetch in stage 1.
            min_relevance: Minimum relevance score to include.
            use_reranking: Whether to use LLM re-ranking (stage 2).

        Returns:
            List of (memory, relevance_score) tuples.
        """
        # Stage 1: Embedding similarity
        query_embedding = self.embeddings.embed(query)
        candidates = self.store.search_by_embedding(query_embedding, limit=candidate_limit)

        if not candidates:
            return []

        # Stage 2: LLM re-ranking (if available)
        if use_reranking and self.llm is not None and len(candidates) > limit:
            results = self._rerank(query, candidates)
        else:
            results = candidates

        # Filter by minimum relevance and limit
        results = [(mem, score) for mem, score in results if score >= min_relevance]
        results = results[:limit]

        # Record access for returned memories
        for mem, _ in results:
            self.store.record_access(mem.id)

        return results

    def _rerank(
        self, query: str, candidates: list[tuple[Memory, float]]
    ) -> list[tuple[Memory, float]]:
        """Use LLM to re-rank candidate memories by relevance to query."""
        # Build prompt with candidates
        memory_list = []
        memory_map: dict[str, Memory] = {}
        for mem, sim_score in candidates:
            memory_list.append({"id": mem.id, "content": mem.content})
            memory_map[mem.id] = mem

        prompt = f"""Query: {query}

Candidate memories:
{json.dumps(memory_list, indent=2)}

Judge which memories are relevant to the query."""

        response = self.llm.complete(prompt, system=RERANK_SYSTEM)

        # Parse re-ranking response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            return candidates  # Fallback to original ranking

        try:
            rankings = json.loads(json_match.group())
        except json.JSONDecodeError:
            return candidates

        # Build re-ranked results
        reranked: list[tuple[Memory, float]] = []
        for item in rankings:
            mem_id = item.get("id")
            relevance = float(item.get("relevance", 0))
            if mem_id in memory_map:
                reranked.append((memory_map[mem_id], relevance))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
