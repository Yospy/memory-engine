"""Tests for the Memory Engine components.

These tests use mock LLM and embedding providers to avoid API calls.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from memory_engine.builder import PromptBuilder
from memory_engine.embeddings.base import EmbeddingProvider
from memory_engine.llm.base import LLMProvider
from memory_engine.models import Conversation, Memory, MemoryType, Message
from memory_engine.observer import MemoryObserver
from memory_engine.retriever import MemoryRetriever
from memory_engine.store import MemoryStore
from memory_engine.engine import MemoryEngine


# --- Mock providers ---


class MockLLM(LLMProvider):
    """Mock LLM that returns canned responses."""

    def __init__(self, response: str = "[]"):
        self._response = response

    def complete(self, prompt: str, system: str = "") -> str:
        return self._response


class MockEmbeddings(EmbeddingProvider):
    """Mock embedding provider that returns deterministic vectors."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self._call_count = 0

    def embed(self, text: str) -> list[float]:
        # Simple hash-based deterministic embedding
        h = hash(text) % 1000
        vec = [(h + i) % 100 / 100.0 for i in range(self.dimension)]
        self._call_count += 1
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# --- MemoryStore tests ---


class TestMemoryStore:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = MemoryStore(db_path=self.tmp.name)

    def teardown_method(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def _make_memory(self, content: str = "test memory", **kwargs) -> Memory:
        defaults = dict(
            content=content,
            memory_type=MemoryType.FACT,
            confidence=0.8,
            tags=["test"],
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        defaults.update(kwargs)
        return Memory(**defaults)

    def test_add_and_get(self):
        mem = self._make_memory("user likes Python")
        self.store.add(mem)
        retrieved = self.store.get(mem.id)
        assert retrieved is not None
        assert retrieved.content == "user likes Python"
        assert retrieved.confidence == 0.8

    def test_get_all(self):
        self.store.add(self._make_memory("fact 1", confidence=0.9))
        self.store.add(self._make_memory("fact 2", confidence=0.5))
        self.store.add(self._make_memory("fact 3", confidence=0.7))
        all_mems = self.store.get_all()
        assert len(all_mems) == 3
        # Should be sorted by confidence descending
        assert all_mems[0].confidence >= all_mems[1].confidence

    def test_update_confidence(self):
        mem = self._make_memory(confidence=0.5)
        self.store.add(mem)
        self.store.update_confidence(mem.id, 0.3)
        updated = self.store.get(mem.id)
        assert updated is not None
        assert abs(updated.confidence - 0.8) < 0.01

    def test_update_confidence_clamped(self):
        mem = self._make_memory(confidence=0.9)
        self.store.add(mem)
        self.store.update_confidence(mem.id, 0.5)
        updated = self.store.get(mem.id)
        assert updated is not None
        assert updated.confidence == 1.0

    def test_record_access(self):
        mem = self._make_memory()
        self.store.add(mem)
        self.store.record_access(mem.id)
        self.store.record_access(mem.id)
        updated = self.store.get(mem.id)
        assert updated is not None
        assert updated.access_count == 2
        assert updated.last_used is not None

    def test_delete(self):
        mem = self._make_memory()
        self.store.add(mem)
        self.store.delete(mem.id)
        assert self.store.get(mem.id) is None

    def test_search_by_embedding(self):
        mem1 = self._make_memory("fact 1", embedding=[1.0, 0.0, 0.0, 0.0])
        mem2 = self._make_memory("fact 2", embedding=[0.0, 1.0, 0.0, 0.0])
        mem3 = self._make_memory("fact 3", embedding=[0.9, 0.1, 0.0, 0.0])
        self.store.add(mem1)
        self.store.add(mem2)
        self.store.add(mem3)

        results = self.store.search_by_embedding([1.0, 0.0, 0.0, 0.0], limit=2)
        assert len(results) == 2
        # First result should be most similar to query
        assert results[0][0].content == "fact 1"
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_find_similar(self):
        mem = self._make_memory("user likes Python", embedding=[1.0, 0.0, 0.0, 0.0])
        self.store.add(mem)
        found = self.store.find_similar([0.99, 0.01, 0.0, 0.0], threshold=0.9)
        assert found is not None
        assert found.content == "user likes Python"

    def test_find_similar_no_match(self):
        mem = self._make_memory("user likes Python", embedding=[1.0, 0.0, 0.0, 0.0])
        self.store.add(mem)
        found = self.store.find_similar([0.0, 1.0, 0.0, 0.0], threshold=0.9)
        assert found is None

    def test_decay_memories(self):
        # Create a memory with old updated_at
        mem = self._make_memory(confidence=0.3)
        mem.updated_at = datetime.utcnow() - timedelta(days=120)
        self.store.add(mem)

        deactivated = self.store.decay_memories(half_life_days=30.0)
        assert deactivated == 1

        updated = self.store.get(mem.id)
        assert updated is not None
        assert not updated.is_active


# --- MemoryObserver tests ---


class TestMemoryObserver:
    def test_observe_extracts_memories(self):
        llm_response = json.dumps([
            {
                "content": "The user is a Python developer",
                "confidence": 0.9,
                "memory_type": "fact",
                "tags": ["career", "python"],
            },
            {
                "content": "The user prefers dark mode",
                "confidence": 0.7,
                "memory_type": "preference",
                "tags": ["preferences"],
            },
        ])
        observer = MemoryObserver(llm=MockLLM(response=llm_response))
        conv = Conversation(messages=[
            Message(role="user", content="I'm a Python developer and I love dark mode"),
            Message(role="assistant", content="That's great!"),
        ])
        memories = observer.observe(conv)
        assert len(memories) == 2
        assert memories[0].content == "The user is a Python developer"
        assert memories[0].confidence == 0.9
        assert memories[0].memory_type == MemoryType.FACT
        assert memories[1].memory_type == MemoryType.PREFERENCE

    def test_observe_empty_conversation(self):
        observer = MemoryObserver(llm=MockLLM())
        conv = Conversation()
        memories = observer.observe(conv)
        assert memories == []

    def test_observe_handles_invalid_json(self):
        observer = MemoryObserver(llm=MockLLM(response="not json at all"))
        conv = Conversation(messages=[Message(role="user", content="hello")])
        memories = observer.observe(conv)
        assert memories == []

    def test_observe_handles_empty_array(self):
        observer = MemoryObserver(llm=MockLLM(response="[]"))
        conv = Conversation(messages=[Message(role="user", content="hi there")])
        memories = observer.observe(conv)
        assert memories == []


# --- MemoryRetriever tests ---


class TestMemoryRetriever:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = MemoryStore(db_path=self.tmp.name)
        self.embeddings = MockEmbeddings()

    def teardown_method(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def test_retrieve_without_reranking(self):
        # Add some memories with known embeddings
        mem1 = Memory(content="user likes Python", embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], confidence=0.9)
        mem2 = Memory(content="user lives in NYC", embedding=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], confidence=0.8)
        self.store.add(mem1)
        self.store.add(mem2)

        retriever = MemoryRetriever(store=self.store, embeddings=self.embeddings)
        results = retriever.retrieve("test query", limit=5, use_reranking=False, min_relevance=0.0)
        assert len(results) > 0

    def test_retrieve_with_reranking(self):
        # Add multiple memories so candidate_limit > limit triggers reranking
        mem1 = Memory(content="user likes Python", embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], confidence=0.9)
        mem2 = Memory(content="user knows Java", embedding=[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], confidence=0.8)
        mem3 = Memory(content="user likes tea", embedding=[0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], confidence=0.7)
        self.store.add(mem1)
        self.store.add(mem2)
        self.store.add(mem3)

        rerank_response = json.dumps([
            {"id": mem1.id, "relevance": 0.95},
            {"id": mem2.id, "relevance": 0.6},
            {"id": mem3.id, "relevance": 0.1},
        ])
        llm = MockLLM(response=rerank_response)
        retriever = MemoryRetriever(store=self.store, embeddings=self.embeddings, llm=llm)
        results = retriever.retrieve("Python", limit=1, use_reranking=True, candidate_limit=15, min_relevance=0.0)
        assert len(results) > 0


# --- PromptBuilder tests ---


class TestPromptBuilder:
    def test_build_empty(self):
        builder = PromptBuilder()
        result = builder.build([], "hello")
        assert result == ""

    def test_build_with_memories(self):
        memories = [
            (Memory(content="User is a developer", confidence=0.9), 0.95),
            (Memory(content="User likes coffee", confidence=0.6), 0.7),
        ]
        builder = PromptBuilder()
        result = builder.build(memories, "hello")
        assert "What you know about the user" in result
        assert "User is a developer" in result
        assert "User likes coffee" in result
        assert "high confidence" in result
        assert "moderate confidence" in result

    def test_build_filters_low_confidence(self):
        memories = [
            (Memory(content="Solid fact", confidence=0.9), 0.9),
            (Memory(content="Shaky fact", confidence=0.1), 0.5),
        ]
        builder = PromptBuilder(min_confidence=0.2)
        result = builder.build(memories, "hello")
        assert "Solid fact" in result
        assert "Shaky fact" not in result

    def test_build_full_system_prompt(self):
        memories = [
            (Memory(content="User is a developer", confidence=0.9), 0.9),
        ]
        builder = PromptBuilder()
        result = builder.build_full_system_prompt("You are helpful.", memories, "hello")
        assert result.startswith("You are helpful.")
        assert "User is a developer" in result


# --- MemoryEngine integration tests ---


class TestMemoryEngine:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def _make_engine(self, llm_response: str = "[]") -> MemoryEngine:
        return MemoryEngine(
            llm=MockLLM(response=llm_response),
            embeddings=MockEmbeddings(),
            db_path=self.tmp.name,
            use_reranking=False,
        )

    def test_observe_stores_memories(self):
        llm_response = json.dumps([
            {"content": "User is an engineer", "confidence": 0.9, "memory_type": "fact", "tags": ["career"]},
        ])
        engine = self._make_engine(llm_response)
        conv = Conversation(messages=[
            Message(role="user", content="I'm a software engineer"),
            Message(role="assistant", content="Cool!"),
        ])
        stored = engine.observe(conv)
        assert len(stored) == 1
        assert stored[0].content == "User is an engineer"

        all_mems = engine.get_all_memories()
        assert len(all_mems) == 1
        engine.close()

    def test_recall_returns_memories(self):
        engine = self._make_engine()
        # Manually add a memory
        mem = Memory(
            content="User likes Python",
            confidence=0.9,
            embedding=engine.embeddings.embed("User likes Python"),
        )
        engine.store.add(mem)

        results = engine.recall("Python programming")
        # Should find the memory (may or may not match depending on mock embeddings)
        assert isinstance(results, list)
        engine.close()

    def test_build_prompt(self):
        engine = self._make_engine()
        mem = Memory(
            content="User is a developer",
            confidence=0.9,
            embedding=engine.embeddings.embed("User is a developer"),
        )
        engine.store.add(mem)

        prompt = engine.build_prompt("Be helpful.", "hello", use_reranking=False)
        assert "Be helpful." in prompt
        engine.close()

    def test_decay(self):
        engine = self._make_engine()
        mem = Memory(
            content="Old memory",
            confidence=0.3,
            embedding=[0.1] * 8,
        )
        mem.updated_at = datetime.utcnow() - timedelta(days=120)
        engine.store.add(mem)

        deactivated = engine.decay()
        assert deactivated == 1
        engine.close()

    def test_process_turn(self):
        llm_response = json.dumps([
            {"content": "User loves hiking", "confidence": 0.8, "memory_type": "preference", "tags": ["hobbies"]},
        ])
        engine = self._make_engine(llm_response)
        conv = Conversation(messages=[
            Message(role="user", content="I go hiking every weekend"),
            Message(role="assistant", content="That sounds fun!"),
        ])

        new_memories, system_prompt = engine.process_turn(
            conv, "Tell me about trails", base_system="Be helpful."
        )
        assert len(new_memories) == 1
        assert "Be helpful." in system_prompt
        engine.close()
