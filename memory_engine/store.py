from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from memory_engine.models import Memory, MemoryType


class MemoryStore:
    """SQLite-backed memory storage with vector similarity search."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS memories (
        id          TEXT PRIMARY KEY,
        content     TEXT NOT NULL,
        memory_type TEXT DEFAULT 'fact',
        confidence  REAL DEFAULT 0.5,
        tags        TEXT,
        embedding   BLOB,
        source      TEXT,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_used   TIMESTAMP,
        access_count INTEGER DEFAULT 0,
        is_active   BOOLEAN DEFAULT 1
    );
    """

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(self.SCHEMA)
        self.conn.commit()

    def add(self, memory: Memory) -> None:
        """Store a memory."""
        self.conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, content, memory_type, confidence, tags, embedding, source,
                created_at, updated_at, last_used, access_count, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.id,
                memory.content,
                memory.memory_type.value,
                memory.confidence,
                json.dumps(memory.tags),
                self._serialize_embedding(memory.embedding),
                memory.source,
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                memory.last_used.isoformat() if memory.last_used else None,
                memory.access_count,
                memory.is_active,
            ),
        )
        self.conn.commit()

    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def get_all(self, active_only: bool = True) -> list[Memory]:
        """Retrieve all memories."""
        query = "SELECT * FROM memories"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY confidence DESC"
        rows = self.conn.execute(query).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search_by_embedding(
        self, query_embedding: list[float], limit: int = 10, min_confidence: float = 0.1
    ) -> list[tuple[Memory, float]]:
        """Find memories by cosine similarity to query embedding.

        Returns list of (memory, similarity_score) tuples, sorted by similarity descending.
        """
        memories = self.get_all(active_only=True)
        scored: list[tuple[Memory, float]] = []

        query_vec = np.array(query_embedding)

        for mem in memories:
            if mem.embedding is None:
                continue
            if mem.confidence < min_confidence:
                continue
            mem_vec = np.array(mem.embedding)
            similarity = self._cosine_similarity(query_vec, mem_vec)
            scored.append((mem, float(similarity)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def update_confidence(self, memory_id: str, delta: float) -> None:
        """Adjust a memory's confidence by delta. Clamps to [0, 1]."""
        mem = self.get(memory_id)
        if mem is None:
            return
        new_conf = max(0.0, min(1.0, mem.confidence + delta))
        self.conn.execute(
            "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
            (new_conf, datetime.utcnow().isoformat(), memory_id),
        )
        self.conn.commit()

    def record_access(self, memory_id: str) -> None:
        """Record that a memory was accessed (used in retrieval)."""
        self.conn.execute(
            """UPDATE memories
               SET last_used = ?, access_count = access_count + 1, updated_at = ?
               WHERE id = ?""",
            (datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), memory_id),
        )
        self.conn.commit()

    def decay_memories(self, half_life_days: float = 30.0) -> int:
        """Apply time-based confidence decay to all memories.

        Uses exponential decay: new_conf = conf * 2^(-days_since_update / half_life)
        Deactivates memories that fall below 0.05 confidence.

        Returns the number of memories deactivated.
        """
        now = datetime.utcnow()
        memories = self.get_all(active_only=True)
        deactivated = 0

        for mem in memories:
            days_elapsed = (now - mem.updated_at).total_seconds() / 86400
            if days_elapsed < 1:
                continue

            decay_factor = math.pow(2, -days_elapsed / half_life_days)
            new_conf = mem.confidence * decay_factor

            if new_conf < 0.05:
                self.conn.execute(
                    "UPDATE memories SET is_active = 0, confidence = ?, updated_at = ? WHERE id = ?",
                    (new_conf, now.isoformat(), mem.id),
                )
                deactivated += 1
            else:
                self.conn.execute(
                    "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
                    (new_conf, now.isoformat(), mem.id),
                )

        self.conn.commit()
        return deactivated

    def find_similar(
        self, embedding: list[float], threshold: float = 0.85
    ) -> Optional[Memory]:
        """Find a memory with embedding similarity above threshold.

        Used for deduplication / merging.
        """
        results = self.search_by_embedding(embedding, limit=1)
        if results and results[0][1] >= threshold:
            return results[0][0]
        return None

    def delete(self, memory_id: str) -> None:
        """Hard delete a memory."""
        self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    # --- Private helpers ---

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _serialize_embedding(embedding: Optional[list[float]]) -> Optional[bytes]:
        if embedding is None:
            return None
        return np.array(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _deserialize_embedding(data: Optional[bytes]) -> Optional[list[float]]:
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.float32).tolist()

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            confidence=row["confidence"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            embedding=self._deserialize_embedding(row["embedding"]),
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_used=datetime.fromisoformat(row["last_used"]) if row["last_used"] else None,
            access_count=row["access_count"],
            is_active=bool(row["is_active"]),
        )
