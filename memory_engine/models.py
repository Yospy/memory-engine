from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    CONTEXT = "context"
    EPISODIC = "episodic"


@dataclass
class Memory:
    content: str
    memory_type: MemoryType = MemoryType.FACT
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    source: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    access_count: int = 0
    is_active: bool = True


@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
