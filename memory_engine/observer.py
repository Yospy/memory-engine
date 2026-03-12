from __future__ import annotations

import json
import re
from typing import Optional

from memory_engine.llm.base import LLMProvider
from memory_engine.models import Conversation, Memory, MemoryType


EXTRACTION_SYSTEM = """You are a memory extraction system. Your job is to extract stable,
long-term facts about the user from a conversation.

Rules:
1. Extract facts that will likely still be true in 1 year
2. Ignore greetings, filler, temporary states ("I'm tired today")
3. Focus on: preferences, background, expertise, relationships, goals, habits
4. Rate each fact's confidence from 0.0 to 1.0
5. Classify each as: fact, preference, context, or episodic

Respond ONLY with a JSON array. Each element must have:
- "content": the memory text (write in third person about "the user")
- "confidence": float 0.0-1.0
- "memory_type": one of "fact", "preference", "context", "episodic"
- "tags": list of short topic tags

If there is nothing worth remembering, respond with an empty array: []

Example output:
[
  {"content": "The user is a software engineer working at a startup", "confidence": 0.9, "memory_type": "fact", "tags": ["career", "engineering"]},
  {"content": "The user prefers Python over JavaScript", "confidence": 0.7, "memory_type": "preference", "tags": ["programming", "languages"]}
]"""


class MemoryObserver:
    """Extracts memories from conversations using an LLM."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def observe(self, conversation: Conversation) -> list[Memory]:
        """Extract memories from a conversation.

        Args:
            conversation: The conversation to analyze.

        Returns:
            A list of Memory objects extracted from the conversation.
        """
        if not conversation.messages:
            return []

        prompt = self._format_conversation(conversation)
        response = self.llm.complete(prompt, system=EXTRACTION_SYSTEM)
        return self._parse_response(response, source=conversation.id)

    def _format_conversation(self, conversation: Conversation) -> str:
        lines = []
        for msg in conversation.messages:
            lines.append(f"{msg.role}: {msg.content}")
        return "\n".join(lines)

    def _parse_response(self, response: str, source: Optional[str] = None) -> list[Memory]:
        """Parse LLM response into Memory objects."""
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            return []

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        memories = []
        for item in items:
            if not isinstance(item, dict) or "content" not in item:
                continue

            try:
                memory_type = MemoryType(item.get("memory_type", "fact"))
            except ValueError:
                memory_type = MemoryType.FACT

            memories.append(
                Memory(
                    content=item["content"],
                    confidence=float(item.get("confidence", 0.5)),
                    memory_type=memory_type,
                    tags=item.get("tags", []),
                    source=source,
                )
            )

        return memories
