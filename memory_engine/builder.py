from __future__ import annotations

from memory_engine.models import Memory


class PromptBuilder:
    """Formats retrieved memories for injection into LLM prompts."""

    def __init__(self, max_memories: int = 10, min_confidence: float = 0.2):
        self.max_memories = max_memories
        self.min_confidence = min_confidence

    def build(self, memories: list[tuple[Memory, float]], user_message: str) -> str:
        """Build a system prompt section with relevant memories.

        Args:
            memories: List of (memory, relevance_score) tuples.
            user_message: The user's current message.

        Returns:
            Formatted system prompt section.
        """
        if not memories:
            return ""

        # Filter by confidence and limit
        filtered = [
            (mem, score)
            for mem, score in memories
            if mem.confidence >= self.min_confidence
        ]

        # Sort by relevance * confidence (combined score)
        filtered.sort(key=lambda x: x[1] * x[0].confidence, reverse=True)
        filtered = filtered[: self.max_memories]

        if not filtered:
            return ""

        lines = ["## What you know about the user:"]
        for mem, score in filtered:
            conf_label = self._confidence_label(mem.confidence)
            lines.append(f"- {mem.content} ({conf_label})")

        return "\n".join(lines)

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        if confidence >= 0.8:
            return "high confidence"
        if confidence >= 0.5:
            return "moderate confidence"
        return "low confidence"


    def build_full_system_prompt(
        self,
        base_system: str,
        memories: list[tuple[Memory, float]],
        user_message: str,
    ) -> str:
        """Build a complete system prompt with base instructions + memories.

        Args:
            base_system: The base system prompt.
            memories: Retrieved memories with relevance scores.
            user_message: The user's current message.

        Returns:
            Combined system prompt.
        """
        memory_section = self.build(memories, user_message)
        if not memory_section:
            return base_system
        return f"{base_system}\n\n{memory_section}"
