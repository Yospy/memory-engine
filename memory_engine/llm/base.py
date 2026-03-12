from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.

        Returns:
            The LLM's response text.
        """
        ...
