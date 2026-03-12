from __future__ import annotations

import anthropic

from memory_engine.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, system: str = "") -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system if system else anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
