# Memory Engine

A Python library that gives AI systems persistent memory.

Observe conversations → extract stable facts → store with confidence → retrieve intelligently → inject into prompts.

## Why

LLMs don't remember anything between conversations. Current memory solutions suffer from retrieval chaos, memory pollution, context overload, and temporal staleness. This project experiments with what AI *should* remember and how.

## Architecture

```
Conversation → Observer → Store → Retriever → Builder → LLM
```

- **Observer** — LLM-powered extraction of stable facts from conversations
- **Store** — Local persistence with confidence scoring and time decay
- **Retriever** — Hybrid search: embedding similarity + LLM re-ranking
- **Builder** — Token-aware prompt assembly with relevant memories

## Stack

- Python 3.11+
- Provider-agnostic LLM layer (OpenAI, Anthropic, etc.)
- Local storage (SQLite)

## Status

Early experimental stage.
