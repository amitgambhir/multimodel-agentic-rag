"""Provider-neutral streaming interface used by the agent loop.

Each provider yields ProviderEvents:
  - {"type": "text", "delta": "..."}            partial answer text
  - {"type": "tool_call", "name": ..., "input": {...}, "id": ...}
  - {"type": "usage", "input_tokens": int, "output_tokens": int,
                       "cache_read_tokens": int, "cache_creation_tokens": int}
  - {"type": "stop", "reason": str}

The agent runs the call again after appending tool_result messages until the
model produces a stop without further tool calls or hops are exhausted."""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, TypedDict


class ProviderEvent(TypedDict, total=False):
    type: str
    delta: str
    name: str
    input: dict[str, Any]
    id: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    reason: str


class LLMProvider(Protocol):
    name: str

    async def stream(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> AsyncIterator[ProviderEvent]: ...
