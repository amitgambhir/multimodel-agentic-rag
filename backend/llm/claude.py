"""Claude provider — streaming + prompt caching.

We mark the system prompt as a cache breakpoint so repeated questions over the
same source set hit the cache. The retrieved-evidence block (passed inside the
user message) is also a frequent target — cache breakpoint there too."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str | None = None):
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model or CLAUDE_MODEL
        self.client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async def stream(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> AsyncIterator[dict[str, Any]]:
        system_blocks = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        async with self.client.messages.stream(**kwargs) as stream:
            current_tool: dict[str, Any] | None = None
            current_tool_json = ""
            async for event in stream:
                etype = getattr(event, "type", "")
                if etype == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool = {"id": block.id, "name": block.name}
                        current_tool_json = ""
                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield {"type": "text", "delta": delta.text}
                    elif delta.type == "input_json_delta":
                        current_tool_json += delta.partial_json
                elif etype == "content_block_stop":
                    if current_tool is not None:
                        import json
                        try:
                            tool_input = json.loads(current_tool_json) if current_tool_json else {}
                        except json.JSONDecodeError:
                            tool_input = {}
                        yield {
                            "type": "tool_call",
                            "id": current_tool["id"],
                            "name": current_tool["name"],
                            "input": tool_input,
                        }
                        current_tool = None
                        current_tool_json = ""

            final = await stream.get_final_message()
            usage = final.usage
            yield {
                "type": "usage",
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            }
            yield {"type": "stop", "reason": final.stop_reason or ""}
