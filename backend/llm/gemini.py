"""Gemini provider — streaming text + function calling.

Uses google-genai. We translate the provider-neutral messages/tools shape into
Gemini's contents/tools format and yield the same ProviderEvents."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from google import genai
from google.genai import types as gt

from config import GEMINI_MODEL, GOOGLE_API_KEY


def _to_gemini_contents(messages: list[dict[str, Any]]) -> list[gt.Content]:
    # tool_result blocks carry only `tool_use_id`, not a name (Anthropic schema).
    # Walk forward and remember the name from each tool_use so we can resolve it.
    tool_id_to_name: dict[str, str] = {}
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for block in c:
                if block.get("type") == "tool_use" and block.get("id"):
                    tool_id_to_name[block["id"]] = block.get("name", "tool")

    contents: list[gt.Content] = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        parts: list[gt.Part] = []
        content = m["content"]
        if isinstance(content, str):
            parts.append(gt.Part(text=content))
        else:
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    parts.append(gt.Part(text=block["text"]))
                elif btype == "tool_use":
                    parts.append(
                        gt.Part(
                            function_call=gt.FunctionCall(
                                name=block["name"], args=block.get("input", {})
                            )
                        )
                    )
                elif btype == "tool_result":
                    payload = block.get("content")
                    if isinstance(payload, list):
                        for sub in payload:
                            if sub.get("type") == "text":
                                payload = sub.get("text", "")
                                break
                    if not isinstance(payload, str):
                        payload = json.dumps(payload)
                    try:
                        response = json.loads(payload)
                        if not isinstance(response, dict):
                            response = {"result": response}
                    except json.JSONDecodeError:
                        response = {"result": payload}
                    name = (
                        block.get("name")
                        or tool_id_to_name.get(block.get("tool_use_id", ""), "tool")
                    )
                    parts.append(
                        gt.Part(
                            function_response=gt.FunctionResponse(
                                name=name, response=response
                            )
                        )
                    )
        contents.append(gt.Content(role=role, parts=parts))
    return contents


_JSON_TO_GEMINI_TYPE = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _normalize_schema(schema: Any) -> Any:
    """Recursively upper-case JSON Schema `type` values to Gemini's enum.

    google-genai validates `type` against an uppercase Type enum
    (`STRING`, `OBJECT`, …). Tools defined for Anthropic use JSON Schema
    lowercase, so translate on the fly."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                out[k] = _JSON_TO_GEMINI_TYPE.get(v.lower(), v.upper())
            else:
                out[k] = _normalize_schema(v)
        return out
    if isinstance(schema, list):
        return [_normalize_schema(x) for x in schema]
    return schema


def _to_gemini_tools(tools: list[dict[str, Any]]) -> list[gt.Tool]:
    if not tools:
        return []
    decls = []
    for t in tools:
        params = _normalize_schema(
            t.get("input_schema") or {"type": "object", "properties": {}}
        )
        decls.append(
            gt.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=params,
            )
        )
    return [gt.Tool(function_declarations=decls)]


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str | None = None):
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        self.model = model or GEMINI_MODEL
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    async def stream(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> AsyncIterator[dict[str, Any]]:
        contents = _to_gemini_contents(messages)
        gtools = _to_gemini_tools(tools)
        cfg = gt.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=gtools or None,
        )

        stream = self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=cfg,
        )
        if hasattr(stream, "__await__"):
            stream = await stream

        in_tok = out_tok = 0
        stop_reason = ""
        async for chunk in stream:
            for cand in chunk.candidates or []:
                if not cand.content or not cand.content.parts:
                    continue
                for part in cand.content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        yield {"type": "text", "delta": text}
                    fc = getattr(part, "function_call", None)
                    if fc and fc.name:
                        yield {
                            "type": "tool_call",
                            "id": f"{fc.name}-{id(fc)}",
                            "name": fc.name,
                            "input": dict(fc.args or {}),
                        }
                if cand.finish_reason:
                    stop_reason = str(cand.finish_reason).lower()
            if chunk.usage_metadata:
                in_tok = chunk.usage_metadata.prompt_token_count or in_tok
                out_tok = chunk.usage_metadata.candidates_token_count or out_tok

        yield {
            "type": "usage",
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }
        yield {"type": "stop", "reason": stop_reason}
