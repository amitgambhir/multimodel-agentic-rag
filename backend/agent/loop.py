"""Agentic RAG loop.

Up to N hops of (model -> optional retrieval tool call -> model). The first
retrieval is precomputed before the agent runs (so the citation panel always
matches a real retrieval). On hop 2+, the model may call `refine_search` with
an alternative query — we run that against the same store and feed the results
back, then ask for the final answer.

Yields a stream of frontend-shaped events:
  {"type": "trace", "step": "...", "detail": "..."}
  {"type": "answer_delta", "delta": "..."}
  {"type": "tool", "name": "...", "input": {...}, "result_summary": "..."}
  {"type": "usage", ...}
  {"type": "done", "answer": "...", "matches": [...], "trace": [...], "usage": {...}}
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from starlette.concurrency import run_in_threadpool

from app_state import RAG_STORE
from config import MAX_AGENT_HOPS
from llm import LLMProvider


SYSTEM_PROMPT = """You are an evidence-grounded research assistant.

You receive a question along with a numbered list of retrieved evidence chunks.
Each chunk has an integer id like [1], [2], etc.

Rules for your final answer:
- Use ONLY the supplied evidence. If the evidence is insufficient, say so plainly.
- Do NOT include inline citation markers like [1] in the answer — the UI shows
  citations in a separate panel.
- Do NOT use Markdown bold, italics, or asterisk bullets. Use short paragraphs.
- Be concise (3-6 sentences typical). Lead with the direct answer.

You may call `refine_search` AT MOST ONCE if the initial evidence is clearly
off-topic or missing a key sub-question. Pass a tightly scoped alternative query
that targets the gap. After tool results return, write the final answer."""


def _format_evidence(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "No evidence retrieved."
    lines = []
    for i, m in enumerate(matches, start=1):
        lines.append(
            f"[{i}] ({m['modality']}) {m['source']} — score {m['score']}\n{m['snippet']}"
        )
    return "\n\n".join(lines)


def _refine_tool_def() -> dict[str, Any]:
    return {
        "name": "refine_search",
        "description": (
            "Run one additional retrieval against the source index using a more "
            "specific query. Use this only if the initial evidence is missing "
            "key information needed to answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Refined query, more specific than the user's original question.",
                },
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
        },
    }


async def run_agent(
    provider: LLMProvider,
    question: str,
    top_k: int,
    max_hops: int | None = None,
) -> AsyncIterator[dict[str, Any]]:
    max_hops = max(1, min(max_hops or MAX_AGENT_HOPS, 3))

    # Hop 1: precomputed retrieval, exposed as the system context.
    initial = await run_in_threadpool(RAG_STORE.search, question, top_k)
    matches: list[dict[str, Any]] = list(initial["matches"])
    yield {
        "type": "trace",
        "step": "retrieve",
        "detail": f"Initial retrieval: {len(matches)} chunks (top_k={top_k}).",
    }

    user_payload = (
        f"Question: {question}\n\n"
        f"Evidence:\n{_format_evidence(matches)}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": user_payload}]}
    ]

    tools = [_refine_tool_def()] if max_hops > 1 else []

    final_answer_parts: list[str] = []
    usage_total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    trace: list[dict[str, Any]] = [
        {"step": "retrieve", "detail": f"Initial retrieval: {len(matches)} chunks."}
    ]

    for hop in range(1, max_hops + 1):
        yield {"type": "trace", "step": f"think:{hop}", "detail": f"Calling {provider.name} (hop {hop})."}

        pending_tool_calls: list[dict[str, Any]] = []
        assistant_blocks: list[dict[str, Any]] = []
        text_buffer = ""

        async for event in provider.stream(
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools if hop < max_hops else [],
        ):
            etype = event["type"]
            if etype == "text":
                text_buffer += event["delta"]
                yield {"type": "answer_delta", "delta": event["delta"]}
            elif etype == "tool_call":
                pending_tool_calls.append(event)
            elif etype == "usage":
                for k in usage_total:
                    usage_total[k] += event.get(k, 0) or 0
            elif etype == "stop":
                pass

        if text_buffer:
            assistant_blocks.append({"type": "text", "text": text_buffer})
        for tc in pending_tool_calls:
            assistant_blocks.append(
                {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
            )

        if not pending_tool_calls:
            # Terminal hop — this is the real answer.
            if text_buffer:
                final_answer_parts.append(text_buffer)
            trace.append({"step": "answer", "detail": "Model produced final answer."})
            break

        # Tool-using hop. Any text the model emitted before the tool call was
        # reasoning, not the answer — tell the UI to discard the streamed
        # tokens so they don't bleed into the final answer text.
        if text_buffer:
            yield {"type": "reset_answer"}

        # Append assistant turn so the next call is well-formed.
        messages.append({"role": "assistant", "content": assistant_blocks})

        tool_result_blocks: list[dict[str, Any]] = []
        for tc in pending_tool_calls:
            if tc["name"] == "refine_search":
                refined_query = (tc["input"] or {}).get("query", question)
                refined_k = int((tc["input"] or {}).get("top_k", top_k))
                refined = await run_in_threadpool(RAG_STORE.search, refined_query, refined_k)
                new_matches = refined["matches"]
                # Merge — preserve order, dedupe by source_id+chunk_index.
                seen = {(m["source_id"], m["chunk_index"]) for m in matches}
                for m in new_matches:
                    key = (m["source_id"], m["chunk_index"])
                    if key not in seen:
                        matches.append(m)
                        seen.add(key)
                summary = f"Refined query='{refined_query}' returned {len(new_matches)} chunks."
                trace.append({"step": "refine_search", "detail": summary})
                yield {
                    "type": "tool",
                    "name": "refine_search",
                    "input": {"query": refined_query, "top_k": refined_k},
                    "result_summary": summary,
                }
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {"matches": new_matches}, ensure_ascii=False
                                ),
                            }
                        ],
                    }
                )
            else:
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": [{"type": "text", "text": "Unknown tool"}],
                        "is_error": True,
                    }
                )

        messages.append({"role": "user", "content": tool_result_blocks})

    # Snapshot the final space with the query projected into the same basis.
    space = await run_in_threadpool(RAG_STORE.snapshot, initial["query_vector"])

    yield {
        "type": "done",
        "answer": "".join(final_answer_parts).strip()
        or "I could not produce an answer from the current sources.",
        "matches": matches,
        "trace": trace,
        "usage": usage_total,
        "space": space,
    }
