"""FastAPI server: ingest endpoints + SSE-streamed /ask."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl
from starlette.concurrency import run_in_threadpool

from agent import run_agent
from app_state import RAG_STORE
from config import ALLOWED_ORIGINS, MAX_AGENT_HOPS, PORT, available_providers
from ingest import extract_pdf_text, fetch_url_text, image_preview_data_url
from llm import get_provider

app = FastAPI(title="Multi-LLM Multimodal Agentic RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextSourceRequest(BaseModel):
    title: str
    text: str
    modality: Literal["text"] = "text"


class UrlSourceRequest(BaseModel):
    url: HttpUrl
    title: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    provider: Literal["claude", "gemini"] = "claude"
    top_k: int = Field(6, ge=1, le=12)
    max_hops: int = Field(MAX_AGENT_HOPS, ge=1, le=3)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "providers": available_providers(),
        "max_hops_default": MAX_AGENT_HOPS,
        **await run_in_threadpool(RAG_STORE.space_tool),
    }


@app.get("/space")
async def space():
    return await run_in_threadpool(RAG_STORE.snapshot)


@app.post("/sources/text")
async def add_text_source(req: TextSourceRequest):
    try:
        source = await run_in_threadpool(
            RAG_STORE.add_text_source, req.title, req.text, req.modality
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"source": source.__dict__, "space": await run_in_threadpool(RAG_STORE.snapshot)}


@app.post("/sources/url")
async def add_url_source(req: UrlSourceRequest):
    try:
        text, page_title = await fetch_url_text(str(req.url))
        title = req.title or page_title or str(req.url)[:80]
        source = await run_in_threadpool(RAG_STORE.add_text_source, title, text, "url")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, f"Could not ingest URL: {exc}") from exc
    return {"source": source.__dict__, "space": await run_in_threadpool(RAG_STORE.snapshot)}


@app.post("/sources/file")
async def add_file_source(
    file: UploadFile = File(...),
    title: str = Form(""),
    notes: str = Form(""),
):
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large; keep under 50 MB.")
    mime = (file.content_type or "").lower()
    name = title or (file.filename or "Uploaded")

    try:
        if mime.startswith("image/"):
            preview = await run_in_threadpool(image_preview_data_url, data)
            source = await run_in_threadpool(
                RAG_STORE.add_image_source, name, data, notes, preview
            )
        elif mime == "application/pdf" or (file.filename or "").lower().endswith(".pdf"):
            text = await run_in_threadpool(extract_pdf_text, data)
            if not text:
                raise HTTPException(400, "Could not extract text from this PDF.")
            source = await run_in_threadpool(
                RAG_STORE.add_text_source, name, text, "pdf", notes
            )
        else:
            raise HTTPException(
                400,
                "Unsupported file type. MVP supports images and PDFs.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    return {"source": source.__dict__, "space": await run_in_threadpool(RAG_STORE.snapshot)}


@app.delete("/sources/{source_id}")
async def delete_source(source_id: str):
    removed = await run_in_threadpool(RAG_STORE.remove_source, source_id)
    if not removed:
        raise HTTPException(404, "Source not found.")
    return {"deleted": source_id, "space": await run_in_threadpool(RAG_STORE.snapshot)}


@app.post("/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is required.")
    try:
        provider = get_provider(req.provider)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc

    async def event_source():
        try:
            async for event in run_agent(
                provider=provider,
                question=req.question,
                top_k=req.top_k,
                max_hops=req.max_hops,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            err = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
