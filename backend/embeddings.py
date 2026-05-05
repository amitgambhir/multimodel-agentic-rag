"""Embedding adapter.

Text/URL/PDF: OpenAI `text-embedding-3-*` when OPENAI_API_KEY is set.
Images: caption with Claude vision (or Gemini), then embed the caption.
Without OPENAI_API_KEY: a deterministic local hash so the UI still works."""

from __future__ import annotations

import base64
import hashlib
import logging
import math
from io import BytesIO
from typing import Optional

from PIL import Image

from config import (
    EMBED_DIM_FALLBACK, OPENAI_API_KEY, OPENAI_EMBED_MODEL,
    ANTHROPIC_API_KEY, GOOGLE_API_KEY, CLAUDE_FAST_MODEL, GEMINI_MODEL,
)

log = logging.getLogger("embeddings")

_openai_client = None
_openai_dim: Optional[int] = None
_openai_warned = False
_locked_dim: Optional[int] = None  # established once and never changes


def lock_dim(dim: int) -> None:
    """Pin the embedding dim for this process. Call after restoring an
    existing Chroma collection so any later fallbacks produce same-dim vectors
    instead of corrupting the index."""
    global _locked_dim
    if dim and dim > 0:
        _locked_dim = dim


def active_dim() -> int:
    return _locked_dim or _openai_dim or EMBED_DIM_FALLBACK


def _client():
    global _openai_client
    if _openai_client is None and OPENAI_API_KEY:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def has_openai() -> bool:
    return bool(OPENAI_API_KEY) and not _openai_warned


def dimensions() -> int:
    """text-embedding-3-small=1536, -large=3072. Probed once and cached.

    If a dim was locked from an existing collection, that wins so we can't
    accidentally write a different-dim vector into the index."""
    global _openai_dim
    if _locked_dim:
        return _locked_dim
    if not has_openai():
        return EMBED_DIM_FALLBACK
    if _openai_dim:
        return _openai_dim
    try:
        vec = _embed_openai(["probe"])[0]
        _openai_dim = len(vec)
        return _openai_dim
    except Exception as exc:
        _warn_openai(exc)
        return EMBED_DIM_FALLBACK


def _warn_openai(exc: Exception) -> None:
    global _openai_warned
    if _openai_warned:
        return
    _openai_warned = True
    log.warning(
        "OpenAI embeddings failed (%s: %s). Falling back to local-hash.",
        type(exc).__name__, exc,
    )


def _local_embedding(text: str, dim: Optional[int] = None) -> list[float]:
    target = dim or active_dim()
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * ((target // len(digest)) + 1))[:target]
    vec = [(b - 128) / 128.0 for b in raw]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _embed_openai(texts: list[str]) -> list[list[float]]:
    client = _client()
    if client is None:
        raise RuntimeError("OpenAI client not initialized")
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    vecs = [list(d.embedding) for d in resp.data]
    if _locked_dim and vecs and len(vecs[0]) != _locked_dim:
        raise RuntimeError(
            f"OpenAI returned dim={len(vecs[0])} but collection is locked to "
            f"{_locked_dim}. Did OPENAI_EMBED_MODEL change? Wipe data/chroma/ "
            f"or revert the model setting."
        )
    return vecs


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if not has_openai():
        return [_local_embedding(t) for t in texts]
    try:
        vecs = _embed_openai(texts)
        if vecs and not _locked_dim:
            lock_dim(len(vecs[0]))
        return vecs
    except Exception as exc:
        _warn_openai(exc)
        return [_local_embedding(t) for t in texts]


def embed_query(text: str) -> list[float]:
    if not has_openai():
        return _local_embedding(text)
    try:
        vec = _embed_openai([text])[0]
        if not _locked_dim:
            lock_dim(len(vec))
        return vec
    except Exception as exc:
        _warn_openai(exc)
        return _local_embedding(text)


# ---------------- image captioning ----------------

_IMAGE_CAPTION_PROMPT = (
    "Describe this image in 1-2 short sentences for retrieval. "
    "Focus on subject, setting, key text, and notable details. "
    "Do not preface; output only the description."
)


def _caption_with_claude(image_bytes: bytes, mime: str = "image/jpeg") -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        b64 = base64.b64encode(image_bytes).decode()
        msg = client.messages.create(
            model=CLAUDE_FAST_MODEL,
            max_tokens=120,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": mime, "data": b64,
                        }},
                        {"type": "text", "text": _IMAGE_CAPTION_PROMPT},
                    ],
                }
            ],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return text.strip() or None
    except Exception as exc:
        log.warning("Claude image caption failed: %s", exc)
        return None


def _caption_with_gemini(image_bytes: bytes, mime: str = "image/jpeg") -> Optional[str]:
    if not GOOGLE_API_KEY:
        return None
    try:
        from google import genai
        from google.genai import types as gt
        client = genai.Client(api_key=GOOGLE_API_KEY)
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                gt.Part.from_bytes(data=image_bytes, mime_type=mime),
                gt.Part(text=_IMAGE_CAPTION_PROMPT),
            ],
            config=gt.GenerateContentConfig(max_output_tokens=200),
        )
        text = ""
        for cand in resp.candidates or []:
            for part in (cand.content.parts or []):
                if getattr(part, "text", None):
                    text += part.text
        return text.strip() or None
    except Exception as exc:
        log.warning("Gemini image caption failed: %s", exc)
        return None


def caption_image(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Return a short caption for retrieval. Falls back across providers."""
    return (
        _caption_with_claude(image_bytes, mime)
        or _caption_with_gemini(image_bytes, mime)
        or "image"
    )


def _normalize_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Return JPEG bytes and mime so vision APIs accept the file."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def embed_image(image_bytes: bytes, annotation: str = "") -> tuple[list[float], str]:
    """Caption then embed. Returns (vector, caption_used_for_retrieval)."""
    norm_bytes, mime = _normalize_image(image_bytes)
    caption = caption_image(norm_bytes, mime)
    text_for_embed = " ".join(filter(None, [annotation, caption])).strip() or "image"
    return embed_documents([text_for_embed])[0], caption
