import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_FAST_MODEL = os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-4-5-20251001")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

MAX_AGENT_HOPS = max(1, min(int(os.getenv("MAX_AGENT_HOPS", "2")), 3))

PORT = int(os.getenv("PORT", "8897"))
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if o.strip()
]
ALLOW_PRIVATE_URLS = os.getenv("ALLOW_PRIVATE_URLS", "").lower() == "true"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_chroma_raw = os.getenv("CHROMA_DIR", "./data/chroma")
_chroma_path = Path(_chroma_raw)
if not _chroma_path.is_absolute():
    _chroma_path = _PROJECT_ROOT / _chroma_path
CHROMA_DIR = _chroma_path.resolve()
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EMBED_DIM_FALLBACK = 512


def available_providers() -> list[str]:
    out = []
    if ANTHROPIC_API_KEY:
        out.append("claude")
    if GOOGLE_API_KEY:
        out.append("gemini")
    return out
