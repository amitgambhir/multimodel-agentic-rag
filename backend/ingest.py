"""Helpers for URL/PDF/image ingestion."""

from __future__ import annotations

import base64
import ipaddress
import socket
from io import BytesIO
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from PIL import Image
from pypdf import PdfReader

from config import ALLOW_PRIVATE_URLS


def validate_fetch_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")
    if ALLOW_PRIVATE_URLS:
        return
    host = parsed.hostname.lower()
    if host == "localhost":
        raise ValueError("Private and localhost URLs are disabled.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {parsed.hostname}") from exc
    for item in infos:
        ip = ipaddress.ip_address(item[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("Private and localhost URLs are disabled.")


_MAX_REDIRECTS = 5


async def fetch_url_text(url: str) -> tuple[str, Optional[str]]:
    """Fetch with manual redirect handling so we re-validate every hop. httpx's
    follow_redirects only validates the initial URL, which would let a public
    URL 302 to localhost / link-local / cloud metadata IPs."""
    validate_fetch_url(url)
    async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
        current = url
        seen: set[str] = set()
        for _ in range(_MAX_REDIRECTS + 1):
            r = await client.get(current, headers={"User-Agent": "multimodel-rag/0.1"})
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("location")
                if not loc:
                    raise ValueError("Redirect with no Location header")
                nxt = urljoin(current, loc)
                if nxt in seen:
                    raise ValueError("Redirect loop")
                seen.add(nxt)
                validate_fetch_url(nxt)
                current = nxt
                continue
            r.raise_for_status()
            break
        else:
            raise ValueError("Too many redirects")
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else None)
    text = " ".join(soup.get_text(" ").split())
    return text[:30000], title


def extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages).strip()


def image_preview_data_url(data: bytes, max_side: int = 320) -> str:
    img = Image.open(BytesIO(data)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
