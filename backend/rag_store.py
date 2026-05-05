"""Persistent multimodal RAG store backed by Chroma + OpenAI embeddings.

Sources hold metadata; chunks are the embedded units. We compute a 3D PCA
projection over the current source vectors (mean of their chunk vectors) so
the frontend can render them in a shared space with the query point."""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR
import embeddings as E

Modality = Literal["text", "url", "pdf", "image"]


@dataclass
class Source:
    id: str
    title: str
    modality: Modality
    chunk_count: int
    created_at: float = field(default_factory=time.time)
    notes: str = ""
    preview: str = ""


def _chunk_text(text: str, target: int = 700, overlap: int = 80) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= target:
        return [text]
    step = target - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start : start + target])
        if start + target >= len(text):
            break
    return chunks


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    n = len(vectors[0])
    out = [0.0] * n
    for v in vectors:
        for i, x in enumerate(v):
            out[i] += x
    return [x / len(vectors) for x in out]


def _power_iteration(matrix: list[list[float]], n_components: int = 3, iters: int = 60) -> list[list[float]]:
    if not matrix:
        return []
    dim = len(matrix[0])
    n_components = min(n_components, dim)
    centered = [list(row) for row in matrix]
    means = [sum(row[j] for row in centered) / len(centered) for j in range(dim)]
    for row in centered:
        for j in range(dim):
            row[j] -= means[j]

    components: list[list[float]] = []
    for _ in range(n_components):
        b = [1.0 / math.sqrt(dim)] * dim
        for _ in range(iters):
            new_b = [0.0] * dim
            for row in centered:
                dot = sum(row[j] * b[j] for j in range(dim))
                for j in range(dim):
                    new_b[j] += dot * row[j]
            for c in components:
                proj = sum(new_b[j] * c[j] for j in range(dim))
                for j in range(dim):
                    new_b[j] -= proj * c[j]
            norm = math.sqrt(sum(x * x for x in new_b)) or 1.0
            b = [x / norm for x in new_b]
        components.append(b)
    return components


def _project(vec: list[float], components: list[list[float]], means: list[float]) -> list[float]:
    centered = [vec[j] - means[j] for j in range(len(vec))]
    return [sum(centered[j] * c[j] for j in range(len(vec))) for c in components]


class MultimodalRagStore:
    """Chroma-backed source/chunk store with a 3D PCA projection helper."""

    def __init__(self):
        self._lock = threading.RLock()
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
        )
        self._chunks_coll = self._client.get_or_create_collection(
            name="chunks", metadata={"hnsw:space": "cosine"}
        )
        self.sources: dict[str, Source] = {}
        self.events: list[dict[str, Any]] = []
        self._restore_sources()
        self._lock_existing_dim()

    def _lock_existing_dim(self) -> None:
        """If Chroma already holds vectors, pin embeddings to that dim so a
        transient OpenAI failure can't poison the collection with mismatched
        vectors."""
        try:
            data = self._chunks_coll.get(limit=1, include=["embeddings"])
        except Exception:
            return
        embs = data.get("embeddings")
        if embs is None or len(embs) == 0:
            return
        first = embs[0]
        if first is None:
            return
        E.lock_dim(len(first))

    # ---------- restore ----------

    def _restore_sources(self) -> None:
        try:
            data = self._chunks_coll.get(include=["metadatas"])
        except Exception:
            return
        metas = data.get("metadatas")
        if metas is None:
            return
        seen: dict[str, dict[str, Any]] = {}
        for meta in metas:
            sid = meta.get("source_id")
            if not sid or sid in seen:
                continue
            seen[sid] = meta
        for sid, meta in seen.items():
            self.sources[sid] = Source(
                id=sid,
                title=meta.get("title", "Source"),
                modality=meta.get("modality", "text"),
                chunk_count=int(meta.get("source_chunk_count", 0) or 0),
                created_at=float(meta.get("created_at", time.time())),
                notes=meta.get("notes", ""),
                preview=meta.get("preview", ""),
            )

    # ---------- ingestion ----------

    def add_text_source(
        self,
        title: str,
        text: str,
        modality: Modality = "text",
        notes: str = "",
    ) -> Source:
        text = text.strip()
        if not text:
            raise ValueError("Empty text body")
        chunks = _chunk_text(text)
        vectors = E.embed_documents(chunks)
        source_id = uuid.uuid4().hex[:12]
        source = Source(
            id=source_id,
            title=title.strip() or "Untitled source",
            modality=modality,
            chunk_count=len(chunks),
            notes=notes,
            preview=chunks[0][:240] if chunks else "",
        )
        self._persist_chunks(source, chunks, vectors)
        self._record_event("add", source)
        return source

    def add_image_source(
        self,
        title: str,
        image_bytes: bytes,
        notes: str = "",
        preview_data_url: str = "",
    ) -> Source:
        annotation = " ".join(filter(None, [title, notes])).strip()
        vec, caption = E.embed_image(image_bytes, annotation=annotation)
        source_id = uuid.uuid4().hex[:12]
        source = Source(
            id=source_id,
            title=title.strip() or "Image",
            modality="image",
            chunk_count=1,
            notes=notes or caption,
            preview=preview_data_url or caption,
        )
        chunk_text = (annotation + " " + caption).strip() or "image"
        self._persist_chunks(source, [chunk_text], [vec])
        self._record_event("add", source)
        return source

    def _persist_chunks(self, source: Source, chunks: list[str], vectors: list[list[float]]) -> None:
        ids = [f"{source.id}:{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source_id": source.id,
                "title": source.title,
                "modality": source.modality,
                "chunk_index": i,
                "source_chunk_count": source.chunk_count,
                "created_at": source.created_at,
                "notes": source.notes,
                "preview": source.preview,
            }
            for i, _ in enumerate(chunks)
        ]
        with self._lock:
            self._chunks_coll.add(
                ids=ids,
                documents=chunks,
                embeddings=vectors,
                metadatas=metadatas,
            )
            self.sources[source.id] = source

    def remove_source(self, source_id: str) -> bool:
        with self._lock:
            if source_id not in self.sources:
                return False
            self._chunks_coll.delete(where={"source_id": source_id})
            removed = self.sources.pop(source_id)
        self._record_event("remove", removed)
        return True

    # ---------- retrieval ----------

    def search(self, query: str, top_k: int = 6) -> dict[str, Any]:
        q_vec = E.embed_query(query)
        if not self.sources:
            return {"matches": [], "query_vector": q_vec, "query": query}
        with self._lock:
            res = self._chunks_coll.query(
                query_embeddings=[q_vec],
                n_results=min(top_k * 2, max(top_k, 12)),
                include=["documents", "metadatas", "distances"],
            )
        matches: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            sid = meta.get("source_id")
            if sid in seen_sources:
                continue
            seen_sources.add(sid)
            matches.append(
                {
                    "source_id": sid,
                    "source": meta.get("title", "Source"),
                    "modality": meta.get("modality", "text"),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "snippet": (doc or "")[:480],
                    "score": round(1.0 - float(dist), 4),
                }
            )
            if len(matches) >= top_k:
                break
        return {"matches": matches, "query_vector": q_vec, "query": query}

    # ---------- snapshot / projection ----------

    def _all_source_vectors(self) -> dict[str, list[float]]:
        with self._lock:
            data = self._chunks_coll.get(include=["embeddings", "metadatas"])
        embs = data.get("embeddings")
        metas = data.get("metadatas")
        if embs is None or metas is None:
            return {}
        groups: dict[str, list[list[float]]] = {}
        for vec, meta in zip(embs, metas):
            sid = meta.get("source_id")
            if not sid:
                continue
            groups.setdefault(sid, []).append([float(x) for x in vec])
        return {sid: _mean_vector(vs) for sid, vs in groups.items() if vs}

    def snapshot(self, query_vector: Optional[list[float]] = None) -> dict[str, Any]:
        source_vectors = self._all_source_vectors()
        ordered_ids = list(source_vectors.keys())
        matrix = [source_vectors[sid] for sid in ordered_ids]
        if query_vector is not None and matrix:
            matrix.append(list(query_vector))

        components: list[list[float]] = []
        means: list[float] = []
        if matrix:
            dim = len(matrix[0])
            means = [sum(row[j] for row in matrix) / len(matrix) for j in range(dim)]
            components = _power_iteration(matrix, n_components=3)

        points = []
        for sid in ordered_ids:
            src = self.sources.get(sid)
            if not src:
                continue
            proj = _project(source_vectors[sid], components, means) if components else [0.0, 0.0, 0.0]
            while len(proj) < 3:
                proj.append(0.0)
            points.append(
                {
                    "source_id": sid,
                    "title": src.title,
                    "modality": src.modality,
                    "preview": src.preview,
                    "x": proj[0],
                    "y": proj[1],
                    "z": proj[2],
                }
            )

        query_point = None
        if query_vector is not None and components:
            qp = _project(list(query_vector), components, means)
            while len(qp) < 3:
                qp.append(0.0)
            query_point = {"x": qp[0], "y": qp[1], "z": qp[2]}

        return {
            "sources": [asdict(s) for s in self.sources.values()],
            "points": points,
            "query_point": query_point,
            "events": list(self.events[-30:]),
            "dimensions": len(matrix[0]) if matrix else E.dimensions(),
            "embedding_provider": "openai" if E.has_openai() else "local-hash",
        }

    # ---------- events ----------

    def _record_event(self, kind: str, source: Source) -> None:
        self.events.append(
            {
                "kind": kind,
                "source_id": source.id,
                "title": source.title,
                "modality": source.modality,
                "at": time.time(),
            }
        )

    def space_tool(self) -> dict[str, Any]:
        return {
            "sources_total": len(self.sources),
            "embedding_provider": "openai" if E.has_openai() else "local-hash",
            "dimensions": E.dimensions(),
        }
