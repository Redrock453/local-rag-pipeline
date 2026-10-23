"""Retrieval: local embeddings (Ollama) + vector search (Qdrant REST API).

Any failure (service down, collection missing, nothing above the threshold)
returns an empty context so the pipeline degrades to plain generation instead
of failing.
"""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger("rag_pipeline.retrieve")


def _cfg() -> dict:
    """Read settings at call time so tests and callers can override env vars."""
    return {
        "ollama_url": os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
        "qdrant_url": os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
        "collection": os.environ.get("RAG_COLLECTION", "knowledge"),
        "embed_model": os.environ.get("RAG_EMBED_MODEL", "bge-m3"),
        "top_k": int(os.environ.get("RAG_TOP_K", "3")),
        # bge-m3 cosine scores sit in a narrow band; tune this on your own data.
        "threshold": float(os.environ.get("RAG_SCORE_THRESHOLD", "0.62")),
        "snippet_chars": int(os.environ.get("RAG_SNIPPET_CHARS", "500")),
        # 0 unloads the embedder right after use - useful on small-RAM machines.
        "keep_alive": os.environ.get("RAG_EMBED_KEEP_ALIVE", "0"),
    }


def embed(text: str) -> list[float]:
    cfg = _cfg()
    try:
        resp = requests.post(
            f"{cfg['ollama_url']}/api/embed",
            json={"model": cfg["embed_model"], "input": text,
                  "keep_alive": cfg["keep_alive"]},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except Exception as exc:  # noqa: BLE001 - retrieval must not break the pipeline
        log.warning("embedding failed: %s", exc)
        return []


def search(vector: list[float], limit: int | None = None) -> list[dict]:
    if not vector:
        return []
    cfg = _cfg()
    limit = limit or cfg["top_k"]
    base = f"{cfg['qdrant_url']}/collections/{cfg['collection']}"
    try:
        resp = requests.post(
            f"{base}/points/search",
            json={"vector": vector, "limit": limit, "with_payload": True,
                  "score_threshold": cfg["threshold"]},
            timeout=5,
        )
        if resp.status_code == 404:  # collection not created yet
            return []
        resp.raise_for_status()
        return resp.json().get("result", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("vector search failed: %s", exc)
        return []


def build_context(results: list[dict], snippet_chars: int | None = None) -> str:
    """Format search hits as a prompt block ('' when there is nothing to add)."""
    snippet_chars = snippet_chars or _cfg()["snippet_chars"]
    parts = []
    for hit in results:
        payload = hit.get("payload", {})
        text = (payload.get("text") or "").strip()
        if text:
            source = payload.get("source", "unknown")
            parts.append(f"[source: {source}]\n{text[:snippet_chars]}")
    if not parts:
        return ""
    header = (
        "Relevant fragments from the knowledge base follow. Use them when they "
        "relate to the question and ignore them otherwise. Do not invent facts "
        "beyond them."
    )
    return "\n\n".join([header, *parts])


def retrieve(query: str) -> str:
    return build_context(search(embed(query)))
