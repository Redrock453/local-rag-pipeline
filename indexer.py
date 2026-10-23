"""Index a directory of text files into Qdrant.

walk -> chunk -> secret filter -> embed (Ollama) -> upsert. Point ids are
deterministic (uuid5 of path + chunk number), so re-running the indexer
updates existing points instead of duplicating them.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import requests

from .retrieve import _cfg, embed
from .secret_scan import is_clean

log = logging.getLogger("rag_pipeline.indexer")

TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".py", ".sh", ".yml", ".yaml", ".json", ".toml"}
EXCLUDED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
MAX_CHUNK_CHARS = 1500
_NAMESPACE = uuid.UUID("6f1d3c52-2f0e-4a53-9a39-5f0c7b0a1d11")


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split on blank lines, packing paragraphs up to ``max_chars``.

    A single paragraph longer than the limit is cut into fixed-size pieces.
    """
    chunks: list[str] = []
    current = ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        while len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def point_id(source: str, number: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{source}#{number}"))


def ensure_collection(dim: int) -> None:
    cfg = _cfg()
    url = f"{cfg['qdrant_url']}/collections/{cfg['collection']}"
    if requests.get(url, timeout=5).status_code == 200:
        return
    resp = requests.put(url, json={"vectors": {"size": dim, "distance": "Cosine"}},
                        timeout=10)
    resp.raise_for_status()


def index_directory(root: str) -> dict:
    """Index ``root``. Returns counts: files, indexed, skipped_secret, failed."""
    base = Path(root).resolve()
    cfg = _cfg()
    stats = {"files": 0, "indexed": 0, "skipped_secret": 0, "failed": 0}
    ready = False
    for path in iter_files(base):
        stats["files"] += 1
        source = str(path.relative_to(base))
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            log.warning("cannot read %s: %s", source, exc)
            stats["failed"] += 1
            continue
        points = []
        for number, chunk in enumerate(chunk_text(text)):
            if not is_clean(chunk):
                stats["skipped_secret"] += 1
                continue
            vector = embed(chunk)
            if not vector:
                stats["failed"] += 1
                continue
            if not ready:
                ensure_collection(len(vector))
                ready = True
            points.append({"id": point_id(source, number), "vector": vector,
                           "payload": {"text": chunk, "source": source}})
        if points:
            resp = requests.put(
                f"{cfg['qdrant_url']}/collections/{cfg['collection']}/points",
                params={"wait": "true"}, json={"points": points}, timeout=30)
            resp.raise_for_status()
            stats["indexed"] += len(points)
    return stats
