"""Local-first RAG pipeline: classify -> retrieve -> generate -> validate."""

from .pipeline import run

__all__ = ["run"]
__version__ = "0.1.0"
