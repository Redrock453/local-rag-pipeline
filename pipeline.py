"""classify -> retrieve -> generate -> validate."""
from __future__ import annotations

import json
import logging
import os
import time

from .classify import classify
from .generate import generate
from .retrieve import retrieve

log = logging.getLogger("rag_pipeline")

SYSTEM_PROMPT = (
    "You are a concise technical assistant. Answer directly and accurately. "
    "Put code in fenced blocks. Reply strictly as JSON with two fields: "
    "'message' (the answer for the user) and 'thoughts' (a short reasoning "
    "outline)."
)


class ValidationError(ValueError):
    """The model reply does not satisfy the answer contract."""


def validate(result: dict) -> dict:
    """Check the reply against the contract; raise ValidationError otherwise."""
    for key in ("message", "thoughts"):
        value = result.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"missing or empty field: {key!r}")
    return {"message": result["message"].strip(),
            "thoughts": result["thoughts"].strip()}


def _trace(entry: dict) -> None:
    """Append one JSON line per request when RAG_TRACE_FILE is set."""
    path = os.environ.get("RAG_TRACE_FILE")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run(user_text: str, history: list[dict] | None = None,
        max_attempts: int = 2) -> dict:
    """Answer one question. Returns {'message', 'thoughts'}.

    Intents other than ``technical``/``system`` are rejected so this pipeline
    stays focused; route them elsewhere in your application.
    """
    started = time.time()
    intent = classify(user_text)
    if intent in ("general", "unknown"):
        raise NotImplementedError(f"intent {intent!r} is out of scope for this pipeline")

    context = retrieve(user_text)
    system = SYSTEM_PROMPT + (f"\n\n{context}" if context else "")
    messages = [{"role": "system", "content": system}, *(history or []),
                {"role": "user", "content": user_text}]

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = validate(generate(messages))
            _trace({"ts": started, "intent": intent, "rag": bool(context),
                    "attempts": attempt, "ok": True,
                    "latency_ms": int((time.time() - started) * 1000)})
            return result
        except ValidationError as exc:  # retry: the model may fix its format
            last_error = exc
            log.warning("validation failed (attempt %d): %s", attempt, exc)
    _trace({"ts": started, "intent": intent, "rag": bool(context), "ok": False,
            "error": str(last_error)})
    raise ValidationError(str(last_error))
