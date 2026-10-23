"""Generation with structured outputs and a provider cascade.

Primary provider: any OpenAI-compatible ``/chat/completions`` endpoint
(OpenRouter, vLLM, llama.cpp server, ...). Three attempts per request, from
strictest to loosest:

  1. ``response_format = json_schema`` (strict structured outputs)
  2. ``response_format = json_object`` with the schema injected in the prompt
  3. plain text, with JSON parsed out of a markdown block

If the primary provider fails, the request goes to a local Ollama model using
Ollama's ``format`` (JSON schema) parameter.
"""
from __future__ import annotations

import json
import logging
import os

import requests

from .schema import ANSWER_SCHEMA

log = logging.getLogger("rag_pipeline.generate")


def _cfg() -> dict:
    return {
        "base_url": os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key": os.environ.get("LLM_API_KEY", ""),
        "model": os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
        "ollama_url": os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
        "ollama_model": os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"),
        "timeout": float(os.environ.get("LLM_TIMEOUT", "60")),
    }


def parse_json_answer(content: str) -> dict:
    """Parse a model reply into the answer schema.

    Strips markdown fences; if the text is not valid JSON, wraps it so the
    caller still receives both required keys.
    """
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {"message": content, "thoughts": "(unstructured response)"}


def _post_chat(cfg: dict, payload: dict) -> str:
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    resp = requests.post(f"{cfg['base_url']}/chat/completions", headers=headers,
                         json=payload, timeout=cfg["timeout"])
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _with_schema_hint(messages: list[dict]) -> list[dict]:
    hint = ("\n\nReply with a single JSON object matching this schema: "
            + json.dumps(ANSWER_SCHEMA))
    return [
        {**m, "content": m["content"] + hint} if m["role"] == "system" else m
        for m in messages
    ]


def openai_compatible_chat(messages: list[dict], model: str | None = None,
                           temperature: float = 0.3) -> dict:
    cfg = _cfg()
    model = model or cfg["model"]
    base = {"model": model, "temperature": temperature}

    attempts = [
        {**base, "messages": messages, "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True,
                            "schema": ANSWER_SCHEMA}}},
        {**base, "messages": _with_schema_hint(messages),
         "response_format": {"type": "json_object"}},
        {**base, "messages": messages},
    ]
    last_error: Exception | None = None
    for payload in attempts:
        try:
            return parse_json_answer(_post_chat(cfg, payload))
        except Exception as exc:  # noqa: BLE001 - try the next, looser format
            last_error = exc
            log.debug("attempt failed: %s", exc)
    raise RuntimeError(f"all structured-output attempts failed: {last_error}")


def ollama_chat(messages: list[dict], temperature: float = 0.3) -> dict:
    cfg = _cfg()
    resp = requests.post(
        f"{cfg['ollama_url']}/api/chat",
        json={"model": cfg["ollama_model"], "messages": messages,
              "format": ANSWER_SCHEMA, "stream": False,
              "options": {"temperature": temperature}},
        timeout=cfg["timeout"],
    )
    resp.raise_for_status()
    return parse_json_answer(resp.json()["message"]["content"])


def generate(messages: list[dict], temperature: float = 0.3,
             model: str | None = None) -> dict:
    """Primary provider first, local Ollama as the last resort."""
    errors: list[str] = []
    for name, call in (
        ("primary", lambda: openai_compatible_chat(messages, model, temperature)),
        ("ollama", lambda: ollama_chat(messages, temperature)),
    ):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            log.warning("%s provider failed: %s", name, exc)
    raise RuntimeError("all providers failed - " + " | ".join(errors))
