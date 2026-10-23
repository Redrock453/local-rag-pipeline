"""Rule-based intent classifier (no LLM call).

Intents:
  system     - explicit commands ("/status") or operational verbs ("restart")
  technical  - questions that mention tools, code or infrastructure
  general    - everything else

Matching is done on lemmas when ``pymorphy3`` is installed (handles Russian
word forms), otherwise on lower-cased tokens. English keywords work in both
modes.
"""
from __future__ import annotations

import re

try:  # optional: better matching for Russian inflections
    import pymorphy3

    _MORPH = pymorphy3.MorphAnalyzer()
except Exception:  # noqa: BLE001 - library missing or dictionaries not installed
    _MORPH = None

TECHNICAL_TERMS = {
    "code", "bug", "server", "config", "deploy", "docker", "git", "nginx",
    "python", "api", "endpoint", "script", "bash", "systemd", "qdrant",
    "ollama", "pip", "venv", "linux", "database",
    # Russian
    "код", "баг", "сервер", "настройка", "деплой", "скрипт", "конфиг",
    "архитектура",
}
SYSTEM_TERMS = {
    "restart", "status", "stop", "reboot",
    "перезапуск", "рестарт", "статус", "стоп",
}
SYSTEM_PREFIXES = ("/", "!", "#")

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _normalize(word: str) -> str:
    if _MORPH is not None:
        return _MORPH.parse(word)[0].normal_form
    return word


def tokens(text: str) -> set[str]:
    return {_normalize(w) for w in _WORD.findall(text.lower())}


def classify(text: str) -> str:
    text = text.strip()
    if not text:
        return "unknown"
    if text.startswith(SYSTEM_PREFIXES):
        return "system"
    found = tokens(text)
    if found & TECHNICAL_TERMS:
        return "technical"
    if found & SYSTEM_TERMS:
        return "system"
    return "general"
