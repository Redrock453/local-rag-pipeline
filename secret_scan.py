"""Light regex filter that keeps credentials out of the vector index.

It is deliberately conservative: false positives are fine (a clean chunk is
skipped), false negatives are not. It is a safety net, not a replacement for
a proper scanner such as gitleaks or trufflehog.
"""
from __future__ import annotations

import re

_PATTERNS = [
    r"AWS\s*access\s*key\s*id\s*[:=]\s*AKIA[0-9A-Z]{16}",
    r"AWS\s*secret\s*access\s*key\s*[:=]\s*[0-9a-z/+]{40}",
    r"Bearer\s+[A-Za-z0-9\-._~+/]{10,}=*",
    r"(?:password|passwd|secret|pwd)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?",
    r"(?:api[_-]?key|apikey|token)\s*[:=]\s*['\"]?[^\s'\"]{16,}['\"]?",
    r"-----BEGIN\s+(?:[A-Z]+\s+)?PRIVATE\s+KEY-----",
    r"ghp_[A-Za-z0-9]{36}",
    r"glpat-[A-Za-z0-9_-]{20}",
    r"sk_(?:live|test)_[A-Za-z0-9]{24,}",
    r"sk-[A-Za-z0-9_-]{20,}",
    r"xox[bop]-[A-Za-z0-9-]{10,48}",
]
_COMPILED = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def is_clean(text: str) -> bool:
    """True when no secret-like pattern is found."""
    return not text or _COMPILED.search(text) is None
