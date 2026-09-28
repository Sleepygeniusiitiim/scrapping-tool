"""
Normalization: clean text, language detection, content hashing (exact duplicates), near-duplicate
detection (word shingles), and PII minimization before any text is sent to an LLM.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

import rule_extractor

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Tiny stop-word profiles — enough to tell the common languages of these discussions apart.
_LANG_WORDS = {
    "en": {"the", "and", "is", "to", "for", "in", "i", "my", "you", "with", "looking", "job", "of"},
    "de": {"und", "ich", "die", "der", "das", "ist", "nicht", "mit", "für", "suche", "eine", "arbeit", "zu"},
    "fr": {"le", "la", "et", "je", "les", "des", "pour", "est", "une", "dans", "travail", "cherche"},
    "es": {"el", "la", "y", "que", "de", "en", "los", "para", "busco", "trabajo", "una", "por"},
    "hi-latn": {"hai", "mujhe", "chahiye", "kya", "nahi", "aur", "mein", "ke", "ki", "hoon", "bhai", "naukri"},
}


def clean(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("​", " ")).strip()


def language(text: str) -> str:
    t = text or ""
    if re.search(r"[ऀ-ॿ]", t):
        return "hi"
    if re.search(r"[؀-ۿ]", t):
        return "ar"
    words = [w.lower() for w in _WORD.findall(t)][:400]
    if not words:
        return "unknown"
    scores = {lang: sum(1 for w in words if w in vocab) for lang, vocab in _LANG_WORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "unknown"


def content_hash(text: str) -> str:
    """Level-2 duplicate key: SHA-256 of the lower-cased, whitespace-normalized text."""
    return hashlib.sha256(clean(text).lower().encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 4) -> set:
    words = [w.lower() for w in _WORD.findall(text or "")]
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def near_duplicate(a: str, b: str, threshold: float = 0.8) -> bool:
    """Level-3 duplicate: word-shingle Jaccard similarity (a cheap stand-in for embedding similarity)."""
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= threshold


def redact(text: str) -> Tuple[str, Dict[str, str]]:
    """PII minimization for LLM calls: emails / phone numbers → [EMAIL_1] / [PHONE_1].
    The mapping stays on the server; contacts are re-attached from the author's own text, never from
    the model's output."""
    mapping: Dict[str, str] = {}
    out = text or ""
    for i, e in enumerate(rule_extractor.emails_in(out), start=1):
        tag = f"[EMAIL_{i}]"
        mapping[tag] = e
        out = re.sub(re.escape(e), tag, out, flags=re.IGNORECASE)
    phones = [m.group() for m in rule_extractor.PHONE_RE.finditer(out)]
    for i, p in enumerate(dict.fromkeys(phones), start=1):
        if len(re.sub(r"\D", "", p)) >= 8:
            tag = f"[PHONE_{i}]"
            mapping[tag] = p
            out = out.replace(p, tag)
    return out, mapping


def mask(value: str) -> str:
    """Display form for logs: +91 98•••••210 / r•••@gmail.com."""
    if not value:
        return value
    if "@" in value:
        user, _, dom = value.partition("@")
        return f"{user[:1]}•••@{dom}"
    digits = re.sub(r"\D", "", value)
    return f"{digits[:4]}•••••{digits[-3:]}" if len(digits) > 7 else "•••"
