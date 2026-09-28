"""
Deterministic + semantic parts of the intent score, freshness, source quality and the final score.

    intent score = 25% keyword intent + 20% semantic similarity + 20% explicit need
                 + 15% timeline + 10% location match + 10% LLM confidence
    lead score   = 80% intent score + 10% freshness + 10% source quality
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
from typing import Dict, List, Optional

import httpx

import rule_extractor

from .models import QuerySpec, Unit, UnitIntent

WEIGHTS = {"keyword": 0.25, "semantic": 0.20, "explicit": 0.20, "timeline": 0.15, "location": 0.10, "llm": 0.10}
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_TIMELINE = [
    (re.compile(r"\b(?:immediate(?:ly)?|asap|urgent(?:ly)?|right now|this month|within (?:a|one|1|2|two|3|three) "
                r"(?:weeks?|months?)|can join (?:now|immediately)|sofort)\b", re.I), 100, "immediately / within 3 months"),
    (re.compile(r"\b(?:this year|next (?:few )?months?|within (?:\d{1,2}|six|twelve) months|in (?:\d{1,2}|six) months|"
                r"by (?:the )?end of (?:the )?year|next year|20(?:2[6-9]|3\d))\b", re.I), 85, "within about 12 months"),
    (re.compile(r"\b(?:planning|plan to|want to move|thinking (?:of|about)|someday|in future|future plans?|"
                r"eventually)\b", re.I), 50, "planning, no date"),
]


def _terms_rx(terms: List[str]) -> Optional[re.Pattern]:
    terms = sorted({t.strip().lower() for t in terms if t and len(t.strip()) > 1}, key=len, reverse=True)
    if not terms:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)", re.I)


class Scorer:
    def __init__(self, spec: QuerySpec):
        self.spec = spec
        self.prof = _terms_rx(spec.professions)
        # distinctive single words of multi-word titles also count ("machinist" in "CNC machinist")
        words = {w for p in spec.professions for w in p.lower().split() if len(w) > 3}
        self.prof_words = _terms_rx(list(words - {"operator", "worker", "engineer", "technician", "staff"}))
        self.intent = _terms_rx(spec.high_intent_terms)
        self.negative = _terms_rx(spec.negative_terms)
        self.vocab = self._vocab()

    def _vocab(self) -> Dict[str, float]:
        v: Dict[str, float] = {}
        for group, weight in ((self.spec.professions, 3.0), (self.spec.destination, 2.0), (self.spec.origin, 1.5),
                              (self.spec.high_intent_terms, 1.5), ([self.spec.summary, self.spec.industry], 1.0)):
            for phrase in group:
                for w in _WORD.findall((phrase or "").lower()):
                    if len(w) > 2:
                        v[w] = max(v.get(w, 0), weight)
        return v

    # ---- components (0–100) -------------------------------------------------------------------
    def keyword(self, text: str) -> tuple[int, List[str]]:
        hits: List[str] = []
        score = 0
        prof = self.prof.findall(text) if self.prof else []
        if prof:
            score += 45
            hits.append(f"{self.spec.intent_type.replace('_', ' ') if not prof else prof[0]} identified")
        elif self.prof_words and self.prof_words.search(text):
            score += 25
            hits.append("related role / product words")
        intent = self.intent.findall(text) if self.intent else []
        if intent or rule_extractor.shows_interest(text):
            score += 40 if len(set(intent)) >= 2 else 30
            hits.append("intent phrases: " + ", ".join(dict.fromkeys(x.lower() for x in intent[:3])) if intent
                        else "says they are interested / looking")
        if self.negative and self.negative.search(text):
            score -= 30
        if rule_extractor._HIRING.search(text) and not rule_extractor.shows_interest(text):
            score -= 40                                    # a job ad / recruiter, not a person with the need
        return max(0, min(100, score + (15 if prof and intent else 0))), hits

    def lexical_similarity(self, text: str) -> int:
        words = [w for w in _WORD.findall(text.lower()) if len(w) > 2]
        if not words or not self.vocab:
            return 0
        found = {w for w in words if w in self.vocab}
        covered = sum(self.vocab[w] for w in found)
        total = sum(sorted(self.vocab.values(), reverse=True)[:12])
        return int(min(100, 100 * covered / max(total * 0.5, 1)))

    def explicit(self, unit: Unit) -> int:
        t = unit.text
        if rule_extractor._HIRING.search(t) and not rule_extractor.shows_interest(t):
            return 0
        first_person = re.search(r"\b(?:i|i'm|i am|my|me|ich|mujhe|main)\b", t, re.I)
        return 100 if rule_extractor.shows_interest(t) and first_person else (60 if first_person else 20)

    def timeline(self, text: str) -> tuple[int, Optional[str]]:
        for rx, score, label in _TIMELINE:
            if rx.search(text):
                return score, label
        return 0, None

    def location(self, text: str) -> tuple[int, List[str]]:
        dest, orig = self.spec.destination, self.spec.origin
        if not dest and not orig:
            return 100, []
        d = bool(dest) and rule_extractor.mentions_any(text, dest)
        o = bool(orig) and rule_extractor.mentions_any(text, orig)
        why = ([f"{', '.join(dest)} mentioned"] if d else []) + ([f"{', '.join(orig)} mentioned"] if o else [])
        if d and (o or not orig):
            return 100, why
        return (70 if d else 45 if o else 0), why


def freshness(date_iso: Optional[str]) -> int:
    if not date_iso:
        return 30                                           # unknown date: neither fresh nor stale
    try:
        age = (dt.datetime.now(dt.timezone.utc).date() - dt.date.fromisoformat(date_iso[:10])).days
    except ValueError:
        return 30
    return 100 if age <= 7 else 85 if age <= 30 else 65 if age <= 90 else 40 if age <= 180 else 20


def combine(parts: Dict[str, float]) -> int:
    return int(round(sum(parts.get(k, 0) * w for k, w in WEIGHTS.items())))


def lead_score(intent_score: int, fresh: int, quality: int) -> int:
    return int(round(0.8 * intent_score + 0.1 * fresh + 0.1 * quality))


def tier(score: int) -> str:
    return "HIGH" if score >= 80 else "MEDIUM" if score >= 60 else "LOW" if score >= 40 else "NONE"


# ---- optional embeddings (semantic similarity) ------------------------------------------------
async def embed(texts: List[str], keys: Dict[str, str]) -> Optional[List[List[float]]]:
    """Embeddings from Mistral (mistral-embed) or Gemini when a key is available; None otherwise."""
    texts = [t[:2000] for t in texts]
    mistral = keys.get("mistral") or os.getenv("MISTRAL_API_KEY", "").strip()
    try:
        if mistral:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post("https://api.mistral.ai/v1/embeddings",
                                 headers={"Authorization": f"Bearer {mistral}"},
                                 json={"model": "mistral-embed", "input": texts})
            if r.status_code == 200:
                return [d["embedding"] for d in r.json()["data"]]
        gemini = keys.get("gemini") or os.getenv("GEMINI_API_KEY", "").strip()
        if gemini:
            from google import genai
            client = genai.Client(api_key=gemini)
            res = await client.aio.models.embed_content(model="gemini-embedding-001", contents=texts)
            return [e.values for e in res.embeddings]
    except Exception:
        return None
    return None


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def semantic_from_cosine(c: float) -> int:
    """Map embedding cosine (≈0.6 unrelated … ≈0.9 same meaning for these models) to 0–100."""
    return int(max(0, min(100, (c - 0.62) / (0.88 - 0.62) * 100)))


def llm_parts(r: Optional[UnitIntent]) -> Dict[str, float]:
    if r is None:
        return {}
    return {"llm": r.confidence * 100 if r.relevant else 0,
            "explicit": 100 if r.explicit_need and r.relevant else None}
