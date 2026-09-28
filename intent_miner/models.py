"""
Data model of the Intent Miner:

    user command ─► QuerySpec ─► providers ─► RawDocument ─► Unit (one post / comment)
                 ─► scores + LLM IntentResult ─► Lead (person) with evidence from one or more sources
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

INTENT_TYPES = (
    "job_search", "product_purchase", "service_request", "recruitment", "vendor_search", "partnership",
    "investment", "travel", "education", "immigration", "relocation", "complaint", "comparison",
    "price_inquiry", "informational", "irrelevant",
)
SOURCES = ("search", "reddit", "quora", "forums", "rss", "linkedin", "facebook")


def _clean(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = re.split(r"[,;\n]", v)
    out: Dict[str, None] = {}
    for x in v:
        x = re.sub(r"\s+", " ", str(x or "")).strip()
        if x:
            out.setdefault(x, None)
    return list(out)


# ---------------------------------------------------------------------------
# Query understanding (LLM output, validated)
# ---------------------------------------------------------------------------
class SourcedQuery(BaseModel):
    source: str = Field("search", description="search | reddit | quora | forums | linkedin | facebook")
    query: str


class QuerySpec(BaseModel):
    summary: str = Field("", description="One sentence restating what is being looked for")
    intent_type: str = Field("job_search", description="One of the intent types")
    industry: str = ""
    professions: List[str] = Field(default_factory=list, description="Job titles / products / services + synonyms")
    origin: List[str] = Field(default_factory=list, description="Where the people are now")
    destination: List[str] = Field(default_factory=list, description="Where they want to go / buy / work")
    high_intent_terms: List[str] = Field(default_factory=list)
    negative_terms: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list, description="ISO codes, e.g. en, de, hi")
    timeline_months: Optional[int] = Field(None, description="Planning horizon stated in the command")
    max_age_days: Optional[int] = Field(None, description="How recent the discussions must be")
    subreddits: List[str] = Field(default_factory=list, description="Relevant subreddit names, no r/")
    queries: List[SourcedQuery] = Field(default_factory=list)

    @field_validator("professions", "origin", "destination", "high_intent_terms", "negative_terms",
                     "languages", "subreddits", mode="before")
    @classmethod
    def _lists(cls, v):
        return _clean(v)

    @field_validator("intent_type", mode="before")
    @classmethod
    def _intent(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in INTENT_TYPES else "job_search"


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
@dataclass
class Unit:
    """One piece of text by one author: a post, a comment or an answer."""
    kind: str                         # post | comment | answer | snippet | profile
    author: Optional[str]
    text: str
    author_url: Optional[str] = None
    date: Optional[str] = None        # YYYY-MM-DD


@dataclass
class RawDocument:
    url: str
    source: str                       # reddit | quora | forums | linkedin | facebook | rss | search
    title: str = ""
    units: List[Unit] = field(default_factory=list)
    date: Optional[str] = None
    via: str = "direct"               # how it was obtained: api | direct | unblocker | snippet | feed
    metadata: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    status: str = "ok"                # ok | blocked | failed | skipped

    @property
    def text(self) -> str:
        return "\n".join(u.text for u in self.units)


# ---------------------------------------------------------------------------
# LLM intent classification (structured output)
# ---------------------------------------------------------------------------
class UnitIntent(BaseModel):
    unit: int = Field(..., description="Index of the unit in the list")
    relevant: bool = False
    intent_type: str = "irrelevant"
    intent_strength: int = Field(0, ge=0, le=100)
    buying_stage: Literal["awareness", "research", "consideration", "evaluation", "ready", "unknown"] = "unknown"
    urgency: Literal["low", "medium", "high", "unknown"] = "unknown"
    explicit_need: bool = Field(False, description="The author states their own need / plan")
    profession: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    timeline: Optional[str] = Field(None, description="e.g. 'within 12 months', 'next year', 'immediately'")
    budget: Optional[str] = None
    confidence: float = Field(0.0, ge=0, le=1)
    evidence: List[str] = Field(default_factory=list, description="Verbatim quotes from the unit")

    @field_validator("intent_type", mode="before")
    @classmethod
    def _intent(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in INTENT_TYPES else "irrelevant"

    @field_validator("intent_strength", mode="before")
    @classmethod
    def _strength(cls, v):
        try:
            return max(0, min(100, int(float(v))))
        except (TypeError, ValueError):
            return 0

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, v / 100 if v > 1 else v))


class DocumentIntent(BaseModel):
    units: List[UnitIntent] = Field(default_factory=list)
