"""
Pydantic data contracts shared by every module.

* CandidateRecord          – the row stored in Supabase `candidates`.
* SearchWave               – one wave of the search plan (one platform family).
* ComprehensiveSearchPlan  – the full multi-wave plan Gemini produces.

Two helper models (ExtractedCandidate / PageExtraction) are what Gemini
returns during extraction. They deliberately omit `source_url` and
`platform`: those are set by the pipeline from the real crawled URL so the
model can never invent or mis-attribute a source.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# Hard cap so one very long quote can't bloat the table.
MAX_EVIDENCE_CHARS = 600


def _clean_str(value: Optional[str]) -> Optional[str]:
    """Trim whitespace; turn empty / placeholder strings into None."""
    if value is None or (isinstance(value, float) and value != value):  # None / NaN
        return None
    value = " ".join(str(value).split())
    if value.lower() in {"", "n/a", "na", "nan", "none", "null", "unknown", "not mentioned", "-"}:
        return None
    return value


def _clean_list(values: Optional[List[str]]) -> List[str]:
    """Trim items, drop blanks, de-duplicate case-insensitively, keep order."""
    out: List[str] = []
    seen = set()
    for item in values or []:
        item = _clean_str(item)
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")


def clean_email(value: Optional[str]) -> Optional[str]:
    """Trim punctuation around an address; None unless it looks like one email."""
    value = _clean_str(value)
    if not value:
        return None
    value = value.strip(" .,;:()<>[]\"'").removeprefix("mailto:")
    return value.lower() if _EMAIL_RE.match(value) else None


def clean_phone(value: Optional[str]) -> Optional[str]:
    """Keep a leading + and digits; None unless it has 8–15 digits (E.164 range)."""
    value = _clean_str(value)
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if not 8 <= len(digits) <= 15:
        return None
    first = re.search(r"\+?\d", value)
    return ("+" if first and first.group().startswith("+") else "") + digits


# ---------------------------------------------------------------------------
# Stored record
# ---------------------------------------------------------------------------
class CandidateRecord(BaseModel):
    """A single sourced candidate, exactly matching the `candidates` table."""

    name: Optional[str] = Field(None, description="Person's name or public handle/username")
    current_role: Optional[str] = Field(None, description="Current job title / trade")
    skills: List[str] = Field(default_factory=list, description="Technical skills, machines, certifications")
    current_location: Optional[str] = Field(None, description="City / state / country the person is in now")
    target_countries: List[str] = Field(default_factory=list, description="Countries they want to work in")
    evidence_snippet: Optional[str] = Field(None, description="Verbatim quote proving the match")
    email: Optional[str] = Field(None, description="Email the person themselves posted on the page")
    phone: Optional[str] = Field(None, description="Phone / WhatsApp number the person themselves posted")
    source_url: str = Field(..., description="Canonical page the record was extracted from")
    platform: Optional[str] = Field(None, description="linkedin / reddit / quora / forum / job_portal / …")

    @field_validator("name", "current_role", "current_location", "platform", mode="before")
    @classmethod
    def _norm_str(cls, v):
        return _clean_str(v)

    @field_validator("skills", "target_countries", mode="before")
    @classmethod
    def _norm_list(cls, v):
        return _clean_list(v)

    @field_validator("email", mode="before")
    @classmethod
    def _norm_email(cls, v):
        return clean_email(v)

    @field_validator("phone", mode="before")
    @classmethod
    def _norm_phone(cls, v):
        return clean_phone(v)

    @field_validator("evidence_snippet", mode="before")
    @classmethod
    def _cap_evidence(cls, v: Optional[str]) -> Optional[str]:
        v = _clean_str(v)
        if v and len(v) > MAX_EVIDENCE_CHARS:
            v = v[: MAX_EVIDENCE_CHARS - 1].rstrip() + "…"
        return v

    @field_validator("source_url")
    @classmethod
    def _require_url(cls, v: str) -> str:
        v = (v or "").strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("source_url must be an absolute http(s) URL")
        return v

    def to_db_row(self) -> dict:
        """Dict ready for Supabase upsert (lists stay lists → TEXT[])."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Search plan
# ---------------------------------------------------------------------------
class SearchWave(BaseModel):
    wave_name: str = Field(..., description='Human label, e.g. "LinkedIn Profiles"')
    platform: str = Field(..., description='Platform family, e.g. "linkedin", "reddit_quora", "forums_job_portals"')
    queries: List[str] = Field(default_factory=list, description="Search-engine dork queries for this wave")

    @field_validator("queries", mode="before")
    @classmethod
    def _dedupe_queries(cls, v):
        return _clean_list(v)


class ComprehensiveSearchPlan(BaseModel):
    waves: List[SearchWave] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Gemini extraction output (internal)
# ---------------------------------------------------------------------------
class ExtractedCandidate(BaseModel):
    """What Gemini returns per person found on a page (no URL/platform)."""

    name: Optional[str] = None
    current_role: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    current_location: Optional[str] = None
    target_countries: List[str] = Field(default_factory=list)
    evidence_snippet: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class PageExtraction(BaseModel):
    candidates: List[ExtractedCandidate] = Field(default_factory=list)
