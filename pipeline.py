"""
Step-wise sourcing pipeline for serverless use.

The local Streamlit version runs the whole loop on a background thread.
A Vercel function can only run for a limited time per request, so the
same loop is split into short steps that the browser calls in order:

    plan_search()  → one Gemini call, returns the multi-wave plan
    run_query()    → one search query (canonicalised hits)
    dedup_urls()   → drop URLs already in Supabase `scraped_urls`
    process_batch()→ record URLs → fetch pages → Gemini extraction →
                     grounding check → Supabase upsert   (≈5 URLs per call)

Prompts, grounding check and source-URL rules are the same as agent_pipeline.py.
"""

from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Optional

import supabase_db as db
from fetcher import fetch_batch
from gemini_client import Gemini, GeminiError
from schema import CandidateRecord, ComprehensiveSearchPlan, PageExtraction
from search_module import platform_from_url, search_query

MAX_CONTENT_CHARS_FOR_LLM = 30_000
GROUNDING_MIN_OVERLAP = 0.6   # share of evidence words that must appear on the page


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PLAN_SYSTEM = """You are a senior technical recruiter and OSINT search specialist.
You design search-engine dork queries that surface INDIVIDUAL PEOPLE (not companies,
not job ads) who match a sourcing intent. Queries run on DuckDuckGo / Bing-style engines,
which support: site:, "exact phrases", OR, -exclusion, intitle:, inurl:.
Rules for every query:
- 3 to 12 terms, at most one site: operator, no Google-only operators (no AROUND, no daterange).
- Mix role keywords, skill/machine keywords, location signals (India, Indian, city names),
  and intent signals (abroad, overseas, relocate, Gulf, Europe, Germany, visa, "open to work").
- Vary phrasing across queries so they return different pages.
"""

WAVE_BLUEPRINT = [
    ("LinkedIn Profiles & Posts", "linkedin",
     "site:linkedin.com/in or site:linkedin.com/posts dorks for individual profiles/posts"),
    ("Reddit & Quora Discussions", "reddit_quora",
     "site:reddit.com and site:quora.com dorks for people discussing their own plans/experience"),
    ("Engineering Forums & Job Portals", "forums_job_portals",
     "trade/engineering forums (e.g. practicalmachinist.com, cnczone.com, eng-tips.com) and "
     "Indian job portals/community pages (naukri.com, shine.com, apna.co, indeed) where candidates post"),
    ("Facebook & Community Groups", "facebook_community",
     "site:facebook.com public groups/posts, Telegram/WhatsApp group directories, community pages"),
    ("X / Twitter & Blogs", "x_blogs",
     "site:x.com or site:twitter.com posts, personal blogs, Medium posts written by candidates"),
    ("Long-tail & Regional Sources", "long_tail",
     "regional-language phrasing, Indian city-specific pages, ITI/polytechnic alumni pages"),
]

EXTRACT_SYSTEM = """You extract candidate leads for a recruiter from ONE web page.
Return every INDIVIDUAL PERSON on the page who matches the sourcing intent.
Strict rules:
- Only real individuals speaking about themselves or whose own profile this is.
  EXCLUDE companies, recruiters, agencies, consultants selling services, job advertisements,
  and people who are merely giving advice to others.
- A person qualifies only if the page text itself supports that they match the intent
  (e.g. their trade/skills AND a signal of interest in working abroad, or whatever the intent requires).
- evidence_snippet: copy a VERBATIM sentence or phrase from the page (max ~300 characters)
  that proves the match. Do not paraphrase. Do not invent.
- name: the person's name, or their public username/handle on forums. Null if absent.
- Fill a field only if the page states it. Never guess locations, countries, or skills.
- skills: concrete skills, machines, controllers, software, certifications (e.g. "CNC turning",
  "Fanuc", "VMC", "G-code", "ITI Machinist").
- target_countries: countries/regions they want to work in, as stated (e.g. "Germany", "Gulf").
- If nobody qualifies, return {"candidates": []}.
"""


def _plan_prompt(intent: str, num_waves: int, queries_per_wave: int) -> str:
    lines = [f"Sourcing intent: {intent.strip()}", "",
             f"Produce exactly {num_waves} waves, in this order, each with exactly {queries_per_wave} queries:"]
    for i, (name, platform, hint) in enumerate(WAVE_BLUEPRINT[:num_waves], start=1):
        lines.append(f'Wave {i}: wave_name="{name}", platform="{platform}" — {hint}.')
    lines.append("")
    lines.append("Return JSON matching the schema. Queries only — no explanations.")
    return "\n".join(lines)


def _extract_prompt(intent: str, url: str, platform: str, content: str, snippet_only: bool) -> str:
    note = ("NOTE: The page could not be opened (login wall / blocked). The content below is ONLY the "
            "search-engine title and snippet for this URL. Extract only what it explicitly states.\n\n"
            if snippet_only else "")
    return (f"Sourcing intent: {intent.strip()}\n"
            f"Page URL: {url}\nPlatform: {platform}\n\n{note}"
            f"----- PAGE CONTENT -----\n{content[:MAX_CONTENT_CHARS_FOR_LLM]}\n"
            f"----- END -----")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")


def _is_grounded(evidence: Optional[str], content: str) -> bool:
    """True if most words of the evidence quote actually appear in the page."""
    if not evidence:
        return False
    ev_words = _WORD.findall(evidence.lower())
    if not ev_words:
        return False
    page_words = set(_WORD.findall(content.lower()))
    hits = sum(1 for w in ev_words if w in page_words)
    return hits / len(ev_words) >= GROUNDING_MIN_OVERLAP


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:60] or "person"


def _assign_source_urls(url: str, people: List[dict]) -> List[dict]:
    """First person keeps the page URL; others get a stable '#candidate-<name>' fragment."""
    out, used = [], set()
    for i, person in enumerate(people):
        if i == 0:
            src = url
        else:
            base = f"{url}#candidate-{_slug(person.get('name') or '')}"
            src, n = base, 2
            while src in used:
                src, n = f"{base}-{n}", n + 1
        used.add(src)
        out.append({**person, "source_url": src})
    return out


async def _extract_page(gemini: Gemini, intent: str, url: str, content: str,
                        snippet_only: bool) -> tuple[List[CandidateRecord], int]:
    """Returns (valid records, number dropped as ungrounded/invalid)."""
    platform = platform_from_url(url)
    result = await gemini.generate_structured(
        _extract_prompt(intent, url, platform, content, snippet_only),
        PageExtraction, system_instruction=EXTRACT_SYSTEM, temperature=0.1,
        max_retries=3,
    )
    people, dropped = [], 0
    for c in result.candidates:
        d = c.model_dump()
        if not _is_grounded(d.get("evidence_snippet"), content):
            dropped += 1
            continue
        people.append(d)
    records: List[CandidateRecord] = []
    for row in _assign_source_urls(url, people):
        try:
            records.append(CandidateRecord(**row, platform=platform))
        except Exception:
            dropped += 1
    return records, dropped


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
async def plan_search(gemini: Gemini, intent: str, num_waves: int, queries_per_wave: int) -> dict:
    plan = await gemini.generate_structured(
        _plan_prompt(intent, num_waves, queries_per_wave), ComprehensiveSearchPlan,
        system_instruction=PLAN_SYSTEM, temperature=0.6, thinking_budget=512, max_retries=3,
    )
    waves = [w for w in plan.waves if w.queries][:num_waves]
    for w in waves:
        w.queries = w.queries[:queries_per_wave]
    return {"waves": [w.model_dump() for w in waves]}


def run_query(query: str, max_results: int, region: str, backend: str) -> dict:
    outcome = search_query(query, max_results=max_results, region=region, backend=backend)
    return {
        "query": query,
        "error": outcome.error,
        "rate_limited": outcome.rate_limited,
        "hits": [{"url": h.url, "title": h.title, "snippet": h.snippet} for h in outcome.hits],
    }


def dedup_urls(urls: List[str]) -> List[str]:
    return db.filter_fresh_urls(urls)


async def process_batch(gemini: Gemini, intent: str, items: List[dict], wave_tag: str,
                        page_timeout_s: int, respect_robots: bool, snippet_fallback: bool) -> dict:
    """items: [{url, title, snippet}] — record, fetch, extract, save."""
    urls = [i["url"] for i in items]
    hits: Dict[str, dict] = {i["url"]: i for i in items}
    # Record right before crawling so a stopped run leaves unreached URLs unmarked.
    await asyncio.to_thread(db.record_scraped_urls, urls, wave_tag)

    outcomes = await fetch_batch(urls, page_timeout_s=page_timeout_s, respect_robots=respect_robots)
    stats = {"crawled": sum(o.ok for o in outcomes), "blocked": sum(o.blocked for o in outcomes)}
    stats["failed"] = len(outcomes) - stats["crawled"] - stats["blocked"]

    jobs = []
    for o in outcomes:
        if o.ok:
            jobs.append((o.url, o.markdown, False))
        elif snippet_fallback:
            h = hits.get(o.url, {})
            snippet = h.get("snippet") or ""
            if len(snippet) > 40:
                jobs.append((o.url, f"Title: {h.get('title', '')}\nSnippet: {snippet}".strip(), True))

    results = await asyncio.gather(
        *(_extract_page(gemini, intent, u, c, snip) for (u, c, snip) in jobs), return_exceptions=True
    )
    records: List[CandidateRecord] = []
    warnings: List[str] = []
    dropped = 0
    for (u, _, _), r in zip(jobs, results):
        if isinstance(r, Exception):
            if isinstance(r, GeminiError) and "API key" in str(r):
                raise r
            warnings.append(f"Extraction failed for {u}: {str(r)[:160]}")
            continue
        recs, d = r
        dropped += d
        records.extend(recs)

    if records:
        await asyncio.to_thread(db.save_candidates, records)

    return {
        "stats": {**stats, "records": len(records), "dropped": dropped, "sent_to_gemini": len(jobs)},
        "records": [r.model_dump() for r in records],
        "errors": {o.url: o.error for o in outcomes if o.error},
        "warnings": warnings,
    }
