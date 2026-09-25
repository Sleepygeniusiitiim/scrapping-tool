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
from fetcher import PHONE_RE, fetch_batch, scrapedo_token
from gemini_client import Gemini, GeminiError
from schema import CandidateRecord, ComprehensiveSearchPlan, PageExtraction
from search_module import google_search_scrapedo, platform_from_url, search_query

MAX_CONTENT_CHARS_FOR_LLM = 45_000
GROUNDING_MIN_OVERLAP = 0.6   # share of evidence words that must appear on the page


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PLAN_SYSTEM = """You are a senior technical recruiter and OSINT search specialist.
You design search-engine queries that surface INDIVIDUAL CANDIDATES (not companies, not job ads
on their own) who match a sourcing intent — ideally pages where candidates have publicly posted
their own contact details.

Where candidates are found (use this knowledge):
- Overseas hiring posts on LinkedIn (linkedin.com/posts) and Facebook job groups, where candidates
  reply in the comments with "interested", their experience, email (gmail.com / yahoo.com) or
  WhatsApp / mobile number. These are the richest source — target the POST, the comments come with it.
- LinkedIn profiles (linkedin.com/in) with "open to work", "looking for opportunities abroad".
- Gulf / Europe job-group pages, job-seeker forums, CV/resume listing pages, Reddit and Quora threads
  where people describe their own trade and plans.

Query rules:
- Keep queries SHORT: 4 to 8 terms. Long queries with many quoted phrases return nothing.
- At most ONE quoted candidate signal (like "interested" or "gmail.com") per query, and at least half
  of each wave's queries should have none — just site: + role + destination/hiring words. The hiring
  posts themselves are what we want; their comment sections hold the candidates.
- At most one site: operator. Supported: site:, "exact phrase", OR, -exclusion,
  intitle:, inurl:. No Google-only operators (no AROUND, no daterange).
- Expand the role into its real-world synonyms, machines, controllers and brands (e.g. for press brake:
  "press brake operator", "CNC bending", "sheet metal bending", Amada, Trumpf, Bystronic, "Delem").
- Combine: role/synonym + destination signals (Germany, Europe, Dubai, UAE, Saudi, Qatar, Kuwait, Oman,
  Gulf, abroad, overseas, relocation, visa) + candidate signals ("interested", "my CV", "gmail.com",
  "whatsapp", "contact number", "open to work", "experience in India", Indian city names).
- Vary phrasing and synonyms across queries so they return different pages.
"""

WAVE_BLUEPRINT = [
    ("LinkedIn Hiring Posts & Comments", "linkedin_posts",
     "site:linkedin.com/posts overseas hiring posts for this role — especially Indian overseas-recruitment "
     "agency posts (\"urgent requirement\", \"interview in Mumbai/Delhi/Chennai\", \"free recruitment\", "
     "\"Gulf jobs\", Saudi/Qatar/UAE/Kuwait/Oman project hiring), whose comments hold Indian candidates' "
     "contact details. Short queries: site: + role + one destination or agency phrase"),
    ("Facebook & Job Groups", "facebook_groups",
     "site:facebook.com public posts in Gulf/Europe job groups for this role where candidates reply with "
     "WhatsApp numbers or emails"),
    ("LinkedIn Profiles", "linkedin_profiles",
     "site:linkedin.com/in individual profiles for this role that are open to work / relocation abroad"),
    ("Job Portals, CVs & Forums", "portals_forums",
     "job-seeker pages, CV/resume listings and trade forums (naukri.com, indeed.com, shine.com, apna.co, "
     "gulftalent.com, bayt.com, practicalmachinist.com) where candidates post their own profile"),
    ("Reddit & Quora Discussions", "reddit_quora",
     "site:reddit.com and site:quora.com threads where people discuss their own plans to work abroad in this trade"),
    ("Long-tail & Regional Sources", "long_tail",
     "Indian city / state specific pages, ITI and polytechnic alumni pages, Telegram/WhatsApp group directories"),
]

EXTRACT_SYSTEM = """You extract candidate leads for a recruiter from ONE web page.
The page may start with a structured "Post and comments" block (one line per author) and a
"Contact details found on the page" block listing every email / phone with its surrounding text.

Return every INDIVIDUAL PERSON on the page who matches the sourcing intent:
- The author of a post or profile, AND every commenter — each commenter is a separate person.
- EXCLUDE companies, recruiters, agencies, the person advertising the job, and people only giving
  advice. A recruiter's own post is a source of candidates (its commenters), not a candidate.
- A person qualifies if the page supports that they match the intent. Replying to an overseas job post
  for the role with interest ("interested", sharing a CV/email/number, asking how to apply) counts as
  both interest in working abroad and relevance to that role, even if they don't restate their trade.
- Do NOT infer nationality or location from a person's name. Fill current_location only if stated
  (e.g. "experience in India", a city). Leave it null otherwise — do not drop the person for that.
- evidence_snippet: copy a VERBATIM sentence from the page — preferably the person's own words — that
  proves the match (max ~300 characters). Do not paraphrase. Do not invent.
- email / phone: ONLY a contact detail that THIS SAME PERSON wrote in their own comment, post or profile
  on this page. Copy it exactly. Never give a person the recruiter's, company's or another commenter's
  contact. Null if they didn't post one. Phone includes WhatsApp / mobile numbers.
- name: the person's name or public handle. Fill other fields only if the page states them.
- skills: concrete skills, machines, controllers, software, certifications.
- target_countries: countries/regions they want to work in, as stated or as given by the job post
  they replied to.
- If nobody qualifies, return {"candidates": []}.
"""


def _plan_prompt(intent: str, num_waves: int, queries_per_wave: int, round_no: int = 1,
                 exclude_queries: Optional[List[str]] = None) -> str:
    # Each new round starts further along the blueprint, so rounds cover different source types.
    offset = ((round_no - 1) * num_waves) % len(WAVE_BLUEPRINT)
    blueprint = (WAVE_BLUEPRINT[offset:] + WAVE_BLUEPRINT[:offset])[:num_waves]
    lines = [f"Sourcing intent: {intent.strip()}", "",
             f"Produce exactly {num_waves} waves, in this order, each with exactly {queries_per_wave} queries:"]
    for i, (name, platform, hint) in enumerate(blueprint, start=1):
        lines.append(f'Wave {i}: wave_name="{name}", platform="{platform}" — {hint}.')
    if exclude_queries:
        lines += ["", f"This is search round {round_no}. These queries were already run — do NOT repeat them "
                      "or near-copies; use different synonyms, machine brands, destinations, cities and phrasings:"]
        lines += [f"- {q}" for q in exclude_queries[-120:]]
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


_AT = re.compile(r"\s*(?:\[at\]|\(at\)|\{at\}|\bat\b)\s*", re.IGNORECASE)
_DOT = re.compile(r"\s*(?:\[dot\]|\(dot\)|\{dot\}|\bdot\b)\s*", re.IGNORECASE)


def _compact(text: str) -> str:
    """Lower-case, "at"/"dot" spelled out → @ / ., all whitespace removed."""
    return re.sub(r"\s+", "", _DOT.sub(".", _AT.sub("@", text.lower())))


def _email_on_page(email: Optional[str], content: str) -> bool:
    """The address is on the page, allowing for stray spaces ("name 47@yahoo.com") and
    spelled-out forms ("name at gmail dot com") that people use in comments."""
    if not email:
        return False
    e = email.strip().lower()
    return e in content.lower() or e in _compact(content)


def _phone_on_page(phone: Optional[str], content: str) -> bool:
    """True if the number's digits (ignoring spaces/dashes/country prefix) occur on the page."""
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) < 8:
        return False
    page_digits = [re.sub(r"\D", "", m.group()) for m in PHONE_RE.finditer(content)]
    # Compare the last 10 digits so "+91 98765 43210" matches "9876543210".
    return any(p[-10:] == digits[-10:] for p in page_digits if len(p) >= 8)


_THREAD_LINE = re.compile(r"^(?:POST|COMMENT) by (.+?): (.*)$", re.MULTILINE)


def _thread_lines(content: str, name: Optional[str]) -> tuple[str, str]:
    """(this person's lines, everyone else's lines) from the structured post/comment block."""
    who = (name or "").strip().lower()
    own, others = [], []
    for author, body in _THREAD_LINE.findall(content):
        (own if who and author.strip().lower() == who else others).append(body)
    return "\n".join(own), "\n".join(others)


def _owned(value: Optional[str], on_page, content: str, own: str, others: str) -> bool:
    """A contact is kept only if it's on the page and not someone else's in the thread."""
    if not value or not on_page(value, content):
        return False
    if own and on_page(value, own):
        return True
    # Written by another author (e.g. the recruiter's own number) → not this person's.
    return not (others and on_page(value, others))


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
        # Contact details must appear on the page exactly — never trust a generated one —
        # and, where the page has an author-attributed thread, in this person's own lines.
        own, others = _thread_lines(content, d.get("name"))
        if not _owned(d.get("email"), _email_on_page, content, own, others):
            d["email"] = None
        if not _owned(d.get("phone"), _phone_on_page, content, own, others):
            d["phone"] = None
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
async def plan_search(gemini: Gemini, intent: str, num_waves: int, queries_per_wave: int,
                      round_no: int = 1, exclude_queries: Optional[List[str]] = None) -> dict:
    plan = await gemini.generate_structured(
        _plan_prompt(intent, num_waves, queries_per_wave, round_no, exclude_queries), ComprehensiveSearchPlan,
        system_instruction=PLAN_SYSTEM, temperature=0.6 if round_no == 1 else 0.9, thinking_budget=512,
        max_retries=3,
    )
    done = {q.strip().lower() for q in exclude_queries or []}
    waves = [w for w in plan.waves if w.queries][:num_waves]
    for w in waves:
        w.queries = [q for q in w.queries if q.strip().lower() not in done][:queries_per_wave]
    waves = [w for w in waves if w.queries]
    return {"waves": [w.model_dump() for w in waves]}


def run_query(query: str, max_results: int, region: str, backend: str) -> dict:
    """backend: auto (DuckDuckGo, topped up from Google when thin) | duckduckgo | google."""
    token = scrapedo_token()
    sources, errors, rate_limited = [], [], False
    hits: Dict[str, dict] = {}

    def add(outcome, label):
        nonlocal rate_limited
        sources.append(label)
        rate_limited = rate_limited or outcome.rate_limited
        if outcome.error:
            errors.append(f"{label}: {outcome.error}")
        for h in outcome.hits:
            hits.setdefault(h.url, {"url": h.url, "title": h.title, "snippet": h.snippet})

    if backend != "google" or not token:
        ddg_backend = "duckduckgo" if backend == "duckduckgo" else "auto"
        add(search_query(query, max_results=max_results, region=region, backend=ddg_backend), "ddg")
    # Datacenter IPs get thin DuckDuckGo results; Google via Scrape.do fills the gap.
    if token and (backend == "google" or (backend == "auto" and len(hits) < 3)):
        add(google_search_scrapedo(query, token, max_results=max_results, region=region), "google")

    return {
        "query": query,
        "error": "; ".join(errors) if errors and not hits else None,
        "rate_limited": rate_limited,
        "sources": sources,
        "hits": list(hits.values())[:max_results * 2],
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
        "stats": {**stats, "records": len(records), "dropped": dropped, "sent_to_gemini": len(jobs),
                  "with_phone": sum(1 for r in records if r.phone),
                  "with_email": sum(1 for r in records if r.email),
                  "via_scrapedo": sum(1 for o in outcomes if o.via == "scrape.do" and o.ok)},
        "records": [r.model_dump() for r in records],
        "errors": {o.url: o.error for o in outcomes if o.error},
        "warnings": warnings,
    }
