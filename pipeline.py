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
import rule_extractor
import dates
import integrations
from fetcher import PHONE_RE, fetch_batch, scrapedo_token
from gemini_client import Gemini, GeminiError, GeminiQuotaError
from schema import CandidateRecord, ComprehensiveSearchPlan, PageExtraction, clean_email, clean_phone
from search_module import google_search_scrapedo, platform_from_url, search_query

MAX_CONTENT_CHARS_FOR_LLM = 45_000
GROUNDING_MIN_OVERLAP = 0.6   # share of evidence words that must appear on the page


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PLAN_SYSTEM = """You are a senior recruiter and OSINT search specialist.
You design search-engine queries that surface INDIVIDUAL PEOPLE (not companies, not job ads on their
own) who match the sourcing intent EXACTLY as written — any role (trades, drivers, nurses, teachers,
trainers, engineers, …) and any country. Follow the intent literally: its role, where the people are
or want to work, and what they want. Ideally find pages where people publicly show interest and post
their own contact details.

Where such people are found:
- Hiring / opportunity posts on LinkedIn (linkedin.com/posts) and Facebook groups, where people reply in
  the comments with "interested", their experience, email or WhatsApp / mobile number. Target the POST;
  its comments hold the candidates.
- Profiles with "open to work", "looking for opportunities", "seeking a position" (linkedin.com/in, portals).
- Job-seeker forums, CV / resume pages, Reddit and Quora threads, expat forums, community boards where
  people describe their own profession and plans.

Query rules:
- Keep queries SHORT: 4 to 8 terms. Long queries with many quoted phrases return nothing.
- At most ONE quoted signal (like "interested" or "gmail.com") per query; at least half of each wave's
  queries have none — just site: + role + place / hiring words.
- At most one site: operator. Supported: site:, "exact phrase", OR, -exclusion, intitle:, inurl:.
- Expand the role into real-world synonyms and job titles, INCLUDING the local language of the place in
  the intent (e.g. Germany: "Lehrer", "Trainer", "Ausbilder", "Dozent"; Gulf: English and common Indian
  spellings), plus the key skills / certifications for that role.
- Use the places, nationalities and destinations named in the intent — do not add unrelated countries.
- Combine role/synonym + place signals + interest signals ("interested", "looking for job", "open to
  work", "my CV", "gmail.com", "whatsapp", or the local-language equivalent).
- Vary phrasing and synonyms across queries so they return different pages.

Also return:
- role_keywords: 10–25 short job titles / synonyms / key skills (in English AND the local language) that
  identify a matching person on a page.
- locations: the places the people must be in or want to work in, ONLY if the intent restricts this
  (e.g. ["Germany"] for "trainers in Germany"); [] when the intent does not restrict location.
"""

# Order matters: round 1 takes the first `num_waves` entries, so source types are interleaved
# (a 3-wave run covers LinkedIn, Reddit/Quora and portals/forums, not just LinkedIn + Facebook).
WAVE_BLUEPRINT = [
    ("LinkedIn Hiring Posts & Comments", "linkedin_posts",
     "site:linkedin.com/posts hiring / opportunity posts for this role and place (recruiter and agency posts, "
     "\"urgent requirement\", \"we are hiring\", \"vacancy\" in the relevant language) whose comments hold "
     "people replying with interest and contact details. Short queries: site: + role + place or hiring phrase"),
    ("Reddit & Quora Discussions", "reddit_quora",
     "site:reddit.com and site:quora.com threads where people discuss their own plans to work in this role / place"),
    ("Job Portals, CVs & Forums", "portals_forums",
     "job-seeker pages, CV/resume listings and profession forums relevant to this role and country (pick portals "
     "used in that country, e.g. naukri.com / apna.co / shine.com for India, stepstone.de / indeed.de / xing.com "
     "for Germany, bayt.com / gulftalent.com for the Gulf) where people post their own profile"),
    ("Facebook & Job Groups", "facebook_groups",
     "site:facebook.com public posts in job groups for this role / place where people reply with WhatsApp "
     "numbers or emails"),
    ("Expat & Profession Forums", "forums",
     "public expat and profession forums, Q&A boards and blog comment sections for this role / place where people "
     "post their own experience and contact details. No site: on LinkedIn, Facebook, Reddit or Quora in this wave"),
    ("LinkedIn Profiles", "linkedin_profiles",
     "site:linkedin.com/in individual profiles for this role (and place) that are open to work / looking for a job"),
    ("Long-tail & Regional Sources", "long_tail",
     "city / region specific pages, alumni and training-institute pages, Telegram / WhatsApp group directories "
     "for this role and place"),
]

# Waves whose sites disallow crawlers in robots.txt. With "Respect robots.txt" on, only their search
# snippets can be read, so the plan puts the openly crawlable waves first.
ROBOTS_CLOSED_PLATFORMS = {"linkedin_posts", "linkedin_profiles", "facebook_groups", "reddit_quora"}

EXTRACT_SYSTEM = """You extract candidate leads for a recruiter from ONE web page.
The page may start with a structured "Post and comments" block (one line per author) and a
"Contact details found on the page" block listing every email / phone with its surrounding text.

Return every INDIVIDUAL PERSON on the page who matches the sourcing intent exactly as written:
- The author of a post or profile, AND every commenter — each commenter is a separate person.
- EXCLUDE companies, recruiters, agencies, the person advertising the job, and people only giving
  advice. A recruiter's own post is a source of candidates (its commenters), not a candidate.
- A person qualifies if the page supports that they match the intent's role (and place, when the intent
  names one). Replying to a job post for the role with interest ("interested", sharing a CV/email/number,
  asking how to apply) counts as relevance to that role, even if they don't restate their profession.
- shows_interest: true only if THIS person says they are interested, keen, looking / seeking / open to
  work, ready or willing to join / relocate, shares their CV, or asks how to apply. False otherwise.
- Do NOT infer nationality or location from a person's name. Fill current_location only if stated.
- evidence_snippet: copy a VERBATIM sentence from the page — preferably the person's own words — that
  proves the match (max ~300 characters). Do not paraphrase. Do not invent.
- email / phone: ONLY a contact detail that THIS SAME PERSON wrote in their own comment, post or profile
  on this page. Copy it exactly. Never give a person the recruiter's, company's or another commenter's
  contact. Null if they didn't post one. Phone includes WhatsApp / mobile numbers.
- name: the person's name or public handle. Fill other fields only if the page states them.
- skills: concrete skills, tools, subjects, certifications, languages.
- target_countries: countries/regions they want to work in, as stated or as given by the job post
  they replied to.
- If nobody qualifies, return {"candidates": []}.
"""


def _plan_prompt(intent: str, num_waves: int, queries_per_wave: int, round_no: int = 1,
                 exclude_queries: Optional[List[str]] = None, respect_robots: bool = False) -> str:
    order = WAVE_BLUEPRINT
    if respect_robots:
        order = sorted(WAVE_BLUEPRINT, key=lambda w: w[1] in ROBOTS_CLOSED_PLATFORMS)
    # Each new round starts further along the blueprint, so rounds cover different source types.
    offset = ((round_no - 1) * num_waves) % len(order)
    blueprint = (order[offset:] + order[:offset])[:num_waves]
    lines = [f"Sourcing intent: {intent.strip()}", "",
             f"Produce exactly {num_waves} waves, in this order, each with exactly {queries_per_wave} queries:"]
    for i, (name, platform, hint) in enumerate(blueprint, start=1):
        lines.append(f'Wave {i}: wave_name="{name}", platform="{platform}" — {hint}.')
    if exclude_queries:
        lines += ["", f"This is search round {round_no}. These queries were already run — do NOT repeat them "
                      "or near-copies; use different synonyms, machine brands, destinations, cities and phrasings:"]
        lines += [f"- {q}" for q in exclude_queries[-120:]]
    lines.append("")
    lines.append("Return JSON matching the schema: the waves, role_keywords and locations. No explanations.")
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
# Signs that a search snippet is a person talking about themselves (worth a Gemini call).
_CANDIDATE_SIGNAL = re.compile(
    r"interested|my cv|resume|\bcv\b|looking for (?:a )?(?:job|opportunit)|open to work|years? (?:of )?experience|"
    r"\bi am\b|\biam\b|\bi'm\b|\bmy (?:name|number|mail|email)|whatsapp|contact|@\w+\.\w+|\+?\d[\d\s-]{8,}\d",
    re.IGNORECASE)


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


_THREAD_LINE = re.compile(r"^(?:POST|COMMENT) by (.+?)(?: <(https?://[^>\s]+)>)?(?: \[(\d{4}-\d{2}-\d{2})\])?: (.*)$",
                          re.MULTILINE)
_PROFILE_URL = re.compile(r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/in/[^/?#\s]+", re.IGNORECASE)


def _thread_lines(content: str, name: Optional[str]) -> tuple[str, str]:
    """(this person's lines, everyone else's lines) from the structured post/comment block."""
    who = (name or "").strip().lower()
    own, others = [], []
    for author, _, _, body in _THREAD_LINE.findall(content):
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


def _profile_url(url: str, content: str, name: Optional[str]) -> Optional[str]:
    """The person's own profile link: the page itself for a profile page, else their thread author link."""
    if _PROFILE_URL.match(url):
        return url
    who = (name or "").strip().lower()
    for author, link, _, _ in _THREAD_LINE.findall(content):
        if link and who and author.strip().lower() == who:
            return link
    return None


_PAGE_DATE = re.compile(r"^Page date: (\d{4}-\d{2}-\d{2})$", re.MULTILINE)


def _activity_date(url: str, content: str, name: Optional[str], hit_date: Optional[str]) -> Optional[str]:
    """When this person was active: their own comment's date, else the post / page date,
    else the LinkedIn activity id in the URL, else the date the search engine showed."""
    who = (name or "").strip().lower()
    post_date = None
    for author, _, when, _ in _THREAD_LINE.findall(content):
        if when and who and author.strip().lower() == who:
            return when
        post_date = post_date or when
    m = _PAGE_DATE.search(content)
    return (post_date or (m.group(1) if m else None) or dates.from_linkedin_url(url) or hit_date)


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


async def _extract_page(gemini: Gemini, intent: str, url: str, content: str, snippet_only: bool,
                        mode: str = "ai", keywords: Optional[List[str]] = None,
                        hit_date: Optional[str] = None, locations: Optional[List[str]] = None
                        ) -> tuple[List[CandidateRecord], int, bool, int]:
    """Returns (valid records, number dropped as ungrounded/invalid, whether the AI was called,
    number dropped because they are not in / not heading to the intent's locations).

    mode: "rules" — regex/keyword extraction only, no AI tokens;
          "hybrid" — rules first, the AI only for pages where rules find nobody but the page
                     looks like it has candidates (contacts or candidate phrases);
          "ai" — the AI reads every page.
    """
    platform = platform_from_url(url)
    found: List[dict] = []
    used_ai = False
    if mode in ("rules", "hybrid"):
        found = rule_extractor.extract_people(content, url, keywords or [], snippet_only)
    if mode == "ai" or (mode == "hybrid" and not found and _CANDIDATE_SIGNAL.search(content)):
        result = await gemini.generate_structured(
            _extract_prompt(intent, url, platform, content, snippet_only),
            PageExtraction, system_instruction=EXTRACT_SYSTEM, temperature=0.1,
            max_retries=3,
        )
        found = [c.model_dump() for c in result.candidates]
        used_ai = True
    people, dropped, off_target = [], 0, 0
    post_text = " ".join(re.findall(r"^POST by .*?: (.*)$", content, re.MULTILINE))
    for d in found:
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
        if not d.get("profile_url"):
            d["profile_url"] = _profile_url(url, content, d.get("name"))
        d["activity_date"] = d.get("activity_date") or _activity_date(url, content, d.get("name"), hit_date)
        # Interest: the AI's judgement or the person's own words ("interested", "looking for a job", …).
        said = own or d.get("evidence_snippet") or ""
        d["shows_interest"] = bool(d.get("shows_interest")) or rule_extractor.shows_interest(said)
        if locations:
            where = " ".join([said, d.get("evidence_snippet") or "", d.get("current_location") or "",
                              " ".join(d.get("target_countries") or []), post_text,
                              content[:3000] if d.get("profile_url") == url else ""])
            if not rule_extractor.mentions_any(where, locations):
                off_target += 1
                continue
        people.append(d)
    records: List[CandidateRecord] = []
    for row in _assign_source_urls(url, people):
        try:
            records.append(CandidateRecord(**row, platform=platform))
        except Exception:
            dropped += 1
    return records, dropped, used_ai, off_target


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
async def plan_search(gemini: Gemini, intent: str, num_waves: int, queries_per_wave: int,
                      round_no: int = 1, exclude_queries: Optional[List[str]] = None,
                      respect_robots: bool = False) -> dict:
    plan = await gemini.generate_structured(
        _plan_prompt(intent, num_waves, queries_per_wave, round_no, exclude_queries, respect_robots),
        ComprehensiveSearchPlan,
        system_instruction=PLAN_SYSTEM, temperature=0.6 if round_no == 1 else 0.9, thinking_budget=512,
        max_retries=3,
    )
    done = {q.strip().lower() for q in exclude_queries or []}
    waves = [w for w in plan.waves if w.queries][:num_waves]
    for w in waves:
        w.queries = [q for q in w.queries if q.strip().lower() not in done][:queries_per_wave]
    waves = [w for w in waves if w.queries]
    return {"waves": [w.model_dump() for w in waves], "role_keywords": plan.role_keywords[:40],
            "locations": plan.locations[:10]}


def run_query(query: str, max_results: int, region: str, backend: str, keys: Optional[dict] = None,
              max_age_months: int = 0) -> dict:
    """backend: auto (DuckDuckGo + a Google API when one has a key) | duckduckgo | google."""
    keys = keys or {}
    token = scrapedo_token(keys)
    if token:
        keys = {**keys, "scrapedo": token}
    google_apis = integrations.search_available(keys)
    sources, errors, rate_limited = [], [], False
    hits: Dict[str, dict] = {}

    def add(outcome, label):
        nonlocal rate_limited
        sources.append(label if outcome.hits or not outcome.error else f"{label} ✗")
        rate_limited = rate_limited or outcome.rate_limited
        if outcome.error:
            errors.append(f"{label}: {outcome.error}")
        for h in outcome.hits:
            hit = hits.setdefault(h.url, {"url": h.url, "title": h.title, "snippet": h.snippet})
            hit["date"] = hit.get("date") or getattr(h, "date", None) or _hit_date(hit)

    if backend != "google" or not google_apis:
        ddg_backend = "duckduckgo" if backend == "duckduckgo" else "auto"
        add(search_query(query, max_results=max_results, region=region, backend=ddg_backend,
                         max_age_months=max_age_months), "ddg")
        if backend in ("auto", "google") and not google_apis:
            # No Google API key: try Google's own results page for free (often refused from cloud IPs).
            add(search_query(query, max_results=max_results, region=region, backend="google",
                             max_age_months=max_age_months), "google-free")
    # Google finds pages DuckDuckGo misses (Reddit, Quora, forums), so auto always adds it when possible.
    # The first configured API is used; if it fails (no credits, bad key) the next one is tried.
    if backend in ("auto", "google"):
        for name in google_apis:
            if name == "scrapedo":
                outcome = google_search_scrapedo(query, token, max_results=max_results, region=region)
            else:
                found, err, limited = integrations.web_search(name, keys, query, max_results, region,
                                                              max_age_months)
                outcome = _Outcome(found, err, limited)
            add(outcome, name)
            if not outcome.error:
                break
        if backend == "google" and google_apis and not hits:
            add(search_query(query, max_results=max_results, region=region, backend="auto",
                             max_age_months=max_age_months), "ddg")

    return {
        "query": query,
        "error": "; ".join(errors) if errors and not hits else None,
        "rate_limited": rate_limited,
        "sources": sources,
        "google_missing": backend in ("auto", "google") and not google_apis,
        "hits": list(hits.values())[:max_results * 2],
    }


MAX_ENRICH_PER_BATCH = 10


def _identifiable(r: CandidateRecord) -> bool:
    """Lead databases find people by LinkedIn profile, or by a real full name (then a LinkedIn search).
    Handles like 'shrikantsingh640' or 'soulpsychic_tarot' cannot be looked up."""
    if r.profile_url and "linkedin.com/in/" in r.profile_url:
        return True
    words = re.findall(r"[A-Za-z]{2,}", r.name or "")
    return len(words) >= 2 and not re.search(r"[_\d@]", r.name or "")


async def _enrich_records(records: List[CandidateRecord], keys: dict, require_both: bool) -> tuple[int, List[str]]:
    """Look up missing phone / email in the lead databases for every interested lead. With require_both,
    a looked-up contact is kept only when the lead then has BOTH a phone number and an email.
    Returns (leads filled, log notes) — the first note always says what happened."""
    wanting = [r for r in records if r.shows_interest and not (r.phone and r.email)]
    if not wanting:
        return 0, []
    if not integrations.summary(keys)["enrich"]:
        return 0, [f"Lead databases: {len(wanting)} interested leads are missing a phone / email, but no Apollo / "
                   "Lusha / ContactOut / RocketReach key is set."]
    todo = [r for r in wanting if _identifiable(r)][:MAX_ENRICH_PER_BATCH]
    no_identity = len([r for r in wanting if not _identifiable(r)])
    if not todo:
        return 0, [f"Lead databases: {len(wanting)} interested leads missing contacts, but none has a LinkedIn "
                   "profile or a real full name (only handles like 'user123'), so they cannot be looked up."]
    stopped: set = set()
    sem = asyncio.Semaphore(3)

    async def one(r: CandidateRecord):
        async with sem:
            return await integrations.enrich_person(
                keys, {"name": r.name, "linkedin_url": r.profile_url if "linkedin.com/in/" in (r.profile_url or "") else "",
                       "hints": " ".join(x for x in (r.current_role, r.current_location) if x)}, None, stopped)

    results = await asyncio.gather(*(one(r) for r in todo))
    done, partial, notes = 0, 0, set()
    for r, f in zip(todo, results):
        notes.update(f["errors"][:2])
        r.profile_url = r.profile_url or f.get("profile_url")
        email, phone = r.email or f["email"], r.phone or f["phone"]
        if (f["email"] or f["phone"]) and (not require_both or (email and phone)):
            r.email, r.phone = clean_email(email), clean_phone(phone)
            r.contact_source = f"enriched:{f['provider']}"
            done += 1
        elif f["email"] or f["phone"]:
            partial += 1
    summary = (f"Lead databases: {len(wanting)} interested leads missing contacts → looked up {len(todo)}"
               + (f" ({no_identity} skipped: no LinkedIn profile / full name)" if no_identity else "")
               + f" → filled {done}"
               + (f" ({partial} found only a phone or only an email — not kept because 'both phone & email' is ticked)"
                  if partial else ""))
    return done, [summary] + [f"Lead database: {n}" for n in list(notes)[:3]]


def _hit_date(hit: dict) -> Optional[str]:
    """The date the search engine showed for a result (API field, or the start of the snippet)."""
    return dates.parse(hit.get("date")) or dates.snippet_date(hit.get("snippet") or "")


class _Outcome:
    """integrations.web_search result in the shape run_query's add() expects."""

    def __init__(self, found: List[dict], error: Optional[str], rate_limited: bool):
        from search_module import SearchHit, canonicalize_url, is_useful_url
        self.error, self.rate_limited, self.hits = error, rate_limited, []
        for h in found:
            canon = canonicalize_url(h["url"])
            if canon and is_useful_url(canon):
                self.hits.append(SearchHit(url=canon, title=h["title"], snippet=h["snippet"], date=h.get("date")))


def dedup_urls(urls: List[str]) -> List[str]:
    return db.filter_fresh_urls(urls)


async def process_batch(gemini: Gemini, intent: str, items: List[dict], wave_tag: str,
                        page_timeout_s: int, respect_robots: bool, snippet_fallback: bool,
                        extraction: str = "rules", plan_queries: Optional[List[str]] = None,
                        keys: Optional[dict] = None, max_age_months: int = 0,
                        role_keywords: Optional[List[str]] = None, locations: Optional[List[str]] = None,
                        only_interested: bool = False, enrich: bool = False, require_both: bool = True) -> dict:
    """items: [{url, title, snippet}] — record, fetch, extract, save."""
    urls = [i["url"] for i in items]
    hits: Dict[str, dict] = {i["url"]: i for i in items}
    # Record right before crawling so a stopped run leaves unreached URLs unmarked.
    await asyncio.to_thread(db.record_scraped_urls, urls, wave_tag)

    outcomes = await fetch_batch(urls, page_timeout_s=page_timeout_s, respect_robots=respect_robots, keys=keys)
    stats = {"crawled": sum(o.ok for o in outcomes), "blocked": sum(o.blocked for o in outcomes)}
    stats["failed"] = len(outcomes) - stats["crawled"] - stats["blocked"]

    jobs = []
    for o in outcomes:
        if o.ok:
            jobs.append((o.url, o.markdown, False))
        elif snippet_fallback:
            h = hits.get(o.url, {})
            text = f"Title: {h.get('title', '')}\nSnippet: {h.get('snippet') or ''}".strip()
            # Only snippets that look like a person's own post are worth a Gemini call.
            if len(h.get("snippet") or "") > 40 and _CANDIDATE_SIGNAL.search(text):
                jobs.append((o.url, text, True))

    keywords = list(dict.fromkeys([k.lower() for k in role_keywords or [] if k.strip()] +
                                  rule_extractor.keywords_from(intent, plan_queries or [])))
    results = await asyncio.gather(
        *(_extract_page(gemini, intent, u, c, snip, extraction, keywords, _hit_date(hits.get(u, {})), locations)
          for (u, c, snip) in jobs),
        return_exceptions=True,
    )
    ai_pages = too_old = off_target = not_interested = 0
    records: List[CandidateRecord] = []
    warnings: List[str] = []
    dropped = 0
    quota_error: Optional[str] = None
    extracted, quota_hit = set(), set()
    for (u, _, _), r in zip(jobs, results):
        if isinstance(r, GeminiQuotaError):
            quota_error = str(r)
            quota_hit.add(u)
            continue
        if isinstance(r, Exception):
            if isinstance(r, GeminiError) and "API key" in str(r):
                raise r
            warnings.append(f"Extraction failed for {u}: {str(r)[:160]}")
            continue
        extracted.add(u)
        recs, d, used_ai, off = r
        dropped += d
        off_target += off
        ai_pages += used_ai
        fresh = [x for x in recs if not dates.older_than(x.activity_date, max_age_months)]
        too_old += len(recs) - len(fresh)
        if only_interested:
            keen = [x for x in fresh if x.shows_interest]
            not_interested += len(fresh) - len(keen)
            fresh = keen
        records.extend(fresh)

    enriched, enrich_notes = 0, []
    if enrich and records:
        enriched, enrich_notes = await _enrich_records(records, keys or {}, require_both)
        warnings.extend(enrich_notes)

    if records:
        await asyncio.to_thread(db.save_candidates, records)

    # How each URL ended. robots / quota are retried by later runs (see supabase_db).
    statuses = {}
    for o in outcomes:
        if o.url in quota_hit:
            statuses[o.url] = "quota"
        elif o.error == "disallowed by robots.txt":
            statuses[o.url] = "robots"
        elif o.ok:
            statuses[o.url] = "ok"
        elif o.url in extracted:
            statuses[o.url] = "snippet"
        else:
            statuses[o.url] = "walled" if o.blocked else "failed"
    await asyncio.to_thread(db.set_url_status, statuses)

    reasons: Dict[str, int] = {}
    for o in outcomes:
        if o.error:
            key = "disallowed by robots.txt" if o.error == "disallowed by robots.txt" else o.error.split(";")[0][:60]
            reasons[key] = reasons.get(key, 0) + 1

    return {
        "stats": {**stats, "records": len(records), "dropped": dropped, "sent_to_gemini": ai_pages, "pages_read": len(jobs), "too_old": too_old,
                  "off_target": off_target, "not_interested": not_interested, "enriched": enriched,
                  "interested": sum(1 for r in records if r.shows_interest),
                  "dated": sum(1 for r in records if r.activity_date),
                  "with_phone": sum(1 for r in records if r.phone),
                  "with_email": sum(1 for r in records if r.email),
                  "via_scrapedo": sum(1 for o in outcomes if o.via != "direct" and o.ok),
                  "via": {v: sum(1 for o in outcomes if o.via == v and o.ok)
                          for v in {o.via for o in outcomes if o.via != "direct" and o.ok}}},
        "records": [r.model_dump() for r in records],
        "errors": {o.url: o.error for o in outcomes if o.error},
        "block_reasons": reasons,
        "warnings": warnings,
        "quota_error": quota_error,
    }
