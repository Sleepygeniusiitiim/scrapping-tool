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
import os
import re
from typing import Dict, List, Optional, Tuple

import supabase_db as db
import rule_extractor
import dates
import integrations
from fetcher import PHONE_RE, fetch_batch, scrapedo_token
from gemini_client import Gemini, GeminiError, GeminiQuotaError
from schema import CandidateRecord, ComprehensiveSearchPlan, PageExtraction, clean_email, clean_phone
from search_module import google_search_scrapedo, platform_from_url, search_query

# Page text sent to the AI per page. Long pages are cut down to their start plus the lines that carry contact
# details or "interested / CV / looking for a job" signals (see _focus), which is where candidates are.
MAX_CONTENT_CHARS_FOR_LLM = int(os.getenv("LLM_PAGE_CHARS", "20000") or 20000)
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
  "Interested candidates send CV to …" is the RECRUITER speaking — never interest.
- person_type: "candidate" for a job seeker; "recruiter" for HR, recruiters, agencies, consultancies,
  employers or anyone posting / advertising the job (their profile headline says HR / Talent Acquisition /
  Recruiter / hiring, or they ask others to send CVs); "other" for anyone else. If unsure whether someone is a
  recruiter, prefer "recruiter" when their own words describe a job rather than themselves.
- Do NOT infer nationality or location from a person's name. Fill current_location only if stated
  (city and country as written, e.g. "Lahore, Pakistan").
- If the intent names the candidates' nationality / home country (e.g. "Indian candidates"), EXCLUDE people
  whose page states another nationality, home country or home city (e.g. Pakistani, Lahore, Dhaka, Nepal,
  a +92 / +880 / +977 number).
- If the intent asks for experienced people ("with experience", "experienced", "X years", "worked in
  companies"), include only people whose page shows work experience: years, past / current employers,
  projects or sites.
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


def _extract_prompt(intent: str, url: str, platform: str, content: str, snippet_only: bool,
                    keywords: Optional[List[str]] = None) -> str:
    note = ("NOTE: The page could not be opened (login wall / blocked). The content below is ONLY the "
            "search-engine title and snippet for this URL. Extract only what it explicitly states.\n\n"
            if snippet_only else "")
    return (f"Sourcing intent: {intent.strip()}\n"
            f"Page URL: {url}\nPlatform: {platform}\n\n{note}"
            f"----- PAGE CONTENT -----\n{_focus(content, MAX_CONTENT_CHARS_FOR_LLM, keywords)}\n"
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


def _focus(content: str, budget: int, keywords: Optional[List[str]] = None) -> str:
    """The page start plus every line (with its neighbours) that looks like a person, a contact or an intent
    signal, in page order, within `budget` characters — instead of blindly cutting long pages."""
    if len(content) <= budget:
        return content
    head = min(2500, budget // 4)
    lines = content[head:].split("\n")
    kw = [k.lower() for k in keywords or [] if len(k) > 2]
    keep = set()
    for i, line in enumerate(lines):
        low = line.lower()
        if _CANDIDATE_SIGNAL.search(line) or any(k in low for k in kw):
            keep.update((i - 1, i, i + 1))
    out, used = [content[:head], "\n[…]"], head + 5
    last = -2
    for i in sorted(k for k in keep if 0 <= k < len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if used + len(line) + 6 > budget:
            break
        if i != last + 1:
            out.append("[…]")
        out.append(line)
        used += len(line) + 6
        last = i
    return "\n".join(out)


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
                        hit_date: Optional[str] = None, locations: Optional[List[str]] = None,
                        counts: Optional[Dict[str, int]] = None
                        ) -> tuple[List[CandidateRecord], int, bool, int]:
    """Returns (valid records, number dropped as ungrounded/invalid, whether the AI was called,
    number dropped because they are not in / not heading to the intent's locations).
    `counts["recruiters"]` (when given) is raised for every HR / recruiter / job ad left out.

    mode: "rules" — regex/keyword extraction only, no AI tokens;
          "hybrid" — rules first, the AI only for pages where rules find nobody but the page
                     looks like it has candidates (contacts or candidate phrases);
          "ai" — the AI reads every page.
    """
    origin = rule_extractor.origin_of(intent)
    platform = platform_from_url(url)
    found: List[dict] = []
    used_ai = False
    if mode in ("rules", "hybrid"):
        found = rule_extractor.extract_people(content, url, keywords or [], snippet_only)
    if mode == "ai" or (mode == "hybrid" and not found and _CANDIDATE_SIGNAL.search(content)):
        result = await gemini.generate_structured(
            _extract_prompt(intent, url, platform, content, snippet_only, keywords),
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
        # Candidate or hiring side? The AI's person_type, or the person's own words / name / profile headline.
        headline = content[:600] if d.get("profile_url") == url else ""
        if d.pop("person_type", "candidate") == "recruiter" or \
                rule_extractor.is_recruiter(said, d.get("name") or "") or \
                (headline and rule_extractor._HR_SELF.search(headline)):
            if counts is not None:
                counts["recruiters"] = counts.get("recruiters", 0) + 1
            continue                                  # HR / agency / job ad: a source of candidates, not one
        d["shows_interest"] = bool(d.get("shows_interest")) or rule_extractor.shows_interest(said)
        if origin:
            mine = " ".join([said, d.get("evidence_snippet") or "", d.get("current_location") or "",
                             d.get("phone") or "", content[:3000] if d.get("profile_url") == url else ""])
            if rule_extractor.other_origin(mine, origin):
                off_target += 1
                continue
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
    """backend: auto (DuckDuckGo + a Google API when one has a key) | duckduckgo | google |
    all (every configured engine in parallel, results merged by reciprocal-rank fusion)."""
    keys = keys or {}
    if backend == "all":
        return _run_query_fanout(query, max_results, region, keys, max_age_months)
    token = scrapedo_token(keys)
    if token:
        keys = {**keys, "scrapedo": token}
    google_apis = integrations.search_available(keys)
    sources, errors, rate_limited = [], [], False
    hits: Dict[str, dict] = {}
    extras: dict = {"entities": [], "questions": [], "related": []}

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
                if outcome.error and not outcome.hits and (outcome.rate_limited or "rejected" in outcome.error):
                    integrations.mark_exhausted("scrapedo", outcome.error)
            else:
                found, err, limited = integrations.web_search(name, keys, query, max_results, region,
                                                              max_age_months, extras)
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
        "engine_errors": errors,
        "google_missing": backend in ("auto", "google") and not google_apis,
        "hits": list(hits.values())[:max_results * 2],
        **_dedupe_extras(extras),
    }


def _dedupe_extras(extras: dict) -> dict:
    seen, ents = set(), []
    for e in extras.get("entities", []):
        k = re.sub(r"\W+", "", e.get("name", "").lower())
        if k and k not in seen:
            seen.add(k)
            ents.append(e)
    return {"entities": ents, "related": list(dict.fromkeys(q for q in extras.get("related", []) if q))[:8],
            "questions": [q for q in extras.get("questions", []) if q.get("question")][:8]}


def _run_query_fanout(query: str, max_results: int, region: str, keys: dict, max_age_months: int) -> dict:
    """Every configured engine at once (DuckDuckGo, Serper, Google CSE, SerpApi, Brave, Scrape.do) — each one
    indexes pages the others miss. Results are merged with reciprocal-rank fusion: a page ranked high by
    several engines comes first."""
    from concurrent.futures import ThreadPoolExecutor
    token = scrapedo_token(keys)
    if token:
        keys = {**keys, "scrapedo": token}
    engines = ["ddg"] + integrations.search_available(keys)
    extras: dict = {"entities": [], "questions": [], "related": []}

    def one(name: str):
        if name == "ddg":
            r = search_query(query, max_results=max_results, region=region, backend="auto",
                             max_age_months=max_age_months)
            return name, [{"url": h.url, "title": h.title, "snippet": h.snippet, "date": getattr(h, "date", None)}
                          for h in r.hits], r.error
        if name == "scrapedo":
            r = google_search_scrapedo(query, token, max_results=max_results, region=region)
            return name, [{"url": h.url, "title": h.title, "snippet": h.snippet} for h in r.hits], r.error
        found, err, _ = integrations.web_search(name, keys, query, max_results, region, max_age_months, extras)
        o = _Outcome(found, err, False)
        return name, [{"url": h.url, "title": h.title, "snippet": h.snippet, "date": getattr(h, "date", None)}
                      for h in o.hits], err

    with ThreadPoolExecutor(max_workers=len(engines)) as ex:
        results = list(ex.map(lambda n: _safe_engine(one, n), engines))
    fused: Dict[str, dict] = {}
    score: Dict[str, float] = {}
    sources, errors = [], []
    for name, hits, err in results:
        sources.append(name if hits or not err else f"{name} ✗")
        if err and not hits:
            errors.append(f"{name}: {err}")
        for rank, h in enumerate(hits):
            score[h["url"]] = score.get(h["url"], 0.0) + 1.0 / (60 + rank)
            cur = fused.setdefault(h["url"], {**h, "engines": []})
            cur["engines"].append(name)
            cur["snippet"] = cur.get("snippet") or h.get("snippet")
            cur["date"] = cur.get("date") or h.get("date") or _hit_date(h)
    ranked = sorted(fused.values(), key=lambda h: -score[h["url"]])
    return {"query": query, "error": "; ".join(errors) if errors and not ranked else None, "rate_limited": False,
            "sources": sources, "engine_errors": errors, "google_missing": len(engines) == 1, "hits": ranked[:max_results * 3],
            **_dedupe_extras(extras)}


def _safe_engine(fn, name):
    try:
        return fn(name)
    except Exception as exc:
        return name, [], f"{type(exc).__name__}: {str(exc)[:120]}"


def _scaled(n: int) -> int:
    """Per-batch limits: a background worker (WORKER_SCALE, default 3 there) does more per batch than a
    web request that must finish within the serverless time limit."""
    return max(1, int(n * float(os.getenv("WORKER_SCALE", "1") or 1)))


MAX_ENRICH_PER_BATCH = _scaled(10)


def _identifiable(r: CandidateRecord) -> bool:
    """Lead databases need to know exactly who the person is: their LinkedIn profile."""
    return bool(r.profile_url and "linkedin.com/in/" in r.profile_url)


async def _visit_profiles(records: List[CandidateRecord], respect_robots: bool) -> dict:
    """Open each interested commenter's own public profile (Instagram, X, forum profile, …) to get their real
    name, a LinkedIn link from their bio, and any phone / email they published there."""
    import profile_visit
    targets = {r.profile_url: r for r in records
               if r.profile_url and "linkedin.com/in/" not in r.profile_url and "reddit.com/user/" not in r.profile_url}
    if not targets:
        return {"visited": 0}
    found = await profile_visit.visit(list(targets), respect_robots=respect_robots)
    stats = {"visited": len(targets), "named": 0, "linkedin": 0, "contacts": 0, "walled": 0}
    for url, r in targets.items():
        p = found.get(url) or {}
        if p.get("status") != "ok":
            stats["walled"] += 1
            continue
        if p.get("name") and (not r.name or re.search(r"[_\d.]", r.name) or len(r.name.split()) < 2):
            r.name = p["name"]
            stats["named"] += 1
        if p.get("linkedin"):
            r.profile_url = p["linkedin"]
            stats["linkedin"] += 1
        email = next(iter(p.get("emails") or []), None)
        phone = next(iter(p.get("phones") or []), None)
        if (email and not r.email) or (phone and not r.phone):
            r.email, r.phone = r.email or clean_email(email), r.phone or clean_phone(phone)
            r.contact_source = r.contact_source or "profile_page"
            stats["contacts"] += 1
    return stats


MAX_BIO_SEARCHES = _scaled(8)
MAX_EMAIL_GUESSES = _scaled(3)


async def _search_bios(records: List[CandidateRecord], keys: dict, notes: List[str]) -> Dict[int, str]:
    """Search engines index public profile pages even when they show servers a login wall. Use ONLY the result
    for the person's exact profile URL: their name, bio (phone / email they published) and, for LinkedIn,
    their current company (for the work-email guess). Returns {id(record): company}."""
    import social_lookup
    targets = {r.profile_url: r for r in records if r.profile_url and "reddit.com/user/" not in r.profile_url}
    if not targets:
        return {}
    found = await social_lookup.search_profiles(keys, list(targets), MAX_BIO_SEARCHES)
    companies, got = {}, 0
    for url, res in found.items():
        r = targets[url]
        if res.get("name") and (not r.name or re.search(r"[_\d.]", r.name) or len(r.name.split()) < 2 or
                                (len(res["name"]) > len(r.name) and res["name"].lower().startswith(r.name.split()[0].lower()))):
            r.name = res["name"]
        phone, email = next(iter(res.get("phones") or []), None), next(iter(res.get("emails") or []), None)
        if (phone and not r.phone) or (email and not r.email):
            r.phone, r.email = r.phone or clean_phone(phone), r.email or clean_email(email)
            r.contact_source = r.contact_source or "profile_bio"
            got += 1
        if res.get("company"):
            companies[id(r)] = res["company"]
            r.current_role = r.current_role or res.get("headline")
    notes.append(f"Profile bios (search results for their exact profile): {len(targets)} searched → "
                 f"{len(found)} profiles found → {got} contacts from bios, {len(companies)} employers identified")
    return companies


MAX_DOMAIN_LOOKUPS = _scaled(4)
MAX_VERIFY_PER_BATCH = _scaled(15)
_EMPLOYER = re.compile(r"\b(?:work(?:ing|s)?|employed|job|nurse|engineer|operator|driver|teacher|trainer|manager|"
                       r"executive|technician|staff)\s+(?:at|with|in)\s+([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,4})")
_ROLE_AT = re.compile(r"\s(?:at|@)\s+([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z0-9][A-Za-z0-9&.'-]*){0,4})")


def _employer(r: CandidateRecord) -> Optional[str]:
    """Employer named in the person's own headline ("Staff Nurse at Fortis Hospital") or post ("working at …")."""
    for rx, text in ((_ROLE_AT, r.current_role or ""), (_EMPLOYER, r.evidence_snippet or "")):
        m = rx.search(" " + text)
        if m and not re.fullmatch(r"(?:India|Home|Present|Dubai|Germany|UAE|Canada|UK|USA)", m.group(1).strip()):
            return m.group(1).strip()
    return None


async def _company_profiles(records: List[CandidateRecord], companies: Dict[int, str], keys: dict,
                            respect_robots: bool, notes: List[str]) -> Dict[int, dict]:
    """Waterfall step: the employer's website → mail domain, email format, and the person's own address if
    the company publishes it (a real contact). One crawl per company. Returns {id(record): profile}."""
    import company_contacts
    import email_patterns
    for r in records:                               # employer from the person's own text when no bio gave one
        if id(r) not in companies and (emp := _employer(r)):
            companies[id(r)] = emp
    wanted = list(dict.fromkeys(companies[id(r)] for r in records if id(r) in companies and not r.email))
    wanted = wanted[:MAX_DOMAIN_LOOKUPS]
    if not wanted:
        return {}
    res = await asyncio.gather(*(company_contacts.company_profile(keys, c, respect_robots) for c in wanted),
                               return_exceptions=True)
    by_company = {c: p for c, p in zip(wanted, res) if isinstance(p, dict)}
    out, published = {}, 0
    for r in records:
        prof = by_company.get(companies.get(id(r), ""))
        if not prof:
            continue
        out[id(r)] = prof
        parts = email_patterns.name_parts(r.name or "")
        if r.email or not parts:
            continue
        for person in prof["people"]:              # the company lists this person with their address
            if person.get("email") and email_patterns.name_parts(person.get("name", "")) == parts:
                r.email = clean_email(person["email"])
                r.contact_source = r.contact_source or "company_website"
                published += 1
                break
    notes.append(f"Employer websites: {len(wanted)} employers → {len(by_company)} sites found → "
                 f"{sum(1 for p in by_company.values() if p['domain_from_emails'])} mail domains confirmed, "
                 f"{sum(1 for p in by_company.values() if p['format'])} email formats learned, "
                 f"{published} leads' own addresses published there")
    return out


def _candidates_for(name: str, domain: str, pattern: Optional[dict]) -> List[Tuple[str, str]]:
    """Addresses to test: the company's known format first, then the most common formats."""
    import email_patterns
    parts = email_patterns.name_parts(name)
    if not parts:
        return []
    order = ([pattern["format"]] if pattern else []) + ["first.last", "first", "firstlast", "flast", "f.last",
                                                          "first_last", "last.first", "firstl"]
    seen, out = set(), []
    for fmt in order:
        e = f"{email_patterns.render(fmt, *parts)}@{domain}"
        if e not in seen:
            seen.add(e)
            out.append((fmt, e))
    return out[:7]


async def _guess_work_emails(records: List[CandidateRecord], profiles: Dict[int, dict], keys: dict,
                             notes: List[str]) -> None:
    """LAST RESORT, only for leads who are working (employer known) and for whom no phone number and no email
    was found anywhere else: build the likely addresses from their name and the employer's mail domain, then
    test them with a zero-send SMTP check. A mailbox the server confirms is reported as verified; on a catch-all
    domain only the format the company itself uses is offered, marked inconclusive."""
    import email_verify
    todo = [r for r in records if id(r) in profiles and not r.email_guess and not r.phone and not r.email
            ][:MAX_EMAIL_GUESSES]
    if not todo:
        return
    stats = {"verified": 0, "catch_all": 0, "rejected": 0, "unverified": 0}
    for r in todo:
        prof = profiles[id(r)]
        cands = _candidates_for(r.name or "", prof["domain"], prof["format"])
        if not cands:
            continue
        res = await email_verify.verify_many([e for _, e in cands], keys)
        valid = [(f, e) for f, e in cands if res.get(e, {}).get("status") == "valid"]
        fmt_guess = cands[0] if prof["format"] else None
        if valid:
            f, e = valid[0]
            r.email_guess = f"{e} (mailbox verified by SMTP, format {f}; not published by the person)"
            stats["verified"] += 1
        elif all(res.get(e, {}).get("status") in ("invalid", "no_mail") for _, e in cands):
            stats["rejected"] += 1                  # every likely address bounced: no guess at all
        elif fmt_guess and any(res.get(e, {}).get("status") == "catch_all" for _, e in cands):
            r.email_guess = (f"{fmt_guess[1]} (guessed, {prof['format']['confidence']} confidence, format "
                             f"{fmt_guess[0]}; domain accepts all mail — cannot be verified)")
            stats["catch_all"] += 1
        elif fmt_guess and res.get(fmt_guess[1], {}).get("status") != "invalid":
            r.email_guess = (f"{fmt_guess[1]} (guessed, {prof['format']['confidence']} confidence, format "
                             f"{fmt_guess[0]}; mailbox not verified)")
            stats["unverified"] += 1
    notes.append(f"Work-email guesses (last resort — employed, no phone / email anywhere): {len(todo)} leads → "
                 f"{stats['verified']} mailboxes confirmed by SMTP, {stats['catch_all']} on catch-all domains "
                 f"(inconclusive), {stats['unverified']} unverified format guesses, {stats['rejected']} with every "
                 "likely address rejected (no guess given)")


async def _verify_found(records: List[CandidateRecord], keys: dict, notes: List[str]) -> None:
    """Check every email the waterfall produced: DNS / MX, zero-send SMTP, catch-all and disposable flags."""
    import email_verify
    todo = [r for r in records if r.email and not r.email_status][:MAX_VERIFY_PER_BATCH]
    if not todo:
        return
    res = await email_verify.verify_many([r.email for r in todo], keys)
    counts: Dict[str, int] = {}
    for r in todo:
        v = res.get(r.email)
        if v:
            r.email_status = email_verify.label(v)
            counts[v["status"]] = counts.get(v["status"], 0) + 1
    smtp = any((res.get(r.email) or {}).get("method", "").startswith("smtp") for r in todo)
    notes.append("Email check (DNS / MX" + (" + SMTP" if smtp else "; SMTP port 25 not reachable from this server")
                 + f"): {len(todo)} addresses → " + ", ".join(f"{n} {k.replace('_', '-')}" for k, n in counts.items()))


async def _enrich_records(records: List[CandidateRecord], keys: dict, require_both: bool,
                          respect_robots: bool = True) -> tuple[int, List[str]]:
    """Waterfall enrichment for every interested lead whose page gave no complete contact. Each step runs only
    for the leads the previous steps left without one:
        1. their own profile page        (real name, LinkedIn link, published phone / email)
        2. search result of that profile (bio contacts, employer from the headline)
        3. employer's website            (mail domain, email format, the person's address if published)
        4. lead databases                (LinkedIn URL, or name + company / domain: ContactOut, Lusha,
                                          RocketReach, Apollo, Hunter)
        5. work-email guess + SMTP check (last resort: employed, nothing found anywhere)
        6. verification of every email   (DNS / MX, SMTP mailbox check, catch-all, disposable)
    With require_both, a looked-up contact is kept only when the lead then has BOTH a phone and an email.
    Returns (leads filled from databases, log notes)."""
    import email_patterns
    wanting = [r for r in records if r.shows_interest and not (r.phone and r.email)]
    notes_out: List[str] = []
    if not wanting:
        await _verify_found(records, keys, notes_out)
        return 0, notes_out
    missing = lambda rs: [r for r in rs if not (r.phone and r.email)]
    pv = await _visit_profiles(wanting[:MAX_ENRICH_PER_BATCH * 2], respect_robots)
    if pv.get("visited"):
        notes_out.append(f"Profiles: opened {pv['visited']} commenters' own profiles → {pv['named']} real names, "
                         f"{pv['linkedin']} LinkedIn links, {pv['contacts']} published contacts"
                         + (f" ({pv['walled']} behind a login wall / blocked)" if pv["walled"] else ""))
    wanting = missing(wanting)
    companies = await _search_bios(wanting, keys, notes_out) if wanting else {}
    wanting = missing(wanting)
    profiles = await _company_profiles(wanting, companies, keys, respect_robots, notes_out) if wanting else {}
    wanting = missing(wanting)
    done = 0
    if wanting and not integrations.summary(keys)["enrich"]:
        notes_out.append(f"Lead databases: {len(wanting)} interested leads still miss a phone / email, but no "
                         "Apollo / Lusha / ContactOut / RocketReach / Hunter key is set.")
    elif wanting:
        def lookup_ok(r):
            return _identifiable(r) or (email_patterns.name_parts(r.name or "") and
                                        (id(r) in profiles or id(r) in companies))
        todo = [r for r in wanting if lookup_ok(r)][:MAX_ENRICH_PER_BATCH]
        skipped = len([r for r in wanting if not lookup_ok(r)])
        stopped: set = set()
        sem = asyncio.Semaphore(3)

        async def one(r: CandidateRecord):
            prof = profiles.get(id(r)) or {}
            async with sem:
                return await integrations.enrich_person(
                    keys, {"name": r.name, "linkedin_url": r.profile_url if _identifiable(r) else None,
                           "company": companies.get(id(r)), "domain": prof.get("domain")}, None, stopped)

        results = await asyncio.gather(*(one(r) for r in todo))
        partial, notes = 0, set()
        for r, f in zip(todo, results):
            notes.update(f["errors"][:2])
            email, phone = r.email or f["email"], r.phone or f["phone"]
            if (f["email"] or f["phone"]) and (not require_both or (email and phone)):
                r.email, r.phone = clean_email(email), clean_phone(phone)
                r.contact_source = f"enriched:{f['provider']}"
                done += 1
            elif f["email"] or f["phone"]:
                partial += 1
        by_li = sum(1 for r in todo if _identifiable(r))
        notes_out.append(
            f"Lead databases: {len(wanting)} interested leads missing contacts → looked up {len(todo)} "
            f"({by_li} by LinkedIn profile, {len(todo) - by_li} by name + employer / domain)"
            + (f", {skipped} skipped (no LinkedIn and no employer — too ambiguous to look up)" if skipped else "")
            + f" → filled {done}"
            + (f" ({partial} found only a phone or only an email — not kept because 'both phone & email' is ticked)"
               if partial else ""))
        notes_out += [f"Lead database: {n}" for n in list(notes)[:3]]
    await _guess_work_emails(missing(wanting), profiles, keys, notes_out)     # last step, after lookups
    await _verify_found(records, keys, notes_out)
    return done, notes_out


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
                        only_interested: bool = False, enrich: bool = False, require_both: bool = True,
                        category: str = "") -> dict:
    """items: [{url, title, snippet}] — record, fetch, extract, save (tagged with `category` when given)."""
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
    counts: Dict[str, int] = {}
    results = await asyncio.gather(
        *(_extract_page(gemini, intent, u, c, snip, extraction, keywords, _hit_date(hits.get(u, {})), locations,
                        counts)
          for (u, c, snip) in jobs),
        return_exceptions=True,
    )
    # Every AI provider out of credits: read those pages with the rules instead of dropping them (the AI
    # re-reads them in a later run — their status stays "quota").
    ai_off = next((str(r) for r in results if isinstance(r, GeminiQuotaError)), None)
    if ai_off:
        redo = [i for i, r in enumerate(results) if isinstance(r, GeminiQuotaError)]
        again = await asyncio.gather(
            *(_extract_page(gemini, intent, jobs[i][0], jobs[i][1], jobs[i][2], "rules", keywords,
                            _hit_date(hits.get(jobs[i][0], {})), locations, counts) for i in redo),
            return_exceptions=True)
        for i, r in zip(redo, again):
            results[i] = r
        quota_hit_rules = {jobs[i][0] for i in redo}
    else:
        quota_hit_rules = set()
    ai_pages = too_old = off_target = not_interested = 0
    records: List[CandidateRecord] = []
    warnings: List[str] = []
    dropped = 0
    quota_error: Optional[str] = None
    extracted, quota_hit = set(), set(quota_hit_rules)
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
        enriched, enrich_notes = await _enrich_records(records, keys or {}, require_both, respect_robots)
        warnings.extend(enrich_notes)

    if records:
        await asyncio.to_thread(db.save_candidates, records)
        if category:
            import categories
            await asyncio.to_thread(categories.tag_candidates, [r.source_url for r in records], category)

    # Pages with many contacts but (almost) no candidates for this search → 💡 Suggested sites.
    import suggestions
    per_page: Dict[str, int] = {}
    for rec in records:
        base = (rec.source_url or "").split("#candidate-")[0]
        per_page[base] = per_page.get(base, 0) + 1
    suggested = await asyncio.to_thread(suggestions.record, [
        {"url": o.url, "title": (hits.get(o.url) or {}).get("title") or "", "text": o.markdown,
         "matched": per_page.get(o.url, 0)} for o in outcomes if o.ok], intent)

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
                  "recruiters": counts.get("recruiters", 0),
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
        "ai_off": ai_off,
        "suggested": [{k: r[k] for k in ("url", "title", "n_contacts", "matched")} for r in suggested],
    }
