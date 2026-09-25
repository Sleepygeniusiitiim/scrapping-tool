"""
Orchestrator — the autonomous sourcing loop.

    intent ──► Gemini plan (multi-wave dorks)
                 │
                 ▼  for each wave, sequentially
         search (jittered) ─► canonicalise ─► Supabase dedup ─► record URLs
                 ─► Crawl4AI batch ─► Gemini extraction ─► grounding check
                 ─► Supabase upsert ─► yield events to UI

`run_pipeline()` is an async generator of PipelineEvent objects.
`BackgroundRun` runs that generator on its own thread + event loop so the
Streamlit script can poll it without blocking, and can be stopped cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import random
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import supabase_db as db
from crawler_module import CrawlOutcome, crawl_in_batches
from gemini_client import DEFAULT_MODEL, Gemini, GeminiError
from schema import CandidateRecord, ComprehensiveSearchPlan, PageExtraction, SearchWave
from search_module import (
    JITTER_RANGE,
    SearchHit,
    domain_of,
    platform_from_url,
    search_query,
)

log = logging.getLogger(__name__)

MAX_CONTENT_CHARS_FOR_LLM = 30_000
GROUNDING_MIN_OVERLAP = 0.6   # share of evidence words that must appear on the page


# ---------------------------------------------------------------------------
# Settings & events
# ---------------------------------------------------------------------------
@dataclass
class PipelineSettings:
    gemini_api_key: str
    supabase_url: str
    supabase_key: str
    model: str = DEFAULT_MODEL
    gemini_mode: str = "auto"            # auto | gemini | vertex
    num_waves: int = 3
    queries_per_wave: int = 5
    max_results_per_query: int = 10
    max_urls_per_wave: int = 30          # cap on fresh URLs crawled per wave
    batch_size: int = 5
    page_timeout_s: int = 30
    respect_robots: bool = True
    snippet_fallback: bool = True        # use SERP snippet when a page is walled
    region: str = "in-en"
    search_backend: str = "auto"         # auto | duckduckgo
    pause_between_waves_s: float = 4.0


@dataclass
class PipelineEvent:
    kind: str                 # plan | wave_start | query | dedup | crawl | extract | saved | wave_end | info | warning | error | done
    message: str
    wave: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return asdict(self)


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


def _plan_prompt(intent: str, num_waves: int, queries_per_wave: int) -> str:
    lines = [f"Sourcing intent: {intent.strip()}", "",
             f"Produce exactly {num_waves} waves, in this order, each with exactly {queries_per_wave} queries:"]
    for i, (name, platform, hint) in enumerate(WAVE_BLUEPRINT[:num_waves], start=1):
        lines.append(f'Wave {i}: wave_name="{name}", platform="{platform}" — {hint}.')
    lines.append("")
    lines.append("Return JSON matching the schema. Queries only — no explanations.")
    return "\n".join(lines)


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


def _extract_prompt(intent: str, url: str, platform: str, content: str, snippet_only: bool) -> str:
    note = ("NOTE: The page could not be opened (login wall / blocked). The content below is ONLY the "
            "search-engine title and snippet for this URL. Extract only what it explicitly states.\n\n"
            if snippet_only else "")
    return (f"Sourcing intent: {intent.strip()}\n"
            f"Page URL: {url}\nPlatform: {platform}\n\n{note}"
            f"----- PAGE CONTENT (markdown) -----\n{content[:MAX_CONTENT_CHARS_FOR_LLM]}\n"
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
    """
    `candidates.source_url` is UNIQUE. A profile page has one person, but a
    Reddit/Quora thread can hold several. The first person keeps the page URL;
    additional people get a stable '#candidate-<name>' fragment of the same URL
    so every lead is kept and still opens the right page.
    """
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


async def _search_wave(wave: SearchWave, s: PipelineSettings, stop: threading.Event):
    """Run a wave's queries sequentially with jitter; yields (event, hits) per query."""
    for i, q in enumerate(wave.queries):
        if stop.is_set():
            return
        outcome = await asyncio.to_thread(
            search_query, q, s.max_results_per_query, s.region, s.search_backend
        )
        yield outcome
        if i < len(wave.queries) - 1:
            await asyncio.sleep(random.uniform(*JITTER_RANGE))


async def _extract_page(
    gemini: Gemini, intent: str, url: str, content: str, snippet_only: bool
) -> tuple[List[CandidateRecord], int]:
    """Returns (valid records, number dropped as ungrounded/invalid)."""
    platform = platform_from_url(url)
    result = await gemini.generate_structured(
        _extract_prompt(intent, url, platform, content, snippet_only),
        PageExtraction,
        system_instruction=EXTRACT_SYSTEM,
        temperature=0.1,
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
# Main pipeline
# ---------------------------------------------------------------------------
async def run_pipeline(
    intent: str, s: PipelineSettings, stop: Optional[threading.Event] = None
) -> AsyncIterator[PipelineEvent]:
    stop = stop or threading.Event()
    totals = {"found": 0, "duplicates": 0, "crawled": 0, "blocked": 0, "records": 0}

    # --- init clients --------------------------------------------------
    try:
        db.init_supabase(s.supabase_url, s.supabase_key)
        gemini = Gemini(s.gemini_api_key, model=s.model, mode=s.gemini_mode)
    except Exception as exc:
        yield PipelineEvent("error", f"Setup failed: {exc}")
        return

    # --- Step 1: plan --------------------------------------------------
    yield PipelineEvent("info", "Asking Gemini for a multi-wave search plan…")
    try:
        plan = await gemini.generate_structured(
            _plan_prompt(intent, s.num_waves, s.queries_per_wave),
            ComprehensiveSearchPlan,
            system_instruction=PLAN_SYSTEM,
            temperature=0.6,
            thinking_budget=512,
        )
    except GeminiError as exc:
        yield PipelineEvent("error", f"Planning failed: {exc}")
        return

    waves = [w for w in plan.waves if w.queries][: s.num_waves]
    for w in waves:
        w.queries = w.queries[: s.queries_per_wave]
    if not waves:
        yield PipelineEvent("error", "Gemini returned an empty plan — try rephrasing the intent.")
        return
    yield PipelineEvent("plan", f"Plan ready: {len(waves)} waves, {sum(len(w.queries) for w in waves)} queries.",
                        data={"plan": plan.model_dump()})

    seen_this_run: set[str] = set()

    # --- Step 2: waves, strictly sequential -----------------------------
    for wi, wave in enumerate(waves, start=1):
        if stop.is_set():
            break
        tag = f"W{wi}:{wave.platform}"
        stats = {"wave": wave.wave_name, "found": 0, "duplicates": 0, "fresh": 0,
                 "crawled": 0, "blocked": 0, "failed": 0, "records": 0, "dropped": 0}
        yield PipelineEvent("wave_start", f"Wave {wi}/{len(waves)} — {wave.wave_name}", wave=wave.wave_name,
                            data={"queries": wave.queries})

        # a. search
        hits: Dict[str, SearchHit] = {}
        async for outcome in _search_wave(wave, s, stop):
            new = 0
            for h in outcome.hits:
                if h.url not in hits:
                    hits[h.url] = h
                    new += 1
            if outcome.error:
                yield PipelineEvent("warning", f"Query skipped ({outcome.error[:120]}): {outcome.query}",
                                    wave=wave.wave_name)
            else:
                yield PipelineEvent("query", f"{len(outcome.hits)} results (+{new} new) ← {outcome.query}",
                                    wave=wave.wave_name)
        stats["found"] = len(hits)

        # b. dedup — within this run, then against Supabase
        candidates_urls = [u for u in hits if u not in seen_this_run]
        try:
            fresh = await asyncio.to_thread(db.filter_fresh_urls, candidates_urls)
        except db.SupabaseError as exc:
            yield PipelineEvent("error", str(exc), wave=wave.wave_name)
            return
        seen_this_run.update(hits)
        stats["duplicates"] = len(hits) - len(fresh)
        fresh = fresh[: s.max_urls_per_wave]
        stats["fresh"] = len(fresh)
        yield PipelineEvent("dedup", f"{len(hits)} URLs found → {stats['duplicates']} already seen → "
                                     f"{len(fresh)} fresh to crawl", wave=wave.wave_name, data=dict(stats))

        if not fresh:
            yield PipelineEvent("warning", "Wave produced only already-seen URLs — moving to next wave.",
                                wave=wave.wave_name)
            yield PipelineEvent("wave_end", f"Wave {wi} done (nothing new).", wave=wave.wave_name, data=dict(stats))
            _accumulate(totals, stats)
            continue

        # c–g. per batch: record → crawl → extract → save → yield
        try:
            batch_no = 0
            # Record the first batch right before it is crawled.
            await asyncio.to_thread(db.record_scraped_urls, fresh[: s.batch_size], tag)
            # aclosing() guarantees the browser is shut down on break/return.
            async with contextlib.aclosing(crawl_in_batches(
                fresh, batch_size=s.batch_size, page_timeout_s=s.page_timeout_s, respect_robots=s.respect_robots
            )) as batch_iter:
                async for outcomes in batch_iter:
                    batch_no += 1
                    crawl_ok = sum(o.ok for o in outcomes)
                    blocked = sum(o.blocked for o in outcomes)
                    failed = len(outcomes) - crawl_ok - blocked
                    stats["crawled"] += crawl_ok
                    stats["blocked"] += blocked
                    stats["failed"] += failed
                    yield PipelineEvent("crawl", f"Batch {batch_no}: {crawl_ok} crawled, {blocked} blocked, "
                                                 f"{failed} failed", wave=wave.wave_name,
                                        data={"errors": {o.url: o.error for o in outcomes if o.error}})

                    # Build extraction jobs: real page content, or SERP snippet fallback.
                    jobs = []
                    for o in outcomes:
                        if o.ok:
                            jobs.append((o.url, o.markdown, False))
                        elif s.snippet_fallback and o.url in hits:
                            h = hits[o.url]
                            text = f"Title: {h.title}\nSnippet: {h.snippet}".strip()
                            if len(h.snippet) > 40:
                                jobs.append((o.url, text, True))

                    results = await asyncio.gather(
                        *(_extract_page(gemini, intent, u, c, snip) for (u, c, snip) in jobs),
                        return_exceptions=True,
                    )
                    batch_records: List[CandidateRecord] = []
                    for (u, _, _), r in zip(jobs, results):
                        if isinstance(r, Exception):
                            yield PipelineEvent("warning", f"Extraction failed for {u}: {str(r)[:160]}",
                                                wave=wave.wave_name)
                            if isinstance(r, GeminiError) and "API key" in str(r):
                                yield PipelineEvent("error", str(r), wave=wave.wave_name)
                                return
                            continue
                        recs, dropped = r
                        stats["dropped"] += dropped
                        batch_records.extend(recs)

                    if batch_records:
                        try:
                            await asyncio.to_thread(db.save_candidates, batch_records)
                        except db.SupabaseError as exc:
                            yield PipelineEvent("error", str(exc), wave=wave.wave_name)
                            return
                    stats["records"] += len(batch_records)
                    yield PipelineEvent(
                        "saved",
                        f"Batch {batch_no}: {len(jobs)} pages sent to Gemini → {len(batch_records)} candidates saved",
                        wave=wave.wave_name,
                        data={"records": [r.model_dump() for r in batch_records], "stats": dict(stats)},
                    )
                    if stop.is_set():
                        break
                    # Record the NEXT batch just before the crawler starts it, so URLs
                    # never reached (because of Stop) are not marked as visited.
                    next_start = batch_no * s.batch_size
                    if next_start < len(fresh):
                        await asyncio.to_thread(
                            db.record_scraped_urls, fresh[next_start : next_start + s.batch_size], tag
                        )
        except db.SupabaseError as exc:
            yield PipelineEvent("error", str(exc), wave=wave.wave_name)
            return
        except Exception as exc:  # browser failed to launch, etc. — skip the wave, keep going
            yield PipelineEvent("warning", f"Wave aborted: {type(exc).__name__}: {str(exc)[:200]}",
                                wave=wave.wave_name)

        _accumulate(totals, stats)
        yield PipelineEvent("wave_end", f"Wave {wi} done — {stats['records']} candidates.", wave=wave.wave_name,
                            data=dict(stats))

        if wi < len(waves) and not stop.is_set():
            await asyncio.sleep(s.pause_between_waves_s)

    msg = "Stopped by user." if stop.is_set() else "All waves complete."
    yield PipelineEvent("done", f"{msg} {totals['records']} candidates saved from "
                                f"{totals['crawled']} crawled pages.", data=totals)


def _accumulate(totals: dict, stats: dict) -> None:
    for k in ("found", "duplicates", "crawled", "blocked", "records"):
        totals[k] += stats.get(k, 0)


# ---------------------------------------------------------------------------
# Background runner for Streamlit
# ---------------------------------------------------------------------------
def _new_event_loop() -> asyncio.AbstractEventLoop:
    # Playwright needs subprocess support; on Windows only the Proactor loop has it
    # (Streamlit/Tornado may have switched the global policy to Selector).
    if sys.platform == "win32":
        return asyncio.ProactorEventLoop()  # type: ignore[attr-defined]
    return asyncio.new_event_loop()


class BackgroundRun:
    """Runs `run_pipeline` on a dedicated thread; the UI polls `drain()`."""

    def __init__(self, intent: str, settings: PipelineSettings):
        self.intent = intent
        self.settings = settings
        self.stop_event = threading.Event()
        self._q: "queue.Queue[PipelineEvent]" = queue.Queue()
        self._thread = threading.Thread(target=self._main, name="sourcing-pipeline", daemon=True)
        self.started_at = time.time()

    def start(self) -> "BackgroundRun":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def drain(self) -> List[PipelineEvent]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def _main(self) -> None:
        loop = _new_event_loop()
        asyncio.set_event_loop(loop)

        async def consume():
            async for ev in run_pipeline(self.intent, self.settings, self.stop_event):
                self._q.put(ev)

        try:
            loop.run_until_complete(consume())
        except Exception as exc:
            self._q.put(PipelineEvent("error", f"Pipeline crashed: {exc}",
                                      data={"trace": traceback.format_exc()[-2000:]}))
            self._q.put(PipelineEvent("done", "Run ended with an error."))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
