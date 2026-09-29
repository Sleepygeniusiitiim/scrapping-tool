"""
A whole search run on the server: understand the command → plan sources → search every source → read the pages in
batches → leads, contacts, enrichment. The same steps the browser drives in the web app, for background workers.

`emit(line)` receives progress lines; `stopped()` is polled between steps (cancel from the page).
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, List, Optional

import supabase_db as db

from . import engine, store
from .models import QuerySpec
from .understand import understand

SOURCE_CAP = {"maps": 400, "directories": 120}


async def run(ai, command: str, options: dict, keys: Dict[str, str],
              emit: Callable[[str], None], stopped: Callable[[], bool],
              on_stats: Optional[Callable[[dict], None]] = None) -> dict:
    settings = dict(options.get("settings") or {})
    sources = options.get("sources") or []
    auto = bool(options.get("auto_sources", True))
    totals = {"pages": 0, "leads": 0, "records": 0, "with_phone": 0, "with_email": 0, "blocked": 0, "failed": 0,
              "searches": 0}

    emit("Understanding the command and choosing sources…")
    spec: QuerySpec = await understand(ai, command, sources, options.get("max_age_days"),
                                       int(options.get("num_queries") or 16), options.get("exclude_queries") or [],
                                       auto)
    run_id = await asyncio.to_thread(store.create_run, command, spec.model_dump())
    totals["run_id"] = run_id
    if auto:
        emit("Source plan: " + " · ".join(f"{c.source} {c.weight} ({c.reason})" for c in spec.source_plan
                                          if c.weight >= 50))
    if spec.places:
        emit(f"{len(spec.places)} places: {', '.join(spec.places[:15])}{'…' if len(spec.places) > 15 else ''}")
    settings["plan_queries"] = [q.query for q in spec.queries]
    by_src: Dict[str, List[str]] = {}
    for q in spec.queries:
        by_src.setdefault(q.source, []).append(q.query)
    if "rss" in sources and settings.get("feeds"):
        by_src["rss"] = ["feeds"]
    emit(f"Plan: {len(spec.queries)} searches — " + ", ".join(f"{k} {len(v)}" for k, v in by_src.items()))

    seen: set = set()
    _related_cache: Dict[str, List[str]] = {}          # Google's related searches, per source (this run only)
    max_urls = int(options.get("max_urls") or 60)
    batch = max(1, min(8, int(options.get("batch") or 5)))
    concurrency = int(options.get("search_concurrency") or 4)
    for src, queries in by_src.items():
        if stopped():
            break
        hits: Dict[str, dict] = {}
        sem = asyncio.Semaphore(concurrency)

        async def one(q: str):
            async with sem:
                if stopped():
                    return
                r = await engine.discover(spec, src, q, keys, settings)
                _related_cache.setdefault(src, []).extend(x for x in r.get("related") or []
                                                          if x not in _related_cache.get(src, []))
                totals["searches"] += 1
                new = 0
                for h in r["hits"]:
                    if h["url"] not in hits and h["url"] not in seen:
                        hits[h["url"]] = h
                        new += 1
                emit(f"[{src}] {len(r['hits'])} results (+{new}) ← {q}" +
                     (f"  ⚠️ {r['error'][:120]}" if r.get("error") else ""))

        ran = {q.lower() for q in queries}

        await asyncio.gather(*(one(q) for q in queries), return_exceptions=True)
        if settings.get("expand_related", True) and _related_cache:
            extra = [x for x in _related_cache.pop(src, []) if x.lower() not in ran][:min(6, len(queries))]
            if extra:
                emit(f"[{src}] +{len(extra)} related searches Google suggested: {' · '.join(extra)}")
                ran.update(x.lower() for x in extra)
                await asyncio.gather(*(one(q) for q in extra), return_exceptions=True)
        urls = list(hits)
        seen.update(urls)
        fresh = urls if settings.get("reprocess") else await asyncio.to_thread(db.filter_fresh_urls, urls)
        fresh = fresh[:max(max_urls, SOURCE_CAP.get(src, 0))]
        emit(f"[{src}] {len(urls)} found → {len(urls) - len(fresh) if len(fresh) < len(urls) else 0} read before / "
             f"over the limit → {len(fresh)} to read")
        for b in range(0, len(fresh), batch):
            if stopped():
                break
            items = [hits[u] for u in fresh[b:b + batch]]
            try:
                res = await engine.process(ai, spec, items, keys, {**settings, "wave_tag": f"JOB:{src}"}, run_id)
            except Exception as exc:
                emit(f"[{src}] batch {b // batch + 1} failed: {type(exc).__name__}: {str(exc)[:160]}")
                continue
            st = res["stats"]
            totals["pages"] += st.get("fetched", 0)
            totals["blocked"] += st.get("blocked", 0)
            totals["failed"] += st.get("failed", 0)
            totals["leads"] += st.get("leads", 0)
            totals["records"] += st.get("records", 0)
            totals["with_phone"] += st.get("with_phone", 0)
            totals["with_email"] += st.get("with_email", 0)
            emit(f"[{src}] batch {b // batch + 1}: {st.get('fetched', 0)} read, {st.get('blocked', 0)} blocked → "
                 f"{st.get('leads', 0)} leads, {st.get('records', 0)} contacts saved "
                 f"({st.get('with_phone', 0)} phone, {st.get('with_email', 0)} email)")
            for w in res.get("warnings", [])[:4]:
                emit(f"[{src}]   {w[:300]}")
            if on_stats:
                on_stats(totals)
    emit(("Stopped. " if stopped() else "Done. ") +
         f"{totals['leads']} leads, {totals['records']} contacts ({totals['with_phone']} with phone, "
         f"{totals['with_email']} with email) from {totals['pages']} pages and {totals['searches']} searches.")
    return totals
