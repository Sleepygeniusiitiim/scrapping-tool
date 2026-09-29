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
    import ai_router
    claude_key = (options.get("llm_keys") or {}).get("claude", "")
    thinker = ai_router.reasoning(ai, claude_key)                     # Claude plans (when it has a key)
    if options.get("claude_reading"):
        ai = thinker                                                  # …and reads pages, when ticked
    spec: QuerySpec = await understand(thinker, command, sources, options.get("max_age_days"),
                                       int(options.get("num_queries") or 16), options.get("exclude_queries") or [],
                                       auto)
    run_id = await asyncio.to_thread(store.create_run, command, spec.model_dump())
    totals["run_id"] = run_id
    totals["queries"] = [q.query for q in spec.queries]
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
            for x in res.get("suggested") or []:
                emit(f"[{src}]   💡 suggested site ({x['n_contacts']} contacts, not a match for this search): {x['url']}")
            for w in res.get("warnings", [])[:4]:
                emit(f"[{src}]   {w[:300]}")
            if on_stats:
                on_stats(totals)
    emit(("Stopped. " if stopped() else "Done. ") +
         f"{totals['leads']} leads, {totals['records']} contacts ({totals['with_phone']} with phone, "
         f"{totals['with_email']} with email) from {totals['pages']} pages and {totals['searches']} searches.")
    return totals


_SUM = ("pages", "leads", "records", "with_phone", "with_email", "blocked", "failed", "searches")


async def run_rounds(ai, command: str, options: dict, keys: Dict[str, str],
                     emit: Callable[[str], None], stopped: Callable[[], bool],
                     on_stats: Optional[Callable[[dict], None]] = None) -> dict:
    """`rounds` runs of the same command one after another; every round plans NEW searches (all queries of
    earlier rounds are excluded), optionally pausing `pause_minutes` between rounds."""
    rounds = max(1, min(20, int(options.get("rounds") or 1)))
    pause = max(0.0, float(options.get("pause_minutes") or 0))
    exclude: List[str] = list(options.get("exclude_queries") or [])
    grand: dict = {k: 0 for k in _SUM}
    grand.update(rounds_done=0, rounds=rounds, run_ids=[])
    for r in range(rounds):
        if stopped():
            break
        if rounds > 1:
            emit(f"━━━ Round {r + 1} of {rounds} ━━━")

        def partial(t, _g=dict(grand)):
            if on_stats:
                on_stats({**_g, **{k: _g[k] + t.get(k, 0) for k in _SUM},
                          "run_ids": _g["run_ids"] + [t.get("run_id")], "run_id": t.get("run_id")})

        t = await run(ai, command, {**options, "exclude_queries": exclude}, keys, emit, stopped, partial)
        exclude += t.get("queries", [])
        for k in _SUM:
            grand[k] += t.get(k, 0)
        grand["run_ids"].append(t.get("run_id"))
        grand["run_id"] = t.get("run_id")
        grand["rounds_done"] = r + 1
        if on_stats:
            on_stats(dict(grand))
        if r < rounds - 1 and pause and not stopped():
            emit(f"Pausing {pause:g} min before round {r + 2}…")
            waited = 0.0
            while waited < pause * 60 and not stopped():
                await asyncio.sleep(5)
                waited += 5
    if rounds > 1:
        emit(f"All rounds done: {grand['rounds_done']} rounds, {grand['leads']} leads, {grand['records']} contacts "
             f"({grand['with_phone']} with phone, {grand['with_email']} with email).")
    return grand
