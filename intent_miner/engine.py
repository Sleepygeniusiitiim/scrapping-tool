"""
One processing step of an Intent Miner run (called per batch of discovered URLs by the page):

    fetch (per provider) → normalize → dedupe (URL, content hash, near-duplicate) →
    Stage 1 keyword filter → Stage 2 semantic filter → Stage 3 LLM (shortlist only, PII redacted) →
    score + evidence + "why" → leads (+ copy of qualified leads into the main candidates table)
"""

from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Optional

import dates
import rule_extractor
import supabase_db as db
from gemini_client import GeminiError, GeminiQuotaError
from schema import CandidateRecord
from search_module import canonicalize_url

from . import processing, scoring, store
from .classify import classify
from .models import QuerySpec, RawDocument, Unit
from .providers.base import SOURCE_QUALITY, source_of
from .providers.reddit import RedditProvider
from .providers.web import QuoraProvider, RssProvider, SearchProvider, WebProvider, snippet_doc

STAGE1_MIN = 25          # keyword score needed to go further (or an explicit first-person need)
STAGE2_MIN = 30          # semantic similarity needed for the LLM (or a strong keyword score)
MAX_LLM_UNITS_PER_DOC = 12
MAX_LLM_UNITS_PER_BATCH = 40


def providers(keys: Dict[str, str], settings: dict) -> Dict[str, object]:
    return {
        "search": SearchProvider(keys, settings.get("backend", "auto"), settings.get("region", "wt-wt"),
                                 settings.get("max_results", 20)),
        "reddit": RedditProvider(keys),
        "quora": QuoraProvider(keys),
        "web": WebProvider(keys, settings.get("respect_robots", True), settings.get("timeout", 15)),
        "rss": RssProvider(keys, settings.get("feeds", [])),
    }


def _health(provs: Dict[str, object]) -> List[dict]:
    return [{"provider": p.name, "health": p.health, **p.stats} for p in provs.values()]


# ---------------------------------------------------------------------------
# Discovery (one query)
# ---------------------------------------------------------------------------
async def discover(spec: QuerySpec, source: str, query: str, keys: Dict[str, str], settings: dict) -> dict:
    provs = providers(keys, settings)
    limit = settings.get("max_results", 20)
    hits, error = [], None
    try:
        if source == "reddit":
            try:
                hits = await provs["reddit"].search(query, spec, limit)
            except RuntimeError as exc:
                # Reddit API unavailable → search-engine discovery of Reddit threads.
                error = str(exc)
                hits = await provs["search"].search(f"site:reddit.com {query}", spec, limit)
        elif source == "rss":
            hits = await provs["rss"].search(query, spec, limit)
        else:
            hits = await provs["search"].search(query, spec, limit)
    except RuntimeError as exc:
        error = str(exc)
    out = []
    for h in hits:
        if h.get("error"):
            error = error or h["error"]
            continue
        url = canonicalize_url(h.get("url") or "")
        if not url:
            continue
        item = {"url": url, "title": h.get("title", ""), "snippet": h.get("snippet", ""),
                "date": h.get("date"), "source": source_of(url)}
        doc = h.get("doc")
        if isinstance(doc, RawDocument):     # API / feed already returned the content — send it along
            item["doc"] = _doc_to_json(doc)
        out.append(item)
    return {"hits": out, "error": error, "health": _health(provs)}


def _doc_to_json(d: RawDocument) -> dict:
    return {"url": d.url, "source": d.source, "title": d.title, "date": d.date, "via": d.via,
            "units": [u.__dict__ for u in d.units]}


def _doc_from_json(j: dict) -> RawDocument:
    d = RawDocument(url=j["url"], source=j.get("source", "rss"), title=j.get("title", ""), date=j.get("date"),
                    via=j.get("via", "api"))
    d.units = [Unit(**u) for u in j.get("units", [])]
    return d


# ---------------------------------------------------------------------------
# Processing (one batch of discovered items)
# ---------------------------------------------------------------------------
async def fetch_docs(items: List[dict], provs: Dict[str, object]) -> List[RawDocument]:
    reddit, web, quora = [], [], []
    for it in items:
        src = it.get("source") or source_of(it["url"])
        (reddit if src == "reddit" else quora if src == "quora" else web).append(it)
    docs: List[RawDocument] = []
    for it in items:
        if it.get("doc") and (it.get("source") or source_of(it["url"])) not in ("reddit",):
            docs.append(_doc_from_json(it["doc"]))
    web = [it for it in web if not it.get("doc")]
    tasks = []
    for it in reddit:
        hit = {**it, "doc": _doc_from_json(it["doc"]) if it.get("doc") else None}
        tasks.append(_reddit_doc(provs["reddit"], provs["web"], it, hit))
    tasks += [provs["quora"].fetch(it["url"], it) for it in quora]
    results = await asyncio.gather(*tasks)
    docs += list(results)
    if web:
        docs += await provs["web"].fetch_many(web)
    return docs


async def _reddit_doc(reddit, web, item, hit) -> RawDocument:
    doc = await reddit.fetch(item["url"], hit)
    if doc.status == "ok" and doc.units:
        return doc
    fallback = snippet_doc(item["url"], "reddit", item)
    fallback.status, fallback.error = doc.status, doc.error
    return fallback


async def process(ai, spec: QuerySpec, items: List[dict], keys: Dict[str, str], settings: dict,
                  run_id: str = "") -> dict:
    provs = providers(keys, settings)
    min_score = int(settings.get("min_score", 60))
    stats = {"fetched": 0, "failed": 0, "blocked": 0, "duplicates": 0, "unchanged": 0, "units": 0,
             "stage1": 0, "stage2": 0, "llm_calls": 0, "llm_units": 0, "relevant": 0,
             "high": 0, "medium": 0, "low": 0, "too_old": 0, "leads": 0}
    warnings: List[str] = []

    docs = await fetch_docs(items, provs)
    # ---- normalize + dedupe -----------------------------------------------------------------------
    hashes = {}
    for d in docs:
        d.url = canonicalize_url(d.url) or d.url
        for u in d.units:
            u.text = processing.clean(u.text)
        d.units = [u for u in d.units if len(u.text) >= 20]
        hashes[d.url] = processing.content_hash(d.title + "\n" + d.text) if d.units else None
        stats["fetched" if d.status == "ok" else d.status if d.status in ("failed", "blocked") else "failed"] += 1
    known = await asyncio.to_thread(store.known_hashes, [h for h in hashes.values() if h])

    shortlist: List[tuple] = []          # (doc, unit_index, unit, parts, why)
    scorer = scoring.Scorer(spec)
    for d in docs:
        lang = processing.language(d.text)
        h = hashes.get(d.url)
        store_row = {"url": d.url, "source": d.source, "title": d.title[:500], "language": lang, "date": d.date,
                     "via": d.via, "hash": h, "status": d.status if d.units or d.status != "ok" else "empty",
                     "error": d.error, "units": len(d.units)}
        await asyncio.to_thread(store.save_document, run_id, store_row)
        if h and h in known and not settings.get("reprocess"):
            stats["unchanged"] += 1           # same content already classified in an earlier run
            continue
        kept: List[Unit] = []
        for i, u in enumerate(d.units):
            if any(processing.near_duplicate(u.text, k.text) for k in kept):
                stats["duplicates"] += 1
                continue
            kept.append(u)
            stats["units"] += 1
            if dates.older_than(u.date or d.date, max(1, round((spec.max_age_days or 0) / 30))
                                if spec.max_age_days else 0):
                stats["too_old"] += 1
                continue
            context = f"{d.title}\n{u.text}" if u.kind in ("post", "snippet", "answer") else u.text
            kw, kw_why = scorer.keyword(context)
            explicit = scorer.explicit(u)
            if kw < STAGE1_MIN and explicit < 100:
                continue
            stats["stage1"] += 1
            tl, tl_label = scorer.timeline(u.text)
            loc, loc_why = scorer.location(f"{d.title} {u.text}" if u.kind != "comment" else
                                           f"{u.text} {d.units[0].text if d.units else ''}")
            parts = {"keyword": kw, "semantic": scorer.lexical_similarity(context), "explicit": explicit,
                     "timeline": tl, "location": loc, "llm": 0}
            why = kw_why + loc_why + ([f"timeline: {tl_label}"] if tl_label else [])
            shortlist.append((d, i, u, parts, why, lang))

    # ---- Stage 2: semantic similarity (embeddings when available) ------------------------------------
    if shortlist:
        vecs = await scoring.embed([spec.summary or " ".join(spec.professions + spec.destination)] +
                                   [s[2].text for s in shortlist], keys)
        if vecs:
            for s, v in zip(shortlist, vecs[1:]):
                s[3]["semantic"] = scoring.semantic_from_cosine(scoring.cosine(vecs[0], v))
    passed = [s for s in shortlist if s[3]["semantic"] >= STAGE2_MIN or s[3]["keyword"] >= 55]
    stats["stage2"] = len(passed)
    passed.sort(key=lambda s: scoring.combine(s[3]), reverse=True)
    passed = passed[:MAX_LLM_UNITS_PER_BATCH]

    # ---- Stage 3: LLM on the shortlist only, contacts redacted -------------------------------------
    by_doc: Dict[str, List[tuple]] = {}
    for s in passed:
        by_doc.setdefault(s[0].url, []).append(s)
    llm: Dict[tuple, object] = {}
    use_llm = settings.get("use_llm", True) and ai is not None

    async def run_llm(url, group):
        units = []
        for d, i, u, *_ in group[:MAX_LLM_UNITS_PER_DOC]:
            red, _ = processing.redact(u.text)
            units.append((i, u.kind, u.author or "", red))
        res = await classify(ai, spec, group[0][0].title, units)
        stats["llm_calls"] += 1
        stats["llm_units"] += len(units)
        for i, r in res.items():
            llm[(url, i)] = r

    if use_llm and by_doc:
        results = await asyncio.gather(*(run_llm(u, g) for u, g in by_doc.items()), return_exceptions=True)
        for r in results:
            if isinstance(r, GeminiQuotaError):
                warnings.append(f"AI limit reached — scored without the LLM: {str(r)[:160]}")
            elif isinstance(r, Exception):
                warnings.append(f"LLM classification failed for one page: {str(r)[:160]}")

    # ---- Scores, evidence, leads ------------------------------------------------------------------
    leads, events, candidates = [], [], []
    for d, i, u, parts, why, lang in passed:
        r = llm.get((d.url, i))
        if r is not None:
            if not r.relevant or r.intent_type in ("recruitment", "irrelevant"):
                events.append(_event(d, i, u, parts, r, lang, None))
                continue
            parts["llm"] = r.confidence * 100
            if r.explicit_need:
                parts["explicit"] = 100
            parts["keyword"] = max(parts["keyword"], min(100, r.intent_strength))
        intent_score = scoring.combine(parts)
        fresh = scoring.freshness(u.date or d.date)
        quality = SOURCE_QUALITY["snippet"] if d.via == "snippet" else SOURCE_QUALITY.get(d.source, 35)
        score = scoring.lead_score(intent_score, fresh, quality)
        t = scoring.tier(score)
        stats["relevant"] += 1
        if t == "NONE" or score < min_score:
            events.append(_event(d, i, u, {**parts, "intent": intent_score, "lead": score}, r, lang, None))
            continue
        stats[t.lower()] += 1
        evidence = (r.evidence if r and r.evidence else [rule_extractor._evidence(u.text, rule_extractor._INTEREST)])
        why_list = _why(why, parts, r, fresh, u.date or d.date, d)
        emails, phones = rule_extractor.emails_in(u.text), rule_extractor.phones_in(u.text)
        own_contact = not (rule_extractor._HIRING.search(u.text) and not rule_extractor.shows_interest(u.text))
        author = u.author if u.author and u.kind != "snippet" or d.source == "quora" else None
        key = (f"{d.source}:{author.lower()}" if author else
               f"contact:{(emails or phones or [''])[0]}" if (emails or phones) and own_contact else f"url:{d.url}#{i}")
        lead = {
            "lead_key": key, "display_name": author,
            "platform": d.source, "profile_url": u.author_url,
            "email": emails[0] if emails and own_contact else None,
            "phone": phones[0] if phones and own_contact else None,
            "profession": (r.profession if r else None) or (spec.professions[0] if spec.professions else None),
            "origin": (r.origin if r else None), "destination": (r.destination if r else None),
            "timeline": (r.timeline if r else None), "intent_type": (r.intent_type if r else spec.intent_type),
            "intent_score": intent_score, "lead_score": score, "tier": t,
            "confidence": round((r.confidence if r else 0.5), 2), "freshness": fresh, "source_quality": quality,
            "evidence": evidence[:3], "why": why_list, "status": "QUALIFIED",
            "last_activity": u.date or d.date,
            "sources": [{"url": d.url, "source": d.source, "kind": u.kind, "date": u.date or d.date,
                         "score": score, "evidence": evidence[0] if evidence else ""}],
        }
        leads.append(lead)
        events.append(_event(d, i, u, {**parts, "intent": intent_score, "lead": score}, r, lang, key))
        if settings.get("save_to_candidates", True):
            candidates.append(_candidate(lead, d, i, u))

    saved = await asyncio.to_thread(store.upsert_leads, run_id, leads) if leads else []
    await asyncio.to_thread(store.save_events, run_id, events)
    if candidates:
        try:
            await asyncio.to_thread(db.save_candidates, candidates)
        except db.SupabaseError as exc:
            warnings.append(f"Could not copy leads to the candidates table: {exc}")
    stats["leads"] = len(saved)
    health = _health(provs)
    await asyncio.to_thread(store.add_run_stats, run_id, stats)
    await asyncio.to_thread(store.save_health, run_id, health)
    return {"stats": stats, "leads": saved, "warnings": warnings, "health": health,
            "failed": [{"url": d.url, "status": d.status, "error": d.error} for d in docs if d.status != "ok"]}


def _event(d, i, u, parts, r, lang, key) -> dict:
    red, _ = processing.redact(u.text)
    return {"url": d.url, "unit": i, "kind": u.kind, "author": u.author, "date": u.date or d.date,
            "language": lang, "redacted": red, "scores": parts, "intent": r.model_dump() if r else None,
            "lead_key": key}


def _why(why: List[str], parts: dict, r, fresh: int, date: Optional[str], d: RawDocument) -> List[str]:
    out = list(dict.fromkeys(why))
    if r is not None:
        if r.explicit_need:
            out.insert(0, f"Explicitly states their own need ({r.intent_type.replace('_', ' ')})")
        if r.profession:
            out.append(f"{r.profession} identified")
        if r.origin:
            out.append(f"Origin: {r.origin}")
        if r.destination:
            out.append(f"Destination: {r.destination}")
        if r.timeline:
            out.append(f"Timeline: {r.timeline}")
        out.append(f"AI confidence {int(r.confidence * 100)}% · stage: {r.buying_stage} · urgency: {r.urgency}")
    else:
        out.append("Scored by rules only (no AI classification)")
    if date:
        out.append(f"Activity date {date}" + (" (recent)" if fresh >= 85 else ""))
    if parts.get("semantic", 0) >= 60:
        out.append("High semantic similarity to the request")
    if d.via == "snippet":
        out.append("Based on the search-result snippet only (page not readable)")
    return ["✓ " + x for x in dict.fromkeys(out)][:10]


def _candidate(lead: dict, d: RawDocument, i: int, u: Unit) -> CandidateRecord:
    slug = re.sub(r"[^a-z0-9]+", "-", (lead.get("display_name") or f"unit-{i}").lower()).strip("-")[:50]
    return CandidateRecord(
        name=lead.get("display_name") if lead["lead_key"] and not lead["lead_key"].startswith("url:") else None,
        current_role=lead.get("profession"), skills=[], current_location=lead.get("origin"),
        target_countries=[lead["destination"]] if lead.get("destination") else [],
        evidence_snippet=(lead["evidence"][0] if lead["evidence"] else u.text[:300]),
        email=lead.get("email"), phone=lead.get("phone"),
        source_url=f"{d.url}#im-{slug}" if i else d.url, platform=d.source,
        profile_url=u.author_url, activity_date=lead.get("last_activity"),
        shows_interest=True, contact_source="posted_on_page" if (lead.get("email") or lead.get("phone")) else None)
