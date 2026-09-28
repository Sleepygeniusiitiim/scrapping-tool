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
from .providers.instagram import InstagramProvider, post_code
from .providers.maps import MapsProvider
from .providers.reddit import RedditProvider
from .providers.youtube import YouTubeProvider
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
        "youtube": YouTubeProvider(keys),
        "quora": QuoraProvider(keys),
        "web": WebProvider(keys, settings.get("respect_robots", True), settings.get("timeout", 15)),
        "rss": RssProvider(keys, settings.get("feeds", [])),
        "instagram": InstagramProvider(keys, settings.get("respect_robots", True)),
        "maps": MapsProvider(keys),
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
        elif source == "youtube":
            hits = await provs["youtube"].search(query, spec, limit)
        elif source == "maps":
            try:
                hits = await provs["maps"].search(query, spec, limit)
            except RuntimeError as exc:
                # No Maps key / API down → web search for the same businesses (their sites and listings).
                error = str(exc)
                hits = await provs["search"].search(f"{query} contact number", spec, limit)
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
        doc = h.get("doc")
        item = {"url": url, "title": h.get("title", ""), "snippet": h.get("snippet", ""),
                "date": h.get("date"), "source": doc.source if isinstance(doc, RawDocument) else source_of(url)}
        if isinstance(doc, RawDocument):     # API / feed already returned the content — send it along
            item["doc"] = _doc_to_json(doc)
        out.append(item)
    return {"hits": out, "error": error, "health": _health(provs)}


def _doc_to_json(d: RawDocument) -> dict:
    return {"url": d.url, "source": d.source, "title": d.title, "date": d.date, "via": d.via,
            "units": [u.__dict__ for u in d.units], "metadata": dict(d.metadata)}


def _doc_from_json(j: dict) -> RawDocument:
    d = RawDocument(url=j["url"], source=j.get("source", "rss"), title=j.get("title", ""), date=j.get("date"),
                    via=j.get("via", "api"))
    d.units = [Unit(**u) for u in j.get("units", [])]
    d.metadata = {str(k): str(v) for k, v in (j.get("metadata") or {}).items()}
    return d


# ---------------------------------------------------------------------------
# Processing (one batch of discovered items)
# ---------------------------------------------------------------------------
async def fetch_docs(items: List[dict], provs: Dict[str, object]) -> List[RawDocument]:
    reddit, web, quora, youtube, insta = [], [], [], [], []
    for it in items:
        src = it.get("source") or source_of(it["url"])
        if post_code(it["url"]) and not it.get("doc"):
            insta.append(it)
            continue
        (reddit if src == "reddit" else quora if src == "quora" else youtube if src == "youtube" else web).append(it)
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
    tasks += [provs["youtube"].fetch(it["url"], it) for it in youtube]
    tasks += [provs["instagram"].fetch(it["url"], it) for it in insta]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    fetched = [it for it in reddit] + [it for it in quora] + [it for it in youtube] + [it for it in insta]
    for it, r in zip(fetched, results):
        if isinstance(r, Exception):           # one broken page never fails the whole batch
            d = snippet_doc(it["url"], it.get("source") or source_of(it["url"]), it)
            d.status, d.error = "failed", f"{type(r).__name__}: {str(r)[:120]}"
            docs.append(d)
        elif it in insta and r.status != "ok" and "robots" not in (r.error or "") and not r.units:
            web.append(it)                     # embed page refused → normal reader with the unblockers
        else:
            docs.append(r)
    if web:
        try:
            docs += await provs["web"].fetch_many(web)
        except Exception as exc:
            for it in web:
                d = snippet_doc(it["url"], it.get("source") or source_of(it["url"]), it)
                d.status, d.error = "failed", f"{type(exc).__name__}: {str(exc)[:120]}"
                docs.append(d)
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
    min_score = int(settings.get("min_score", 50))
    stats = {"fetched": 0, "failed": 0, "blocked": 0, "duplicates": 0, "unchanged": 0, "units": 0,
             "stage1": 0, "stage2": 0, "llm_calls": 0, "llm_units": 0, "relevant": 0,
             "high": 0, "medium": 0, "low": 0, "too_old": 0, "leads": 0}
    warnings: List[str] = []

    # Shared dedup ledger with the classic search: pages read here are skipped by later runs of either.
    await asyncio.to_thread(db.record_scraped_urls, [i["url"] for i in items], settings.get("wave_tag") or "IM")
    docs = await fetch_docs(items, provs)
    # ---- normalize + dedupe -----------------------------------------------------------------------
    hashes = {}
    for d in docs:
        d.url = canonicalize_url(d.url) or d.url
        for u in d.units:
            u.text = processing.clean(u.text)
        # short replies ("Interested", "DM me") are real signals in comments; posts need some substance
        d.units = [u for u in d.units if len(u.text) >= (3 if u.kind == "comment" else 20)]
        hashes[d.url] = processing.content_hash(d.title + "\n" + d.text) if d.units else None
        stats["fetched" if d.status == "ok" else d.status if d.status in ("failed", "blocked") else "failed"] += 1
    known = await asyncio.to_thread(store.known_hashes, [h for h in hashes.values() if h])

    shortlist: List[tuple] = []          # (doc, unit_index, unit, parts, why)
    scorer = scoring.Scorer(spec)
    orgs = spec.target == "organizations"      # B2B: businesses / institutes, not individuals with a need
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
            if not orgs and dates.older_than(u.date or d.date, max(1, round((spec.max_age_days or 0) / 30))
                                             if spec.max_age_days else 0):
                stats["too_old"] += 1
                continue
            context = f"{d.title}\n{u.text}" if u.kind in ("post", "snippet", "answer", "organization") else u.text
            if orgs:
                kw, kw_why = scorer.org_keyword(context)
                explicit = scorer.org_contact(u.text)
                if kw < STAGE1_MIN:
                    continue
            else:
                kw, kw_why = scorer.keyword(context)
                if u.kind == "comment":
                    post = f"{d.title} {d.units[0].text[:600] if d.units and d.units[0] is not u else ''}"
                    bonus, bonus_why = scorer.reply_context(u.text, post)
                    kw, kw_why = min(100, kw + bonus), kw_why + bonus_why
                explicit = scorer.explicit(u)
                if kw < STAGE1_MIN and explicit < 100:
                    continue
                if rule_extractor._HIRING.search(u.text) and explicit < 100:
                    continue                      # the job ad / recruiter post itself, not a candidate
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
    passed.sort(key=lambda s: scoring.combine(s[3], orgs, with_llm=False), reverse=True)
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
    cand_lead: Dict[str, dict] = {}
    for d, i, u, parts, why, lang in passed:
        r = llm.get((d.url, i))
        if r is not None:
            if not r.relevant or r.intent_type == "irrelevant" or (not orgs and r.intent_type == "recruitment"):
                events.append(_event(d, i, u, parts, r, lang, None))
                continue
            parts["llm"] = r.confidence * 100
            if r.explicit_need:
                parts["explicit"] = 100
            parts["keyword"] = max(parts["keyword"], min(100, r.intent_strength))
        intent_score = scoring.combine(parts, orgs, with_llm=r is not None)   # AI weight only when the AI judged it
        fresh = scoring.freshness(u.date or d.date)
        if orgs:
            fresh = max(fresh, 60)             # a business listing does not go stale like a post
        quality = SOURCE_QUALITY["snippet"] if d.via == "snippet" else SOURCE_QUALITY.get(d.source, 35)
        score = scoring.lead_score(intent_score, fresh, quality)
        t = scoring.tier(score)
        stats["relevant"] += 1
        # the author's own interest: first person or a reply comment ("Interested"), not an article's "nurses who want…"
        interested = (r is not None and r.explicit_need) or parts["explicit"] >= 100
        if settings.get("only_interested") and not interested and not orgs:
            stats["not_interested"] = stats.get("not_interested", 0) + 1
            events.append(_event(d, i, u, {**parts, "intent": intent_score, "lead": score}, r, lang, None))
            continue
        if t == "NONE" or score < min_score:
            events.append(_event(d, i, u, {**parts, "intent": intent_score, "lead": score}, r, lang, None))
            continue
        stats[t.lower()] += 1
        evidence = (r.evidence if r and r.evidence else [rule_extractor._evidence(u.text, rule_extractor._INTEREST)])
        why_list = _why(why, parts, r, fresh, u.date or d.date, d, orgs)
        emails, phones = rule_extractor.emails_in(u.text), rule_extractor.phones_in(u.text)
        own_contact = orgs or not (rule_extractor._HIRING.search(u.text) and not rule_extractor.shows_interest(u.text))
        author = u.author if u.author and u.kind != "snippet" or d.source == "quora" else None
        if orgs:
            author = (r.organization if r and r.organization else None) or \
                (u.author if u.kind == "organization" else None) or _site_name(d)
        key = (_org_key(author, d.url) if orgs and author else
               f"{d.source}:{author.lower()}" if author else
               f"contact:{(emails or phones or [''])[0]}" if (emails or phones) and own_contact else f"url:{d.url}#{i}")
        lead = {
            "lead_key": key, "display_name": author,
            "platform": d.source, "profile_url": u.author_url,
            "email": emails[0] if emails and own_contact else None,
            "phone": phones[0] if phones and own_contact else None,
            "profession": (((r.org_type or r.profession) if orgs else r.profession) if r else None)
                          or (spec.professions[0] if spec.professions else None),
            "origin": ((r.city or r.origin) if orgs else r.origin) if r else None,
            "destination": (r.destination if r else None),
            "timeline": (r.timeline if r else None), "intent_type": (r.intent_type if r else spec.intent_type),
            "intent_score": intent_score, "lead_score": score, "tier": t,
            "confidence": round((r.confidence if r else 0.5), 2), "freshness": fresh, "source_quality": quality,
            "evidence": evidence[:3], "why": why_list, "status": "QUALIFIED",
            "last_activity": u.date or d.date,
            "sources": [{"url": d.url, "source": d.source, "kind": u.kind, "date": u.date or d.date,
                         "score": score, "evidence": evidence[0] if evidence else ""}],
        }
        if orgs:
            lead["website"] = d.metadata.get("website") or None
            lead["origin"] = lead["origin"] or d.metadata.get("city") or None
        leads.append(lead)
        events.append(_event(d, i, u, {**parts, "intent": intent_score, "lead": score}, r, lang, key))
        if settings.get("save_to_candidates", True):
            cand = _candidate(lead, d, i, u)
            cand_lead[cand.source_url] = lead
            candidates.append(cand)

    # ---- Classic per-page extraction on the same pages (the original pipeline's reader) ----------------
    classic_records: List[CandidateRecord] = []
    if settings.get("classic", True) and orgs:
        warnings.append("Organization search: the classic page reader looks for individual candidates, so it "
                        "was skipped for this command.")
    if settings.get("classic", True) and not orgs:
        classic_records, classic_ai = await _classic(ai, spec, docs, hits_by_url(items), settings, stats, warnings)
        stats["llm_calls"] += classic_ai
        before = len(candidates)
        candidates = _merge_classic(candidates, classic_records)
        stats["classic_added"] = len(candidates) - before
        for c in candidates:                    # details the page reader found for Intent Miner leads
            L = cand_lead.get(c.source_url)
            if L is not None:
                L["email"], L["phone"] = L["email"] or c.email, L["phone"] or c.phone

    # ---- Organizations: public contacts from their own website (the Hunter / Apollo crawl approach) ----
    if orgs and leads and settings.get("company_contacts", True):
        await _company_contacts(leads, candidates, cand_lead, keys, settings, stats, warnings)

    # ---- Lead databases (Apollo / Lusha / ContactOut / RocketReach) for interested leads -------------
    if settings.get("enrich") and candidates:
        import pipeline
        n, notes = await pipeline._enrich_records(candidates, keys, settings.get("require_both", True),
                                                  settings.get("respect_robots", True))
        stats["enriched"] = n
        warnings.extend(notes)
        for c in candidates:                    # contacts found for Intent Miner leads go back onto the lead
            L = cand_lead.get(c.source_url)
            if L is None:
                continue
            if c.contact_source and c.contact_source.startswith(("enriched", "profile_page")):
                L["email"], L["phone"], L["status"] = L["email"] or c.email, L["phone"] or c.phone, "ENRICHED"
            if c.name and c.name != L.get("display_name") and len((c.name or "").split()) >= 2:
                L["why"] = L.get("why", []) + [f"✓ Real name from their profile: {c.name}"]
                L["display_name"] = c.name
            if c.profile_url and "linkedin.com/in/" in c.profile_url:
                L["profile_url"] = c.profile_url

    # ---- Government directories (imported lists): link each lead to its official record --------------
    if leads and settings.get("gov_match", True):
        await _gov_match(ai, leads, candidates, cand_lead, settings, stats, warnings)

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
    try:
        await asyncio.to_thread(db.set_url_status, {
            d.url: ("ok" if d.status == "ok" and d.via != "snippet" else "snippet" if d.units else
                    "robots" if (d.error or "") == "disallowed by robots.txt" else
                    "walled" if d.status == "blocked" else "failed") for d in docs})
    except db.SupabaseError:
        pass
    stats["classic"] = stats.get("classic_added", 0)
    stats["records"] = len(candidates)
    stats["with_phone"] = sum(1 for c in candidates if c.phone)
    stats["with_email"] = sum(1 for c in candidates if c.email)
    return {"stats": stats, "leads": saved, "warnings": warnings, "health": health,
            "records": [c.model_dump() for c in candidates],
            "failed": [{"url": d.url, "status": d.status, "error": d.error} for d in docs if d.status != "ok"]}


MAX_COMPANY_CRAWLS_PER_BATCH = 6
MAX_MAPS_LOOKUPS_PER_BATCH = 6
MAX_OWNER_SEARCHES_PER_BATCH = 4
_gov_count = {"n": None, "at": 0.0}


async def _gov_match(ai, leads: List[dict], candidates: List[CandidateRecord], cand_lead: Dict[str, dict],
                     settings: dict, stats: dict, warnings: List[str]) -> None:
    import time
    import gov_registry
    if _gov_count["n"] is None or time.time() - _gov_count["at"] > 300:
        _gov_count.update(n=await asyncio.to_thread(gov_registry.count), at=time.time())
    if not _gov_count["n"]:
        return
    try:
        st = await gov_registry.match_leads(leads, ai, settings.get("use_llm", True))
    except Exception as exc:
        warnings.append(f"Government-directory matching failed: {type(exc).__name__}: {str(exc)[:120]}")
        return
    stats["gov_matched"], stats["gov_contacts"] = st["matched"], st["contacts"]
    stats["llm_calls"] += st["ai"]
    for c in candidates:
        L = cand_lead.get(c.source_url)
        if L is not None and (L.get("gov_match") or {}).get("status") == "matched":
            if (not c.phone and L.get("phone")) or (not c.email and L.get("email")):
                c.contact_source = c.contact_source or f"government_list:{L['gov_match']['dataset']}"[:80]
            c.phone, c.email = c.phone or L.get("phone"), c.email or L.get("email")
    warnings.append(f"Government lists: {st['checked']} leads checked → {st['matched']} matched to an official "
                    f"record ({st['ai']} confirmed by AI), {st['possible']} possible, {st['contacts']} got a new "
                    "phone / email")


async def _company_contacts(leads: List[dict], candidates: List[CandidateRecord], cand_lead: Dict[str, dict],
                            keys: Dict[str, str], settings: dict, stats: dict, warnings: List[str]) -> None:
    """For organization leads:
        1. no phone yet (found on a directory / website) → its Google Maps listing (phone, website)
        2. its own website → published phones / emails / WhatsApp, and people named with their role
        3. owners / directors / founders → LinkedIn profiles found by search (name + title)"""
    import company_contacts
    from schema import clean_email, clean_phone
    from .providers import maps
    named = [L for L in leads if L.get("display_name")]
    cand_of = {id(L): c for c, L in ((c, cand_lead.get(c.source_url)) for c in candidates) if L is not None}
    before = {id(L): (L.get("email"), L.get("phone")) for L in named}

    # 1. Maps listing for businesses that came without a phone
    need = [L for L in named if not L.get("phone") and L.get("platform") != "maps"][:MAX_MAPS_LOOKUPS_PER_BATCH]
    if need and maps.available(keys):
        res = await asyncio.gather(*(maps.lookup(keys, L["display_name"], L.get("origin") or "") for L in need),
                                   return_exceptions=True)
        got = 0
        for L, p in zip(need, res):
            if isinstance(p, dict):
                L["phone"] = L.get("phone") or clean_phone(p.get("phone"))
                L["website"] = L.get("website") or p.get("website") or None
                L["why"] = L.get("why", []) + [f"✓ Google Maps listing: {p['name']}, {p.get('address') or ''}"]
                got += bool(p.get("phone"))
        stats["maps_lookups"] = len(need)
        warnings.append(f"Google Maps: looked up {len(need)} businesses found without a phone → {got} phones")

    # 2. own website
    todo = sorted(named, key=lambda L: (bool(L.get("email")), not L.get("website")))[:MAX_COMPANY_CRAWLS_PER_BATCH]
    page_of = {L["lead_key"]: L.get("website") or (L.get("sources") or [{}])[0].get("url", "") for L in todo}
    results = await asyncio.gather(*(company_contacts.for_organization(
        keys, L["display_name"], L.get("origin") or "", page_of[L["lead_key"]], settings.get("respect_robots", True))
        for L in todo), return_exceptions=True)
    sites = 0
    for L, res in zip(todo, results):
        if isinstance(res, Exception) or not res:
            continue
        sites += 1
        L["website"] = res["website"]
        good = [e["email"] for e in res["emails"] if e["domain_accepts_mail"] is not False]
        L["email"] = L.get("email") or (clean_email(good[0]) if good else None)
        L["phone"] = L.get("phone") or (clean_phone(res["phones"][0]) if res["phones"] else
                                        clean_phone(res["whatsapp"][0]) if res["whatsapp"] else None)
        L["org_contacts"] = {k: res[k] for k in ("emails", "phones", "whatsapp", "social", "people")}
        why = [f"✓ Website: {res['website']} ({res['pages']} pages read)"]
        if res["people"]:
            why.append("✓ People named on the site: " +
                       ", ".join(f"{p['name']} ({p['role']})" for p in res["people"][:3]))
        if good:
            why.append(f"✓ {len(good)} published email(s); domain accepts mail")
        L["why"] = L.get("why", []) + why

    # 3. decision makers on LinkedIn (owner / director / founder …) for leads whose site named nobody senior
    senior = re.compile(r"owner|founder|director|proprietor|partner|principal|chairman|ceo|md\b", re.I)
    who = [L for L in named if not any(senior.search(p.get("role", "")) for p in
                                       ((L.get("org_contacts") or {}).get("people") or []))]
    who = who[:MAX_OWNER_SEARCHES_PER_BATCH]
    owners = 0
    if who and settings.get("owner_search", True):
        found = await asyncio.gather(*(company_contacts.decision_makers(keys, L["display_name"], L.get("origin") or "")
                                       for L in who), return_exceptions=True)
        for L, people in zip(who, found):
            if isinstance(people, list) and people:
                oc = L.get("org_contacts") or {"emails": [], "phones": [], "whatsapp": [], "social": [], "people": []}
                oc["people"] = people + list(oc.get("people") or [])
                L["org_contacts"] = oc
                L["why"] = L.get("why", []) + ["✓ Decision makers (LinkedIn): " +
                                               ", ".join(f"{p['name']} ({p['role']})" for p in people[:3])]
                owners += 1
    found_new = 0
    for L in named:
        if (L.get("email"), L.get("phone")) != before[id(L)]:
            found_new += 1
            c = cand_of.get(id(L))
            if c is not None:
                c.email, c.phone = c.email or L["email"], c.phone or L["phone"]
                c.contact_source = c.contact_source or "company_website"
    stats["company_sites"] = sites
    stats["company_contacts"] = found_new
    stats["decision_makers"] = owners
    warnings.append(f"Businesses: {len(todo)} websites tried → {sites} read, {owners} with owners / directors "
                    f"found on LinkedIn, {found_new} got a new phone / email")


def _person_keys(c: CandidateRecord) -> set:
    keys = set()
    if c.email:
        keys.add("e:" + c.email.lower())
    if c.phone:
        keys.add("p:" + re.sub(r"\D", "", c.phone)[-10:])
    if c.name:
        keys.add("n:" + re.sub(r"[^a-z]+", " ", c.name.lower()).strip() + "|" + c.source_url.split("#")[0])
    return keys


def _merge_classic(im_cands: List[CandidateRecord], classic: List[CandidateRecord]) -> List[CandidateRecord]:
    """One record per person: a classic-reader record for someone the Intent Miner already has (same email,
    phone, or same name on the same page) only fills in missing details; the rest are added."""
    out = list(im_cands)
    index: Dict[str, CandidateRecord] = {}
    for c in out:
        for k in _person_keys(c):
            index.setdefault(k, c)
    for c in classic:
        match = next((index[k] for k in _person_keys(c) if k in index), None)
        if match is None:
            out.append(c)
            for k in _person_keys(c):
                index.setdefault(k, c)
            continue
        for f in ("email", "phone", "current_location", "current_role", "profile_url", "activity_date"):
            if not getattr(match, f) and getattr(c, f):
                setattr(match, f, getattr(c, f))
        match.skills = list(dict.fromkeys((match.skills or []) + (c.skills or [])))[:15]
        match.target_countries = list(dict.fromkeys((match.target_countries or []) + (c.target_countries or [])))
    return out


def _site_name(d: RawDocument) -> Optional[str]:
    """Organization name from a page title: 'ABC Driving School - Ludhiana | Justdial' → 'ABC Driving School'."""
    head = re.split(r"\s[-–|:]\s|\s\|", d.title or "")[0].strip()
    # a directory / search listing ("Top 50 Driving Schools in Ludhiana") is not one business
    if re.match(r"(?:top|best|list of|\d+\+?|popular|famous|all)\b", head, re.I) or \
            re.search(r"\bnear me\b|\b(?:schools|centres|centers|institutes|academies|companies|dealers|"
                      r"services|agencies|trainers)\s+(?:in|near|at)\b", head, re.I):
        return None
    return head[:120] or None


def _org_key(name: str, url: str) -> str:
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").removeprefix("www.")
    directory = any(x in host for x in ("justdial", "indiamart", "sulekha", "tradeindia", "yellowpages", "linkedin",
                                         "facebook", "google", "quora", "reddit"))
    norm = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return f"org:{norm}" if directory else f"org:{norm}|{host}"


def hits_by_url(items: List[dict]) -> Dict[str, dict]:
    return {canonicalize_url(i["url"]) or i["url"]: i for i in items}


def _markdown(d: RawDocument) -> str:
    """The text the classic reader expects: the fetched page, or a thread rebuilt from API units."""
    if d.metadata.get("markdown"):
        return d.metadata["markdown"]
    lines = [f"# {d.title}"] if d.title else []
    if d.date:
        lines.append(f"Page date: {d.date}")
    thread = [f"{'POST' if u.kind in ('post', 'answer', 'profile') else 'COMMENT'} by {u.author or 'unknown'}"
              f"{f' <{u.author_url}>' if u.author_url else ''}{f' [{u.date}]' if u.date else ''}: {u.text}"
              for u in d.units if u.kind != "snippet"]
    if thread:
        lines += ["", "## Post and comments (structured, with authors)", *thread]
    lines += ["", "## Page text", d.text]
    return "\n".join(lines)


async def _classic(ai, spec: QuerySpec, docs: List[RawDocument], hits: Dict[str, dict], settings: dict,
                   stats: dict, warnings: List[str]) -> tuple[List[CandidateRecord], int]:
    """Run the original pipeline's page reader (rules / hybrid / AI, contact ownership checks, dates,
    interest and location filters) on the pages the Intent Miner already fetched."""
    import pipeline
    mode = settings.get("extraction", "rules")
    intent = settings.get("intent") or spec.summary
    keywords = list(dict.fromkeys([k.lower() for k in spec.professions] +
                                  rule_extractor.keywords_from(intent, settings.get("plan_queries") or [])))
    from .models import concrete_places
    places = concrete_places(spec.destination) or concrete_places(spec.origin)
    months = max(1, round(spec.max_age_days / 30)) if spec.max_age_days else 0
    jobs = []
    for d in docs:
        if not d.units:
            continue
        snippet_only = d.via == "snippet"
        hit = hits.get(d.url, {})
        text = (f"Title: {hit.get('title') or d.title}\nSnippet: {d.text}" if snippet_only else _markdown(d))
        jobs.append((d, pipeline._extract_page(ai, intent, d.url, text, snippet_only, mode, keywords,
                                               d.date or pipeline._hit_date(hit), places)))
    results = await asyncio.gather(*(j for _, j in jobs), return_exceptions=True)
    out, ai_calls = [], 0
    for (d, _), r in zip(jobs, results):
        if isinstance(r, Exception):
            if isinstance(r, GeminiQuotaError):
                warnings.append(f"AI limit reached in the page reader: {str(r)[:120]}")
            continue
        recs, _, used_ai, _ = r
        ai_calls += used_ai
        for rec in recs:
            if dates.older_than(rec.activity_date, months):
                stats["too_old"] += 1
                continue
            if settings.get("only_interested") and not rec.shows_interest:
                stats["not_interested"] = stats.get("not_interested", 0) + 1
                continue
            out.append(rec)
    return out, ai_calls


def _event(d, i, u, parts, r, lang, key) -> dict:
    red, _ = processing.redact(u.text)
    return {"url": d.url, "unit": i, "kind": u.kind, "author": u.author, "date": u.date or d.date,
            "language": lang, "redacted": red, "scores": parts, "intent": r.model_dump() if r else None,
            "lead_key": key}


def _why(why: List[str], parts: dict, r, fresh: int, date: Optional[str], d: RawDocument,
         orgs: bool = False) -> List[str]:
    out = list(dict.fromkeys(why))
    if orgs:
        if r is not None and r.organization:
            out.insert(0, f"Organization: {r.organization}" + (f" ({r.org_type})" if r.org_type else ""))
        if parts.get("explicit", 0) >= 100:
            out.append("Public business phone / email on the page")
        if r is not None and r.explicit_need:
            out.append("Actively operating / enrolling / open to partners")
        if r is not None and r.city:
            out.append(f"Located in {r.city}")
        out.append(f"AI confidence {int(r.confidence * 100)}%" if r is not None
                   else "Scored by rules only (no AI classification)")
    elif r is not None:
        if r.explicit_need:
            out.insert(0, f"Explicitly states their own need ({r.intent_type.replace('_', ' ')})")
        if r.profession and not orgs:
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
        shows_interest=None if lead["lead_key"].startswith("org:") else True, contact_source="posted_on_page" if (lead.get("email") or lead.get("phone")) else None)
