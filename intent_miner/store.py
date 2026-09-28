"""
Storage for the Intent Miner (PostgreSQL / Neon, same database as the main tool; new im_* tables only).

    im_search_runs      one row per command run, with the parsed QuerySpec and run metrics
    im_documents        every document seen: canonical URL, content hash, status, attempts (dead-letter)
    im_intent_events    every scored post / comment: component scores, LLM result, evidence
    im_leads            one row per person / entity, best scores, evidence, lifecycle status
    im_lead_sources     every source a lead was seen in (a lead can have many)
    im_provider_health  per-run provider health
"""

from __future__ import annotations

import json
import re
import threading
from typing import Dict, List, Optional

import psycopg2.extras

import supabase_db as db

LIFECYCLE = ["DISCOVERED", "RAW", "NORMALIZED", "RELEVANT", "INTENT_CLASSIFIED", "QUALIFIED", "ENRICHED",
             "EXPORTED", "CONTACTED", "RESPONDED", "CONVERTED"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS im_search_runs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    command TEXT NOT NULL,
    spec JSONB,
    stats JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS im_documents (
    canonical_url TEXT PRIMARY KEY,
    source TEXT, title TEXT, language TEXT, doc_date DATE, via TEXT,
    content_hash TEXT, status TEXT, error TEXT,
    attempts INT DEFAULT 0, units INT DEFAULT 0,
    run_id UUID, first_seen TIMESTAMPTZ DEFAULT NOW(), last_seen TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_im_documents_hash ON im_documents (content_hash);
CREATE TABLE IF NOT EXISTS im_intent_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id UUID, url TEXT, unit_index INT, unit_kind TEXT, author TEXT, activity_date DATE,
    language TEXT, redacted_text TEXT, scores JSONB, intent JSONB, lead_key TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS im_leads (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    lead_key TEXT UNIQUE NOT NULL,
    display_name TEXT, platform TEXT, profile_url TEXT, email TEXT, phone TEXT,
    profession TEXT, origin TEXT, destination TEXT, timeline TEXT, intent_type TEXT,
    intent_score INT, lead_score INT, tier TEXT, confidence REAL, freshness INT, source_quality INT,
    evidence JSONB DEFAULT '[]'::jsonb, why JSONB DEFAULT '[]'::jsonb, possible_matches JSONB DEFAULT '[]'::jsonb,
    status TEXT DEFAULT 'QUALIFIED', last_activity DATE, run_id UUID, salesforce_id TEXT,
    first_seen TIMESTAMPTZ DEFAULT NOW(), last_seen TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS im_lead_sources (
    lead_key TEXT NOT NULL, url TEXT NOT NULL, source TEXT, unit_kind TEXT, activity_date DATE,
    score INT, evidence TEXT, seen_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (lead_key, url)
);
CREATE TABLE IF NOT EXISTS im_provider_health (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id UUID, provider TEXT, health TEXT, requests INT, ok INT, blocked INT, failed INT,
    at TIMESTAMPTZ DEFAULT NOW()
);
"""

_ready = False
_lock = threading.Lock()


def _ensure() -> None:
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        db._ensure_schema()

        def run():
            with db._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(SCHEMA)
                conn.commit()
        db._with_retry(run, "Creating Intent Miner tables")
        _ready = True


def _q(sql: str, params=None, fetch: str = "", what: str = "Intent Miner query"):
    _ensure()

    def run():
        with db._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else None
            conn.commit()
            return out
    return db._with_retry(run, what)


# ---------------------------------------------------------------------------
def create_run(command: str, spec: dict) -> str:
    row = _q("INSERT INTO im_search_runs (command, spec) VALUES (%s, %s) RETURNING id",
             (command, json.dumps(spec)), "one", "Creating search run")
    return str(row["id"])


def add_run_stats(run_id: str, stats: Dict[str, int]) -> None:
    if not run_id:
        return
    _q("""UPDATE im_search_runs SET stats = COALESCE((
              SELECT jsonb_object_agg(k, COALESCE((stats->>k)::int, 0) + COALESCE((%s::jsonb->>k)::int, 0))
              FROM jsonb_object_keys(COALESCE(stats, '{}'::jsonb) || %s::jsonb) AS k), '{}'::jsonb)
          WHERE id::text = %s""", (json.dumps(stats), json.dumps(stats), run_id), what="Updating run stats")


def list_runs(limit: int = 20) -> List[dict]:
    rows = _q("SELECT id, command, stats, created_at FROM im_search_runs ORDER BY created_at DESC LIMIT %s",
              (limit,), "all")
    return [db._serialize_row(dict(r)) for r in rows]


def known_hashes(hashes: List[str]) -> set:
    """Documents already classified with identical content (incremental crawling: skip the LLM)."""
    if not hashes:
        return set()
    rows = _q("SELECT content_hash FROM im_documents WHERE content_hash = ANY(%s) AND status = 'ok'",
              (hashes,), "all")
    return {r["content_hash"] for r in rows}


def save_document(run_id: str, d: dict) -> None:
    _q("""INSERT INTO im_documents (canonical_url, source, title, language, doc_date, via, content_hash, status,
                                    error, attempts, units, run_id)
          VALUES (%(url)s, %(source)s, %(title)s, %(language)s, %(date)s, %(via)s, %(hash)s, %(status)s, %(error)s,
                  1, %(units)s, %(run)s)
          ON CONFLICT (canonical_url) DO UPDATE SET
              source = EXCLUDED.source, title = COALESCE(EXCLUDED.title, im_documents.title),
              language = EXCLUDED.language, doc_date = COALESCE(EXCLUDED.doc_date, im_documents.doc_date),
              via = EXCLUDED.via, content_hash = COALESCE(EXCLUDED.content_hash, im_documents.content_hash),
              status = EXCLUDED.status, error = EXCLUDED.error, units = EXCLUDED.units,
              attempts = im_documents.attempts + 1, last_seen = NOW(), run_id = EXCLUDED.run_id""",
       {**d, "run": run_id or None}, what="Saving document")


def failed_documents(limit: int = 100) -> List[dict]:
    rows = _q("""SELECT canonical_url AS url, source, title, status, error, attempts, last_seen FROM im_documents
                 WHERE status IN ('failed', 'blocked') ORDER BY last_seen DESC LIMIT %s""", (limit,), "all")
    return [db._serialize_row(dict(r)) for r in rows]


def save_events(run_id: str, events: List[dict]) -> None:
    if not events:
        return
    _ensure()

    def run():
        with db._connect() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO im_intent_events (run_id, url, unit_index, unit_kind, author, activity_date,
                                                  language, redacted_text, scores, intent, lead_key) VALUES %s""",
                    [(run_id or None, e["url"], e["unit"], e["kind"], e.get("author"), e.get("date"),
                      e.get("language"), e.get("redacted", "")[:4000], json.dumps(e["scores"]),
                      json.dumps(e.get("intent")), e.get("lead_key")) for e in events])
            conn.commit()
    db._with_retry(run, "Saving intent events")


def _norm_name(n: Optional[str]) -> str:
    return re.sub(r"[^a-z ]+", " ", (n or "").lower()).strip()


def upsert_leads(run_id: str, leads: List[dict]) -> List[dict]:
    """Insert or merge leads. Same key → same lead; same email / phone → merged into the existing lead;
    same name on another platform → recorded as a *possible* match needing verification (never merged)."""
    out = []
    for L in leads:
        contact_match = None
        if L.get("email") or L.get("phone"):
            digits = re.sub(r"\D", "", L.get("phone") or "")[-10:]
            contact_match = _q("""SELECT lead_key FROM im_leads WHERE (%s <> '' AND lower(email) = lower(%s))
                                  OR (%s <> '' AND right(regexp_replace(COALESCE(phone,''), '\\D', '', 'g'), 10) = %s)
                                  LIMIT 1""", (L.get("email") or "", L.get("email") or "", digits, digits), "one")
        key = contact_match["lead_key"] if contact_match else L["lead_key"]
        possible = []
        if L.get("display_name") and not contact_match:
            rows = _q("""SELECT lead_key, display_name, platform FROM im_leads
                         WHERE lower(regexp_replace(COALESCE(display_name,''), '[^a-zA-Z ]+', ' ', 'g')) = %s
                           AND lead_key <> %s AND platform <> %s LIMIT 5""",
                      (_norm_name(L["display_name"]), key, L.get("platform") or ""), "all")
            possible = [{"lead_key": r["lead_key"], "name": r["display_name"], "platform": r["platform"],
                         "identity_confidence": 0.4, "requires_verification": True} for r in rows]
        row = _q("""
            INSERT INTO im_leads (lead_key, display_name, platform, profile_url, email, phone, profession, origin,
                                  destination, timeline, intent_type, intent_score, lead_score, tier, confidence,
                                  freshness, source_quality, evidence, why, possible_matches, status, last_activity,
                                  run_id)
            VALUES (%(key)s, %(display_name)s, %(platform)s, %(profile_url)s, %(email)s, %(phone)s, %(profession)s,
                    %(origin)s, %(destination)s, %(timeline)s, %(intent_type)s, %(intent_score)s, %(lead_score)s,
                    %(tier)s, %(confidence)s, %(freshness)s, %(source_quality)s, %(evidence)s, %(why)s,
                    %(possible)s, %(status)s, %(last_activity)s, %(run)s)
            ON CONFLICT (lead_key) DO UPDATE SET
                display_name = COALESCE(im_leads.display_name, EXCLUDED.display_name),
                profile_url = COALESCE(im_leads.profile_url, EXCLUDED.profile_url),
                email = COALESCE(im_leads.email, EXCLUDED.email), phone = COALESCE(im_leads.phone, EXCLUDED.phone),
                profession = COALESCE(EXCLUDED.profession, im_leads.profession),
                origin = COALESCE(EXCLUDED.origin, im_leads.origin),
                destination = COALESCE(EXCLUDED.destination, im_leads.destination),
                timeline = COALESCE(EXCLUDED.timeline, im_leads.timeline),
                intent_type = COALESCE(EXCLUDED.intent_type, im_leads.intent_type),
                intent_score = GREATEST(im_leads.intent_score, EXCLUDED.intent_score),
                lead_score = GREATEST(im_leads.lead_score, EXCLUDED.lead_score),
                tier = CASE WHEN EXCLUDED.lead_score > im_leads.lead_score THEN EXCLUDED.tier ELSE im_leads.tier END,
                confidence = GREATEST(im_leads.confidence, EXCLUDED.confidence),
                freshness = GREATEST(im_leads.freshness, EXCLUDED.freshness),
                evidence = COALESCE((SELECT jsonb_agg(DISTINCT e) FROM (
                    SELECT jsonb_array_elements(im_leads.evidence || EXCLUDED.evidence) AS e LIMIT 12) x), '[]'::jsonb),
                why = CASE WHEN EXCLUDED.lead_score >= im_leads.lead_score THEN EXCLUDED.why ELSE im_leads.why END,
                possible_matches = CASE WHEN jsonb_array_length(EXCLUDED.possible_matches) > 0
                                        THEN EXCLUDED.possible_matches ELSE im_leads.possible_matches END,
                last_activity = GREATEST(im_leads.last_activity, EXCLUDED.last_activity),
                last_seen = NOW(), run_id = EXCLUDED.run_id
            RETURNING *""", {**L, "key": key, "evidence": json.dumps(L.get("evidence", [])),
                             "why": json.dumps(L.get("why", [])), "possible": json.dumps(possible),
                             "run": run_id or None}, "one", "Saving lead")
        for s in L.get("sources", []):
            _q("""INSERT INTO im_lead_sources (lead_key, url, source, unit_kind, activity_date, score, evidence)
                  VALUES (%s, %s, %s, %s, %s, %s, %s)
                  ON CONFLICT (lead_key, url) DO UPDATE SET score = GREATEST(im_lead_sources.score, EXCLUDED.score),
                      evidence = EXCLUDED.evidence, seen_at = NOW()""",
               (key, s["url"], s["source"], s["kind"], s.get("date"), s.get("score"), (s.get("evidence") or "")[:600]),
               what="Saving lead source")
        out.append(db._serialize_row(dict(row)))
    return out


def list_leads(min_score: int = 0, limit: int = 1000, run_id: str = "") -> List[dict]:
    rows = _q(f"""SELECT l.*, COALESCE((SELECT jsonb_agg(jsonb_build_object('url', s.url, 'source', s.source,
                          'kind', s.unit_kind, 'date', s.activity_date, 'score', s.score) ORDER BY s.score DESC)
                          FROM im_lead_sources s WHERE s.lead_key = l.lead_key), '[]'::jsonb) AS sources
                  FROM im_leads l WHERE l.lead_score >= %s {"AND l.run_id::text = %s" if run_id else ""}
                  ORDER BY l.lead_score DESC, l.last_activity DESC NULLS LAST LIMIT %s""",
              (min_score, run_id, limit) if run_id else (min_score, limit), "all")
    return [db._serialize_row(dict(r)) for r in rows]


def set_status(lead_ids: List[str], status: str, salesforce_id: Optional[str] = None) -> None:
    if status not in LIFECYCLE:
        raise ValueError("unknown status")
    _q("UPDATE im_leads SET status = %s, salesforce_id = COALESCE(%s, salesforce_id) WHERE id::text = ANY(%s)",
       (status, salesforce_id, lead_ids), what="Updating lead status")


def get_leads(lead_ids: List[str]) -> List[dict]:
    rows = _q("SELECT * FROM im_leads WHERE id::text = ANY(%s)", (lead_ids,), "all")
    return [db._serialize_row(dict(r)) for r in rows]


def save_health(run_id: str, providers: List[dict]) -> None:
    for p in providers:
        if not p.get("requests"):
            continue
        _q("""INSERT INTO im_provider_health (run_id, provider, health, requests, ok, blocked, failed)
              VALUES (%s, %s, %s, %s, %s, %s, %s)""",
           (run_id or None, p["provider"], p["health"], p["requests"], p["ok"], p["blocked"], p["failed"]),
           what="Saving provider health")


def provider_health() -> List[dict]:
    rows = _q("""SELECT provider, health, requests, ok, blocked, failed, at FROM (
                    SELECT *, row_number() OVER (PARTITION BY provider ORDER BY at DESC) AS rn
                    FROM im_provider_health) x WHERE rn = 1 ORDER BY provider""", fetch="all")
    return [db._serialize_row(dict(r)) for r in rows]
