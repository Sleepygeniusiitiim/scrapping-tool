"""
Suggested sites: pages a run read that list many phone numbers / emails but gave few or no leads for that
search (a Gulf walk-in list, a recruiters directory, a job board's contact page…). They are often useful anyway,
so they are kept — with the contacts found on them — across runs, and shown under 💡 Suggested sites.

    site_suggestions   one row per page URL: title, contacts, how many matched the search, which search found it
"""

from __future__ import annotations

import json
import threading
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import psycopg2.extras

import rule_extractor
import supabase_db as db

SCHEMA = """
CREATE TABLE IF NOT EXISTS site_suggestions (
    url TEXT PRIMARY KEY, domain TEXT, title TEXT,
    emails JSONB DEFAULT '[]'::jsonb, phones JSONB DEFAULT '[]'::jsonb,
    n_contacts INT DEFAULT 0, matched INT DEFAULT 0, command TEXT, times_seen INT DEFAULT 1,
    dismissed BOOLEAN DEFAULT FALSE, first_seen TIMESTAMPTZ DEFAULT NOW(), last_seen TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_site_suggestions_seen ON site_suggestions (dismissed, last_seen DESC);
"""
MIN_CONTACTS = 3          # at least this many distinct phones + emails on the page
MAX_KEPT = 300            # contacts kept per page
_ready: set = set()          # schemas (organizations) whose tables exist
_lock = threading.Lock()


def _q(sql: str, params=None, fetch: str = ""):
    if db.current_schema() not in _ready:
        with _lock:
            if db.current_schema() not in _ready:
                db._ensure_schema()

                def mk():
                    with db._connect() as conn:
                        with conn.cursor() as cur:
                            cur.execute(SCHEMA)
                        conn.commit()
                db._with_retry(mk, "Creating suggestions table")
                _ready.add(db.current_schema())

    def run():
        with db._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else None
            conn.commit()
            return out
    return db._with_retry(run, "Suggested sites")


def pick(pages: Iterable[dict]) -> List[dict]:
    """pages: {url, title, text, matched}. The ones with many contacts but (almost) no leads for the search."""
    try:
        import privacy
        lists = privacy._load()
    except Exception:
        lists = None
    out = []
    for p in pages:
        text = p.get("text") or ""
        if not text:
            continue
        emails = rule_extractor.emails_in(text)
        phones = rule_extractor.phones_in(text)
        if lists:
            emails = [e for e in emails if not privacy.suppressed(email=e, lists=lists)]
            phones = [x for x in phones if not privacy.suppressed(phone=x, lists=lists)]
        n = len(emails) + len(phones)
        matched = int(p.get("matched") or 0)
        if n >= MIN_CONTACTS and matched < max(1, n // 3):
            out.append({"url": p["url"], "domain": urlparse(p["url"]).netloc.removeprefix("www."),
                        "title": (p.get("title") or "")[:300], "emails": emails[:MAX_KEPT],
                        "phones": phones[:MAX_KEPT], "n_contacts": n, "matched": matched})
    return out


def save(rows: List[dict], command: str) -> int:
    for r in rows:
        _q("""INSERT INTO site_suggestions (url, domain, title, emails, phones, n_contacts, matched, command)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
              ON CONFLICT (url) DO UPDATE SET title = COALESCE(NULLIF(EXCLUDED.title, ''), site_suggestions.title),
                  emails = EXCLUDED.emails, phones = EXCLUDED.phones,
                  n_contacts = GREATEST(site_suggestions.n_contacts, EXCLUDED.n_contacts),
                  matched = EXCLUDED.matched, command = EXCLUDED.command,
                  times_seen = site_suggestions.times_seen + 1, last_seen = NOW()""",
           (r["url"], r["domain"], r["title"], json.dumps(r["emails"]), json.dumps(r["phones"]), r["n_contacts"],
            r["matched"], (command or "")[:500]))
    return len(rows)


def record(pages: Iterable[dict], command: str) -> List[dict]:
    """Pick and save; returns what was saved (never raises — a suggestion must not break a run)."""
    try:
        rows = pick(pages)
        if rows:
            save(rows, command)
        return rows
    except Exception:
        return []


def listing(limit: int = 200, include_dismissed: bool = False) -> List[dict]:
    rows = _q(f"""SELECT url, domain, title, emails, phones, n_contacts, matched, command, times_seen, dismissed,
                         first_seen, last_seen FROM site_suggestions
                  {"" if include_dismissed else "WHERE NOT dismissed"}
                  ORDER BY last_seen DESC, n_contacts DESC LIMIT %s""", (limit,), "all")
    return [db._serialize_row(dict(r)) for r in rows]


def dismiss(url: str, dismissed: bool = True) -> None:
    _q("UPDATE site_suggestions SET dismissed = %s WHERE url = %s", (dismissed, url))
