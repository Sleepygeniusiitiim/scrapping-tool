"""
Neon PostgreSQL persistence layer (replacing Supabase).

Responsibilities
----------------
* `scraped_urls` – the dedup ledger. Every URL is checked here before it is
  crawled and written here right before it is crawled.
* `candidates`   – structured records, upserted on `source_url` so a record
  can never be stored twice.

All functions use Neon PostgreSQL (`DATABASE_URL` / `NEON_DATABASE_URL`) with
automatic connection retry and schema initialization.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
import time
import uuid
from typing import Any, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from schema import CandidateRecord

log = logging.getLogger(__name__)

SCRAPED_URLS_TABLE = "scraped_urls"
CANDIDATES_TABLE = "candidates"

DEFAULT_NEON_DATABASE_URL = (
    "postgresql://neondb_owner:npg_jBms9Rc4oHgD@"
    "ep-fancy-dust-b5rdd2ee-pooler.c-7.us-east-2.aws.neon.tech/"
    "neondb?sslmode=require&channel_binding=require"
)

_IN_CHUNK = 200
_WRITE_CHUNK = 200
_RETRIES = 3

_database_url: Optional[str] = None
_schema_ready: bool = False
_lock = threading.Lock()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scraped_urls (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    domain TEXT,
    wave_tag TEXT,
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS candidates (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT,
    "current_role" TEXT,
    skills TEXT[],
    current_location TEXT,
    target_countries TEXT[],
    evidence_snippet TEXT,
    source_url TEXT UNIQUE NOT NULL,
    platform TEXT,
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scraped_urls_url ON scraped_urls (url);
CREATE INDEX IF NOT EXISTS idx_candidates_source_url ON candidates (source_url);

-- Contact details the candidate posted themselves (added after the first release).
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS email TEXT;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS phone TEXT;
"""


class SupabaseError(RuntimeError):
    """Raised with a human-readable explanation of a Neon PostgreSQL database failure."""


DatabaseError = SupabaseError


# ---------------------------------------------------------------------------
# Client / Connection management
# ---------------------------------------------------------------------------
def _resolve_database_url(url: Optional[str] = None) -> str:
    candidate = (url or "").strip()
    if candidate.startswith(("postgres://", "postgresql://")):
        return candidate
    env_url = (
        os.getenv("DATABASE_URL")
        or os.getenv("NEON_DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or DEFAULT_NEON_DATABASE_URL
    ).strip()
    return env_url


def init_supabase(url: Optional[str] = None, key: Optional[str] = None) -> str:
    """Configure the Neon PostgreSQL database URL and ensure schema tables exist."""
    global _database_url, _schema_ready
    resolved = _resolve_database_url(url)
    if not resolved:
        raise SupabaseError("DATABASE_URL (Neon PostgreSQL connection string) is required.")
    with _lock:
        if _database_url != resolved:
            _database_url = resolved
            _schema_ready = False
    _ensure_schema()
    return _database_url


init_db = init_supabase


def _connect():
    global _database_url
    if not _database_url:
        _database_url = _resolve_database_url()
    return psycopg2.connect(_database_url, connect_timeout=15)


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _lock:
        if _schema_ready:
            return
        def _run():
            with _connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(SCHEMA_SQL)
                conn.commit()
        _with_retry(_run, "Initializing Neon DB schema")
        _schema_ready = True


def _with_retry(fn, what: str):
    """Run a PostgreSQL operation with small exponential backoff for transient errors."""
    last: Optional[Exception] = None
    for attempt in range(_RETRIES):
        try:
            return fn()
        except Exception as exc:
            last = exc
            time.sleep(0.6 * (2 ** attempt))
    raise SupabaseError(f"{what} failed: {last}") from last


def _chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def check_connection() -> str:
    """Cheap sanity check used by the UI 'Test connection' button."""
    _ensure_schema()
    def _ping():
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {SCRAPED_URLS_TABLE};")
                urls_count = cur.fetchone()[0]
                cur.execute(f"SELECT COUNT(*) FROM {CANDIDATES_TABLE};")
                cands_count = cur.fetchone()[0]
                return f"Connected to Neon DB (PostgreSQL) — {cands_count} candidates, {urls_count} URLs tracked."
    return _with_retry(_ping, "Neon DB connection check")


# ---------------------------------------------------------------------------
# URL ledger
# ---------------------------------------------------------------------------
def filter_fresh_urls(urls: List[str]) -> List[str]:
    """Return only URLs that are NOT already in `scraped_urls` (order kept)."""
    unique = list(dict.fromkeys(u for u in urls if u))
    if not unique:
        return []
    _ensure_schema()
    seen: set[str] = set()

    def _lookup():
        with _connect() as conn:
            with conn.cursor() as cur:
                for chunk in _chunks(unique, _IN_CHUNK):
                    cur.execute(
                        f"SELECT url FROM {SCRAPED_URLS_TABLE} WHERE url = ANY(%s);",
                        (list(chunk),),
                    )
                    for (u,) in cur.fetchall():
                        seen.add(u)

    _with_retry(_lookup, "URL dedup lookup")
    return [u for u in unique if u not in seen]


def record_scraped_urls(urls: List[str], wave_tag: str) -> int:
    """Bulk-insert URLs into `scraped_urls`. Already-present URLs are ignored."""
    unique = list(dict.fromkeys(u for u in urls if u))
    if not unique:
        return 0
    _ensure_schema()
    rows = [(u, urlparse(u).netloc.lower(), wave_tag) for u in unique]

    def _insert():
        with _connect() as conn:
            with conn.cursor() as cur:
                for chunk in _chunks(rows, _WRITE_CHUNK):
                    psycopg2.extras.execute_values(
                        cur,
                        f"""
                        INSERT INTO {SCRAPED_URLS_TABLE} (url, domain, wave_tag)
                        VALUES %s
                        ON CONFLICT (url) DO NOTHING;
                        """,
                        list(chunk),
                    )
            conn.commit()

    _with_retry(_insert, "Recording scraped URLs")
    return len(unique)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
def save_candidates(candidates: List[CandidateRecord]) -> int:
    """Upsert candidates on `source_url`. Returns number of rows sent."""
    if not candidates:
        return 0
    _ensure_schema()
    by_url = {c.source_url: c.to_db_row() for c in candidates}
    rows = [
        (
            r.get("name"),
            r.get("current_role"),
            list(r.get("skills") or []),
            r.get("current_location"),
            list(r.get("target_countries") or []),
            r.get("evidence_snippet"),
            r.get("email"),
            r.get("phone"),
            r["source_url"],
            r.get("platform"),
        )
        for r in by_url.values()
    ]

    def _upsert():
        with _connect() as conn:
            with conn.cursor() as cur:
                for chunk in _chunks(rows, _WRITE_CHUNK):
                    psycopg2.extras.execute_values(
                        cur,
                        f"""
                        INSERT INTO {CANDIDATES_TABLE} (
                            name, "current_role", skills, current_location,
                            target_countries, evidence_snippet, email, phone, source_url, platform
                        )
                        VALUES %s
                        ON CONFLICT (source_url) DO UPDATE SET
                            name = EXCLUDED.name,
                            "current_role" = EXCLUDED."current_role",
                            skills = EXCLUDED.skills,
                            current_location = EXCLUDED.current_location,
                            target_countries = EXCLUDED.target_countries,
                            evidence_snippet = EXCLUDED.evidence_snippet,
                            email = COALESCE(EXCLUDED.email, {CANDIDATES_TABLE}.email),
                            phone = COALESCE(EXCLUDED.phone, {CANDIDATES_TABLE}.phone),
                            platform = COALESCE(EXCLUDED.platform, {CANDIDATES_TABLE}.platform),
                            discovered_at = NOW();
                        """,
                        list(chunk),
                    )
            conn.commit()

    _with_retry(_upsert, "Saving candidates")
    return len(rows)


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, uuid.UUID):
            out[k] = str(v)
        elif isinstance(v, (datetime.datetime, datetime.date)):
            out[k] = v.isoformat()
        elif k in ("skills", "target_countries") and v is None:
            out[k] = []
        else:
            out[k] = v
    return out


def fetch_all_candidates() -> List[dict]:
    """Return every stored candidate from Neon DB, newest first."""
    _ensure_schema()

    def _fetch():
        with _connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, name, "current_role", skills, current_location,
                           target_countries, evidence_snippet, email, phone, source_url,
                           platform, discovered_at
                    FROM {CANDIDATES_TABLE}
                    ORDER BY discovered_at DESC;
                    """
                )
                return [_serialize_row(dict(r)) for r in cur.fetchall()]

    return _with_retry(_fetch, "Fetching candidates")


def count_scraped_urls() -> int:
    """Total URLs in the ledger (for the UI header)."""
    _ensure_schema()

    def _count():
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {SCRAPED_URLS_TABLE};")
                row = cur.fetchone()
                return int(row[0] if row else 0)

    return _with_retry(_count, "Counting scraped URLs")
