"""
Supabase persistence layer.

Responsibilities
----------------
* `scraped_urls` – the dedup ledger. Every URL is checked here before it is
  crawled and written here right before it is crawled.
* `candidates`   – structured records, upserted on `source_url` so a record
  can never be stored twice.

All functions use a single module-level client created by `init_supabase()`
(or lazily from SUPABASE_URL / SUPABASE_KEY in the environment).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Iterable, List, Optional, Sequence
from urllib.parse import urlparse

from supabase import Client, create_client

from schema import CandidateRecord

log = logging.getLogger(__name__)

SCRAPED_URLS_TABLE = "scraped_urls"
CANDIDATES_TABLE = "candidates"

# PostgREST sends `.in_()` filters in the query string. Long URLs × many
# values can exceed proxy / server URL-length limits, so we chunk lookups.
_IN_CHUNK = 40
_WRITE_CHUNK = 200
_PAGE_SIZE = 1000
_RETRIES = 3

_client: Optional[Client] = None
_lock = threading.Lock()


class SupabaseError(RuntimeError):
    """Raised with a human-readable explanation of a Supabase failure."""


# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------
def init_supabase(url: Optional[str] = None, key: Optional[str] = None) -> Client:
    """Create (or replace) the shared client. Falls back to env vars."""
    global _client
    url = (url or os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = (key or os.getenv("SUPABASE_KEY") or "").strip()
    if not url or not key:
        raise SupabaseError("SUPABASE_URL and SUPABASE_KEY are required.")
    try:
        with _lock:
            _client = create_client(url, key)
    except Exception as exc:  # malformed URL / key format
        raise SupabaseError(f"Could not create Supabase client: {exc}") from exc
    return _client


def get_client() -> Client:
    global _client
    if _client is None:
        init_supabase()
    assert _client is not None
    return _client


def _explain(exc: Exception) -> str:
    """Translate common PostgREST errors into actionable messages."""
    msg = str(exc)
    if "42P01" in msg or "does not exist" in msg or "PGRST205" in msg:
        return "Table not found — run schema.sql in the Supabase SQL editor first."
    if "42501" in msg or "row-level security" in msg.lower() or "permission denied" in msg.lower():
        return ("Permission denied (Row Level Security). Use the service_role key, "
                "or run the optional RLS policies at the bottom of schema.sql.")
    if "401" in msg or "Invalid API key" in msg or "JWT" in msg:
        return "Supabase rejected the key — check SUPABASE_KEY."
    return msg


def _with_retry(fn, what: str):
    """Run a Supabase call with small exponential backoff for transient errors."""
    last: Optional[Exception] = None
    for attempt in range(_RETRIES):
        try:
            return fn()
        except Exception as exc:  # network hiccups, 5xx, timeouts
            last = exc
            text = str(exc)
            # Errors that will never succeed on retry → fail fast.
            if any(code in text for code in ("42P01", "42501", "PGRST205", "Invalid API key", "42703", "23502")):
                break
            time.sleep(0.8 * (2 ** attempt))
    raise SupabaseError(f"{what} failed: {_explain(last)}") from last


def _chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def check_connection() -> str:
    """Cheap sanity check used by the UI 'Test connection' button."""
    client = get_client()
    _with_retry(lambda: client.table(SCRAPED_URLS_TABLE).select("id").limit(1).execute(), "scraped_urls check")
    _with_retry(lambda: client.table(CANDIDATES_TABLE).select("id").limit(1).execute(), "candidates check")
    return "Connected — both tables are reachable."


# ---------------------------------------------------------------------------
# URL ledger
# ---------------------------------------------------------------------------
def filter_fresh_urls(urls: List[str]) -> List[str]:
    """Return only URLs that are NOT already in `scraped_urls` (order kept)."""
    unique = list(dict.fromkeys(u for u in urls if u))
    if not unique:
        return []
    client = get_client()
    seen: set[str] = set()
    for chunk in _chunks(unique, _IN_CHUNK):
        res = _with_retry(
            lambda c=chunk: client.table(SCRAPED_URLS_TABLE).select("url").in_("url", list(c)).execute(),
            "URL dedup lookup",
        )
        seen.update(row["url"] for row in (res.data or []))
    return [u for u in unique if u not in seen]


def record_scraped_urls(urls: List[str], wave_tag: str) -> int:
    """Bulk-insert URLs into `scraped_urls`. Already-present URLs are ignored."""
    unique = list(dict.fromkeys(u for u in urls if u))
    if not unique:
        return 0
    client = get_client()
    rows = [{"url": u, "domain": urlparse(u).netloc.lower(), "wave_tag": wave_tag} for u in unique]
    for chunk in _chunks(rows, _WRITE_CHUNK):
        _with_retry(
            lambda c=chunk: client.table(SCRAPED_URLS_TABLE)
            .upsert(list(c), on_conflict="url", ignore_duplicates=True)
            .execute(),
            "Recording scraped URLs",
        )
    return len(unique)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
def save_candidates(candidates: List[CandidateRecord]) -> int:
    """Upsert candidates on `source_url`. Returns number of rows sent."""
    if not candidates:
        return 0
    # De-duplicate inside the batch too — Postgres rejects an upsert that
    # touches the same conflict key twice in one statement.
    by_url = {c.source_url: c.to_db_row() for c in candidates}
    rows = list(by_url.values())
    client = get_client()
    for chunk in _chunks(rows, _WRITE_CHUNK):
        _with_retry(
            lambda c=chunk: client.table(CANDIDATES_TABLE).upsert(list(c), on_conflict="source_url").execute(),
            "Saving candidates",
        )
    return len(rows)


def fetch_all_candidates() -> List[dict]:
    """Return every stored candidate, newest first (paginated reads)."""
    client = get_client()
    out: List[dict] = []
    start = 0
    while True:
        res = _with_retry(
            lambda s=start: client.table(CANDIDATES_TABLE)
            .select("*")
            .order("discovered_at", desc=True)
            .range(s, s + _PAGE_SIZE - 1)
            .execute(),
            "Fetching candidates",
        )
        batch = res.data or []
        out.extend(batch)
        if len(batch) < _PAGE_SIZE:
            break
        start += _PAGE_SIZE
    return out


def count_scraped_urls() -> int:
    """Total URLs in the ledger (for the UI header)."""
    client = get_client()
    res = _with_retry(
        lambda: client.table(SCRAPED_URLS_TABLE).select("id", count="exact").limit(1).execute(),
        "Counting scraped URLs",
    )
    return int(res.count or 0)
