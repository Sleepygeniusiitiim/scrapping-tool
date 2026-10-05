"""
Categories: a label you give a search ("Foreign employers hiring", "Gulf welders", …). Every search started
under a category is logged, and every contact / lead it saves is tagged with it — so the database can be filtered
to one category: its searches and all the data they found.

    search_categories   name, when created
    category_searches   one row per search started: category, command, what it looked for, mode, when
    candidates.categories / im_leads.categories   TEXT[] of the categories that found the row
"""

from __future__ import annotations

import re
import threading
from typing import Iterable, List, Optional

import psycopg2.extras

import supabase_db as db

SCHEMA = """
CREATE TABLE IF NOT EXISTS search_categories (
    name TEXT PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS category_searches (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, category TEXT NOT NULL, command TEXT,
    target TEXT, mode TEXT, run_id TEXT, started_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_category_searches ON category_searches (category, started_at DESC);
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS categories TEXT[] DEFAULT '{}';
"""
LEADS_SCHEMA = "ALTER TABLE im_leads ADD COLUMN IF NOT EXISTS categories TEXT[] DEFAULT '{}';"
_ready = False
_lock = threading.Lock()


def clean(name: Optional[str]) -> str:
    """A tidy category name ('' = none)."""
    return re.sub(r"\s+", " ", (name or "")).strip()[:60]


def _q(sql: str, params=None, fetch: str = ""):
    global _ready
    if not _ready:
        with _lock:
            if not _ready:
                db._ensure_schema()

                def mk():
                    with db._connect() as conn:
                        with conn.cursor() as cur:
                            cur.execute(SCHEMA)
                            try:
                                from intent_miner import store
                                store._ensure()
                                cur.execute(LEADS_SCHEMA)
                            except Exception:
                                conn.rollback()
                                cur.execute(SCHEMA)
                        conn.commit()
                db._with_retry(mk, "Creating category tables")
                _ready = True

    def run():
        with db._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else None
            conn.commit()
            return out
    return db._with_retry(run, "Categories")


def create(name: str) -> str:
    """Add a category before any search uses it (the Search page's “➕ Create a new category”)."""
    cat = clean(name)
    if not cat:
        raise ValueError("Give the category a name.")
    _q("INSERT INTO search_categories (name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (cat,))
    return cat


def log_search(category: str, command: str, target: str = "", mode: str = "", run_id: str = "") -> None:
    cat = clean(category)
    if not cat:
        return
    _q("INSERT INTO search_categories (name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (cat,))
    _q("INSERT INTO category_searches (category, command, target, mode, run_id) VALUES (%s, %s, %s, %s, %s)",
       (cat, (command or "")[:2000], target or None, mode or None, run_id or None))


def _append(table: str, key_col: str, keys: Iterable[str], category: str) -> int:
    cat, keys = clean(category), [k for k in dict.fromkeys(keys) if k]
    if not cat or not keys:
        return 0
    _q("INSERT INTO search_categories (name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (cat,))
    r = _q(f"""WITH u AS (UPDATE {table} SET categories = ARRAY(SELECT DISTINCT unnest(COALESCE(categories, '{{}}')
                                                                     || ARRAY[%s]::text[]))
                          WHERE {key_col} = ANY(%s) AND NOT (%s = ANY(COALESCE(categories, '{{}}')))
                          RETURNING 1) SELECT COUNT(*) AS n FROM u""", (cat, keys, cat), "one")
    return int(r["n"])


def tag_candidates(source_urls: Iterable[str], category: str) -> int:
    """Never raises: a failed tag must not lose a batch."""
    try:
        return _append("candidates", "source_url", source_urls, category)
    except Exception:
        return 0


def tag_leads(lead_keys: Iterable[str], category: str) -> int:
    try:
        return _append("im_leads", "lead_key", lead_keys, category)
    except Exception:
        return 0


def listing() -> List[dict]:
    """Every category with how many searches, contacts and leads it has, newest activity first."""
    rows = _q("""SELECT c.name,
                        (SELECT COUNT(*) FROM category_searches s WHERE s.category = c.name) AS searches,
                        (SELECT MAX(started_at) FROM category_searches s WHERE s.category = c.name) AS last_search,
                        (SELECT COUNT(*) FROM candidates x WHERE c.name = ANY(x.categories)) AS contacts
                 FROM search_categories c ORDER BY last_search DESC NULLS LAST, c.name""", fetch="all")
    out = [db._serialize_row(dict(r)) for r in rows]
    try:
        counts = {r["cat"]: int(r["n"]) for r in _q(
            "SELECT unnest(categories) AS cat, COUNT(*) AS n FROM im_leads GROUP BY 1", fetch="all")}
    except Exception:
        counts = {}
    for r in out:
        r["leads"] = counts.get(r["name"], 0)
    return out


def searches(category, limit: int = 200) -> List[dict]:
    """Searches of one category, or of several (a list)."""
    names = [clean(c) for c in (category if isinstance(category, (list, tuple)) else [category]) if clean(c)]
    rows = _q("""SELECT id, category, command, target, mode, run_id, started_at FROM category_searches
                 WHERE category = ANY(%s) ORDER BY started_at DESC LIMIT %s""", (names, limit), "all")
    return [db._serialize_row(dict(r)) for r in rows]
