"""
Privacy controls for personal contact data.

* Do-not-contact list: phone numbers / emails / profile links of people who opted out (a STOP reply to the
  auto-reply, the 🚫 button on a lead, or added by hand). Saving checks it: a suppressed person is never saved
  again, and adding someone erases their saved phone / email everywhere (the rows themselves are kept).
* Retention: `purge(days)` erases the phone / email / guessed email of *individuals* (not businesses) saved more
  than `days` ago — run it from ⚙️ Settings, or on a schedule by a worker (PRIVACY_RETENTION_DAYS).

Provenance is kept per contact: candidates.contact_source and source_url say where each contact came from.
"""

from __future__ import annotations

import re
import threading
from typing import Dict, Iterable, List, Optional

import psycopg2.extras

import supabase_db as db

SCHEMA = """
CREATE TABLE IF NOT EXISTS do_not_contact (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    phone_key TEXT, email TEXT, profile TEXT, name TEXT, reason TEXT, created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dnc_phone ON do_not_contact (phone_key);
CREATE INDEX IF NOT EXISTS idx_dnc_email ON do_not_contact (lower(email));
"""
_ready: set = set()          # schemas (organizations) whose tables exist
_lock = threading.Lock()
_cache: Dict[str, set] = {}


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
                db._with_retry(mk, "Creating do-not-contact table")
                _ready.add(db.current_schema())

    def run():
        with db._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else None
            conn.commit()
            return out
    return db._with_retry(run, "Do-not-contact list")


def phone_key(p: Optional[str]) -> str:
    return re.sub(r"\D", "", p or "")[-10:]


def _load() -> Dict[str, set]:
    rows = _q("SELECT phone_key, lower(email) AS email, lower(profile) AS profile FROM do_not_contact", fetch="all")
    return {"phones": {r["phone_key"] for r in rows if r["phone_key"]},
            "emails": {r["email"] for r in rows if r["email"]},
            "profiles": {r["profile"] for r in rows if r["profile"]}}


def suppressed(phone: Optional[str] = None, email: Optional[str] = None, profile: Optional[str] = None,
               lists: Optional[Dict[str, set]] = None) -> bool:
    lists = lists or _load()
    return bool((phone_key(phone) and phone_key(phone) in lists["phones"]) or
                (email and email.lower() in lists["emails"]) or
                (profile and profile.lower().rstrip("/") in lists["profiles"]))


def filter_rows(rows: Iterable[dict], phone_k: str = "phone", email_k: str = "email",
                profile_k: str = "profile_url") -> List[dict]:
    """Rows of people who did not opt out (one list read per call)."""
    rows = list(rows)
    try:
        lists = _load()
    except Exception:
        return rows                                  # never lose a save because the list is unreachable
    if not any(lists.values()):
        return rows
    return [r for r in rows if not suppressed(r.get(phone_k), r.get(email_k), r.get(profile_k), lists)]


def add(phone: Optional[str] = None, email: Optional[str] = None, profile: Optional[str] = None,
        name: Optional[str] = None, reason: str = "opted out") -> dict:
    """Opt someone out and erase their saved contact details."""
    pk, em, pr = phone_key(phone), (email or "").strip().lower(), (profile or "").strip().lower().rstrip("/")
    if not (pk or em or pr):
        raise ValueError("give a phone number, email or profile link")
    _q("INSERT INTO do_not_contact (phone_key, email, profile, name, reason) VALUES (%s, %s, %s, %s, %s)",
       (pk or None, em or None, pr or None, name, reason))
    erased = 0
    conds = []
    params: list = []
    if pk:
        conds.append("right(regexp_replace(COALESCE(phone,''), '\\D', '', 'g'), 10) = %s")
        params.append(pk)
    if em:
        conds.append("lower(email) = %s")
        params.append(em)
    if pr:
        conds.append("lower(rtrim(profile_url, '/')) = %s")
        params.append(pr)
    where = " OR ".join(conds)
    # The row stays (history is never deleted); only this person's contact details are erased and the row is
    # marked, so outreach skips it.
    for table, sets in (("candidates", "phone = NULL, email = NULL, email_guess = NULL, email_status = NULL, "
                                       "outreach_status = 'do_not_contact'"),
                        ("im_leads", "phone = NULL, email = NULL, status = 'DO_NOT_CONTACT'")):
        try:
            r = _q(f"WITH d AS (UPDATE {table} SET {sets} WHERE {where} RETURNING 1) SELECT COUNT(*) AS n FROM d",
                   params, "one")
            erased += int(r["n"])
        except Exception:
            pass
    return {"erased": erased}


def listing(limit: int = 200) -> List[dict]:
    rows = _q("SELECT id, phone_key, email, profile, name, reason, created_at FROM do_not_contact "
              "ORDER BY created_at DESC LIMIT %s", (limit,), "all")
    return [db._serialize_row(dict(r)) for r in rows]


def remove(entry_id: int) -> None:
    _q("DELETE FROM do_not_contact WHERE id = %s", (entry_id,))


def purge(days: int) -> dict:
    """Erase individuals' phone / email / guessed email saved more than `days` ago (businesses are kept)."""
    days = max(1, int(days))
    c = _q(f"""WITH u AS (UPDATE candidates SET phone = NULL, email = NULL, email_guess = NULL, email_status = NULL
                          WHERE discovered_at < NOW() - INTERVAL '{days} days'
                            AND COALESCE(shows_interest, FALSE) = TRUE
                            AND (phone IS NOT NULL OR email IS NOT NULL OR email_guess IS NOT NULL)
                          RETURNING 1) SELECT COUNT(*) AS n FROM u""", fetch="one")
    try:
        L = _q(f"""WITH u AS (UPDATE im_leads SET phone = NULL, email = NULL
                              WHERE last_seen < NOW() - INTERVAL '{days} days' AND lead_key NOT LIKE 'org:%%'
                                AND (phone IS NOT NULL OR email IS NOT NULL) RETURNING 1)
                   SELECT COUNT(*) AS n FROM u""", fetch="one")
    except Exception:
        L = {"n": 0}
    return {"candidates": int(c["n"]), "leads": int(L["n"]), "days": days}
