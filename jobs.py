"""
Background jobs in Neon: the Vercel app (API gateway) queues a run; persistent workers (worker/worker.py, on any
server or container platform) claim it, run it end to end and write progress back. The page can be closed.

    im_jobs       one row per job: status queued → running → done | failed | cancelled, options, log, stats
    im_workers    heartbeat of each worker (so the page can say whether one is online)

Claiming uses FOR UPDATE SKIP LOCKED, so any number of workers can share the queue safely. A running job whose
worker stops sending heartbeats is put back in the queue (once), then marked failed.
"""

from __future__ import annotations

import json
import socket
import os
import threading
from typing import List, Optional

import psycopg2.extras

import supabase_db as db

SCHEMA = """
CREATE TABLE IF NOT EXISTS im_jobs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'run', status TEXT NOT NULL DEFAULT 'queued',
    command TEXT, options JSONB DEFAULT '{}'::jsonb, run_id TEXT,
    log JSONB DEFAULT '[]'::jsonb, stats JSONB DEFAULT '{}'::jsonb, error TEXT,
    attempts INT DEFAULT 0, cancel_requested BOOLEAN DEFAULT FALSE, worker TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(), started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, heartbeat_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_im_jobs_status ON im_jobs (status, created_at);
ALTER TABLE im_jobs ADD COLUMN IF NOT EXISTS run_after TIMESTAMPTZ;
ALTER TABLE im_jobs ADD COLUMN IF NOT EXISTS org_id TEXT;
CREATE TABLE IF NOT EXISTS im_workers (
    worker TEXT PRIMARY KEY, info JSONB DEFAULT '{}'::jsonb, last_seen TIMESTAMPTZ DEFAULT NOW()
);
"""
STALE_MINUTES = 10
MAX_LOG_LINES = 2000
_ready: set = set()
_lock = threading.Lock()


def _q(sql: str, params=None, fetch: str = "", what: str = "Job queue"):
    with db.use_schema("public"):          # one queue for all organizations
        return _q_public(sql, params, fetch, what)


def _q_public(sql: str, params=None, fetch: str = "", what: str = "Job queue"):
    if "public" not in _ready:
        with _lock:
            if "public" not in _ready:
                db._ensure_schema()

                def mk():
                    with db._connect() as conn:
                        with conn.cursor() as cur:
                            cur.execute(SCHEMA)
                        conn.commit()
                db._with_retry(mk, "Creating job tables")
                _ready.add("public")

    def run():
        with db._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else None
            conn.commit()
            return out
    return db._with_retry(run, what)


def _row(r) -> Optional[dict]:
    return db._serialize_row(dict(r)) if r else None


def worker_id() -> str:
    return os.getenv("WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}"


# ---------------------------------------------------------------------------
def enqueue(kind: str, command: str, options: dict, run_after_hours: float = 0, org_id: str = "") -> dict:
    """Queue a job; with run_after_hours it waits that long before a worker may start it. org_id: the
    organization whose data (schema) and keys the job uses ("" = the master workspace)."""
    return _row(_q("""INSERT INTO im_jobs (kind, command, options, run_after, org_id)
                      VALUES (%s, %s, %s, CASE WHEN %s > 0 THEN NOW() + make_interval(secs => %s) END, NULLIF(%s, ''))
                      RETURNING *""",
                   (kind, command, json.dumps(options), run_after_hours, run_after_hours * 3600, org_id or ""), "one",
                   "Queuing job"))


def get(job_id: str, log_from: int = 0, org_id: str = "") -> Optional[dict]:
    r = _q("""SELECT id, kind, status, command, run_id, stats, error, attempts, cancel_requested, worker, created_at,
                     started_at, finished_at, heartbeat_at, jsonb_array_length(log) AS log_len,
                     COALESCE((SELECT jsonb_agg(e) FROM jsonb_array_elements(log) WITH ORDINALITY AS t(e, i)
                               WHERE i > %s), '[]'::jsonb) AS log
              FROM im_jobs WHERE id::text = %s AND COALESCE(org_id, '') = %s""",
           (log_from, job_id, org_id or ""), "one")
    return _row(r)


def recent(limit: int = 20, org_id: str = "") -> List[dict]:
    rows = _q("""SELECT id, kind, status, command, run_id, stats, error, created_at, started_at, finished_at, worker,
                        run_after, options->>'rounds' AS rounds, options->>'repeat_hours' AS repeat_hours
                 FROM im_jobs WHERE COALESCE(org_id, '') = %s ORDER BY created_at DESC LIMIT %s""",
              (org_id or "", limit), "all")
    return [_row(r) for r in rows]


def cancel(job_id: str, org_id: str = "") -> None:
    _q("""UPDATE im_jobs SET cancel_requested = TRUE,
              status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
              finished_at = CASE WHEN status = 'queued' THEN NOW() ELSE finished_at END
          WHERE id::text = %s AND COALESCE(org_id, '') = %s""", (job_id, org_id or ""))


# --- worker side -------------------------------------------------------------------------------------
def requeue_stale() -> int:
    """Jobs whose worker vanished: back to the queue once, then failed."""
    rows = _q(f"""UPDATE im_jobs SET status = CASE WHEN attempts < 2 THEN 'queued' ELSE 'failed' END,
                     error = CASE WHEN attempts < 2 THEN error ELSE 'worker stopped responding' END,
                     finished_at = CASE WHEN attempts < 2 THEN NULL ELSE NOW() END
                 WHERE status = 'running' AND heartbeat_at < NOW() - INTERVAL '{STALE_MINUTES} minutes'
                 RETURNING id""", fetch="all")
    return len(rows or [])


def claim(worker: str, kinds: Optional[List[str]] = None) -> Optional[dict]:
    r = _q("""UPDATE im_jobs SET status = 'running', worker = %s, attempts = attempts + 1,
                     started_at = COALESCE(started_at, NOW()), heartbeat_at = NOW()
              WHERE id = (SELECT id FROM im_jobs WHERE status = 'queued' AND NOT cancel_requested
                            AND (run_after IS NULL OR run_after <= NOW())
                            AND (%s::text[] IS NULL OR kind = ANY(%s::text[]))
                          ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
              RETURNING *""", (worker, kinds, kinds), "one", "Claiming job")
    return _row(r)


def heartbeat(worker: str, info: dict, job_id: Optional[str] = None) -> bool:
    """Returns True when the running job was asked to stop."""
    _q("""INSERT INTO im_workers (worker, info, last_seen) VALUES (%s, %s, NOW())
          ON CONFLICT (worker) DO UPDATE SET info = EXCLUDED.info, last_seen = NOW()""", (worker, json.dumps(info)))
    if not job_id:
        return False
    r = _q("UPDATE im_jobs SET heartbeat_at = NOW() WHERE id::text = %s RETURNING cancel_requested",
           (job_id,), "one")
    return bool(r and r["cancel_requested"])


def log(job_id: str, lines: List[str]) -> None:
    if lines:
        _q(f"""UPDATE im_jobs SET log = (CASE WHEN jsonb_array_length(log) > {MAX_LOG_LINES}
                                          THEN '[]'::jsonb ELSE log END) || %s::jsonb, heartbeat_at = NOW()
               WHERE id::text = %s""", (json.dumps(lines), job_id))


def update(job_id: str, **fields) -> None:
    sets, vals = [], []
    for k, v in fields.items():
        if k in ("stats", "options"):
            sets.append(f"{k} = %s::jsonb")
            vals.append(json.dumps(v))
        elif k in ("status", "error", "run_id"):
            sets.append(f"{k} = %s")
            vals.append(v)
    if fields.get("status") in ("done", "failed", "cancelled"):
        sets.append("finished_at = NOW()")
        # page keys sent with the job are erased as soon as it ends
        sets.append("options = options - 'keys' - 'llm_keys'")
    if sets:
        _q(f"UPDATE im_jobs SET {', '.join(sets)} WHERE id::text = %s", vals + [job_id])


def workers(active_minutes: int = 3) -> List[dict]:
    rows = _q(f"""SELECT worker, info, last_seen FROM im_workers
                  WHERE last_seen > NOW() - INTERVAL '{int(active_minutes)} minutes' ORDER BY last_seen DESC""",
              fetch="all")
    return [_row(r) for r in rows]
