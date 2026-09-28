"""
Hybrid semantic search on Neon Postgres: dense vectors (pgvector) + full-text (tsvector) + fuzzy names (pg_trgm),
merged with reciprocal-rank fusion.

    search_index(kind, ref, body, tsv, model, embedding)
        kind = lead | candidate | gov      ref = the row id in its own table

Why three signals: vectors find meaning ("HMV trainer" ≈ "heavy vehicle driving instructor"), full-text finds
exact words and registration numbers, trigrams find misspelt names ("Gurprit" ≈ "Gurpreet"). Any of them can be
missing — no embedding key, or no pgvector on the database — and search still works with the others.

Embeddings (stored with their model name; only vectors of the same model are compared):
    1. the self-hosted model server (LOCAL_LLM_URL / LOCAL_EMBED_URL, e.g. Ollama nomic-embed-text, 768-d)
    2. Gemini gemini-embedding-001 at 768 dimensions
    3. Mistral mistral-embed (1024-d)
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import httpx
import psycopg2.extras

import supabase_db as db

_state = {"ready": False, "vector": False, "trgm": False}
_lock = threading.Lock()
EMBED_BATCH = 64
RRF_K = 60
MAX_DISTANCE = float(os.getenv("VECTOR_MAX_DISTANCE", "0.6"))     # cosine distance; unrelated rows are left out


def _exec(sql: str, params=None, fetch: str = "", what: str = "Search index"):
    def run():
        with db._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else None
            conn.commit()
            return out
    return db._with_retry(run, what)


def _try(sql: str) -> bool:
    try:
        with db._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        return True
    except Exception:
        return False


def ensure() -> dict:
    if _state["ready"]:
        return _state
    with _lock:
        if _state["ready"]:
            return _state
        db._ensure_schema()
        _state["vector"] = _try("CREATE EXTENSION IF NOT EXISTS vector")
        _state["trgm"] = _try("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        _exec("""
            CREATE TABLE IF NOT EXISTS search_index (
                kind TEXT NOT NULL, ref TEXT NOT NULL, body TEXT NOT NULL,
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', body)) STORED,
                model TEXT, updated_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (kind, ref));
            CREATE INDEX IF NOT EXISTS idx_search_index_tsv ON search_index USING gin (tsv);""",
              what="Creating search index")
        if _state["vector"]:
            _try("ALTER TABLE search_index ADD COLUMN IF NOT EXISTS embedding vector")
        if _state["trgm"]:
            _try("CREATE INDEX IF NOT EXISTS idx_search_index_trgm ON search_index USING gin (body gin_trgm_ops)")
        _state["ready"] = True
        return _state


# ---------------------------------------------------------------------------
# Embeddings for storage
# ---------------------------------------------------------------------------
async def embed(texts: List[str], keys: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], List[List[float]]]:
    """(model name, vectors) from the first available embedding source, or (None, [])."""
    keys = keys or {}
    texts = [t[:2000] for t in texts]
    if not texts:
        return None, []
    import ai_router
    url = ai_router.embed_endpoint()
    if url:
        model = os.getenv("LOCAL_EMBED_MODEL", "nomic-embed-text")
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(url, json={"model": model, "input": texts},
                                 headers={"Authorization": f"Bearer {os.getenv('LOCAL_LLM_KEY', 'local')}"})
            if r.status_code == 200:
                return f"local:{model}", [d["embedding"] for d in r.json()["data"]]
        except Exception:
            pass
    gemini = keys.get("gemini") or os.getenv("GEMINI_API_KEY", "").strip()
    if gemini:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=gemini)
            res = await client.aio.models.embed_content(
                model="gemini-embedding-001", contents=texts,
                config=types.EmbedContentConfig(output_dimensionality=768))
            return "gemini-embedding-001@768", [list(e.values) for e in res.embeddings]
        except Exception:
            pass
    mistral = keys.get("mistral") or os.getenv("MISTRAL_API_KEY", "").strip()
    if mistral:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post("https://api.mistral.ai/v1/embeddings", headers={"Authorization": f"Bearer {mistral}"},
                                 json={"model": "mistral-embed", "input": texts})
            if r.status_code == 200:
                return "mistral-embed", [d["embedding"] for d in r.json()["data"]]
        except Exception:
            pass
    return None, []


def _vec(v: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def upsert_text(kind: str, rows: List[Tuple[str, str]]) -> int:
    """Full-text / trigram part (immediate, no API calls). rows: [(ref, body)]."""
    st = ensure()
    rows = [(str(r), b[:6000]) for r, b in rows if r and b and b.strip()]
    if not rows:
        return 0

    def run():
        with db._connect() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO search_index (kind, ref, body) VALUES %s
                    ON CONFLICT (kind, ref) DO UPDATE SET body = EXCLUDED.body, updated_at = NOW(),
                        model = CASE WHEN search_index.body = EXCLUDED.body THEN search_index.model END""" +
                    (", embedding = CASE WHEN search_index.body = EXCLUDED.body THEN search_index.embedding END"
                     if st["vector"] else ""),
                    [(kind, r, b) for r, b in rows], page_size=500)
            conn.commit()
    db._with_retry(run, "Updating search index")
    return len(rows)


async def embed_pending(keys: Optional[Dict[str, str]] = None, kinds: Optional[List[str]] = None,
                        limit: int = 256) -> dict:
    """Embed rows that have no vector yet (or a vector from another model). Returns counts."""
    import asyncio
    st = await asyncio.to_thread(ensure)
    if not st["vector"]:
        return {"embedded": 0, "note": "pgvector is not available on this database"}
    model, probe = await embed(["probe"], keys)
    if not model:
        return {"embedded": 0, "note": "no embedding model (local server, Gemini or Mistral key)"}
    rows = await asyncio.to_thread(_exec, """
        SELECT kind, ref, body FROM search_index
        WHERE (model IS DISTINCT FROM %s OR embedding IS NULL) AND (%s::text[] IS NULL OR kind = ANY(%s::text[]))
        ORDER BY updated_at DESC LIMIT %s""", (model, kinds, kinds, limit), "all")
    done = 0
    for i in range(0, len(rows), EMBED_BATCH):
        chunk = rows[i:i + EMBED_BATCH]
        m, vecs = await embed([r["body"] for r in chunk], keys)
        if m != model or len(vecs) != len(chunk):
            break

        def save(chunk=chunk, vecs=vecs):
            with db._connect() as conn:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_batch(cur, """
                        UPDATE search_index SET embedding = %s::vector, model = %s WHERE kind = %s AND ref = %s""",
                        [(_vec(v), model, r["kind"], r["ref"]) for r, v in zip(chunk, vecs)], page_size=200)
                conn.commit()
        await asyncio.to_thread(db._with_retry, save, "Saving embeddings")
        done += len(chunk)
    left = await asyncio.to_thread(_exec, """SELECT COUNT(*) AS n FROM search_index
        WHERE (model IS DISTINCT FROM %s OR embedding IS NULL) AND (%s::text[] IS NULL OR kind = ANY(%s::text[]))""",
                                   (model, kinds, kinds), "one")
    return {"embedded": done, "remaining": int(left["n"]), "model": model}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
async def search(query: str, kinds: List[str], keys: Optional[Dict[str, str]] = None, k: int = 20) -> List[dict]:
    """[{kind, ref, score, signals}] best first — reciprocal-rank fusion of vector, full-text and trigram ranks."""
    import asyncio
    st = await asyncio.to_thread(ensure)
    query = (query or "").strip()
    if not query:
        return []
    model, vecs = (await embed([query], keys)) if st["vector"] else (None, [])
    parts, params = [], []
    if model and vecs:
        parts.append("""SELECT kind, ref, 'vector' AS sig, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS r
                        FROM search_index WHERE kind = ANY(%s) AND model = %s AND embedding IS NOT NULL
                          AND embedding <=> %s::vector < %s
                        ORDER BY embedding <=> %s::vector LIMIT 100""")
        params += [_vec(vecs[0]), kinds, model, _vec(vecs[0]), MAX_DISTANCE, _vec(vecs[0])]
    parts.append("""SELECT kind, ref, 'text' AS sig, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
                    FROM search_index, websearch_to_tsquery('simple', %s) q
                    WHERE kind = ANY(%s) AND tsv @@ q ORDER BY ts_rank_cd(tsv, q) DESC LIMIT 100""")
    words = re.findall(r"[\w-]{2,}", query)
    params += [" or ".join(words) or query, kinds]            # any word matches; more matching words rank higher
    if st["trgm"]:
        parts.append("""SELECT kind, ref, 'fuzzy' AS sig, ROW_NUMBER() OVER (ORDER BY word_similarity(%s, body) DESC) AS r
                        FROM search_index WHERE kind = ANY(%s) AND %s <%% body
                        ORDER BY word_similarity(%s, body) DESC LIMIT 100""")
        params += [query, kinds, query, query]
    sql = ("SELECT kind, ref, SUM(1.0 / (%s + r)) AS score, array_agg(sig) AS signals FROM ("
           + " UNION ALL ".join(f"({p})" for p in parts) + ") x GROUP BY kind, ref ORDER BY score DESC LIMIT %s")
    rows = await asyncio.to_thread(_exec, sql, [RRF_K] + params + [k], "all")
    return [{"kind": r["kind"], "ref": r["ref"], "score": round(float(r["score"]) * 1000, 2),
             "signals": sorted(set(r["signals"]))} for r in rows]


def status() -> dict:
    st = ensure()
    rows = _exec("""SELECT kind, COUNT(*) AS n, COUNT(model) AS embedded, MAX(model) AS model
                    FROM search_index GROUP BY kind""", fetch="all")
    return {"pgvector": st["vector"], "trigram": st["trgm"],
            "index": {r["kind"]: {"rows": r["n"], "embedded": r["embedded"], "model": r["model"]} for r in rows}}


# ---------------------------------------------------------------------------
# What goes into the index
# ---------------------------------------------------------------------------
def lead_body(L: dict) -> str:
    people = " ".join(p.get("name", "") + " " + p.get("role", "")
                      for p in ((L.get("org_contacts") or {}).get("people") or []) if isinstance(p, dict))
    ev = L.get("evidence") or []
    if isinstance(ev, str):
        try:
            ev = json.loads(ev)
        except ValueError:
            ev = [ev]
    return " | ".join(str(x) for x in (L.get("display_name"), L.get("profession"), L.get("origin"),
                                        L.get("destination"), L.get("platform"), L.get("website"), people,
                                        " ".join(map(str, ev[:3]))) if x)


def candidate_body(c: dict) -> str:
    return " | ".join(str(x) for x in (c.get("name"), c.get("current_role"), " ".join(c.get("skills") or []),
                                        c.get("current_location"), " ".join(c.get("target_countries") or []),
                                        c.get("platform"), c.get("evidence_snippet")) if x)


def gov_body(g: dict) -> str:
    return " | ".join(str(x) for x in (g.get("name"), g.get("category"), g.get("reg_no"), g.get("address"),
                                        g.get("city"), g.get("district"), g.get("state"), g.get("dataset")) if x)


def index_leads(leads: List[dict]) -> int:
    return upsert_text("lead", [(str(L.get("id")), lead_body(L)) for L in leads if L.get("id")])


def index_gov_dataset(dataset: str) -> int:
    rows = _exec("SELECT id, name, category, reg_no, address, city, district, state, dataset FROM gov_records "
                 "WHERE dataset = %s", (dataset,), "all")
    return upsert_text("gov", [(str(r["id"]), gov_body(r)) for r in rows])


def index_candidates(limit: int = 5000) -> int:
    rows = _exec("""SELECT id, name, "current_role", skills, current_location, target_countries, platform,
                           evidence_snippet FROM candidates ORDER BY discovered_at DESC LIMIT %s""", (limit,), "all")
    return upsert_text("candidate", [(str(r["id"]), candidate_body(r)) for r in rows])
