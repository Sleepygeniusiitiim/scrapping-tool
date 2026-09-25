"""
Vercel serverless API (FastAPI) for the Candidate Sourcing Agent.

Secrets never reach the browser: GEMINI_API_KEY, SUPABASE_URL and
SUPABASE_KEY are read from Vercel environment variables. Every endpoint
requires the `X-App-Password` header to match APP_PASSWORD — the service_role
key can read and write every stored candidate, so the API must not be open.
"""

from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path
from typing import List, Optional

# Shared modules live in the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import pipeline  # noqa: E402
import supabase_db as db  # noqa: E402
from gemini_client import DEFAULT_MODEL, Gemini, GeminiError  # noqa: E402
from schema import CandidateRecord  # noqa: E402

app = FastAPI(title="Candidate Sourcing Agent API", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# Auth & clients
# ---------------------------------------------------------------------------
def require_password(x_app_password: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("APP_PASSWORD", "")
    if not expected:
        raise HTTPException(503, "APP_PASSWORD is not set in the Vercel environment variables.")
    if not x_app_password or not hmac.compare_digest(x_app_password, expected):
        raise HTTPException(401, "Wrong password.")


def _gemini() -> Gemini:
    try:
        return Gemini(os.getenv("GEMINI_API_KEY", ""),
                      model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
                      mode=os.getenv("GEMINI_MODE", "auto"))
    except GeminiError as exc:
        raise HTTPException(500, str(exc))


def _db() -> None:
    try:
        db.init_supabase()
    except db.SupabaseError as exc:
        raise HTTPException(500, str(exc))


auth = [Depends(require_password)]


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
class PlanIn(BaseModel):
    intent: str = Field(..., min_length=3, max_length=2000)
    num_waves: int = Field(3, ge=1, le=6)
    queries_per_wave: int = Field(5, ge=1, le=10)


class QueryIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(10, ge=1, le=30)
    region: str = "in-en"
    backend: str = "auto"


class DedupIn(BaseModel):
    urls: List[str] = Field(default_factory=list, max_length=1000)


class Hit(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""


class BatchIn(BaseModel):
    intent: str
    items: List[Hit] = Field(..., min_length=1, max_length=8)
    wave_tag: str = "W?"
    page_timeout_s: int = Field(15, ge=5, le=30)
    respect_robots: bool = True
    snippet_fallback: bool = True


class SaveIn(BaseModel):
    records: List[CandidateRecord] = Field(..., max_length=2000)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", dependencies=auth)
async def health():
    out = {}
    try:
        _db()
        out["supabase"] = await run_in_threadpool(db.check_connection)
    except Exception as exc:
        out["supabase_error"] = str(getattr(exc, "detail", exc))
    try:
        out["gemini"] = await _gemini().ping()
    except Exception as exc:
        out["gemini_error"] = str(getattr(exc, "detail", exc))
    return out


@app.post("/api/plan", dependencies=auth)
async def plan(body: PlanIn):
    try:
        result = await pipeline.plan_search(_gemini(), body.intent, body.num_waves, body.queries_per_wave)
    except GeminiError as exc:
        raise HTTPException(502, f"Planning failed: {exc}")
    if not result["waves"]:
        raise HTTPException(422, "Gemini returned an empty plan — try rephrasing the intent.")
    return result


@app.post("/api/search", dependencies=auth)
def search(body: QueryIn):
    backend = body.backend if body.backend in ("auto", "duckduckgo") else "auto"
    return pipeline.run_query(body.query, body.max_results, body.region, backend)


@app.post("/api/dedup", dependencies=auth)
def dedup(body: DedupIn):
    _db()
    try:
        return {"fresh": pipeline.dedup_urls(body.urls)}
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))


@app.post("/api/process", dependencies=auth)
async def process(body: BatchIn):
    _db()
    try:
        return await pipeline.process_batch(
            _gemini(), body.intent, [i.model_dump() for i in body.items], body.wave_tag,
            body.page_timeout_s, body.respect_robots, body.snippet_fallback,
        )
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))
    except GeminiError as exc:
        raise HTTPException(502, str(exc))


@app.get("/api/candidates", dependencies=auth)
def candidates():
    _db()
    try:
        rows = db.fetch_all_candidates()
        return {"candidates": rows, "scraped_urls": db.count_scraped_urls()}
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))


@app.post("/api/candidates", dependencies=auth)
def save(body: SaveIn):
    _db()
    try:
        return {"saved": db.save_candidates(body.records)}
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))


# Local development: serve the page from the same server (Vercel serves public/ itself).
if not os.getenv("VERCEL"):
    from fastapi.staticfiles import StaticFiles  # noqa: E402

    app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "public", html=True), name="static")
