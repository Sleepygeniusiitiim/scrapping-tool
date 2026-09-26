"""
Vercel serverless API (FastAPI) for the Candidate Sourcing Agent.

Connected to Neon PostgreSQL (`DATABASE_URL`) and Google Gemini (`GEMINI_API_KEY`).
Every endpoint verifies `X-App-Password` against `APP_PASSWORD` (defaulting to
`CSA-Neon-Vercel-2026!` if not overridden in environment variables).
Also supports passing `X-Gemini-Key` from the web UI if `GEMINI_API_KEY` is not
set in server environment variables.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs

# Shared modules live in the repo root.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except Exception:
    pass

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import pipeline  # noqa: E402
import supabase_db as db  # noqa: E402
from gemini_client import DEFAULT_MODEL, Gemini, GeminiError, GeminiQuotaError  # noqa: E402
from openrouter_client import PROVIDERS, OpenRouter  # noqa: E402
import outreach  # noqa: E402
from schema import CandidateRecord  # noqa: E402

DEFAULT_APP_PASSWORD = "CSA-Neon-Vercel-2026!"

app = FastAPI(title="Candidate Sourcing Agent API", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# Vercel ASGI path normalization middleware
# ---------------------------------------------------------------------------
class VercelPathNormalizedMiddleware:
    """
    When Vercel rewrites `/api/<route>` to `/api/index`, `@vercel/python` may
    set `scope["path"]` to `/api/index` or `/index` instead of `/api/<route>`.
    This middleware restores the original `/api/<route>` path from:
      1. `X-Endpoint` request header (sent by public/index.html)
      2. `__path` query parameter (sent by vercel.json rewrite & public/index.html)
      3. `x-matched-path` / `x-now-route-matches` Vercel headers
    """

    def __init__(self, app_instance):
        self.app = app_instance

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
            qs = parse_qs(scope.get("query_string", b"").decode("latin1"))

            target_path = None
            if headers.get("x-endpoint"):
                target_path = headers["x-endpoint"].strip()
            elif "__path" in qs and qs["__path"]:
                target_path = qs["__path"][0].strip()
            elif "x-now-route-matches" in headers:
                matches = parse_qs(headers["x-now-route-matches"])
                if "1" in matches and matches["1"]:
                    target_path = matches["1"][0].strip()
                elif "path" in matches and matches["path"]:
                    target_path = matches["path"][0].strip()

            current_path = scope.get("path", "")
            if target_path:
                clean = target_path.lstrip("/")
                if clean.startswith("api/"):
                    scope["path"] = "/" + clean
                else:
                    scope["path"] = "/api/" + clean
            elif current_path in ("/api/index", "/api/index.py", "/index", "/index.py"):
                scope["path"] = "/api/health"
            elif current_path in ("/health", "/plan", "/search", "/dedup", "/process", "/candidates"):
                scope["path"] = "/api" + current_path

        await self.app(scope, receive, send)


app.add_middleware(VercelPathNormalizedMiddleware)


# ---------------------------------------------------------------------------
# Auth & clients
# ---------------------------------------------------------------------------
def require_password(x_app_password: Optional[str] = Header(default=None)) -> None:
    expected = (os.getenv("APP_PASSWORD") or DEFAULT_APP_PASSWORD).strip()
    provided = (x_app_password or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(401, "Wrong password. Use your Vercel APP_PASSWORD.")


def _gemini(
    x_gemini_key: Optional[str] = Header(default=None),
    x_gemini_model: Optional[str] = Header(default=None),
    x_gemini_mode: Optional[str] = Header(default=None),
    x_openrouter_key: Optional[str] = Header(default=None),
    x_openrouter_model: Optional[str] = Header(default=None),
    x_llm_provider: Optional[str] = Header(default=None),
    x_llm_key: Optional[str] = Header(default=None),
    x_llm_model: Optional[str] = Header(default=None),
):
    """The AI for this request: the provider chosen on the page (OpenRouter / Cerebras / Groq) with the
    key from the page or its env var; otherwise the first provider with an env key; otherwise Gemini."""
    provider = (x_llm_provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    key, model = (x_llm_key or "").strip(), (x_llm_model or "").strip()
    if not key and x_openrouter_key and provider in ("", "openrouter"):      # older pages
        provider, key, model = "openrouter", x_openrouter_key.strip(), (x_openrouter_model or "").strip()
    if provider in PROVIDERS and not key:
        key = os.getenv(PROVIDERS[provider]["env"], "").strip()
    if not key and provider in ("", "auto"):
        provider = next((p for p, c in PROVIDERS.items() if os.getenv(c["env"], "").strip()), "")
        key = os.getenv(PROVIDERS[provider]["env"], "").strip() if provider else ""
    if key and provider in PROVIDERS:
        model = model or os.getenv(provider.upper() + "_MODEL", "").strip()
        try:
            return OpenRouter(key, model=model, provider=provider)
        except GeminiError as exc:
            raise HTTPException(500, str(exc))
    api_key = (x_gemini_key or os.getenv("GEMINI_API_KEY") or "").strip()
    model = (x_gemini_model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
    mode = (x_gemini_mode or os.getenv("GEMINI_MODE") or "auto").strip()
    if not api_key:
        raise HTTPException(500, "No AI key: choose a provider and enter its key on the page, or set "
                                 "OPENROUTER_API_KEY / CEREBRAS_API_KEY / GROQ_API_KEY (or GEMINI_API_KEY) in Vercel.")
    try:
        return Gemini(api_key, model=model, mode=mode)
    except GeminiError as exc:
        raise HTTPException(500, str(exc))


def _db(x_database_url: Optional[str] = Header(default=None)) -> None:
    # The connection string comes from the server environment only. Accepting it
    # from a request header would let any caller point the server at another host.
    del x_database_url
    try:
        db.init_supabase()
    except db.SupabaseError as exc:
        raise HTTPException(500, str(exc))


auth = [Depends(require_password)]
router = APIRouter(dependencies=auth)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
class PlanIn(BaseModel):
    intent: str = Field(..., min_length=3, max_length=2000)
    num_waves: int = Field(3, ge=1, le=6)
    queries_per_wave: int = Field(5, ge=1, le=10)
    round: int = Field(1, ge=1, le=100)
    exclude_queries: List[str] = Field(default_factory=list, max_length=500)
    respect_robots: bool = False


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
@router.get("/health")
async def health(
    x_gemini_key: Optional[str] = Header(default=None),
    x_gemini_model: Optional[str] = Header(default=None),
    x_gemini_mode: Optional[str] = Header(default=None),
    x_openrouter_key: Optional[str] = Header(default=None),
    x_openrouter_model: Optional[str] = Header(default=None),
    x_llm_provider: Optional[str] = Header(default=None),
    x_llm_key: Optional[str] = Header(default=None),
    x_llm_model: Optional[str] = Header(default=None),
    x_database_url: Optional[str] = Header(default=None),
):
    out = {}
    try:
        _db(x_database_url)
        status = await run_in_threadpool(db.check_connection)
        out["database"] = status
        out["supabase"] = status
    except Exception as exc:
        err = str(getattr(exc, "detail", exc))
        out["database_error"] = err
        out["supabase_error"] = err
    out["scrapedo"] = "configured" if os.getenv("SCRAPEDO_TOKEN", "").strip() else "not set"
    try:
        gem = _gemini(x_gemini_key=x_gemini_key, x_gemini_model=x_gemini_model, x_gemini_mode=x_gemini_mode,
                      x_openrouter_key=x_openrouter_key, x_openrouter_model=x_openrouter_model,
                      x_llm_provider=x_llm_provider, x_llm_key=x_llm_key, x_llm_model=x_llm_model)
        out["gemini"] = await gem.ping()
    except Exception as exc:
        out["gemini_error"] = str(getattr(exc, "detail", exc))
    return out


@router.post("/plan")
async def plan(
    body: PlanIn,
    gemini = Depends(_gemini),
):
    try:
        result = await pipeline.plan_search(gemini, body.intent, body.num_waves, body.queries_per_wave,
                                             body.round, body.exclude_queries, body.respect_robots)
    except GeminiError as exc:
        raise HTTPException(502, f"Planning failed: {exc}")
    if not result["waves"]:
        raise HTTPException(422, "Gemini returned an empty plan — try rephrasing the intent.")
    return result


@router.post("/search")
def search(body: QueryIn):
    backend = body.backend if body.backend in ("auto", "duckduckgo", "google") else "auto"
    return pipeline.run_query(body.query, body.max_results, body.region, backend)


@router.post("/dedup")
def dedup(body: DedupIn, x_database_url: Optional[str] = Header(default=None)):
    _db(x_database_url)
    try:
        return {"fresh": pipeline.dedup_urls(body.urls)}
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))


@router.post("/process")
async def process(
    body: BatchIn,
    gemini = Depends(_gemini),
    x_database_url: Optional[str] = Header(default=None),
):
    _db(x_database_url)
    try:
        return await pipeline.process_batch(
            gemini,
            body.intent,
            [i.model_dump() for i in body.items],
            body.wave_tag,
            body.page_timeout_s,
            body.respect_robots,
            body.snippet_fallback,
        )
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))
    except GeminiError as exc:
        raise HTTPException(502, str(exc))


@router.get("/candidates")
def candidates(x_database_url: Optional[str] = Header(default=None)):
    _db(x_database_url)
    try:
        rows = db.fetch_all_candidates()
        return {"candidates": rows, "scraped_urls": db.count_scraped_urls()}
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))


@router.post("/candidates")
def save(body: SaveIn, x_database_url: Optional[str] = Header(default=None)):
    _db(x_database_url)
    try:
        return {"saved": db.save_candidates(body.records)}
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))


# ---------------------------------------------------------------------------
# Assisted outreach (nothing is sent from here — recruiters send each message)
# ---------------------------------------------------------------------------
class DraftIn(BaseModel):
    candidate_ids: List[str] = Field(..., min_length=1, max_length=20)
    brief: outreach.OutreachBrief


class StatusIn(BaseModel):
    candidate_id: str
    status: str = Field(..., pattern="^(new|drafted|sent|replied|not_interested)$")
    message: Optional[str] = Field(None, max_length=2000)


class ReplyIn(BaseModel):
    candidate_id: str
    reply_text: str = Field(..., min_length=1, max_length=5000)


@router.post("/outreach/draft")
async def outreach_draft(body: DraftIn, gemini=Depends(_gemini)):
    _db()
    try:
        cands = await run_in_threadpool(db.get_candidates, body.candidate_ids)
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))
    results = await asyncio.gather(*(outreach.draft_message(gemini, c, body.brief) for c in cands),
                                   return_exceptions=True)
    out = []
    for c, r in zip(cands, results):
        if isinstance(r, Exception):
            if isinstance(r, GeminiError) and ("API key" in str(r) or isinstance(r, GeminiQuotaError)):
                raise HTTPException(502, str(r))
            out.append({"id": c["id"], "error": str(r)[:200]})
            continue
        row = await run_in_threadpool(db.update_candidate, c["id"],
                                      {"outreach_message": r, "outreach_status": "drafted"})
        out.append(row)
    return {"drafts": out}


@router.post("/outreach/status")
def outreach_status(body: StatusIn):
    _db()
    fields = {"outreach_status": body.status}
    if body.message is not None:
        fields["outreach_message"] = body.message
    if body.status == "sent":
        fields["outreach_sent_at"] = "now()"
    try:
        return db.update_candidate(body.candidate_id, fields)
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))


@router.post("/outreach/reply")
async def outreach_reply(body: ReplyIn, gemini=Depends(_gemini)):
    _db()
    try:
        reading = await outreach.read_reply(gemini, body.reply_text)
    except GeminiError as exc:
        raise HTTPException(502, str(exc))
    fields = {"reply_text": body.reply_text, "replied_at": "now()",
              "outreach_status": "not_interested" if reading.interested is False else "replied"}
    if reading.email or reading.phone:
        # Shared by the candidate in reply to us — replaces anything scraped.
        if reading.email:
            fields["email"] = reading.email
        if reading.phone:
            fields["phone"] = reading.phone
        fields["contact_source"] = "shared_in_reply"
        fields["contact_shared_at"] = "now()"
    try:
        row = await run_in_threadpool(db.update_candidate, body.candidate_id, fields)
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))
    return {"candidate": row, "reading": reading.model_dump()}


# Mount routes at both `/api/*` and root `/*` so Vercel serverless rewrites
# always resolve regardless of how `@vercel/python` sets `scope["path"]`.
app.include_router(router, prefix="/api")
app.include_router(router, prefix="")

# Local development: serve the page from the same server (Vercel serves public/ itself).
if not os.getenv("VERCEL"):
    from fastapi.staticfiles import StaticFiles  # noqa: E402

    app.mount("/", StaticFiles(directory=ROOT_DIR / "public", html=True), name="static")
