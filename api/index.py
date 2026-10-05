"""
Vercel serverless API (FastAPI) for the Candidate Sourcing Agent.

Connected to Neon PostgreSQL (`DATABASE_URL`) and Google Gemini (`GEMINI_API_KEY`).
Every endpoint verifies `X-App-Password` against `APP_PASSWORD` (defaulting to
set in the APP_PASSWORD environment variable).
Also supports passing `X-Gemini-Key` from the web UI if `GEMINI_API_KEY` is not
set in server environment variables.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import re
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

from fastapi import Query, APIRouter, Depends, FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.responses import PlainTextResponse  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import pipeline  # noqa: E402
import supabase_db as db  # noqa: E402
from gemini_client import DEFAULT_MODEL, Gemini, GeminiError, GeminiQuotaError  # noqa: E402
from openrouter_client import PROVIDERS, OpenRouter  # noqa: E402
from ai_chain import FALLBACK_ORDER, AIChain  # noqa: E402
import integrations  # noqa: E402
import dates  # noqa: E402
import portal_import  # noqa: E402
import meta_autoreply  # noqa: E402
import gov_registry  # noqa: E402
import ai_router  # noqa: E402
from intent_miner import engine as im_engine, export as im_export, store as im_store  # noqa: E402
from intent_miner.models import QuerySpec  # noqa: E402
from intent_miner.understand import understand as im_understand  # noqa: E402
import outreach  # noqa: E402
from schema import CandidateRecord, clean_email, clean_phone  # noqa: E402

DEFAULT_APP_PASSWORD = ""        # no built-in password: APP_PASSWORD must be set in the environment

app = FastAPI(title="Candidate Sourcing Agent API", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# Vercel ASGI path normalization middleware
# ---------------------------------------------------------------------------
class VercelPathNormalizedMiddleware:
    """
    When Vercel rewrites `/api/<route>` to `/api/index`, `@vercel/python` may
    set `scope["path"]` to `/api/index` or `/index` instead of `/api/<route>`.
    This middleware restores the original `/api/<route>` path from:
      1. `X-Endpoint` request header (sent by public/app.html)
      2. `__path` query parameter (sent by vercel.json rewrite & public/app.html)
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
                # the endpoint may carry its query ("im/leads?min_score=0"): the query is already in the URL,
                # only the path part names the route
                target_path = target_path.split("?", 1)[0].split("#", 1)[0]
                clean = target_path.lstrip("/")
                if clean.startswith("api/"):
                    scope["path"] = "/" + clean
                else:
                    scope["path"] = "/api/" + clean
            elif current_path in ("/api/index", "/api/index.py", "/index", "/index.py"):
                scope["path"] = "/api/health"
            elif current_path in ("/health", "/plan", "/search", "/dedup", "/process", "/candidates", "/enrich", "/import"):
                scope["path"] = "/api" + current_path

        await self.app(scope, receive, send)


app.add_middleware(VercelPathNormalizedMiddleware)


# ---------------------------------------------------------------------------
# Auth & clients
# ---------------------------------------------------------------------------
def require_password(x_app_password: Optional[str] = Header(default=None)) -> None:
    expected = (os.getenv("APP_PASSWORD") or DEFAULT_APP_PASSWORD).strip()
    if not expected:
        raise HTTPException(503, "APP_PASSWORD is not set on the server: add it in the Vercel project's environment "
                                 "variables and redeploy.")
    provided = (x_app_password or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(401, "Wrong password. Use your Vercel APP_PASSWORD.")


def _json_header(value: Optional[str]) -> dict:
    try:
        data = json.loads(value or "{}")
    except ValueError:
        return {}
    return {str(k).lower(): str(v).strip() for k, v in data.items() if v} if isinstance(data, dict) else {}


def _gemini(
    x_gemini_key: Optional[str] = Header(default=None),
    x_gemini_model: Optional[str] = Header(default=None),
    x_gemini_mode: Optional[str] = Header(default=None),
    x_openrouter_key: Optional[str] = Header(default=None),
    x_openrouter_model: Optional[str] = Header(default=None),
    x_llm_provider: Optional[str] = Header(default=None),
    x_llm_key: Optional[str] = Header(default=None),
    x_llm_model: Optional[str] = Header(default=None),
    x_llm_keys: Optional[str] = Header(default=None),
    x_llm_models: Optional[str] = Header(default=None),
):
    """The AI for this request: a fallback chain starting with the provider chosen on the page, then
    every other provider that has a key (on the page or in Vercel env vars), free tiers first. When one
    runs out of credits or limits, the next one takes over."""
    provider = (x_llm_provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    keys, models = _json_header(x_llm_keys), _json_header(x_llm_models)
    if x_openrouter_key and provider in ("", "openrouter"):                  # older pages
        keys.setdefault("openrouter", x_openrouter_key.strip())
        if x_openrouter_model:
            models.setdefault("openrouter", x_openrouter_model.strip())
    if provider in PROVIDERS:
        if (x_llm_key or "").strip():
            keys[provider] = x_llm_key.strip()
        if (x_llm_model or "").strip():
            models[provider] = x_llm_model.strip()
    if (x_gemini_key or "").strip():
        keys["gemini"] = x_gemini_key.strip()
    if x_gemini_model:
        models.setdefault("gemini", x_gemini_model.strip())

    def key_for(p: str) -> str:
        env = "GEMINI_API_KEY" if p == "gemini" else PROVIDERS[p]["env"]
        return keys.get(p) or os.getenv(env, "").strip()

    def factory(p: str, key: str):
        model = models.get(p) or os.getenv(p.upper() + "_MODEL", "").strip()
        if p == "gemini":
            mode = (x_gemini_mode or os.getenv("GEMINI_MODE") or "auto").strip()
            return lambda: Gemini(key, model=model or DEFAULT_MODEL, mode=mode)
        return lambda: OpenRouter(key, model=model, provider=p)

    order = ([provider] if provider in FALLBACK_ORDER else []) + [p for p in FALLBACK_ORDER if p != provider]
    entries = [("Gemini" if p == "gemini" else PROVIDERS[p]["label"], factory(p, k))
               for p in order if (k := key_for(p))]
    if not entries:
        raise HTTPException(500, "No AI key: choose a provider and enter its key on the page, or set "
                                 "OPENROUTER_API_KEY / GROQ_API_KEY / CEREBRAS_API_KEY / MISTRAL_API_KEY / "
                                 "SAMBANOVA_API_KEY / NVIDIA_API_KEY / GITHUB_MODELS_TOKEN / DEEPSEEK_API_KEY / "
                                 "MOONSHOT_API_KEY / GEMINI_API_KEY in Vercel.")
    return AIChain(entries)


def _thinker(gemini=Depends(_gemini), x_claude_key: Optional[str] = Header(default=None),
             x_claude_model: Optional[str] = Header(default=None)):
    """The AI for the thinking steps only (understanding the command, planning searches): Claude first when
    its key is set on the page or as ANTHROPIC_API_KEY, then the usual chain. Page reading never uses it."""
    return ai_router.reasoning(gemini, (x_claude_key or "").strip(), (x_claude_model or "").strip())


def _reader(gemini=Depends(_gemini), x_claude_reading: Optional[str] = Header(default=None),
            x_claude_key: Optional[str] = Header(default=None), x_claude_model: Optional[str] = Header(default=None)):
    """The AI for reading pages / classifying / fit checks: the usual chain, or Claude first when the page's
    “Use Claude for reading” box is ticked."""
    if (x_claude_reading or "").strip() == "1":
        return ai_router.reasoning(gemini, (x_claude_key or "").strip(), (x_claude_model or "").strip())
    return gemini


def _keys(x_integrations: Optional[str] = Header(default=None)) -> dict:
    """Third-party service keys (search, unblock, enrichment) from the page, falling back to env vars."""
    return integrations.resolve_keys(_json_header_raw(x_integrations))


def _json_header_raw(value: Optional[str]) -> dict:
    try:
        data = json.loads(value or "{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


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
    category: str = Field("", max_length=60)


class QueryIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(10, ge=1, le=100)
    region: str = "in-en"
    backend: str = "auto"
    max_age_months: int = Field(0, ge=0, le=120)
    max_age_days: int = Field(0, ge=0, le=3650)   # custom days (wins over months)


class DedupIn(BaseModel):
    urls: List[str] = Field(default_factory=list, max_length=1000)


class Hit(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    date: Optional[str] = None


class BatchIn(BaseModel):
    intent: str
    items: List[Hit] = Field(..., min_length=1, max_length=8)
    wave_tag: str = "W?"
    page_timeout_s: int = Field(15, ge=5, le=30)
    respect_robots: bool = True
    snippet_fallback: bool = True
    extraction: str = Field("rules", pattern="^(rules|hybrid|ai)$")
    plan_queries: List[str] = Field(default_factory=list, max_length=200)
    max_age_months: int = Field(0, ge=0, le=120)
    max_age_days: int = Field(0, ge=0, le=3650)   # custom days (wins over months)
    role_keywords: List[str] = Field(default_factory=list, max_length=60)
    locations: List[str] = Field(default_factory=list, max_length=20)
    only_interested: bool = False
    enrich: bool = False
    require_both: bool = True
    category: str = Field("", max_length=60)


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
    x_llm_keys: Optional[str] = Header(default=None),
    x_llm_models: Optional[str] = Header(default=None),
    x_database_url: Optional[str] = Header(default=None),
    x_integrations: Optional[str] = Header(default=None),
    x_claude_key: Optional[str] = Header(default=None),
    x_claude_model: Optional[str] = Header(default=None),
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
    keys = _keys(x_integrations)
    out["scrapedo"] = "configured" if keys.get("scrapedo") else "not set"
    out["integrations"] = integrations.summary(keys)
    try:
        gem = _gemini(x_gemini_key=x_gemini_key, x_gemini_model=x_gemini_model, x_gemini_mode=x_gemini_mode,
                      x_openrouter_key=x_openrouter_key, x_openrouter_model=x_openrouter_model,
                      x_llm_provider=x_llm_provider, x_llm_key=x_llm_key, x_llm_model=x_llm_model,
                      x_llm_keys=x_llm_keys, x_llm_models=x_llm_models)
        out["gemini"] = await gem.ping()
    except Exception as exc:
        out["gemini_error"] = str(getattr(exc, "detail", exc))
    claude_key = (x_claude_key or os.getenv("ANTHROPIC_API_KEY", "")).strip()
    if claude_key:
        try:
            from claude_client import Claude
            out["claude"] = await Claude(claude_key, model=(x_claude_model or "").strip()).ping()
        except Exception as exc:
            out["claude_error"] = str(exc)[:300]
    return out


class SuppressIn(BaseModel):
    phone: Optional[str] = Field(None, max_length=40)
    email: Optional[str] = Field(None, max_length=200)
    profile: Optional[str] = Field(None, max_length=500)
    name: Optional[str] = Field(None, max_length=200)
    reason: str = Field("opted out", max_length=200)


@router.post("/privacy/suppress")
def privacy_suppress(body: SuppressIn):
    """Do not contact this person again: added to the list, their saved contact details erased."""
    import privacy
    _db()
    try:
        return _im_db(privacy.add, body.phone, body.email, body.profile, body.name, body.reason)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/privacy/list")
def privacy_list():
    import privacy
    _db()
    return {"entries": _im_db(privacy.listing, 500)}


@router.post("/privacy/remove")
def privacy_remove(body: dict):
    import privacy
    _db()
    _im_db(privacy.remove, int(body.get("id") or 0))
    return {"ok": True}


@router.get("/suggestions")
def suggestions_ep(include_dismissed: bool = False, limit: int = 200):
    """💡 Pages with many contacts that did not match their search (kept across runs, with the contacts)."""
    import suggestions
    _db()
    return {"sites": _im_db(suggestions.listing, max(1, min(limit, 1000)), include_dismissed)}


@router.post("/suggestions/dismiss")
def suggestions_dismiss_ep(body: dict):
    import suggestions
    _db()
    url = str(body.get("url") or "")
    if not url:
        raise HTTPException(400, "url is required")
    _im_db(suggestions.dismiss, url, bool(body.get("dismissed", True)))
    return {"ok": True}


@router.post("/privacy/purge")
def privacy_purge(body: dict):
    """Erase individuals' contact details older than N days (businesses are kept)."""
    import privacy
    _db()
    return _im_db(privacy.purge, int(body.get("days") or 180))


@router.get("/readiness")
def readiness(x_integrations: Optional[str] = Header(default=None)):
    """What this deployment can do (no AI call): used by the run summary to suggest fixes."""
    from intent_miner.providers import maps
    keys = _keys(x_integrations)
    summ = integrations.summary(keys)
    try:
        db.init_supabase()
        database = "ok"
    except Exception as exc:
        database = str(getattr(exc, "detail", exc))[:300]
    configured = [n for n in integrations.SEARCH_ORDER
                  if (keys.get("google_cse_key") and keys.get("google_cse_cx") if n == "google_cse" else keys.get(n))]
    return {"database": database, "search_configured": configured, "exhausted": integrations.exhausted(),
            "integrations": summ, "maps": maps.available(keys),
            "mailbox_check": summ.get("verify") or [],
            "smtp_remote": bool(os.getenv("SMTP_VERIFY_URL")),
            "reddit_api": bool(keys.get("reddit_client_id") and keys.get("reddit_client_secret")),
            "youtube_api": bool(keys.get("youtube") or os.getenv("YOUTUBE_API_KEY")),
            "meta": bool(os.getenv("META_PAGE_TOKEN"))}


@router.post("/plan")
async def plan(
    body: PlanIn,
    gemini = Depends(_thinker),
):
    try:
        result = await pipeline.plan_search(gemini, body.intent, body.num_waves, body.queries_per_wave,
                                             body.round, body.exclude_queries, body.respect_robots)
    except GeminiError as exc:
        raise HTTPException(502, f"Planning failed: {exc}")
    if not result["waves"]:
        raise HTTPException(422, "Gemini returned an empty plan — try rephrasing the intent.")
    if body.category and body.round == 1:
        import categories
        await run_in_threadpool(_im_db, categories.log_search, body.category, body.intent, "people", "classic")
    result["warnings"] = gemini.notices
    return result


@router.post("/search")
def search(body: QueryIn, keys: dict = Depends(_keys)):
    backend = body.backend if body.backend in ("auto", "duckduckgo", "google") else "auto"
    return pipeline.run_query(body.query, body.max_results, body.region, backend, keys, body.max_age_months,
                              body.max_age_days)


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
    gemini = Depends(_reader),
    keys: dict = Depends(_keys),
    x_database_url: Optional[str] = Header(default=None),
):
    _db(x_database_url)
    try:
        result = await pipeline.process_batch(
            ai_router.bulk(gemini),
            body.intent,
            [i.model_dump() for i in body.items],
            body.wave_tag,
            body.page_timeout_s,
            body.respect_robots,
            body.snippet_fallback,
            body.extraction,
            body.plan_queries,
            keys,
            body.max_age_months,
            body.role_keywords,
            body.locations,
            body.only_interested,
            body.enrich,
            body.require_both,
            body.category,
            body.max_age_days,
        )
        result["warnings"] = gemini.notices + result.get("warnings", [])
        return result
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
# Contact enrichment (Apollo, Lusha, ContactOut, RocketReach — official APIs, your own keys)
# ---------------------------------------------------------------------------
class EnrichIn(BaseModel):
    candidate_ids: List[str] = Field(..., min_length=1, max_length=10)
    providers: List[str] = Field(default_factory=list)
    only_interested: bool = True
    max_age_months: int = Field(0, ge=0, le=120)
    max_age_days: int = Field(0, ge=0, le=3650)   # custom days (wins over months)
    require_both: bool = True


_LINKEDIN_IN = re.compile(r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/in/[^/?#\s]+", re.IGNORECASE)


@router.post("/enrich")
async def enrich(body: EnrichIn, keys: dict = Depends(_keys)):
    """Fill missing phone / email from the lead databases the user has keys for."""
    _db()
    providers = [p for p in (body.providers or integrations.ENRICH_ORDER) if keys.get(p)]
    if not providers:
        raise HTTPException(400, "No enrichment key set. Add an Apollo, Lusha, ContactOut or RocketReach "
                                 "API key under “Lead databases & search APIs”.")
    try:
        cands = await run_in_threadpool(db.get_candidates, body.candidate_ids)
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))
    stopped: set = set()
    sem = asyncio.Semaphore(3)

    async def one(c: dict) -> dict:
        if c.get("email") and c.get("phone"):
            return {"id": c["id"], "name": c.get("name"), "skipped": "already has phone and email"}
        if body.only_interested and not c.get("shows_interest"):
            return {"id": c["id"], "name": c.get("name"), "skipped": "has not said they are interested"}
        if dates.older_than(c.get("activity_date"), body.max_age_months, body.max_age_days):
            return {"id": c["id"], "name": c.get("name"), "skipped": "older than the chosen period"}
        li = c.get("profile_url") or ""
        if not li:
            m = _LINKEDIN_IN.match(c.get("source_url") or "")
            li = m.group() if m else ""
        async with sem:
            found = await integrations.enrich_person(
                keys, {"name": c.get("name"), "linkedin_url": li,
                       "hints": " ".join(x for x in (c.get("current_role"), c.get("current_location")) if x)},
                providers, stopped)
        fields = {}
        if found["email"] and not c.get("email"):
            fields["email"] = clean_email(found["email"])
        if found["phone"] and not c.get("phone"):
            fields["phone"] = clean_phone(found["phone"])
        partial = False
        if body.require_both and not ((c.get("email") or fields.get("email")) and (c.get("phone") or fields.get("phone"))):
            partial = bool(fields)
            fields = {}                    # only keep leads that end up with both a phone number and an email
        out = {"id": c["id"], "name": c.get("name"), "tried": found["tried"], "errors": found["errors"][:3]}
        if partial:
            out["errors"].append("found only a phone or only an email — not saved ('both phone & email' is ticked)")
        if found.get("profile_url") and not c.get("profile_url"):
            fields["profile_url"] = found["profile_url"]       # saved even without contacts: next lookup is cheaper
        if fields.get("email") or fields.get("phone"):
            fields["contact_source"] = f"enriched:{found['provider']}"
            fields["contact_shared_at"] = "now()"
        if fields:
            try:
                row = await run_in_threadpool(db.update_candidate, c["id"], fields)
                if fields.get("email") or fields.get("phone"):
                    out.update({"email": row.get("email"), "phone": row.get("phone"), "provider": found["provider"]})
            except db.SupabaseError as exc:
                out["errors"].append(str(exc)[:160])
        return out

    results = await asyncio.gather(*(one(c) for c in cands))
    return {"results": results, "providers": providers, "stopped": sorted(stopped)}


class ImportIn(BaseModel):
    portal: str = Field("other", pattern="^(naukri|foundit|workindia|indeed|apna|naukrigulf|other)$")
    applied: bool = True
    filename: str = Field("", max_length=300)
    data_b64: str = Field(..., min_length=4, max_length=6_000_000)


@router.post("/import")
def import_export(body: ImportIn):
    """Candidates from an Excel / CSV file downloaded from a job-portal employer account."""
    import base64
    _db()
    try:
        data = base64.b64decode(body.data_b64.split(",")[-1])
    except ValueError:
        raise HTTPException(400, "The file could not be read.")
    try:
        records, info = portal_import.parse_export(body.filename, data, body.portal, body.applied)
    except Exception as exc:
        raise HTTPException(400, f"Could not read this file ({type(exc).__name__}: {str(exc)[:160]}). "
                                 "Save it as .xlsx or .csv and try again.")
    if not records:
        raise HTTPException(400, "No candidates found in the file. Columns recognised: "
                                 f"{info.get('columns') or 'none'} — the file needs a name column and a "
                                 "phone or email column.")
    try:
        saved = db.save_candidates(records)
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))
    return {"saved": saved, **info,
            "with_phone": sum(1 for r in records if r.phone), "with_email": sum(1 for r in records if r.email)}


# ---------------------------------------------------------------------------
# Intent Miner (added alongside the original pipeline)
# ---------------------------------------------------------------------------
class IMSettings(BaseModel):
    backend: str = "auto"
    region: str = "wt-wt"
    max_results: int = Field(20, ge=5, le=100)
    respect_robots: bool = True
    timeout: int = Field(15, ge=5, le=30)
    feeds: List[str] = Field(default_factory=list, max_length=20)
    min_score: int = Field(50, ge=0, le=100)
    use_llm: bool = True
    save_to_candidates: bool = True
    reprocess: bool = False
    # shared with the classic search settings (combined mode)
    intent: str = Field("", max_length=2000)
    extraction: str = Field("rules", pattern="^(rules|hybrid|ai)$")
    only_interested: bool = False
    enrich: bool = False
    require_both: bool = True
    classic: bool = True
    plan_queries: List[str] = Field(default_factory=list, max_length=300)
    wave_tag: str = Field("IM", max_length=60)
    verify_fit: bool = True                 # AI + rules check that each business fits the command
    strict_requirements: bool = False       # drop businesses whose required registration is not shown
    expand_related: bool = True             # also run Google's related searches
    category: str = Field("", max_length=60)  # everything this search saves is tagged with it


class IMUnderstandIn(BaseModel):
    command: str = Field(..., min_length=5, max_length=2000)
    sources: List[str] = Field(default_factory=list)
    max_age_days: Optional[int] = Field(None, ge=1, le=3650)
    num_queries: int = Field(16, ge=4, le=60)
    exclude_queries: List[str] = Field(default_factory=list, max_length=500)
    auto_sources: bool = False
    target: str = Field("", pattern="^(|people|organizations)$")   # "" = let the AI decide
    category: str = Field("", max_length=60)
    mode: str = Field("", max_length=20)


class IMDiscoverIn(BaseModel):
    spec: QuerySpec
    source: str = "search"
    query: str = Field(..., min_length=1, max_length=500)
    settings: IMSettings = Field(default_factory=IMSettings)


class IMProcessIn(BaseModel):
    spec: QuerySpec
    run_id: str = ""
    items: List[dict] = Field(..., min_length=1, max_length=8)
    settings: IMSettings = Field(default_factory=IMSettings)


class IMExportIn(BaseModel):
    format: str = Field("csv", pattern="^(csv|xlsx|json)$")
    min_score: int = Field(0, ge=0, le=100)
    run_id: str = ""
    mark_exported: bool = False


class IMStatusIn(BaseModel):
    lead_ids: List[str] = Field(..., min_length=1, max_length=500)
    status: str


class IMSalesforceIn(BaseModel):
    min_score: int = Field(80, ge=0, le=100)
    run_id: str = ""


def _im_keys(keys: dict, x_llm_keys: Optional[str], x_gemini_key: Optional[str]) -> dict:
    """Integration keys plus the AI keys usable for embeddings (Mistral / Gemini)."""
    llm = _json_header(x_llm_keys)
    return {**keys, **{k: v for k, v in (("mistral", llm.get("mistral")),
                                          ("gemini", (x_gemini_key or "").strip())) if v}}


def _im_db(fn, *a):
    try:
        return fn(*a)
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))


@router.get("/categories")
def categories_ep():
    """Every category with its number of searches, contacts and leads."""
    import categories
    _db()
    return {"categories": _im_db(categories.listing)}


@router.get("/categories/searches")
def category_searches_ep(name: str = "", names: List[str] = Query(default=[])):
    import categories
    _db()
    return {"searches": _im_db(categories.searches, [n for n in names if n] or [name])}


@router.post("/categories")
def category_create_ep(body: dict):
    """Create a category up front (before the first search that uses it)."""
    import categories
    _db()
    try:
        name = _im_db(categories.create, str(body.get("name") or ""))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"name": name, "categories": _im_db(categories.listing)}


class MessagePlanIn(BaseModel):
    records: List[dict] = Field(..., max_length=5000)
    channels: List[str] = Field(default_factory=lambda: ["whatsapp", "email"])
    only_interested: bool = True
    skip_days: int = Field(30, ge=0, le=365)


class MessageSendIn(BaseModel):
    channel: str = Field(..., pattern="^(whatsapp|email)$")
    recipients: List[dict] = Field(..., min_length=1, max_length=25)
    campaign: str = Field("", max_length=200)
    template_id: str = Field("", max_length=120)
    placeholders: List[str] = Field(default_factory=list, max_length=20)
    subject: str = Field("", max_length=200)
    body: str = Field("", max_length=5000)
    opt_out: str = Field("", max_length=300)
    role: str = Field("", max_length=120)
    skip_days: int = Field(30, ge=0, le=365)


@router.post("/messaging/plan")
def messaging_plan_ep(body: MessagePlanIn):
    """Who would get a WhatsApp / an email (after do-not-contact, interest and recently-messaged checks)."""
    import messaging
    _db()
    chans = [c for c in body.channels if c in ("whatsapp", "email")]
    return _im_db(messaging.plan, body.records, chans, body.only_interested, body.skip_days)


@router.post("/messaging/send")
def messaging_send_ep(body: MessageSendIn, keys: dict = Depends(_keys)):
    """Send one batch (≤25) through Pinnacle (WhatsApp) or Brevo (email). Recipients are re-checked here, so a
    stale page can't message someone who opted out or was messaged meanwhile."""
    import messaging
    _db()
    if body.channel == "whatsapp" and not body.template_id.strip():
        raise HTTPException(400, "Enter the approved WhatsApp template ID from the Pinnacle console.")
    if body.channel == "email" and not (body.subject.strip() and body.body.strip()):
        raise HTTPException(400, "Enter the email subject and message.")
    if body.channel == "email" and not body.opt_out.strip():
        raise HTTPException(400, "Keep an opt-out line in the email (e.g. “Reply STOP and we won't contact you again”).")
    checked = _im_db(messaging.plan, body.recipients, [body.channel], False, body.skip_days)[body.channel]
    if not checked:
        return {"sent": 0, "failed": 0, "results": [], "note": "everyone in this batch was opted out, "
                                                               "already messaged or has no address"}
    return _im_db(lambda: messaging.send_batch(
        keys, body.channel, checked, body.campaign, template_id=body.template_id.strip(),
        placeholders=body.placeholders, subject=body.subject, body=body.body, opt_out=body.opt_out,
        defaults={"role": body.role}))


@router.get("/messaging/log")
def messaging_log_ep(limit: int = 200, campaign: str = ""):
    import messaging
    _db()
    return {"log": _im_db(messaging.log, max(1, min(limit, 2000)), campaign)}


@router.post("/command-kind")
def command_kind_ep(body: dict):
    """Cheap, no-AI check: does the command ask for the hiring side (employers / HR / management)?"""
    from intent_miner.understand import hiring_side
    return {"hiring_side": hiring_side(str(body.get("command") or ""))}


@router.post("/im/understand")
async def im_understand_ep(body: IMUnderstandIn, gemini=Depends(_thinker)):
    _db()
    try:
        spec = await im_understand(gemini, body.command, body.sources, body.max_age_days, body.num_queries,
                                   body.exclude_queries, body.auto_sources, body.target)
    except GeminiError as exc:
        raise HTTPException(502, f"Understanding the command failed: {exc}")
    run_id = await run_in_threadpool(_im_db, im_store.create_run, body.command, spec.model_dump())
    if body.category:
        import categories
        await run_in_threadpool(_im_db, categories.log_search, body.category, body.command, spec.target,
                                body.mode or "intent", run_id)
    return {"spec": spec.model_dump(), "run_id": run_id, "warnings": getattr(gemini, "notices", [])}


@router.post("/im/discover")
async def im_discover_ep(body: IMDiscoverIn, keys: dict = Depends(_keys)):
    return await im_engine.discover(body.spec, body.source, body.query, keys, body.settings.model_dump())


@router.post("/im/process")
async def im_process_ep(body: IMProcessIn, gemini=Depends(_reader), keys: dict = Depends(_keys),
                        x_llm_keys: Optional[str] = Header(default=None),
                        x_gemini_key: Optional[str] = Header(default=None)):
    _db()
    try:
        res = await im_engine.process(gemini, body.spec, body.items, _im_keys(keys, x_llm_keys, x_gemini_key),
                                      body.settings.model_dump(), body.run_id)
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))
    except GeminiError as exc:
        raise HTTPException(502, str(exc))
    except Exception as exc:                        # say what broke instead of a bare "HTTP 500"
        import traceback
        where = traceback.extract_tb(exc.__traceback__)[-1]
        raise HTTPException(500, f"{type(exc).__name__}: {str(exc)[:200]} "
                                 f"(at {where.filename.rsplit('/', 1)[-1]}:{where.lineno})")
    res["warnings"] = getattr(gemini, "notices", []) + res["warnings"]
    return res


@router.get("/im/leads")
def im_leads_ep(min_score: int = 0, run_id: str = "", offset: int = 0, limit: int = 2000, category: str = "",
                since: str = "", until: str = "", categories: List[str] = Query(default=[])):
    _db()
    cats = [c for c in categories if c] or ([category] if category else [])
    return {"leads": _im_db(im_store.list_leads, min_score, max(1, min(limit, 5000)), run_id, max(0, offset),
                            cats, since, until)}


@router.get("/storage")
def storage_ep(x_database_url: Optional[str] = Header(default=None)):
    """Which database the app uses and how much it holds (candidates, leads, runs, date range)."""
    _db(x_database_url)
    try:
        out = db.storage_summary()
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))
    try:
        out["intent"] = _im_db(im_store.counts)
    except HTTPException:
        out["intent"] = {}
    return out


@router.post("/im/export")
def im_export_ep(body: IMExportIn):
    import base64
    _db()
    leads = _im_db(im_store.list_leads, body.min_score, 5000, body.run_id)
    data, mime = {"csv": (im_export.to_csv, "text/csv"), "json": (im_export.to_json, "application/json"),
                  "xlsx": (im_export.to_xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                  }[body.format]
    blob = data(leads)
    if body.mark_exported and leads:
        _im_db(im_store.set_status, [L["id"] for L in leads if L.get("status") in ("QUALIFIED", "ENRICHED")],
               "EXPORTED")
    return {"filename": f"intent-leads.{body.format}", "mime": mime, "count": len(leads),
            "data_b64": base64.b64encode(blob).decode()}


@router.post("/im/status")
def im_status_ep(body: IMStatusIn):
    _db()
    if body.status not in im_store.LIFECYCLE:
        raise HTTPException(400, "Unknown status.")
    _im_db(im_store.set_status, body.lead_ids, body.status)
    return {"updated": len(body.lead_ids)}


@router.post("/im/salesforce")
async def im_salesforce_ep(body: IMSalesforceIn, keys: dict = Depends(_keys)):
    _db()
    if not (keys.get("salesforce_instance_url") and keys.get("salesforce_token")):
        raise HTTPException(400, "Add the Salesforce instance URL and access token under the API keys first.")
    leads = await run_in_threadpool(_im_db, im_store.list_leads, body.min_score, 500, body.run_id)
    res = await im_export.push_salesforce(leads, keys["salesforce_instance_url"], keys["salesforce_token"],
                                          body.min_score)
    for c in res["created"]:
        await run_in_threadpool(_im_db, im_store.set_status, [c["id"]], "EXPORTED", c["salesforce_id"])
    return res


@router.get("/im/recheck")
def im_recheck_ep(sources: str = "", hours: int = 24, only_productive: bool = True, limit: int = 60):
    """Pages read before, to re-read for new comments / updates."""
    _db()
    src = [s for s in sources.split(",") if s]
    return {"items": _im_db(im_store.recheck_candidates, src, max(1, min(hours, 24 * 90)), only_productive,
                            max(1, min(limit, 300)))}


@router.get("/im/failed")
def im_failed_ep():
    _db()
    return {"failed": _im_db(im_store.failed_documents, 200)}


@router.get("/im/health")
def im_health_ep():
    _db()
    return {"providers": _im_db(im_store.provider_health), "runs": _im_db(im_store.list_runs, 10)}


# ---------------------------------------------------------------------------
# Government directories (public lists) and matching leads to them
# ---------------------------------------------------------------------------
class GovImportIn(BaseModel):
    dataset: str = Field(..., min_length=2, max_length=120)
    kind: str = Field("org", pattern="^(org|person)$")
    filename: str = Field("", max_length=300)
    data_b64: str = Field(..., min_length=4, max_length=6_000_000)
    source_url: str = Field("", max_length=500)
    replace: bool = False


class GovDatagovIn(BaseModel):
    resource_id: str = Field(..., pattern=r"^[A-Za-z0-9-]{8,64}$")
    dataset: str = Field("", max_length=120)
    kind: str = Field("org", pattern="^(org|person)$")
    max_records: int = Field(5000, ge=100, le=50000)
    replace: bool = False


class GovMatchIn(BaseModel):
    lead_ids: List[str] = Field(default_factory=list, max_length=500)
    only_unmatched: bool = True
    limit: int = Field(40, ge=1, le=200)
    use_ai: bool = True
    after: str = Field("", max_length=64)


@router.post("/gov/import")
def gov_import(body: GovImportIn):
    import base64
    _db()
    try:
        data = base64.b64decode(body.data_b64.split(",")[-1])
    except ValueError:
        raise HTTPException(400, "The file could not be read.")
    try:
        info = gov_registry.import_file(body.dataset.strip(), body.kind, body.filename, data, body.source_url,
                                        body.replace)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"Could not read this file ({type(exc).__name__}: {str(exc)[:160]}).")
    if not info.get("saved") and info.get("rows") and "name" in (info.get("columns") or {}):
        info["note"] = "Every record in this file was already imported (nothing new added)."
        return info
    if not info.get("saved"):
        raise HTTPException(400, "No records found. The file needs a name column (e.g. \"Name of the Institute\", "
                                 f"\"Company Name\"). Columns recognised: {info.get('columns') or 'none'}")
    return info


@router.post("/gov/datagov")
async def gov_datagov(body: GovDatagovIn, keys: dict = Depends(_keys)):
    _db()
    key = keys.get("datagov") or ""
    if not key:
        raise HTTPException(400, "Add your data.gov.in API key (🔑 keys → data.gov.in) first — it is free after "
                                 "signing up at data.gov.in.")
    try:
        return await gov_registry.import_datagov(key, body.resource_id, body.dataset.strip(), body.kind,
                                                 body.max_records, body.replace)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))


class GovUrlIn(BaseModel):
    dataset: str = Field(..., min_length=2, max_length=120)
    kind: str = Field("org", pattern="^(org|person)$")
    url: str = Field(..., pattern=r"^https?://", max_length=1000)
    replace: bool = False


@router.post("/gov/import-url")
async def gov_import_url(body: GovUrlIn):
    """A list published online (PDF, Excel / CSV, HTML table) imported by its link."""
    _db()
    try:
        info = await gov_registry.import_url(body.dataset.strip(), body.kind, body.url, body.replace)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {str(exc)[:200]}")
    return info


class GovLeadsIn(BaseModel):
    dataset: str = Field(..., min_length=2, max_length=120)
    region: str = Field("", max_length=300)          # "North India", "Punjab, Haryana", "Ludhiana" …
    offset: int = Field(0, ge=0)
    limit: int = Field(6, ge=1, le=12)
    only_missing: bool = True                        # skip records already made into leads
    respect_robots: bool = True


@router.post("/gov/leads")
async def gov_leads(body: GovLeadsIn, request: Request, keys: dict = Depends(_keys)):
    """Rows of an official list (in a region) → leads, each enriched: Maps listing, own website, shared inboxes,
    owners on LinkedIn. Call repeatedly with the returned offset."""
    from intent_miner import planner as im_planner, verify as im_verify
    _db()
    places = [p.strip() for p in re.split(r"[,;]", body.region) if p.strip()]
    wanted = list(dict.fromkeys(im_planner.expand_places(places, body.region) +
                                im_verify._states_for(QuerySpec(origin=places), body.region) + places))

    def load():
        conds, params = ["dataset = %s", "kind = 'org'"], [body.dataset]
        if wanted:
            conds.append("(" + " OR ".join(["state ILIKE %s OR district ILIKE %s OR city ILIKE %s OR address ILIKE %s"]
                                           * len(wanted)) + ")")
            for w in wanted:
                params += [f"%{w}%"] * 4
        where = " AND ".join(conds)
        total = gov_registry._q(f"SELECT COUNT(*) AS n FROM gov_records WHERE {where}", params, "one")["n"]
        rows = gov_registry._q(f"SELECT * FROM gov_records WHERE {where} ORDER BY id OFFSET %s LIMIT %s",
                               params + [body.offset, body.limit], "all")
        return int(total), [dict(r) for r in rows]
    total, rows = await run_in_threadpool(load)
    leads = []
    for r in rows:
        where = ", ".join(x for x in (r.get("district") or r.get("city"), r.get("state")) if x)
        L = im_engine.org_lead(r["name"], where, r.get("phone"), r.get("email"), r.get("website"), "gov_list",
                               r.get("source_url") or "", [f"✓ On the official list “{r['dataset']}”"
                                                           + (f", reg {r['reg_no']}" if r.get("reg_no") else "")],
                               r.get("category") or "registered recruiting agent", 85)
        L["gov_match"] = {"status": "matched", "method": "source", "score": 100, "dataset": r["dataset"],
                          "record_id": r["id"], "name": r["name"], "reg_no": r.get("reg_no"),
                          "address": r.get("address"), "district": r.get("district"), "state": r.get("state"),
                          "phone": r.get("phone"), "email": r.get("email"), "signals": ["from the list"]}
        leads.append(L)
    h = request.headers
    try:
        ai = _gemini(*(h.get(k) for k in ("x-gemini-key", "x-gemini-model", "x-gemini-mode", "x-openrouter-key",
                                          "x-openrouter-model", "x-llm-provider", "x-llm-key", "x-llm-model",
                                          "x-llm-keys", "x-llm-models")))
    except HTTPException:
        ai = None
    res = await im_engine.enrich_org_leads(leads, keys, {"respect_robots": body.respect_robots, "gov_match": False},
                                           ai)
    return {"total": total, "next_offset": body.offset + len(rows), "done": body.offset + len(rows) >= total,
            "places": wanted[:40], **res}


class NamesIn(BaseModel):
    names: List[str] = Field(..., min_length=1, max_length=8)
    city: str = Field("", max_length=100)
    respect_robots: bool = True


@router.post("/im/names")
async def im_names(body: NamesIn, request: Request, keys: dict = Depends(_keys)):
    """Businesses the user names ("Magic Billion", "Aimpersand") → Maps listing, website, contacts, owners."""
    _db()
    leads = [im_engine.org_lead(n.strip(), body.city.strip(), why=["Looked up by name"], profession=None)
             for n in body.names if n.strip()]
    h = request.headers
    try:
        ai = _gemini(*(h.get(k) for k in ("x-gemini-key", "x-gemini-model", "x-gemini-mode", "x-openrouter-key",
                                          "x-openrouter-model", "x-llm-provider", "x-llm-key", "x-llm-model",
                                          "x-llm-keys", "x-llm-models")))
    except HTTPException:
        ai = None
    return await im_engine.enrich_org_leads(leads, keys, {"respect_robots": body.respect_robots}, ai)


@router.get("/gov/datasets")
def gov_datasets():
    _db()
    return {"datasets": _im_db(gov_registry.datasets)}


@router.post("/gov/delete")
def gov_delete(body: dict):
    _db()
    name = str(body.get("dataset") or "").strip()
    if not name:
        raise HTTPException(400, "dataset is required")
    return {"deleted": _im_db(gov_registry.delete_dataset, name)}


@router.post("/gov/match")
async def gov_match(body: GovMatchIn, request: Request):
    """Match saved leads to the imported government records (rules + AI check; rules only without an AI key)."""
    _db()
    h = request.headers
    try:
        gemini = _gemini(*(h.get(k) for k in ("x-gemini-key", "x-gemini-model", "x-gemini-mode", "x-openrouter-key",
                                              "x-openrouter-model", "x-llm-provider", "x-llm-key", "x-llm-model",
                                              "x-llm-keys", "x-llm-models"))) if body.use_ai else None
    except HTTPException:
        gemini = None
    leads = await run_in_threadpool(_im_db, gov_registry.leads_to_match, body.lead_ids, body.only_unmatched,
                                    body.limit, body.after)
    if not leads:
        return {"stats": {"checked": 0}, "leads": [], "last_id": "", "more": False}
    last_id = str(leads[-1]["id"])
    stats = await gov_registry.match_leads(leads, gemini, gemini is not None, max_ai=min(len(leads), 25))
    changed = [L for L in leads if L.get("gov_match")]
    for L in changed:
        await run_in_threadpool(_im_db, gov_registry.save_lead_match, L)
    return {"stats": stats, "leads": [db._serialize_row(dict(L)) for L in changed], "last_id": last_id,
            "more": len(leads) == body.limit, "warnings": getattr(gemini, "notices", [])}


# ---------------------------------------------------------------------------
# Background jobs (persistent workers — worker/worker.py)
# ---------------------------------------------------------------------------
class JobIn(BaseModel):
    command: str = Field(..., min_length=5, max_length=2000)
    sources: List[str] = Field(default_factory=list)
    auto_sources: bool = True
    max_age_days: Optional[int] = Field(None, ge=1, le=3650)
    num_queries: int = Field(16, ge=4, le=60)
    max_urls: int = Field(60, ge=5, le=2000)
    batch: int = Field(5, ge=1, le=8)
    settings: IMSettings = Field(default_factory=IMSettings)
    use_page_keys: bool = False
    rounds: int = Field(1, ge=1, le=20)              # automatic rounds, each with new searches
    pause_minutes: float = Field(0, ge=0, le=720)    # wait between rounds
    repeat_hours: float = Field(0, ge=0, le=720)     # run the whole job again every N hours (0 = once)
    target: str = Field("", pattern="^(|people|organizations)$")
    category: str = Field("", max_length=60)


@router.post("/jobs")
def job_create(body: JobIn, x_integrations: Optional[str] = Header(default=None),
               x_llm_keys: Optional[str] = Header(default=None), x_gemini_key: Optional[str] = Header(default=None),
               x_llm_provider: Optional[str] = Header(default=None),
               x_claude_key: Optional[str] = Header(default=None),
               x_claude_reading: Optional[str] = Header(default=None)):
    import jobs
    _db()
    opts = body.model_dump(exclude={"command", "use_page_keys"})
    opts["llm_provider"] = (x_llm_provider or "").strip()
    opts["claude_reading"] = (x_claude_reading or "").strip() == "1"
    if body.use_page_keys:        # stored with the job only until it finishes (then erased)
        opts["keys"] = _json_header_raw(x_integrations)
        opts["llm_keys"] = {**_json_header(x_llm_keys), **({"gemini": x_gemini_key.strip()} if x_gemini_key else {}),
                            **({"claude": x_claude_key.strip()} if x_claude_key else {})}
    job = _im_db(jobs.enqueue, "run", body.command, opts)
    if body.category:
        import categories
        _im_db(categories.log_search, body.category, body.command, body.target, "background job", str(job["id"]))
    return {"job": {k: job[k] for k in ("id", "status", "created_at")}, "workers": _im_db(jobs.workers)}


@router.get("/jobs")
def job_list():
    import jobs
    _db()
    return {"jobs": _im_db(jobs.recent, 20), "workers": _im_db(jobs.workers)}


@router.get("/jobs/{job_id}")
def job_get(job_id: str, log_from: int = 0):
    import jobs
    _db()
    job = _im_db(jobs.get, job_id, max(0, log_from))
    if not job:
        raise HTTPException(404, "No such job")
    return {"job": job}


@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    import jobs
    _db()
    _im_db(jobs.cancel, job_id)
    return {"ok": True}


@router.post("/jobs-index")
def job_index():
    """Queue embedding the whole search index on a worker (no time limit there)."""
    import jobs
    _db()
    return {"job": _im_db(jobs.enqueue, "index", "embed the search index", {}), "workers": _im_db(jobs.workers)}


# ---------------------------------------------------------------------------
# Hybrid semantic search over everything saved (pgvector + full-text + fuzzy names)
# ---------------------------------------------------------------------------
@router.get("/search")
async def search_saved(q: str, kinds: str = "lead,candidate,gov", k: int = 30,
                       x_gemini_key: Optional[str] = Header(default=None),
                       x_llm_keys: Optional[str] = Header(default=None), keys: dict = Depends(_keys)):
    import vectors
    _db()
    kinds_l = [x for x in kinds.split(",") if x in ("lead", "candidate", "gov")] or ["lead"]
    try:
        hits = await vectors.search(q, kinds_l, _im_keys(keys, x_llm_keys, x_gemini_key), min(max(k, 1), 100))
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))

    def load():
        out = []
        by = {}
        for h in hits:
            by.setdefault(h["kind"], []).append(h["ref"])
        rows = {}
        if by.get("lead"):
            for r in im_store._q("SELECT * FROM im_leads WHERE id::text = ANY(%s)", (by["lead"],), "all"):
                rows[("lead", str(r["id"]))] = db._serialize_row(dict(r))
        if by.get("candidate"):
            for r in im_store._q("SELECT * FROM candidates WHERE id::text = ANY(%s)", (by["candidate"],), "all"):
                rows[("candidate", str(r["id"]))] = db._serialize_row(dict(r))
        if by.get("gov"):
            for r in im_store._q("SELECT * FROM gov_records WHERE id::text = ANY(%s)", (by["gov"],), "all"):
                rows[("gov", str(r["id"]))] = db._serialize_row(dict(r))
        for h in hits:
            row = rows.get((h["kind"], h["ref"]))
            if row:
                out.append({**h, "row": row})
        return out
    return {"results": await run_in_threadpool(load)}


@router.post("/search/reindex")
async def search_reindex(x_gemini_key: Optional[str] = Header(default=None),
                         x_llm_keys: Optional[str] = Header(default=None), keys: dict = Depends(_keys)):
    """Put saved leads / candidates / government lists into the search index and embed a slice of what is not
    embedded yet (call again while `remaining` > 0)."""
    import vectors
    _db()

    def texts():
        leads = [dict(r) for r in im_store._q("SELECT * FROM im_leads ORDER BY last_seen DESC LIMIT 5000", None, "all")]
        n = vectors.index_leads(leads) + vectors.index_candidates()
        for d in gov_registry.datasets():
            n += vectors.index_gov_dataset(d["dataset"])
        return n
    indexed = await run_in_threadpool(texts)
    emb = await vectors.embed_pending(_im_keys(keys, x_llm_keys, x_gemini_key), None, limit=256)
    return {"indexed": indexed, **emb, "status": await run_in_threadpool(vectors.status)}


# ---------------------------------------------------------------------------
# Auto-reply on your own Instagram / Facebook posts (Meta official APIs)
# ---------------------------------------------------------------------------
meta_public = APIRouter()        # Meta calls the webhook itself: no app password, signature-checked instead


@meta_public.get("/meta/webhook")
def meta_verify(request: Request):
    q = request.query_params
    token = os.getenv("META_VERIFY_TOKEN", "").strip()
    if q.get("hub.mode") == "subscribe" and token and hmac.compare_digest(q.get("hub.verify_token", ""), token):
        return PlainTextResponse(q.get("hub.challenge", ""))
    raise HTTPException(403, "Verification failed: META_VERIFY_TOKEN does not match.")


@meta_public.post("/meta/webhook")
async def meta_webhook(request: Request):
    raw = await request.body()
    if not meta_autoreply.verify_signature(raw, request.headers.get("x-hub-signature-256")):
        raise HTTPException(403, "Bad signature (check META_APP_SECRET).")
    try:
        payload = json.loads(raw or b"{}")
    except ValueError:
        raise HTTPException(400, "Not JSON")
    try:
        results = await meta_autoreply.process_event(payload)
    except db.SupabaseError as exc:
        results = [{"error": str(exc)[:200]}]
    return {"ok": True, "results": results}


class MetaSettingsIn(BaseModel):
    enabled: Optional[bool] = None
    public_reply: Optional[str] = Field(None, max_length=1000)
    dm_text: Optional[str] = Field(None, max_length=1000)
    thanks_text: Optional[str] = Field(None, max_length=1000)
    extra_keywords: Optional[str] = Field(None, max_length=500)


@router.get("/meta/status")
def meta_status(request: Request):
    _db()
    try:
        return {"env": meta_autoreply.env_status(), "settings": meta_autoreply.get_settings(),
                "threads": meta_autoreply.threads(200)}
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))


@router.post("/meta/settings")
def meta_settings(body: MetaSettingsIn):
    _db()
    values = {k: ("1" if v else "0") if k == "enabled" else v for k, v in body.model_dump().items() if v is not None}
    try:
        return {"settings": meta_autoreply.save_settings(values)}
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))


class EnrichTestIn(BaseModel):
    linkedin_url: str = ""
    name: str = ""
    company: str = ""


@router.post("/enrich/test")
async def enrich_test(body: EnrichTestIn, keys: dict = Depends(_keys)):
    """One real lookup per lead database, with each service's own reply — to see why lookups fail."""
    co = body.company.strip()
    is_domain = bool(re.fullmatch(r"(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+/?", co, re.I))
    person = {"linkedin_url": body.linkedin_url.strip(), "name": body.name.strip(),
              "company": "" if is_domain else co,
              "domain": re.sub(r"^(?:https?://)?(?:www\.)?", "", co).rstrip("/") if is_domain else ""}
    names = [p for p in integrations.ENRICH_ORDER if keys.get(p)]
    if not names:
        raise HTTPException(400, "No lead-database key is set (page or Vercel env vars).")
    results = await asyncio.gather(*(integrations.enrich_one(p, keys, person) for p in names))
    return {"results": [{"provider": integrations.SERVICES[p][1], "status": r.get("status"),
                         "emails": r.get("emails", []), "phones": r.get("phones", []),
                         "error": r.get("error"), "reply": r.get("raw", "")} for p, r in zip(names, results)]}


class VerifyIn(BaseModel):
    emails: List[str] = Field(..., min_length=1, max_length=25)


@router.post("/verify/email")
async def verify_email(body: VerifyIn, keys: dict = Depends(_keys)):
    """Zero-send verification: syntax, disposable, DNS / MX, SMTP RCPT check, catch-all probe."""
    import email_verify
    res = await email_verify.verify_many([e.strip() for e in body.emails], keys)
    return {"results": [{"email": e, **v, "label": email_verify.label(v)} for e, v in res.items()]}


class FindEmailIn(BaseModel):
    name: str = Field(..., min_length=3, max_length=120)
    company: str = Field(..., min_length=2, max_length=200)
    respect_robots: bool = True


@router.post("/email/find")
async def find_email(body: FindEmailIn, keys: dict = Depends(_keys)):
    """The waterfall's last steps for one person: employer site → mail domain + format → likely addresses →
    SMTP check of each."""
    import company_contacts
    import email_verify
    prof = await company_contacts.company_profile(keys, body.company, body.respect_robots)
    if not prof:
        raise HTTPException(404, "Could not find the company's website — enter its domain (e.g. fortishealthcare.com).")
    cands = pipeline._candidates_for(body.name, prof["domain"], prof["format"])
    if not cands:
        raise HTTPException(400, "Enter first and last name.")
    res = await email_verify.verify_many([e for _, e in cands], keys)
    return {"site": prof["site"], "domain": prof["domain"], "domain_from_emails": prof["domain_from_emails"],
            "format": prof["format"], "published": [p for p in prof["people"] if p.get("email")][:10],
            "candidates": [{"email": e, "format": f, **res.get(e, {}), "label": email_verify.label(res.get(e))}
                           for f, e in cands]}


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
app.include_router(meta_public, prefix="/api")

# Local development: serve the page from the same server (Vercel serves public/ itself).
if not os.getenv("VERCEL"):
    from fastapi.staticfiles import StaticFiles  # noqa: E402

    app.mount("/", StaticFiles(directory=ROOT_DIR / "public", html=True), name="static")
