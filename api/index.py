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

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException  # noqa: E402
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
from intent_miner import engine as im_engine, export as im_export, store as im_store  # noqa: E402
from intent_miner.models import QuerySpec  # noqa: E402
from intent_miner.understand import understand as im_understand  # noqa: E402
import outreach  # noqa: E402
from schema import CandidateRecord, clean_email, clean_phone  # noqa: E402

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
            elif current_path in ("/health", "/plan", "/search", "/dedup", "/process", "/candidates", "/enrich", "/import"):
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
                                 "DEEPSEEK_API_KEY / MOONSHOT_API_KEY / GEMINI_API_KEY in Vercel.")
    return AIChain(entries)


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


class QueryIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(10, ge=1, le=100)
    region: str = "in-en"
    backend: str = "auto"
    max_age_months: int = Field(0, ge=0, le=120)


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
    role_keywords: List[str] = Field(default_factory=list, max_length=60)
    locations: List[str] = Field(default_factory=list, max_length=20)
    only_interested: bool = False
    enrich: bool = False
    require_both: bool = True


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
    result["warnings"] = gemini.notices
    return result


@router.post("/search")
def search(body: QueryIn, keys: dict = Depends(_keys)):
    backend = body.backend if body.backend in ("auto", "duckduckgo", "google") else "auto"
    return pipeline.run_query(body.query, body.max_results, body.region, backend, keys, body.max_age_months)


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
    keys: dict = Depends(_keys),
    x_database_url: Optional[str] = Header(default=None),
):
    _db(x_database_url)
    try:
        result = await pipeline.process_batch(
            gemini,
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
        if dates.older_than(c.get("activity_date"), body.max_age_months):
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


class IMUnderstandIn(BaseModel):
    command: str = Field(..., min_length=5, max_length=2000)
    sources: List[str] = Field(default_factory=list)
    max_age_days: Optional[int] = Field(None, ge=1, le=3650)
    num_queries: int = Field(16, ge=4, le=60)
    exclude_queries: List[str] = Field(default_factory=list, max_length=500)


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


@router.post("/im/understand")
async def im_understand_ep(body: IMUnderstandIn, gemini=Depends(_gemini)):
    _db()
    try:
        spec = await im_understand(gemini, body.command, body.sources, body.max_age_days, body.num_queries,
                                   body.exclude_queries)
    except GeminiError as exc:
        raise HTTPException(502, f"Understanding the command failed: {exc}")
    run_id = await run_in_threadpool(_im_db, im_store.create_run, body.command, spec.model_dump())
    return {"spec": spec.model_dump(), "run_id": run_id, "warnings": getattr(gemini, "notices", [])}


@router.post("/im/discover")
async def im_discover_ep(body: IMDiscoverIn, keys: dict = Depends(_keys)):
    return await im_engine.discover(body.spec, body.source, body.query, keys, body.settings.model_dump())


@router.post("/im/process")
async def im_process_ep(body: IMProcessIn, gemini=Depends(_gemini), keys: dict = Depends(_keys),
                        x_llm_keys: Optional[str] = Header(default=None),
                        x_gemini_key: Optional[str] = Header(default=None)):
    _db()
    try:
        res = await im_engine.process(gemini, body.spec, body.items, _im_keys(keys, x_llm_keys, x_gemini_key),
                                      body.settings.model_dump(), body.run_id)
    except db.SupabaseError as exc:
        raise HTTPException(502, str(exc))
    res["warnings"] = getattr(gemini, "notices", []) + res["warnings"]
    return res


@router.get("/im/leads")
def im_leads_ep(min_score: int = 0, run_id: str = ""):
    _db()
    return {"leads": _im_db(im_store.list_leads, min_score, 2000, run_id)}


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


@router.get("/im/failed")
def im_failed_ep():
    _db()
    return {"failed": _im_db(im_store.failed_documents, 200)}


@router.get("/im/health")
def im_health_ep():
    _db()
    return {"providers": _im_db(im_store.provider_health), "runs": _im_db(im_store.list_runs, 10)}


class EnrichTestIn(BaseModel):
    linkedin_url: str = ""
    name: str = ""
    company: str = ""


@router.post("/enrich/test")
async def enrich_test(body: EnrichTestIn, keys: dict = Depends(_keys)):
    """One real lookup per lead database, with each service's own reply — to see why lookups fail."""
    person = {"linkedin_url": body.linkedin_url.strip(), "name": body.name.strip(), "company": body.company.strip()}
    names = [p for p in integrations.ENRICH_ORDER if keys.get(p)]
    if not names:
        raise HTTPException(400, "No lead-database key is set (page or Vercel env vars).")
    results = await asyncio.gather(*(integrations.enrich_one(p, keys, person) for p in names))
    return {"results": [{"provider": integrations.SERVICES[p][1], "status": r.get("status"),
                         "emails": r.get("emails", []), "phones": r.get("phones", []),
                         "error": r.get("error"), "reply": r.get("raw", "")} for p, r in zip(names, results)]}


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
