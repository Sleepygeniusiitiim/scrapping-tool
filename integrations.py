"""
Third-party services, each optional and enabled by its API key (entered on the page or set in Vercel):

* Google / web search APIs — Serper.dev, SerpApi, Google Programmable Search, Brave Search.
  DuckDuckGo from a Vercel IP returns few results; these return real Google results.
* Unblockers — Scrape.do, ScraperAPI, ZenRows, ScrapingBee (residential proxies), plus Jina Reader
  (free, no key needed). Used only for pages that refuse the direct fetch.
* Contact enrichment — Apollo, Lusha, ContactOut, RocketReach. Official APIs only: they look up a
  candidate's LinkedIn profile URL (or name) in the provider's database. Account passwords are never
  used — automated logins break these providers' terms and get accounts banned.

Every function here never raises; failures come back as an error string.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Dict, List, Optional, Tuple

import httpx

import dates
from fetcher import EMAIL_RE

# name → (env var, label, kind)
SERVICES: Dict[str, Tuple[str, str, str]] = {
    "serper": ("SERPER_API_KEY", "Serper.dev (Google)", "search"),
    "serpapi": ("SERPAPI_KEY", "SerpApi (Google)", "search"),
    "google_cse_key": ("GOOGLE_CSE_KEY", "Google Programmable Search key", "search"),
    "google_cse_cx": ("GOOGLE_CSE_CX", "Google Programmable Search engine ID (cx)", "search"),
    "brave": ("BRAVE_API_KEY", "Brave Search", "search"),
    "scrapedo": ("SCRAPEDO_TOKEN", "Scrape.do", "unblock"),
    "scraperapi": ("SCRAPERAPI_KEY", "ScraperAPI", "unblock"),
    "zenrows": ("ZENROWS_API_KEY", "ZenRows", "unblock"),
    "scrapingbee": ("SCRAPINGBEE_API_KEY", "ScrapingBee", "unblock"),
    "jina": ("JINA_API_KEY", "Jina Reader (optional key; works free without one)", "unblock"),
    "apollo": ("APOLLO_API_KEY", "Apollo.io", "enrich"),
    "lusha": ("LUSHA_API_KEY", "Lusha", "enrich"),
    "contactout": ("CONTACTOUT_API_KEY", "ContactOut", "enrich"),
    "rocketreach": ("ROCKETREACH_API_KEY", "RocketReach", "enrich"),
    "reddit_client_id": ("REDDIT_CLIENT_ID", "Reddit API app client id", "source"),
    "reddit_client_secret": ("REDDIT_CLIENT_SECRET", "Reddit API app secret", "source"),
    "salesforce_instance_url": ("SALESFORCE_INSTANCE_URL", "Salesforce instance URL", "crm"),
    "salesforce_token": ("SALESFORCE_ACCESS_TOKEN", "Salesforce access token", "crm"),
}
SEARCH_ORDER = ["serper", "google_cse", "serpapi", "scrapedo", "brave"]
UNBLOCK_ORDER = ["scrapedo", "scraperapi", "zenrows", "scrapingbee", "jina"]
ENRICH_ORDER = ["contactout", "lusha", "rocketreach", "apollo"]


def resolve_keys(page_keys: Optional[dict]) -> Dict[str, str]:
    """Keys from the page, falling back to Vercel env vars."""
    page_keys = {str(k): str(v).strip() for k, v in (page_keys or {}).items() if v}
    out = {}
    for name, (env, _, _) in SERVICES.items():
        v = page_keys.get(name) or os.getenv(env, "").strip()
        if v:
            out[name] = v
    return out


def search_available(keys: Dict[str, str]) -> List[str]:
    out = []
    for name in SEARCH_ORDER:
        if name == "google_cse":
            if keys.get("google_cse_key") and keys.get("google_cse_cx"):
                out.append(name)
        elif keys.get(name):
            out.append(name)
    return out


def summary(keys: Dict[str, str]) -> Dict[str, List[str]]:
    kinds: Dict[str, List[str]] = {"search": search_available(keys), "unblock": [], "enrich": []}
    for name in UNBLOCK_ORDER:
        if keys.get(name) or (name == "jina" and os.getenv("DISABLE_JINA", "") != "1"):
            kinds["unblock"].append(name)
    kinds["enrich"] = [n for n in ENRICH_ORDER if keys.get(n)]
    kinds["sources"] = ["reddit API"] if keys.get("reddit_client_id") and keys.get("reddit_client_secret") else []
    kinds["crm"] = ["salesforce"] if keys.get("salesforce_instance_url") and keys.get("salesforce_token") else []
    return kinds


# ---------------------------------------------------------------------------
# Search APIs → [{"url", "title", "snippet"}]
# ---------------------------------------------------------------------------
def _gl(region: str) -> str:
    return (region.split("-")[0] if region and region != "wt-wt" else "in") or "in"


def web_search(name: str, keys: Dict[str, str], query: str, max_results: int, region: str,
               max_age_months: int = 0) -> Tuple[List[dict], Optional[str], bool]:
    """(hits, error, rate_limited) from one search API. max_age_months > 0 restricts to recent pages."""
    gl = _gl(region)
    n = max(1, min(max_results, 100))
    tbs = {"tbs": f"qdr:m{max_age_months}"} if max_age_months else {}
    try:
        # Short timeout: a slow search API must not hold up the run (DuckDuckGo results are used anyway).
        with httpx.Client(timeout=httpx.Timeout(12.0, connect=6.0)) as c:
            if name == "serper":
                r = c.post("https://google.serper.dev/search", headers={"X-API-KEY": keys["serper"]},
                           json={"q": query, "gl": gl, "hl": "en", "num": n, **tbs})
                items = [(i.get("link"), i.get("title"), i.get("snippet"), i.get("date"))
                         for i in _ok(r).get("organic", [])]
            elif name == "serpapi":
                r = c.get("https://serpapi.com/search.json", params={
                    "engine": "google", "q": query, "gl": gl, "hl": "en", "num": n, "api_key": keys["serpapi"], **tbs})
                items = [(i.get("link"), i.get("title"), i.get("snippet"), i.get("date"))
                         for i in _ok(r).get("organic_results", [])]
            elif name == "google_cse":
                items = []
                for start in range(1, min(n, 30) + 1, 10):      # 10 per call, max 3 calls
                    r = c.get("https://www.googleapis.com/customsearch/v1", params={
                        "key": keys["google_cse_key"], "cx": keys["google_cse_cx"], "q": query,
                        "num": 10, "start": start, "gl": gl,
                        **({"dateRestrict": f"m{max_age_months}"} if max_age_months else {})})
                    page = _ok(r).get("items", [])
                    items += [(i.get("link"), i.get("title"), i.get("snippet"), _cse_date(i)) for i in page]
                    if len(page) < 10:
                        break
            elif name == "brave":
                r = c.get("https://api.search.brave.com/res/v1/web/search",
                          headers={"X-Subscription-Token": keys["brave"], "Accept": "application/json"},
                          params={"q": query, "count": min(n, 20), "country": gl.upper(),
                                  **({"freshness": "pm" if max_age_months <= 1 else "py"} if max_age_months else {})})
                items = [(i.get("url"), i.get("title"), re.sub(r"<[^>]+>", "", i.get("description") or ""),
                          i.get("page_age") or i.get("age"))
                         for i in (_ok(r).get("web") or {}).get("results", [])]
            else:
                return [], f"unknown search service {name}", False
    except _HTTPFail as exc:
        return [], f"{SERVICES.get(name, (0, name))[1]} {exc}", exc.code == 429
    except Exception as exc:
        return [], f"{name} {type(exc).__name__}: {str(exc)[:120]}", False
    hits = [{"url": u, "title": (t or "").strip(), "snippet": (s or "").strip(), "date": dates.parse(d)}
            for u, t, s, d in items if u]
    return hits[:n], None, False


def _cse_date(item: dict) -> Optional[str]:
    for meta in (item.get("pagemap") or {}).get("metatags") or []:
        for k in ("article:published_time", "og:published_time", "datepublished", "date"):
            if meta.get(k):
                return meta[k]
    return None


class _HTTPFail(Exception):
    def __init__(self, code: int, text: str):
        super().__init__(f"HTTP {code}: {text[:150]}")
        self.code = code


def _ok(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        raise _HTTPFail(r.status_code, r.text)
    try:
        return r.json()
    except ValueError:
        raise _HTTPFail(r.status_code, "non-JSON reply")


# ---------------------------------------------------------------------------
# Unblockers → (status, final_url, content_type, body)  — body is HTML, or markdown for Jina
# ---------------------------------------------------------------------------
_STRICT = ("linkedin.com", "facebook.com", "instagram.com", "quora.com", "indeed.com", "glassdoor.",
           "naukri.com", "bayt.com", "gulftalent.com")


async def unblock_fetch(name: str, keys: Dict[str, str], url: str, timeout_s: int = 60
                        ) -> Tuple[int, str, str, str, Optional[str]]:
    strict = any(h in url.lower() for h in _STRICT)
    try:
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as c:
            if name == "scraperapi":
                p = {"api_key": keys["scraperapi"], "url": url, "country_code": "in"}
                if strict:
                    p["premium"] = "true"
                r = await c.get("https://api.scraperapi.com/", params=p)
            elif name == "zenrows":
                p = {"apikey": keys["zenrows"], "url": url}
                if strict:
                    p["premium_proxy"] = "true"
                r = await c.get("https://api.zenrows.com/v1/", params=p)
            elif name == "scrapingbee":
                p = {"api_key": keys["scrapingbee"], "url": url, "render_js": "false"}
                if strict:
                    p["premium_proxy"] = "true"
                r = await c.get("https://app.scrapingbee.com/api/v1/", params=p)
            elif name == "jina":
                h = {"Accept": "text/plain", "X-Return-Format": "text"}
                if keys.get("jina"):
                    h["Authorization"] = f"Bearer {keys['jina']}"
                r = await c.get("https://r.jina.ai/" + url, headers=h)
                return r.status_code, url, "text/markdown", r.text, None
            else:
                return 0, url, "", "", f"unknown unblocker {name}"
    except Exception as exc:
        return 0, url, "", "", f"{name} {type(exc).__name__}: {str(exc)[:120]}"
    if r.status_code in (401, 403) and "api" in r.text[:300].lower() and "key" in r.text[:300].lower():
        return r.status_code, url, "", "", f"{SERVICES[name][1]} rejected its API key"
    return r.status_code, url, r.headers.get("content-type", "text/html"), r.text, None


# ---------------------------------------------------------------------------
# Contact enrichment
# ---------------------------------------------------------------------------
_PHONE_KEY = re.compile(r"phone|mobile|number|tel", re.IGNORECASE)


# Parts of a lookup reply that describe the employer, not the person — their phones / emails are skipped.
_COMPANY_KEYS = {"organization", "organisation", "account", "company", "employer", "current_employer_data",
                 "employment_history", "companies", "org"}
# Placeholder values some services return instead of a real contact (e.g. Apollo's locked emails).
_PLACEHOLDER = re.compile(r"not_unlocked|email_not|noemail|no-email|example\.|@domain\.com$|unavailable|redacted|"
                          r"\*\*\*", re.IGNORECASE)


def _collect(node, emails: List[str], phones: List[str], key: str = "") -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if str(k).lower() in _COMPANY_KEYS:
                continue
            _collect(v, emails, phones, str(k))
    elif isinstance(node, list):
        for v in node:
            _collect(v, emails, phones, key)
    elif isinstance(node, str):
        value = node.strip()
        if _PLACEHOLDER.search(value):
            return
        if EMAIL_RE.fullmatch(value):
            emails.append(value.lower())
        elif _PHONE_KEY.search(key) and not re.search(r"type|status|count|id$", key, re.IGNORECASE) \
                and 8 <= len(re.sub(r"\D", "", value)) <= 15:
            phones.append(value)


def _personal_first(emails: List[str]) -> List[str]:
    free = ("gmail.", "yahoo.", "hotmail.", "outlook.", "rediffmail.", "ymail.", "icloud.", "live.")
    uniq = list(dict.fromkeys(e for e in emails if "@" in e and not e.endswith(("apollo.io", "lusha.com"))))
    return sorted(uniq, key=lambda e: not any(f in e for f in free))


async def enrich_one(name: str, keys: Dict[str, str], person: dict) -> dict:
    """Look one person up. person: {name, linkedin_url, company}. Returns {emails, phones, error}."""
    li = person.get("linkedin_url") or ""
    full = (person.get("name") or "").strip()
    first, _, last = full.partition(" ")
    company = person.get("company") or ""
    try:
        async with httpx.AsyncClient(timeout=25) as c:
            if name == "contactout":
                if not li:
                    return {"error": "ContactOut needs a LinkedIn profile URL"}
                r = await c.get("https://api.contactout.com/v1/people/linkedin",
                                params={"profile": li, "include_phone": "true"},
                                headers={"authorization": "basic", "token": keys["contactout"]})
            elif name == "lusha":
                params = {"linkedinUrl": li} if li else {"firstName": first, "lastName": last, "companyName": company}
                if not li and not (first and last and company):
                    return {"error": "Lusha needs a LinkedIn URL, or name + company"}
                r = await c.get("https://api.lusha.com/v2/person", params=params, headers={"api_key": keys["lusha"]})
            elif name == "rocketreach":
                params = {"linkedin_url": li} if li else {"name": full, "current_employer": company}
                if not li and not (full and company):
                    return {"error": "RocketReach needs a LinkedIn URL, or name + company"}
                r = await c.get("https://api.rocketreach.co/api/v2/person/lookup", params=params,
                                headers={"Api-Key": keys["rocketreach"]})
            elif name == "apollo":
                q = {"reveal_personal_emails": "true"}
                if li:
                    q["linkedin_url"] = li
                elif full and company:
                    q.update({"name": full, "organization_name": company})
                else:
                    return {"error": "Apollo needs a LinkedIn URL, or name + company"}
                # Apollo reads match parameters from the query string.
                r = await c.post("https://api.apollo.io/api/v1/people/match", params=q, json={},
                                 headers={"x-api-key": keys["apollo"], "Content-Type": "application/json",
                                          "Cache-Control": "no-cache"})
            else:
                return {"error": f"unknown provider {name}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    label = SERVICES[name][1]
    why = re.sub(r"\s+", " ", r.text or "")[:200]
    if r.status_code == 401:
        return {"error": f"{label} rejected the API key (401): {why}", "stop": True, "status": 401}
    if r.status_code == 403:
        # Usually: the plan has no API access, or this endpoint needs a higher plan / master key.
        return {"error": f"{label} refused the request (403 — plan without API access?): {why}",
                "stop": True, "status": 403}
    if r.status_code in (402, 429):
        return {"error": f"{label} out of credits / rate limited ({r.status_code}): {why}", "stop": True,
                "status": r.status_code}
    if r.status_code == 404:
        return {"emails": [], "phones": [], "status": 404, "raw": why}
    if r.status_code >= 400:
        return {"error": f"{label} HTTP {r.status_code}: {why}", "status": r.status_code}
    try:
        data = r.json()
    except ValueError:
        return {"error": f"{label} returned non-JSON"}
    emails, phones = [], []
    _collect(data, emails, phones)
    return {"emails": _personal_first(emails), "phones": list(dict.fromkeys(phones)), "status": r.status_code,
            "raw": why}


def find_linkedin_url(keys: Dict[str, str], name: str, hints: str = "") -> Optional[str]:
    """A lead with only a name: look for their LinkedIn profile with a Google search API
    ("Name" role place site:linkedin.com/in) and accept a result only if its title starts with the name."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z.'-]+", name or "") if len(w) > 1]
    if len(words) < 2:
        return None                                  # a single word / handle matches too many people
    apis = [a for a in search_available(keys) if a != "scrapedo"]
    if not apis:
        return None
    query = f'site:linkedin.com/in "{" ".join(words[:3])}" {hints}'.strip()
    hits, _, _ = web_search(apis[0], keys, query, 5, "wt-wt")
    want = " ".join(w.lower() for w in words[:2])
    for h in hits:
        title = re.sub(r"[^a-z ]+", " ", (h.get("title") or "").lower())
        if re.match(r"https?://([a-z]{2,3}\.)?linkedin\.com/in/", h["url"]) and \
                " ".join(title.split()[:2]) == want:
            return h["url"].split("?")[0]
    return None


async def enrich_person(keys: Dict[str, str], person: dict, providers: Optional[List[str]] = None,
                        stopped: Optional[set] = None) -> dict:
    """Try providers in order until both phone and email are found."""
    stopped = stopped if stopped is not None else set()
    found = {"email": None, "phone": None, "provider": None, "tried": [], "errors": [], "profile_url": None}
    if not person.get("linkedin_url") and not person.get("company"):
        url = await asyncio.to_thread(find_linkedin_url, keys, person.get("name") or "", person.get("hints") or "")
        if not url:
            found["errors"].append("no LinkedIn profile or company to look up")
            return found
        person = {**person, "linkedin_url": url}
        found["profile_url"] = url
    for p in providers or ENRICH_ORDER:
        if not keys.get(p) or p in stopped:
            continue
        if found["email"] and found["phone"]:
            break
        res = await enrich_one(p, keys, person)
        found["tried"].append(p)
        if res.get("error"):
            found["errors"].append(f"{p}: {res['error']}")
            if res.get("stop"):
                stopped.add(p)
            continue
        if not found["email"] and res["emails"]:
            found["email"], found["provider"] = res["emails"][0], found["provider"] or p
        if not found["phone"] and res["phones"]:
            found["phone"], found["provider"] = res["phones"][0], found["provider"] or p
        await asyncio.sleep(0.2)
    return found
