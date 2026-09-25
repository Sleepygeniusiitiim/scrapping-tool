"""
Search layer — DuckDuckGo (via the `ddgs` metasearch library) dorking.

* One query at a time with randomized 1.0–2.5 s jitter between queries.
* Rate-limit (HTTP 202/429 "Ratelimit") responses trigger a longer cool-down
  and one retry, then the query is skipped instead of crashing the run.
* Every result URL is canonicalised so that the same page reached through
  different tracking links dedups to one string in Supabase.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlparse, urlunparse

log = logging.getLogger(__name__)

# `duckduckgo-search` was renamed to `ddgs`; support both.
try:  # pragma: no cover - import shim
    from ddgs import DDGS  # type: ignore
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException  # type: ignore
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS  # type: ignore
    from duckduckgo_search.exceptions import (  # type: ignore
        DuckDuckGoSearchException as DDGSException,
        RatelimitException,
        TimeoutException,
    )

JITTER_RANGE = (1.0, 2.5)          # seconds between individual queries
RATE_LIMIT_COOLDOWN = (8.0, 15.0)  # seconds to back off after a 429

# Query parameters that never change page content.
_TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "msclkid", "yclid", "mc_cid", "mc_eid", "igshid",
    "ref", "ref_src", "ref_url", "referrer", "source", "src", "trk", "trkinfo",
    "trackingid", "lipi", "original_referer", "originalsubdomain", "si", "share_id",
    "_ga", "_gl", "spm", "cmpid", "sh", "rdt", "context", "sort", "mibextid", "rcm",
}
_TRACKING_PREFIXES = ("utm_", "pk_", "hsa_", "mtm_", "at_")

# Result domains that are never useful as sources.
_JUNK_DOMAINS = (
    "duckduckgo.com", "google.com/search", "bing.com/search", "youtube.com", "youtu.be",
    "facebook.com/login", "accounts.google.com", "play.google.com", "apps.apple.com",
)
_HTTPS_ONLY = ("linkedin.com", "reddit.com", "quora.com", "facebook.com", "x.com", "twitter.com",
               "naukri.com", "indeed.com", "github.com", "medium.com")
_JUNK_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".mp4", ".zip", ".exe")


# ---------------------------------------------------------------------------
# URL canonicalisation
# ---------------------------------------------------------------------------
def _unwrap_redirect(url: str) -> str:
    """Unwrap DuckDuckGo / Google redirect links to their real target."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = dict(parse_qsl(parsed.query)).get("uddg")
        if target:
            return unquote(target)
    if host.endswith("google.com") and parsed.path == "/url":
        q = dict(parse_qsl(parsed.query))
        target = q.get("q") or q.get("url")
        if target:
            return target
    return url


def canonicalize_url(url: str) -> Optional[str]:
    """
    Normalise a URL so equivalent links compare equal.

    - unwrap search-engine redirects
    - https for major platforms, lowercase host, drop default ports and 'm.' / 'www.' variants
    - LinkedIn country subdomains (in.linkedin.com) → www.linkedin.com
    - Reddit mirrors (old./np./m.) → www.reddit.com
    - strip utm_* and other tracking params, sort the remaining ones
    - drop fragments and trailing slashes
    Returns None for non-http(s) or unparsable input.
    """
    if not url:
        return None
    try:
        url = _unwrap_redirect(url.strip())
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return None

        host = (parsed.hostname or "").lower().rstrip(".")
        if not host:
            return None

        # Platform-specific host normalisation
        if host.endswith("linkedin.com"):
            host = "www.linkedin.com"
        elif host.endswith("reddit.com"):
            host = "www.reddit.com"
        elif host.endswith("quora.com") and host.count(".") == 1:
            host = "www.quora.com"
        elif host.startswith("m.") and host.count(".") >= 2:
            host = "www." + host[2:]

        port = parsed.port
        netloc = host if port in (None, 80, 443) else f"{host}:{port}"

        # Path: collapse duplicate slashes, drop trailing slash, keep case
        # (paths are case-sensitive on many sites), percent-encode quotes/spaces.
        path = re.sub(r"/{2,}", "/", parsed.path or "/")
        if len(path) > 1:
            path = path.rstrip("/")
        path = quote(unquote(path), safe="/:@!$&'()*+,;=-._~%")

        query_pairs = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith(_TRACKING_PREFIXES)
        ]
        query = urlencode(sorted(query_pairs))

        # Keep the original scheme (some sites are http-only) except for the big
        # platforms, which are always https — so http/https variants dedup.
        scheme = "https" if host.endswith(_HTTPS_ONLY) else parsed.scheme
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:  # malformed input should never crash a wave
        return None


def is_useful_url(url: str) -> bool:
    low = url.lower()
    if any(j in low for j in _JUNK_DOMAINS):
        return False
    if urlparse(low).path.endswith(_JUNK_EXTENSIONS):
        return False
    return True


def domain_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def platform_from_url(url: str) -> str:
    """Coarse platform tag used for the `platform` column."""
    host = domain_of(url)
    table = {
        "linkedin.com": "linkedin",
        "reddit.com": "reddit",
        "quora.com": "quora",
        "facebook.com": "facebook",
        "twitter.com": "x",
        "x.com": "x",
        "naukri.com": "job_portal",
        "indeed.": "job_portal",
        "monster": "job_portal",
        "foundit.in": "job_portal",
        "shine.com": "job_portal",
        "apna.co": "job_portal",
        "workindia": "job_portal",
        "glassdoor": "job_portal",
        "timesjobs": "job_portal",
        "practicalmachinist.com": "forum",
        "cnczone.com": "forum",
        "eng-tips.com": "forum",
        "github.com": "github",
    }
    for needle, tag in table.items():
        if needle in host:
            return tag
    if "forum" in host or "/forum" in url.lower() or "/thread" in url.lower():
        return "forum"
    return host.removeprefix("www.") or "web"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
@dataclass
class SearchHit:
    url: str                # canonical URL
    title: str = ""
    snippet: str = ""
    query: str = ""


@dataclass
class QueryOutcome:
    query: str
    hits: List[SearchHit] = field(default_factory=list)
    error: Optional[str] = None
    rate_limited: bool = False


def search_query(
    query: str,
    max_results: int = 10,
    region: str = "in-en",
    backend: str = "auto",
    timeout: int = 15,
) -> QueryOutcome:
    """
    Run ONE query. Never raises: failures are reported on the outcome.

    backend: "auto" lets ddgs rotate across engines (most resilient);
             "duckduckgo" forces DuckDuckGo only.
    """
    outcome = QueryOutcome(query=query)
    for attempt in range(2):
        try:
            raw = DDGS(timeout=timeout).text(
                query, region=region, safesearch="off", max_results=max_results, backend=backend
            ) or []
            seen: set[str] = set()
            for item in raw:
                href = item.get("href") or item.get("url") or item.get("link") or ""
                canon = canonicalize_url(href)
                if not canon or canon in seen or not is_useful_url(canon):
                    continue
                seen.add(canon)
                outcome.hits.append(
                    SearchHit(
                        url=canon,
                        title=(item.get("title") or "").strip(),
                        snippet=(item.get("body") or item.get("snippet") or "").strip(),
                        query=query,
                    )
                )
            outcome.error = None
            return outcome
        except RatelimitException as exc:
            outcome.rate_limited = True
            outcome.error = f"rate limited: {exc}"
            if attempt == 0:
                time.sleep(random.uniform(*RATE_LIMIT_COOLDOWN))
        except TimeoutException as exc:
            outcome.error = f"timeout: {exc}"
            if attempt == 0:
                time.sleep(2.0)
        except DDGSException as exc:
            # ddgs raises DDGSException("No results found.") on empty SERPs.
            if "no results" in str(exc).lower():
                outcome.error = None
                return outcome
            outcome.error = f"search error: {exc}"
            if attempt == 0:
                time.sleep(3.0)
        except Exception as exc:  # network errors, parsing changes, etc.
            outcome.error = f"{type(exc).__name__}: {exc}"
            break
    return outcome


def jitter_sleep(rng: tuple[float, float] = JITTER_RANGE) -> float:
    delay = random.uniform(*rng)
    time.sleep(delay)
    return delay


def search_wave(
    queries: Iterable[str],
    max_results: int = 10,
    region: str = "in-en",
    backend: str = "auto",
    on_query: Optional[Callable[[QueryOutcome], None]] = None,
) -> Dict[str, SearchHit]:
    """
    Run all queries of a wave sequentially with jitter.
    Returns {canonical_url: SearchHit} (first hit wins).
    """
    results: Dict[str, SearchHit] = {}
    queries = list(queries)
    for i, q in enumerate(queries):
        outcome = search_query(q, max_results=max_results, region=region, backend=backend)
        for hit in outcome.hits:
            results.setdefault(hit.url, hit)
        if on_query:
            on_query(outcome)
        if i < len(queries) - 1:
            jitter_sleep()
    return results
