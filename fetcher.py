"""
Fetch layer for serverless (Vercel) — plain HTTP + HTML → text.

Vercel functions can't ship a headless Chromium, so this replaces the
Crawl4AI crawler used by the local Streamlit version. Requests go through
`primp` (already installed by `ddgs`), which impersonates Chrome's TLS
fingerprint — many sites refuse plain Python HTTP clients with a 403. Pages that need
JavaScript or a login (LinkedIn, most of Facebook) come back as "blocked",
and the pipeline falls back to the search-result snippet — same as before.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib import robotparser
from urllib.parse import urlparse

import primp
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

DEFAULT_PAGE_TIMEOUT_S = 15
MIN_USEFUL_CHARS = 250          # below this the page is treated as empty
MAX_TEXT_CHARS = 60_000         # keep memory and LLM cost bounded
ROBOTS_AGENT = "*"               # impersonated Chrome has no bot token; obey the wildcard rules

# Phrases that indicate an auth wall / bot challenge rather than real content.
_WALL_PATTERNS = re.compile(
    r"(authwall|sign in to view|join linkedin|log in to continue|login to continue|"
    r"please enable javascript|verify you are human|are you a robot|captcha|"
    r"access denied|cf-browser-verification|checking your browser|"
    r"you've been blocked|request blocked|unusual traffic)",
    re.IGNORECASE,
)
_DROP_TAGS = ["script", "style", "nav", "footer", "header", "form", "aside", "noscript", "svg", "iframe"]


@dataclass
class CrawlOutcome:
    url: str
    markdown: str = ""
    ok: bool = False
    blocked: bool = False
    error: Optional[str] = None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    body = soup.body or soup
    text = body.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return (f"# {title}\n\n{text}" if title else text).strip()[:MAX_TEXT_CHARS]


class _Robots:
    """Per-run robots.txt cache (one fetch per host)."""

    def __init__(self, client: primp.AsyncClient):
        self._client = client
        self._cache: Dict[str, Optional[robotparser.RobotFileParser]] = {}

    async def allowed(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self._cache:
            try:
                r = await self._client.get(f"{base}/robots.txt", timeout=5)
                if r.status_code >= 400:
                    self._cache[base] = None      # no robots.txt → allowed
                else:
                    rp = robotparser.RobotFileParser()
                    rp.parse(r.text.splitlines())
                    self._cache[base] = rp
            except Exception:
                self._cache[base] = None
        rp = self._cache[base]
        return True if rp is None else rp.can_fetch(ROBOTS_AGENT, url)


async def _fetch_one(client: primp.AsyncClient, robots: Optional[_Robots], url: str,
                     timeout_s: float) -> CrawlOutcome:
    try:
        if robots and not await robots.allowed(url):
            return CrawlOutcome(url=url, blocked=True, error="disallowed by robots.txt")
        r = await asyncio.wait_for(client.get(url, timeout=timeout_s), timeout=timeout_s + 3)
    except (asyncio.TimeoutError, primp.TimeoutError):
        return CrawlOutcome(url=url, error=f"timed out after {timeout_s:.0f}s")
    except Exception as exc:  # DNS failure, TLS error, too many redirects, ...
        return CrawlOutcome(url=url, error=f"{type(exc).__name__}: {str(exc)[:200]}")

    if r.status_code in (401, 403, 429, 999):
        return CrawlOutcome(url=url, blocked=True, error=f"HTTP {r.status_code}")
    if r.status_code >= 400:
        return CrawlOutcome(url=url, error=f"HTTP {r.status_code}")
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return CrawlOutcome(url=url, error=f"unsupported content-type {ctype[:40]}")

    try:
        text = html_to_text(r.text)
    except Exception as exc:
        return CrawlOutcome(url=url, error=f"HTML parsing failed: {exc}")

    head = text[:3000]
    if len(text) < MIN_USEFUL_CHARS or (_WALL_PATTERNS.search(head) and len(text) < 4000):
        return CrawlOutcome(url=url, markdown=text, blocked=True, error="login wall / bot check / empty page")
    return CrawlOutcome(url=url, markdown=text, ok=True)


async def fetch_batch(urls: List[str], page_timeout_s: int = DEFAULT_PAGE_TIMEOUT_S,
                      respect_robots: bool = True) -> List[CrawlOutcome]:
    """Fetch a batch of URLs concurrently; never raises."""
    if not urls:
        return []
    client = primp.AsyncClient(impersonate="chrome", follow_redirects=True, max_redirects=5,
                               timeout=page_timeout_s, headers={"Accept-Language": "en-IN,en;q=0.9"})
    robots = _Robots(client) if respect_robots else None
    return list(await asyncio.gather(*(_fetch_one(client, robots, u, page_timeout_s) for u in urls)))
