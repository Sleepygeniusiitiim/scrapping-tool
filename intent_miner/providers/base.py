"""Provider interface, capabilities, per-provider rate limits and URL → provider routing."""

from __future__ import annotations

import asyncio
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

from ..models import QuerySpec, RawDocument


@dataclass(frozen=True)
class Capability:
    search: bool = True             # can find documents for a query itself
    fetch: bool = True              # can read a full document by URL
    comments: bool = False          # returns comments / answers with authors
    profiles: bool = False
    private_groups: bool = False    # never true: private communities are out of scope
    access: str = "public"          # public | api | api_or_search | search_only | auth_required


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    requests_per_second: float = 1.0
    max_concurrency: int = 3
    retry_attempts: int = 2


class RateLimiter:
    """Independent limiter per provider: concurrency cap + minimum spacing between requests."""

    def __init__(self, cfg: ProviderConfig):
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self._gap = 1.0 / max(cfg.requests_per_second, 0.01)
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._sem.acquire()
        async with self._lock:
            wait = self._next - time.monotonic()
            self._next = max(time.monotonic(), self._next) + self._gap
        if wait > 0:
            await asyncio.sleep(wait + random.uniform(0, 0.2))
        return self

    async def __aexit__(self, *exc):
        self._sem.release()


class BaseProvider(ABC):
    name = "base"
    capability = Capability()
    config = ProviderConfig("base")

    def __init__(self, keys: Dict[str, str]):
        self.keys = keys
        self.limiter = RateLimiter(self.config)
        self.stats = {"requests": 0, "ok": 0, "blocked": 0, "failed": 0}

    def note(self, status: str) -> None:
        self.stats["requests"] += 1
        self.stats[status if status in self.stats else "failed"] += 1

    @property
    def health(self) -> str:
        s = self.stats
        if not s["requests"]:
            return "IDLE"
        bad = (s["blocked"] + s["failed"]) / s["requests"]
        if s["blocked"] and not s["ok"]:
            return "BLOCKED"
        return "HEALTHY" if bad < 0.25 else ("DEGRADED" if bad < 0.75 else "FAILING")

    @abstractmethod
    async def search(self, query: str, spec: QuerySpec, limit: int) -> List[dict]:
        """Candidate hits: [{url, title, snippet, date, doc?}] — `doc` when the API already returned content."""

    @abstractmethod
    async def fetch(self, url: str, hit: Optional[dict] = None) -> RawDocument:
        ...


DIRECTORY_HOSTS = ("justdial.com", "indiamart.com", "sulekha.com", "tradeindia.com", "exportersindia.com",
                   "yellowpages.in", "olx.in", "quikr.com", "asklaila.com", "grotal.com", "yelu.in", "infoisinfo.co.in",
                   "joonsquare.com", "magicpin.in", "urbanpro.com")
BLOG_HOSTS = ("medium.com", "blogspot.com", "blogger.com", "wordpress.com", "substack.com", "tumblr.com",
              "hashnode.dev", "hashnode.com", "dev.to", "wixsite.com", "weebly.com", "ghost.io", "livejournal.com",
              "quora.com/spaces")


def source_of(url: str) -> str:
    """URL classifier: which adapter reads this URL."""
    host = (urlparse(url).hostname or "").lower()
    path = (urlparse(url).path or "").lower()
    if host.endswith("reddit.com") or host == "redd.it":
        return "reddit"
    if host.endswith("quora.com"):
        return "quora"
    if host.endswith("linkedin.com"):
        return "linkedin"
    if host.endswith("facebook.com") or host.endswith("fb.com"):
        return "facebook"
    if host.endswith("instagram.com"):
        return "instagram"
    if (host.startswith(("maps.google.", "www.google.", "google.")) and path.startswith("/maps")) or \
            host.startswith("maps.google."):
        return "maps"
    if any(host == d or host.endswith("." + d) for d in DIRECTORY_HOSTS):
        return "directories"
    if host.endswith("youtube.com") or host == "youtu.be":
        return "youtube"
    if any(host == b or host.endswith("." + b) for b in BLOG_HOSTS):
        return "blogs"
    if any(x in path for x in (".rss", "/feed", ".xml", "/rss")):
        return "rss"
    if re.search(r"/blogs?/|/\d{4}/\d{2}/", path):
        return "blogs"
    return "forums"


# Configurable reliability signal of each source type, used in ranking (not a judgement of the people).
SOURCE_QUALITY = {
    "forums": 85, "reddit": 80, "quora": 70, "linkedin": 75, "facebook": 65, "rss": 70, "youtube": 75, "blogs": 65, "instagram": 65, "maps": 90, "directories": 75,
    "portal": 75, "blog": 55, "web": 35, "snippet": 25,
}
