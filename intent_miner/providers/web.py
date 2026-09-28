"""
Search engines (discovery), Quora (discovery-only), and forums / websites / search-indexed LinkedIn and
Facebook pages (fetch + generic extraction), plus RSS / Atom feeds.

Generic extraction order for a page: schema.org JSON-LD thread (posts / comments / answers with authors
and dates) → page text split into blocks. Pages are fetched with the existing fetcher, so robots.txt,
login walls and the configured unblockers behave exactly as in the main pipeline — nothing here logs in,
solves CAPTCHAs or reads private groups.
"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from typing import List, Optional
from urllib.parse import urlparse

import httpx

import dates
import pipeline
import rule_extractor
from fetcher import fetch_batch

from ..models import QuerySpec, RawDocument, Unit
from .base import BaseProvider, Capability, ProviderConfig, source_of

_THREAD = re.compile(r"^(POST|COMMENT|ORG) by (.+?)(?: <(https?://[^>\s]+)>)?(?: \[(\d{4}-\d{2}-\d{2})\])?: (.*)$",
                     re.MULTILINE)
_PAGE_DATE = re.compile(r"^Page date: (\d{4}-\d{2}-\d{2})$", re.MULTILINE)
_QUORA_AUTHOR = re.compile(r"/(?:answer|profile)/([A-Za-z][A-Za-z0-9-]+?)(?:-\d+)?(?:/|$)")


def _quora_author(url: str) -> Optional[str]:
    """quora.com/How-do-I-…/answer/Ravi-Kumar-12 → 'Ravi Kumar'."""
    m = _QUORA_AUTHOR.search(urlparse(url).path)
    return m.group(1).replace("-", " ") if m else None


class SearchProvider(BaseProvider):
    """Search-engine discovery (DuckDuckGo + the configured Google APIs, via pipeline.run_query)."""
    name = "search"
    capability = Capability(search=True, fetch=False, access="public")
    config = ProviderConfig("search", requests_per_second=0.7, max_concurrency=1)

    def __init__(self, keys, backend: str = "auto", region: str = "wt-wt", max_results: int = 20):
        super().__init__(keys)
        self.backend, self.region, self.max_results = backend, region, max_results

    async def search(self, query: str, spec: QuerySpec, limit: int) -> List[dict]:
        months = max(1, round((spec.max_age_days or 0) / 30)) if spec.max_age_days else 0
        async with self.limiter:
            r = await asyncio.to_thread(pipeline.run_query, query, min(limit, self.max_results), self.region,
                                        self.backend, self.keys, months)
        self.note("failed" if r.get("error") and not r["hits"] else "ok")
        if r.get("error") and not r["hits"]:
            raise RuntimeError(r["error"])
        return r["hits"]

    async def fetch(self, url: str, hit: Optional[dict] = None) -> RawDocument:
        raise NotImplementedError


class QuoraProvider(BaseProvider):
    """Quora blocks crawlers, so it is a discovery adapter: the question / answer found by a search engine
    becomes a document from its title + snippet, with the answer author taken from the URL."""
    name = "quora"
    capability = Capability(search=False, fetch=True, comments=False, access="search_only")
    config = ProviderConfig("quora", requests_per_second=2, max_concurrency=4)

    async def search(self, query: str, spec: QuerySpec, limit: int) -> List[dict]:
        return []

    async def fetch(self, url: str, hit: Optional[dict] = None) -> RawDocument:
        hit = hit or {}
        text = " ".join(x for x in (hit.get("title"), hit.get("snippet")) if x).strip()
        self.note("ok" if text else "failed")
        doc = RawDocument(url=url, source="quora", title=hit.get("title", ""), date=hit.get("date"), via="snippet")
        if text:
            doc.units.append(Unit("answer" if "/answer/" in url else "snippet", _quora_author(url), text,
                                  None, hit.get("date")))
        else:
            doc.status, doc.error = "failed", "no snippet"
        return doc


class WebProvider(BaseProvider):
    """Forums, blogs, public websites and search-indexed LinkedIn / Facebook pages."""
    name = "forums"
    capability = Capability(search=False, fetch=True, comments=True, access="public")
    config = ProviderConfig("forums", requests_per_second=3, max_concurrency=5)

    def __init__(self, keys, respect_robots: bool = True, timeout_s: int = 15):
        super().__init__(keys)
        self.respect_robots, self.timeout_s = respect_robots, timeout_s

    async def search(self, query: str, spec: QuerySpec, limit: int) -> List[dict]:
        return []

    async def fetch_many(self, items: List[dict]) -> List[RawDocument]:
        urls = [i["url"] for i in items]
        outcomes = await fetch_batch(urls, page_timeout_s=self.timeout_s, respect_robots=self.respect_robots,
                                     keys=self.keys)
        docs = []
        for item, o in zip(items, outcomes):
            src = source_of(o.url)
            if o.ok:
                self.note("ok")
                doc = self.parse(o.url, src, o.markdown, item, "direct" if o.via == "direct" else "unblocker")
                doc.metadata["markdown"] = o.markdown          # for the classic per-page extraction
                docs.append(doc)
                continue
            self.note("blocked" if o.blocked else "failed")
            # Walled / refused page: the search snippet is still evidence (marked as such).
            doc = snippet_doc(o.url, src, item)
            doc.status = "blocked" if o.blocked else "failed"
            doc.error = o.error
            docs.append(doc)
        return docs

    async def fetch(self, url: str, hit: Optional[dict] = None) -> RawDocument:
        return (await self.fetch_many([hit or {"url": url}]))[0]

    @staticmethod
    def parse(url: str, source: str, markdown: str, hit: dict, via: str) -> RawDocument:
        title = (re.match(r"#\s*(.+)", markdown) or [None, hit.get("title", "")])[1] or ""
        page_date = (_PAGE_DATE.search(markdown) or [None, None])[1] or hit.get("date") or \
            dates.from_linkedin_url(url)
        doc = RawDocument(url=url, source=source, title=title.strip(), date=page_date, via=via)
        for kind, author, link, when, body in _THREAD.findall(markdown):
            author = None if author.strip().lower() == "unknown" else author.strip()
            doc.units.append(Unit({"POST": "post", "ORG": "organization"}.get(kind, "comment"), author, body,
                                  link or None, when or page_date))
        if doc.units:
            return doc
        comments = social_comments(markdown.split("## Page text", 1)[-1])
        if sum(1 for c in comments if c.kind == "comment") >= 2:
            from profile_visit import profile_url_for
            for c in comments:                  # their own profile page (visited later, before any lookup)
                c.author_url = c.author_url or profile_url_for(url, c.author, None)
            doc.units += comments
            return doc
        # No structured thread: split the page text into blocks; each block keeps the nearest stated name.
        text = markdown.split("## Page text", 1)[-1]
        blocks, cur = [], []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                if cur:
                    blocks.append(" ".join(cur))
                    cur = []
                continue
            cur.append(line)
        if cur:
            blocks.append(" ".join(cur))
        profile = bool(re.search(r"linkedin\.com/in/|/profile/|/user/|/members?/", url))
        if profile:
            name = rule_extractor._name_from_title(title, url)
            doc.units.append(Unit("profile", name, " ".join(blocks)[:6000], url, page_date))
            return doc
        for b in blocks:
            if len(b) >= 60:
                doc.units.append(Unit("post", rule_extractor._name_said(b), b[:3000], None, page_date))
            if len(doc.units) >= 60:
                break
        return doc


_SOCIAL = re.compile(
    r"(?:^|\s)@?([A-Za-z0-9_.]{3,30})\s+(?:Edited\s*·?\s*)?(\d{1,3})\s?(s|m|h|d|w|y|min|mins|hr|hrs|days?|wks?|"
    r"weeks?|yrs?|years?)\b\s*(?:ago)?\s+(.{2,600}?)(?=\s+(?:\d+\s+likes?\s+)?(?:Reply|Like\s+Reply|See translation)\b)",
    re.IGNORECASE)
_NOT_HANDLE = {"likes", "like", "reply", "replies", "view", "views", "more", "comments", "comment", "hide", "and"}


def social_comments(text: str) -> List[Unit]:
    """Comment lists as social sites render them in text: 'shrikantsingh640 15w Can I apply this job 1 like Reply'.
    Each comment becomes its own unit, with the handle as author and a date from the relative age."""
    flat = re.sub(r"\s+", " ", text or "")
    out, seen = [], set()
    first = _SOCIAL.search(flat)
    caption = flat[:first.start()].strip() if first else ""
    if len(caption) >= 20:                   # the post / reel caption the comments reply to
        out.append(Unit("post", None, caption[:2000], None, None))
    for handle, n, unit, body in _SOCIAL.findall(flat):
        if handle.lower() in _NOT_HANDLE or handle.isdigit():
            continue
        body = body.strip()
        key = (handle.lower(), body[:80])
        if key in seen or len(body) < 2:
            continue
        seen.add(key)
        u = unit.lower()
        days = (int(n) * 7 if u.startswith("w") else int(n) * 365 if u.startswith("y") else
                int(n) if u.startswith("d") else 0)
        when = dates.parse(f"{days} days ago") if days else dates.today().isoformat()
        out.append(Unit("comment", handle, body, None, when))
    return out


def snippet_doc(url: str, source: str, hit: dict) -> RawDocument:
    text = " ".join(x for x in (hit.get("title"), hit.get("snippet")) if x).strip()
    doc = RawDocument(url=url, source=source, title=hit.get("title", ""), date=hit.get("date"), via="snippet")
    if text:
        author = _quora_author(url) if source == "quora" else None
        doc.units.append(Unit("snippet", author, text, None, hit.get("date")))
    return doc


class RssProvider(BaseProvider):
    """RSS / Atom feeds the user lists (forum 'new posts' feeds, subreddit .rss, job-board feeds)."""
    name = "rss"
    capability = Capability(search=True, fetch=False, access="public")
    config = ProviderConfig("rss", requests_per_second=2, max_concurrency=3)

    def __init__(self, keys, feeds: List[str]):
        super().__init__(keys)
        self.feeds = [f.strip() for f in feeds if f.strip().startswith("http")][:20]

    async def search(self, query: str, spec: QuerySpec, limit: int) -> List[dict]:
        terms = [t.lower() for t in (spec.professions + spec.high_intent_terms) if t]
        hits = []
        for feed in self.feeds:
            try:
                async with self.limiter:
                    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
                        r = await c.get(feed, headers={"User-Agent": "Mozilla/5.0 (intent-miner feed reader)"})
                r.raise_for_status()
                root = ET.fromstring(r.content)
                self.note("ok")
            except Exception as exc:
                self.note("failed")
                hits.append({"url": feed, "error": f"{type(exc).__name__}: {str(exc)[:80]}"})
                continue
            for it in root.iter():
                tag = it.tag.split("}")[-1]
                if tag not in ("item", "entry"):
                    continue
                def get(*names):
                    # (an Element with no children is falsy, so compare with None explicitly)
                    for n in names:
                        el = next((c for c in it if c.tag.split("}")[-1] == n), None)
                        if el is not None:
                            return el
                    return None
                link = get("link")
                url = (link.get("href") if link is not None and link.get("href") else
                       (link.text if link is not None else "")) or ""
                title = (get("title").text if get("title") is not None else "") or ""
                body_el = get("description", "summary", "content")
                body = re.sub(r"<[^>]+>", " ", (body_el.text if body_el is not None else "") or "")
                author_el = get("author", "creator")
                author = None
                if author_el is not None:
                    name_el = next((c for c in author_el if c.tag.split("}")[-1] == "name"), None)
                    author = (name_el.text if name_el is not None else author_el.text) or None
                date_el = get("pubDate", "published", "updated", "date")
                when = dates.parse(date_el.text if date_el is not None else None)
                text = f"{title}. {body}".strip()
                if terms and not any(t in text.lower() for t in terms):
                    continue
                doc = RawDocument(url=url.strip(), source=source_of(url) if url else "rss", title=title,
                                  date=when, via="feed")
                doc.units.append(Unit("post", author, re.sub(r"\s+", " ", text)[:3000], None, when))
                hits.append({"url": doc.url, "title": title, "snippet": text[:300], "date": when, "doc": doc})
        return hits

    async def fetch(self, url: str, hit: Optional[dict] = None) -> RawDocument:
        return (hit or {}).get("doc") or RawDocument(url=url, source="rss", status="failed", error="not in feed")
