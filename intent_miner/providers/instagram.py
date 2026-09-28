"""
Instagram posts / reels pasted by the user.

A server opening https://www.instagram.com/p/<code>/ gets a login wall, and the comments load later with
JavaScript, so the normal page reader sees nothing. Instagram's own *embed* page
(https://www.instagram.com/p/<code>/embed/captioned/, the one websites use to show a post) is served without
login and carries the caption, the owner and — for many posts — the first comments in a JSON blob.
All comments of a post are only available through the Meta Graph API, and only for posts on your own
account (see meta_autoreply.py).

Instagram's robots.txt does not allow crawlers, so with "Respect robots.txt" ticked this reader stops.
"""

from __future__ import annotations

import datetime as dt
import html as html_lib
import json
import re
from typing import Iterable, List, Optional

import primp

from ..models import RawDocument, Unit
from .base import BaseProvider, Capability, ProviderConfig

_CODE = re.compile(r"instagram\.com/(?:[A-Za-z0-9_.]+/)?(p|reel|reels|tv)/([A-Za-z0-9_-]{5,})", re.IGNORECASE)


def post_code(url: str) -> Optional[str]:
    m = _CODE.search(url or "")
    return m.group(2) if m else None


def _day(ts) -> Optional[str]:
    try:
        return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _walk(node) -> Iterable[dict]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _json_blobs(page: str) -> List[object]:
    """The embed page stores the post as an escaped JSON string ("contextJSON") or as additional data."""
    out = []
    for m in re.finditer(r'"contextJSON"\s*:\s*("(?:[^"\\]|\\.)*")', page):
        try:
            out.append(json.loads(json.loads(m.group(1))))
        except ValueError:
            pass
    for m in re.finditer(r"__additionalDataLoaded\([^,]+,\s*(\{.*?\})\);", page, re.S):
        try:
            out.append(json.loads(m.group(1)))
        except ValueError:
            pass
    return out


def parse_embed(url: str, page: str) -> RawDocument:
    doc = RawDocument(url=url, source="instagram", via="embed")
    seen = set()
    owner = caption = date = None
    for blob in _json_blobs(page):
        for n in _walk(blob):
            if "shortcode" in n and isinstance(n.get("owner"), dict) and not owner:
                owner = n["owner"].get("username")
                date = _day(n.get("taken_at_timestamp"))
                edges = ((n.get("edge_media_to_caption") or {}).get("edges") or [{}])
                caption = ((edges[0] or {}).get("node") or {}).get("text") or caption
            text, who = n.get("text"), (n.get("owner") or {}).get("username") if isinstance(n.get("owner"), dict) else None
            if text and who and "created_at" in n and (who, text[:80]) not in seen:
                seen.add((who, text[:80]))
                doc.units.append(Unit("comment", who, text, f"https://www.instagram.com/{who}/", _day(n["created_at"])))
    if not caption:                              # HTML fallback: the visible caption block
        m = re.search(r'class="Caption"[^>]*>(.*?)</div>', page, re.S)
        if m:
            caption = re.sub(r"<[^>]+>", " ", m.group(1))
            caption = html_lib.unescape(re.sub(r"\s+", " ", caption)).strip()
        who = re.search(r'class="CaptionUsername"[^>]*>([^<]+)<', page)
        owner = owner or (who.group(1).strip() if who else None)
    if caption:
        doc.units.insert(0, Unit("post", owner, caption[:3000], f"https://www.instagram.com/{owner}/" if owner else None,
                                 date))
    doc.date = date
    doc.title = f"Instagram post by @{owner}" if owner else "Instagram post"
    total = re.search(r'"comment_count"\s*:\s*(\d+)|View all ([\d,]+) comments', page)
    if total:
        doc.metadata["comments_total"] = (total.group(1) or total.group(2) or "").replace(",", "")
    return doc


class InstagramProvider(BaseProvider):
    name = "instagram"
    capability = Capability(search=False, fetch=True, comments=True, access="public")
    config = ProviderConfig("instagram", requests_per_second=1, max_concurrency=2)

    def __init__(self, keys, respect_robots: bool = True):
        super().__init__(keys)
        self.respect_robots = respect_robots

    async def search(self, query, spec, limit):
        return []

    async def fetch(self, url: str, hit: Optional[dict] = None) -> RawDocument:
        code = post_code(url)
        if not code:
            return RawDocument(url=url, source="instagram", status="skipped",
                               error="not an Instagram post / reel link (…/p/<code>/ or …/reel/<code>/)")
        canonical = f"https://www.instagram.com/p/{code}/"
        if self.respect_robots:
            return RawDocument(url=canonical, source="instagram", status="blocked",
                               error="disallowed by robots.txt (Instagram does not allow crawlers; untick "
                                     "\"Respect robots.txt\" to read the public embed page)")
        embed = f"https://www.instagram.com/p/{code}/embed/captioned/"
        try:
            async with self.limiter:
                client = primp.AsyncClient(impersonate="chrome", follow_redirects=True, timeout=15,
                                           headers={"Accept-Language": "en-IN,en;q=0.9"})
                r = await client.get(embed)
            page = r.text if r.status_code < 400 else ""
        except Exception as exc:
            self.note("failed")
            return RawDocument(url=canonical, source="instagram", status="failed",
                               error=f"{type(exc).__name__}: {str(exc)[:120]}")
        if not page:
            self.note("blocked")
            return RawDocument(url=canonical, source="instagram", status="blocked",
                               error=f"embed page refused (HTTP {r.status_code})")
        doc = parse_embed(canonical, page)
        if not doc.units:
            self.note("blocked")
            doc.status, doc.error = "blocked", "embed page had no caption or comments (private or removed post)"
            return doc
        self.note("ok")
        n = sum(1 for u in doc.units if u.kind == "comment")
        total = doc.metadata.get("comments_total")
        if total and total.isdigit() and int(total) > n:
            doc.error = (f"Instagram shows only {n} of {total} comments without login — use the auto-reply on "
                         "your own posts (Meta API) to get all of them")
        return doc
