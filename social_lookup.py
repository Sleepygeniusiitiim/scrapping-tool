"""
Find what a commenter published in their own profile bio, via search engines.

Instagram / Facebook often show a login page to servers, but search engines index public profile pages,
and the result's title / snippet usually carries the name and the bio — where people often write their
phone, WhatsApp or email. To avoid attaching someone else's number, ONLY the search result for that exact
profile URL is used (instagram.com/<handle>/, x.com/<handle>, facebook.com/<handle>, linkedin.com/in/<slug>);
pages that merely mention the handle are ignored.
"""

from __future__ import annotations

import asyncio
import re
from typing import Dict, Optional
from urllib.parse import urlparse

import integrations
import rule_extractor
from search_module import search_query

_LI_TITLE = re.compile(r"^(?P<name>[^|\-–]+?)\s*[-–]\s*(?P<headline>[^|]+?)(?:\s*[-–]\s*(?P<company>[^|]+?))?\s*\|\s*LinkedIn",
                       re.IGNORECASE)
_IG_TITLE = re.compile(r"^(?P<name>.+?)\s*\(@(?P<handle>[\w.]+)\)")


def _key(url: str) -> tuple:
    p = urlparse(url or "")
    host = (p.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    host = "x.com" if host == "twitter.com" else host
    seg = [s for s in (p.path or "").split("/") if s]
    if host.endswith("linkedin.com") and len(seg) >= 2 and seg[0] == "in":
        return ("linkedin.com", seg[1].lower())
    if host.endswith("youtube.com") and len(seg) >= 2 and seg[0] in ("channel", "c", "user"):
        return ("youtube.com", seg[1].lower())
    return (host.split(".", host.count(".") - 1)[-1] if host.count(".") > 1 else host, seg[0].lower().lstrip("@")) \
        if seg else (host, "")


def search_profile(keys: Dict[str, str], profile_url: str) -> Optional[dict]:
    """The search-engine result for exactly this profile: name, bio text, phones / emails in it, company."""
    host, handle = _key(profile_url)
    if not handle:
        return None
    site = {"linkedin.com": "linkedin.com/in"}.get(host, host)
    query = f'site:{site} "{handle}"'
    hits = integrations.search_first(keys, query, 10, "wt-wt")
    for h in hits:
        if _key(h.get("url", "")) != (host, handle):
            continue                                  # a page that only mentions the handle — not their profile
        title, snip = h.get("title") or "", h.get("snippet") or ""
        text = f"{title} {snip}"
        out = {"url": h["url"], "bio": snip[:300], "phones": rule_extractor.phones_in(text),
               "emails": rule_extractor.emails_in(text), "name": None, "company": None, "headline": None}
        m = _LI_TITLE.match(title) if host == "linkedin.com" else _IG_TITLE.match(title)
        if m:
            name = m.group("name").strip()
            out["name"] = name if len(name.split()) >= 2 and not re.search(r"[_\d@]", name) else None
            if host == "linkedin.com":
                out["headline"] = (m.group("headline") or "").strip() or None
                out["company"] = (m.group("company") or "").strip() or None
        return out
    return None


async def search_profiles(keys: Dict[str, str], urls, limit: int = 8) -> Dict[str, dict]:
    urls = list(dict.fromkeys(u for u in urls if u))[:limit]
    res = await asyncio.gather(*(asyncio.to_thread(search_profile, keys, u) for u in urls), return_exceptions=True)
    return {u: r for u, r in zip(urls, res) if isinstance(r, dict)}
