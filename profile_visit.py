"""
Open the public profile of a person who showed interest, to learn who they are before any lookup.

A commenter's handle alone ("priya.rn") cannot be looked up in Apollo / Lusha / ContactOut, and guessing
their LinkedIn profile from Google often finds the wrong person. Their own public profile page usually
says who they are: real name, a LinkedIn link in the bio, sometimes a phone / email they published.

Only public pages are read, with the same robots.txt setting and unblockers as the crawler. No logins,
no private data; a profile behind a login wall is reported as such and skipped.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import primp
from bs4 import BeautifulSoup

import rule_extractor
from fetcher import _Robots

_LINKEDIN = re.compile(r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+/?", re.IGNORECASE)
_IG_NAME = re.compile(r"from (.+?) \(@[\w.]+\)")
_TITLE_NAME = re.compile(r"^\s*([^|•\-–(@]{3,60}?)\s*(?:\(@[\w.]+\)|[-–|•])")
_BIO_LINK = re.compile(r"https?://(?:www\.)?(?:linktr\.ee|beacons\.ai|bio\.link|linkin\.bio|taplink\.(?:cc|at)|"
                       r"lnk\.bio|msha\.ke|campsite\.bio|solo\.to|carrd\.co|[a-z0-9-]+\.carrd\.co|linkbio\.co|"
                       r"hoo\.be|stan\.store|allmylinks\.com|about\.me)/[^\s\"'<>)]*", re.IGNORECASE)
_WA_NUM = re.compile(r"(?:wa\.me/|whatsapp\.com/send/?\?phone=)\+?(\d{8,15})", re.IGNORECASE)
_WALL = re.compile(r"log ?in|sign ?in|sign up|join now|create an account", re.IGNORECASE)


def profile_url_for(platform_url: str, handle: Optional[str], author_url: Optional[str]) -> Optional[str]:
    """The person's profile page: the link the page gave, or the platform's profile URL for their handle."""
    if author_url and author_url.startswith("http") and "reddit.com/user/" not in author_url:
        return author_url
    host = (urlparse(platform_url or "").hostname or "").lower()
    h = (handle or "").strip().lstrip("@")
    if not h or not re.fullmatch(r"[A-Za-z0-9_.]{2,30}", h):
        return None
    if host.endswith("instagram.com"):
        return f"https://www.instagram.com/{h}/"
    if host.endswith("tiktok.com"):
        return f"https://www.tiktok.com/@{h}"
    if host.endswith("x.com") or host.endswith("twitter.com"):
        return f"https://x.com/{h}"
    return None


def _real_name(name: str, handle: str = "") -> Optional[str]:
    name = re.sub(r"\s+", " ", (name or "")).strip(" -|•")
    words = re.findall(r"[A-Za-z]{2,}", name)
    if len(words) < 2 or len(name) > 60 or re.search(r"[_\d@]|instagram|linkedin|login|log in", name, re.I):
        return None
    return name


def parse_profile(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    meta = {}
    for m in soup.find_all("meta"):
        k = (m.get("property") or m.get("name") or "").lower()
        if k and m.get("content"):
            meta.setdefault(k, m["content"])
    title = (soup.title.get_text(" ", strip=True) if soup.title else "") or meta.get("og:title", "")
    desc = meta.get("og:description") or meta.get("description") or ""
    name = None
    for obj in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(obj.string or "")
        except (TypeError, ValueError):
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data]) if isinstance(data, dict) else []
        for it in items:
            if isinstance(it, dict) and str(it.get("@type", "")).lower() == "person" and it.get("name"):
                name = name or _real_name(str(it["name"]))
    if not name:
        m = _IG_NAME.search(desc) or _TITLE_NAME.search(meta.get("og:title", "") or title)
        name = _real_name(m.group(1)) if m else None
    body = soup.get_text(" ", strip=True)[:20000]
    links = " ".join(a.get("href", "") for a in soup.find_all("a"))
    text = f"{desc} {body}"
    linkedin = next((m.group(0) for m in _LINKEDIN.finditer(f"{links} {text}")
                     if "linkedin.com/in/" in m.group(0) and m.group(0).rstrip("/") != url.rstrip("/")), None)
    if "linkedin.com/in/" in url:
        linkedin = url
    walled = not name and not linkedin and bool(_WALL.search(title + " " + desc[:200]))
    bio_links = list(dict.fromkeys(m.group(0).rstrip("/.,") for m in _BIO_LINK.finditer(f"{links} {desc}")))[:2]
    phones = rule_extractor.phones_in(desc) + ["+" + m.group(1) for m in _WA_NUM.finditer(f"{links} {desc}")]
    return {"url": url, "name": name, "linkedin": linkedin, "bio": desc[:300],
            "emails": rule_extractor.emails_in(desc), "phones": list(dict.fromkeys(phones)),
            "bio_links": bio_links, "status": "walled" if walled else "ok"}


def parse_link_page(html: str) -> dict:
    """A link-in-bio page (Linktree, bio.link …): the contacts people put there — email, WhatsApp, phone,
    LinkedIn — which Instagram / TikTok bios rarely show directly."""
    from fetcher import reveal_contacts
    soup = BeautifulSoup(html, "html.parser")
    hrefs = " ".join(a.get("href", "") for a in soup.find_all("a"))
    reveal_contacts(soup)
    text = soup.get_text(" ", strip=True)[:20000]
    phones = rule_extractor.phones_in(text) + ["+" + m.group(1) for m in _WA_NUM.finditer(hrefs)]
    linkedin = next((m.group(0) for m in _LINKEDIN.finditer(hrefs) if "linkedin.com/in/" in m.group(0)), None)
    return {"emails": rule_extractor.emails_in(text), "phones": list(dict.fromkeys(phones)), "linkedin": linkedin}


async def visit(urls: List[str], respect_robots: bool = True, timeout_s: int = 12) -> Dict[str, dict]:
    """Fetch public profile pages (a few at a time). Never raises; each result has a status."""
    urls = list(dict.fromkeys(u for u in urls if u))[:15]
    if not urls:
        return {}
    client = primp.AsyncClient(impersonate="chrome", follow_redirects=True, max_redirects=5, timeout=timeout_s,
                               headers={"Accept-Language": "en-IN,en;q=0.9"})
    robots = _Robots(client) if respect_robots else None
    sem = asyncio.Semaphore(3)

    async def one(u: str) -> dict:
        async with sem:
            if robots and not await robots.allowed(u):
                return {"url": u, "status": "robots"}
            try:
                r = await asyncio.wait_for(client.get(u), timeout=timeout_s + 3)
            except Exception as exc:
                return {"url": u, "status": "failed", "error": f"{type(exc).__name__}"}
            if r.status_code in (401, 403, 429, 999):
                return {"url": u, "status": "walled", "error": f"HTTP {r.status_code}"}
            if r.status_code >= 400:
                return {"url": u, "status": "failed", "error": f"HTTP {r.status_code}"}
            return parse_profile(str(r.url), r.text)

    results = await asyncio.gather(*(one(u) for u in urls))

    async def follow(res: dict) -> None:
        """Open the profile's link-in-bio page when the profile itself showed no contact."""
        if res.get("status") != "ok" or res.get("emails") or res.get("phones") or not res.get("bio_links"):
            return
        link = res["bio_links"][0]
        async with sem:
            if robots and not await robots.allowed(link):
                return
            try:
                r = await asyncio.wait_for(client.get(link), timeout=timeout_s + 3)
            except Exception:
                return
        if r.status_code >= 400:
            return
        x = parse_link_page(r.text)
        res["emails"], res["phones"] = x["emails"], x["phones"]
        res["linkedin"] = res.get("linkedin") or x["linkedin"]
        if x["emails"] or x["phones"]:
            res["via_bio_link"] = link

    await asyncio.gather(*(follow(r) for r in results))
    return {r["url"]: r for r in results} | {u: r for u, r in zip(urls, results)}
