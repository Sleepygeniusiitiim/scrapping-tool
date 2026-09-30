"""
Fetch layer for serverless (Vercel) — HTTP fetch + HTML → text.

Vercel functions can't ship a headless Chromium, so this replaces the
Crawl4AI crawler used by the local Streamlit version.

* Requests go through `primp` (already installed by `ddgs`), which
  impersonates Chrome's TLS fingerprint — many sites refuse plain Python
  HTTP clients with a 403.
* If SCRAPEDO_TOKEN is set, pages that come back blocked (login wall,
  403/429, bot check) are fetched again through Scrape.do's proxy network.
* Many social pages (LinkedIn posts, Quora answers, forum threads) embed the
  post and its comments as schema.org JSON-LD, with each comment's author.
  That block is put first so the extractor can tell who wrote what — e.g.
  which commenter posted which email address.
* Every email / phone number on the page is also listed with the text
  around it, so contact details survive even if the page text is cut short.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib import robotparser
from urllib.parse import urlparse

import primp
from bs4 import BeautifulSoup

import dates

log = logging.getLogger(__name__)

DEFAULT_PAGE_TIMEOUT_S = 15
SCRAPEDO_TIMEOUT_S = 60
MIN_USEFUL_CHARS = 250          # below this the page is treated as empty
MAX_TEXT_CHARS = 60_000         # keep memory and LLM cost bounded
ROBOTS_AGENT = "*"               # impersonated Chrome has no bot token; obey the wildcard rules

SCRAPEDO_ENDPOINT = "https://api.scrape.do/"
# Sites that need Scrape.do's residential proxies ("super") rather than datacenter IPs.
_STRICT_HOSTS = ("linkedin.com", "facebook.com", "instagram.com", "quora.com", "x.com", "twitter.com",
                 "indeed.com", "glassdoor.")

# Phrases that indicate an auth wall / bot challenge rather than real content.
_WALL_PATTERNS = re.compile(
    r"(authwall|sign in to view|join linkedin|log in to continue|login to continue|"
    r"please enable javascript|verify you are human|are you a robot|captcha|"
    r"access denied|cf-browser-verification|checking your browser|"
    r"you've been blocked|request blocked|unusual traffic)",
    re.IGNORECASE,
)
_WALL_URL = re.compile(r"(authwall|/login|/signin|/checkpoint|/uas/login|captcha)", re.IGNORECASE)
_DROP_TAGS = ["script", "style", "nav", "footer", "header", "form", "aside", "noscript", "svg", "iframe"]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# Phone-like runs: optional +/00, then 9–16 digits with spaces, dots, dashes or brackets between.
PHONE_RE = re.compile(r"(?<![\d/+])(?:\+|00)?\d(?:[\s().-]?\d){8,15}(?![\d/])")
_CONTACT_CONTEXT = 160
# LinkedIn / Facebook serve a login wall to bursts of parallel requests, so strict
# hosts are fetched one at a time with a pause; other hosts a few at a time.
_STRICT_GAP_S = (1.2, 2.5)
_PER_HOST_CONCURRENCY = 3


@dataclass
class CrawlOutcome:
    url: str
    markdown: str = ""
    ok: bool = False
    blocked: bool = False
    error: Optional[str] = None
    via: str = "direct"          # direct | scrape.do


def scrapedo_token(keys: Optional[dict] = None) -> str:
    return ((keys or {}).get("scrapedo") or os.getenv("SCRAPEDO_TOKEN", "")).strip()


# ---------------------------------------------------------------------------
# HTML → text
# ---------------------------------------------------------------------------
def _walk(node) -> Iterable[dict]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _author_url(obj: dict) -> str:
    a = obj.get("author")
    if isinstance(a, list) and a:
        a = a[0]
    if isinstance(a, dict):
        u = a.get("url") or a.get("sameAs") or ""
        u = u[0] if isinstance(u, list) and u else u
        return str(u).strip() if str(u).startswith("http") else ""
    return ""


def _author_name(obj: dict) -> str:
    a = obj.get("author")
    if isinstance(a, list) and a:
        a = a[0]
    if isinstance(a, dict):
        return str(a.get("name") or a.get("alternateName") or "").strip()
    return str(a or "").strip()


_ORG_TYPES = re.compile(r"Organization|Business|School|College|Institute|Corporation|Company|Store|Service|"
                        r"Agency|Center|Centre|Academy|DrivingSchool|Place", re.IGNORECASE)


def _org_line(obj: dict) -> str:
    """schema.org Organization / LocalBusiness → 'ORG by Name <url>: description | Phone: … | Email: … | Address: …'."""
    kind = obj.get("@type")
    kind = " ".join(kind) if isinstance(kind, list) else str(kind or "")
    name = obj.get("name")
    if not kind or not _ORG_TYPES.search(kind) or not isinstance(name, str) or not name.strip():
        return ""
    if kind in ("WebSite", "WebPage", "SiteNavigationElement", "BreadcrumbList"):
        return ""
    addr = obj.get("address")
    if isinstance(addr, dict):
        addr = ", ".join(str(addr.get(k)) for k in ("streetAddress", "addressLocality", "addressRegion", "postalCode",
                                                   "addressCountry") if isinstance(addr.get(k), str) and addr.get(k))
    tel, mail = obj.get("telephone"), obj.get("email")
    tel = ", ".join(tel) if isinstance(tel, list) else tel
    parts = [re.sub(r"\s+", " ", str(obj.get("description") or kind)).strip()[:600]]
    if isinstance(tel, str) and tel.strip():
        parts.append(f"Phone: {tel.strip()}")
    if isinstance(mail, str) and mail.strip():
        parts.append(f"Email: {mail.strip().removeprefix('mailto:')}")
    if isinstance(addr, str) and addr.strip():
        parts.append(f"Address: {addr.strip()}")
    url = obj.get("url") if isinstance(obj.get("url"), str) and str(obj.get("url")).startswith("http") else ""
    return f"ORG by {name.strip()}{f' <{url}>' if url else ''}: " + " | ".join(parts)


def structured_thread(soup: BeautifulSoup) -> str:
    """Posts, answers and comments from schema.org JSON-LD, one line per author."""
    lines, seen = [], set()
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (TypeError, ValueError):
            continue
        for obj in _walk(data):
            org = _org_line(obj)
            if org and org not in seen:
                seen.add(org)
                lines.append(org)
                continue
            body = obj.get("articleBody") or obj.get("text")
            if not isinstance(body, str) or not body.strip():
                continue
            kind = str(obj.get("@type") or "Item")
            kind = "COMMENT" if kind in ("Comment", "Answer") else "POST"
            author = _author_name(obj) or "unknown"
            link = _author_url(obj)
            body = re.sub(r"\s+", " ", body).strip()
            key = (author, body[:200])
            if key in seen:
                continue
            seen.add(key)
            when = dates.parse(obj.get("datePublished") or obj.get("dateCreated") or obj.get("uploadDate") or "")
            lines.append(f"{kind} by {author}{f' <{link}>' if link else ''}{f' [{when}]' if when else ''}: {body}")
    return "\n".join(lines)


_DATE_META = ("article:published_time", "og:published_time", "article:modified_time", "og:updated_time",
              "datePublished", "dateCreated", "date", "pubdate", "publish-date", "dc.date")


def page_date(soup: BeautifulSoup) -> Optional[str]:
    """When the page (post / thread) was published, from meta tags or the first <time> element."""
    for meta in soup.find_all("meta"):
        key = (meta.get("property") or meta.get("name") or meta.get("itemprop") or "").strip()
        if key in _DATE_META:
            d = dates.parse(meta.get("content"))
            if d:
                return d
    t = soup.find("time")
    if t is not None:
        return dates.parse(t.get("datetime") or t.get_text(" ", strip=True))
    return None


def contact_mentions(text: str) -> str:
    """Every email / phone-like number with the text around it."""
    out, seen = [], set()
    for kind, rx in (("EMAIL", EMAIL_RE), ("PHONE", PHONE_RE)):
        for m in rx.finditer(text):
            value = m.group().strip()
            if kind == "PHONE":
                digits = re.sub(r"\D", "", value)
                if not 9 <= len(digits) <= 15 or len(set(digits)) < 4:
                    continue
            if value.lower() in seen:
                continue
            seen.add(value.lower())
            start, end = max(0, m.start() - _CONTACT_CONTEXT), min(len(text), m.end() + 60)
            ctx = re.sub(r"\s+", " ", text[start:end]).strip()
            out.append(f"{kind} {value} — context: …{ctx}…")
    return "\n".join(out)


_CMT_BLOCKS = {"comment", "comment-body", "wpd-comment", "wc-comment", "comment-item", "comment-block",
               "comment-container", "comment-wrapper", "review", "single-comment", "commentlist-item"}
_CMT_TEXTS = {"comment-content", "comment-text", "comment_text", "wpd-comment-text", "comment-body-text",
              "comment-message", "message", "review-text", "commentbody", "comment-entry"}
_NOT_AUTHOR = re.compile(r"avatar|says|meta|date|time|link", re.I)


def _tokens(el) -> set:
    return {c.lower() for c in (el.get("class") or [])}


def _is_author(el) -> bool:
    toks = _tokens(el)
    return bool(toks & {"fn", "user", "username", "commenter", "name", "author-name", "comment-author-name"} or
                any("author" in t and not _NOT_AUTHOR.search(t) for t in toks))


def html_comments(soup: BeautifulSoup) -> List[str]:
    """Reader comments on any site's blog post / article (WordPress, wpDiscuz, Blogger, generic themes):
    one 'COMMENT by Name [date]: text' line per comment (replies are separate comments)."""
    lines, seen, used = [], set(), set()
    for text_el in soup.find_all(lambda t: bool(_tokens(t) & _CMT_TEXTS)):
        if id(text_el) in used:
            continue
        used.add(id(text_el))
        # the comment this text belongs to: nearest ancestor that is a comment block
        block = text_el.find_parent(lambda t: bool(_tokens(t) & _CMT_BLOCKS))
        if block is None:
            continue
        author_el = next((e for e in block.find_all(_is_author) if not e.find_parent(
            lambda t: bool(_tokens(t) & _CMT_TEXTS))), None)
        if author_el is None:
            continue
        author = re.sub(r"\s+", " ", author_el.get_text(" ", strip=True))
        author = re.sub(r"\s*(?:says|said|replied|wrote)\b.*$|:\s*$", "", author, flags=re.I).strip()[:60]
        text = re.sub(r"\s+", " ", text_el.get_text(" ", strip=True)).strip()
        text = re.sub(r"\s*(?:Reply|Log in to Reply|Like|Report)\s*$", "", text).strip()
        if not author or len(text) < 2 or len(author.split()) > 6:
            continue
        t = block.find("time")
        when = dates.parse(t.get("datetime") or t.get_text(" ", strip=True)) if t is not None else None
        key = (author.lower(), text[:120])
        if key in seen:
            continue
        seen.add(key)
        link = author_el if author_el.name == "a" else author_el.find("a")
        href = link.get("href", "") if link is not None else ""
        href = href if href.startswith("http") and "#" not in href else ""
        lines.append(f"COMMENT by {author}{f' <{href}>' if href else ''}{f' [{when}]' if when else ''}: {text[:1500]}")
    return lines


def decode_cfemail(hexstr: str) -> str:
    """Cloudflare "email protection": the address XOR-encoded with its first byte."""
    try:
        key = int(hexstr[:2], 16)
        out = "".join(chr(int(hexstr[i:i + 2], 16) ^ key) for i in range(2, len(hexstr) - 1, 2))
    except ValueError:
        return ""
    return out if EMAIL_RE.fullmatch(out) else ""


_WA_LINK = re.compile(r"(?:wa\.me/|whatsapp\.com/send/?\?phone=|api\.whatsapp\.com/send/?\?phone=)\+?(\d{8,15})", re.I)


def reveal_contacts(soup: BeautifulSoup) -> None:
    """Make addresses hidden in markup visible where they are: Cloudflare-protected emails, and mailto: / tel: /
    WhatsApp links whose text is only "Email us" / "Call now" / an icon. Kept in place, so a contact stays next
    to the person or business it belongs to."""
    for el in soup.select("[data-cfemail]"):
        e = decode_cfemail(el.get("data-cfemail", ""))
        if e:
            el.replace_with(e)
    for a in soup.find_all("a", href=True):
        h = a["href"].strip()
        low = h.lower()
        add = ""
        if "/cdn-cgi/l/email-protection#" in low:
            add = decode_cfemail(h.split("#", 1)[1])
        elif low.startswith("mailto:"):
            add = h[7:].split("?")[0].strip()
        elif low.startswith("tel:"):
            add = "Phone " + h[4:].strip()
        elif (m := _WA_LINK.search(h)):
            add = "WhatsApp +" + m.group(1)
        if not add:
            continue
        txt = re.sub(r"\s", "", a.get_text(" ", strip=True)).lower()
        core = re.sub(r"\D", "", add) if not "@" in add else add.lower()
        if core and core not in re.sub(r"\s", "", txt) and core not in re.sub(r"\D", "", txt):
            a.append(f" ({add})")


def site_contacts(soup: BeautifulSoup) -> str:
    """Contacts in the header / footer (removed from the page text): the site owner's own, not commenters'."""
    parts = []
    for el in soup.find_all(["footer", "header"]) + soup.select('[class*="footer"], [id*="footer"], [class*="topbar"]'):
        parts.append(el.get_text(" ", strip=True))
    text = " ".join(parts)[:6000]
    if not text:
        return ""
    emails = list(dict.fromkeys(m.group().lower() for m in EMAIL_RE.finditer(text)))[:4]
    phones = []
    for m in PHONE_RE.finditer(text):
        d = re.sub(r"\D", "", m.group())
        if 9 <= len(d) <= 15 and len(set(d)) >= 4 and m.group().strip() not in phones:
            phones.append(m.group().strip())
    out = [f"Email: {e}" for e in emails] + [f"Phone: {p}" for p in phones[:4]]
    return " | ".join(out)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    reveal_contacts(soup)
    thread = structured_thread(soup)
    cmt_soup = BeautifulSoup(html, "html.parser")
    reveal_contacts(cmt_soup)
    blog_comments = html_comments(cmt_soup)
    if blog_comments:
        thread = "\n".join(x for x in [thread] + blog_comments if x)
    published = page_date(soup)
    own = site_contacts(soup)
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    body = soup.body or soup
    text = body.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    contacts = contact_mentions(thread + "\n" + text)
    parts = [f"# {title}" if title else "", f"Page date: {published}" if published else ""]
    if thread:
        parts.append("## Post and comments (structured, with authors)\n" + thread)
    if contacts:
        parts.append("## Contact details found on the page\n" + contacts)
    if own:
        parts.append("## Website's own contact details (header / footer — the site owner, not commenters)\n" + own)
    parts.append("## Page text\n" + text)
    return "\n\n".join(p for p in parts if p).strip()[:MAX_TEXT_CHARS]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
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


def _classify_text(url: str, status: int, text: str, via: str) -> CrawlOutcome:
    """Plain text / markdown from a reader service (Jina) instead of HTML."""
    if status in (401, 403, 407, 429, 451, 999) or status >= 400:
        return CrawlOutcome(url=url, blocked=status in (401, 403, 407, 429, 451, 999), error=f"HTTP {status}", via=via)
    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if len(text) < MIN_USEFUL_CHARS or (_WALL_PATTERNS.search(text[:3000]) and len(text) < 1200):
        return CrawlOutcome(url=url, markdown=text, blocked=True, error="login wall / bot check / empty page", via=via)
    contacts = contact_mentions(text)
    parts = [f"## Contact details found on the page\n{contacts}" if contacts else "", "## Page text\n" + text]
    return CrawlOutcome(url=url, markdown="\n\n".join(p for p in parts if p)[:MAX_TEXT_CHARS], ok=True, via=via)


def _classify(url: str, final_url: str, status: int, ctype: str, html: str, via: str) -> CrawlOutcome:
    if status in (401, 403, 407, 429, 999) or _WALL_URL.search(urlparse(final_url).path or ""):
        return CrawlOutcome(url=url, blocked=True, error=f"HTTP {status}" if status >= 400 else "login wall", via=via)
    if status >= 400:
        return CrawlOutcome(url=url, error=f"HTTP {status}", via=via)
    if ctype and "html" not in ctype and "text" not in ctype:
        return CrawlOutcome(url=url, error=f"unsupported content-type {ctype[:40]}", via=via)
    try:
        text = html_to_text(html)
    except Exception as exc:
        return CrawlOutcome(url=url, error=f"HTML parsing failed: {exc}", via=via)
    # Logged-out social pages always carry "sign in to view more" boilerplate, so a wall
    # phrase only means a wall when there's little else on the page.
    has_thread = "## Post and comments" in text
    if len(text) < MIN_USEFUL_CHARS or (_WALL_PATTERNS.search(text[:3000]) and len(text) < 1200 and not has_thread):
        return CrawlOutcome(url=url, markdown=text, blocked=True, error="login wall / bot check / empty page", via=via)
    return CrawlOutcome(url=url, markdown=text, ok=True, via=via)


async def _direct(client: primp.AsyncClient, url: str, timeout_s: float) -> CrawlOutcome:
    try:
        r = await asyncio.wait_for(client.get(url, timeout=timeout_s), timeout=timeout_s + 3)
    except (asyncio.TimeoutError, primp.TimeoutError):
        return CrawlOutcome(url=url, error=f"timed out after {timeout_s:.0f}s")
    except Exception as exc:  # DNS failure, TLS error, too many redirects, ...
        return CrawlOutcome(url=url, error=f"{type(exc).__name__}: {str(exc)[:200]}")
    return _classify(url, str(r.url), r.status_code, r.headers.get("content-type", ""), r.text, "direct")


async def _scrapedo(client: primp.AsyncClient, url: str, token: str) -> CrawlOutcome:
    host = (urlparse(url).hostname or "").lower()
    params = {"token": token, "url": url, "timeout": str(SCRAPEDO_TIMEOUT_S * 1000)}
    if any(h in host for h in _STRICT_HOSTS):
        params["super"] = "true"
    try:
        r = await asyncio.wait_for(client.get(SCRAPEDO_ENDPOINT, params=params, timeout=SCRAPEDO_TIMEOUT_S + 5),
                                   timeout=SCRAPEDO_TIMEOUT_S + 8)
    except (asyncio.TimeoutError, primp.TimeoutError):
        return CrawlOutcome(url=url, error="Scrape.do timed out", via="scrape.do")
    except Exception as exc:
        return CrawlOutcome(url=url, error=f"Scrape.do {type(exc).__name__}: {str(exc)[:160]}", via="scrape.do")
    if r.status_code == 401:
        return CrawlOutcome(url=url, error="Scrape.do rejected SCRAPEDO_TOKEN", via="scrape.do")
    final = r.headers.get("scrape.do-resolved-url") or url
    return _classify(url, final, r.status_code, r.headers.get("content-type", ""), r.text, "scrape.do")


def _is_strict(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(h in host for h in _STRICT_HOSTS)


async def _paced_direct(client: primp.AsyncClient, sems: Dict[str, asyncio.Semaphore],
                        url: str, timeout_s: float) -> CrawlOutcome:
    host = (urlparse(url).hostname or "").lower()
    strict = _is_strict(url)
    sem = sems.setdefault(host, asyncio.Semaphore(1 if strict else _PER_HOST_CONCURRENCY))
    async with sem:
        outcome = await _direct(client, url, timeout_s)
        if strict and outcome.blocked:
            # One slower retry — a wall is often just burst rate-limiting.
            await asyncio.sleep(random.uniform(3.0, 5.0))
            outcome = await _direct(client, url, timeout_s)
        if strict:
            await asyncio.sleep(random.uniform(*_STRICT_GAP_S))
    return outcome


MAX_UNBLOCK_TRIES = 2     # unblocker services tried per page (keeps a batch inside the function time limit)

# LinkedIn posts have a public embed page (the one websites use to embed a post): it shows the post's author and
# text without a login and is served far more reliably than the post page itself.
_LI_POST_ID = re.compile(r"(?:[-_:](activity|ugcPost|share)[-:](\d{15,22}))", re.IGNORECASE)


def linkedin_embed_url(url: str) -> Optional[str]:
    """https://www.linkedin.com/embed/feed/update/urn:li:<kind>:<id> for a LinkedIn post URL, else None."""
    low = url.lower()
    if "linkedin.com" not in low or not any(p in low for p in ("/posts/", "/feed/update/", "/pulse/")):
        return None
    m = _LI_POST_ID.search(url)
    if not m:
        return None
    kind = {"activity": "activity", "ugcpost": "ugcPost", "share": "share"}[m.group(1).lower()]
    return f"https://www.linkedin.com/embed/feed/update/urn:li:{kind}:{m.group(2)}"


async def _linkedin_embed(client: primp.AsyncClient, url: str, timeout_s: float) -> Optional[CrawlOutcome]:
    embed = linkedin_embed_url(url)
    if not embed:
        return None
    for attempt in range(2):
        try:
            r = await asyncio.wait_for(client.get(embed, timeout=timeout_s), timeout=timeout_s + 3)
        except Exception as exc:
            return CrawlOutcome(url=url, error=f"LinkedIn embed: {type(exc).__name__}", via="linkedin-embed")
        if r.status_code in (429, 999) and attempt == 0:
            await asyncio.sleep(random.uniform(4.0, 7.0))       # rate-limited: one slower retry
            continue
        break
    if r.status_code >= 400 or _WALL_URL.search(urlparse(str(r.url)).path or ""):
        return CrawlOutcome(url=url, blocked=r.status_code in (403, 429, 999), error=f"LinkedIn embed HTTP {r.status_code}",
                            via="linkedin-embed")
    try:
        text = html_to_text(r.text)
    except Exception as exc:
        return CrawlOutcome(url=url, error=f"LinkedIn embed parse: {exc}", via="linkedin-embed")
    if len(text) < 80:                 # an embed is short by nature; below this there is no post text
        return CrawlOutcome(url=url, markdown=text, blocked=True, error="LinkedIn embed: empty", via="linkedin-embed")
    return CrawlOutcome(url=url, markdown=text, ok=True, via="linkedin-embed")


async def _unblock(name: str, keys: dict, proxy_client: Optional[primp.AsyncClient], url: str) -> CrawlOutcome:
    if name == "scrapedo":
        return await _scrapedo(proxy_client or primp.AsyncClient(timeout=SCRAPEDO_TIMEOUT_S + 5), url,
                               scrapedo_token(keys))
    from integrations import SERVICES, unblock_fetch
    via = SERVICES[name][1].split(" (")[0]
    status, final, ctype, body, err = await unblock_fetch(name, keys, url)
    if err:
        return CrawlOutcome(url=url, error=err, via=via)
    if name == "jina":
        return _classify_text(url, status, body, via)
    return _classify(url, final, status, ctype, body, via)


_browser_sem: Optional[asyncio.Semaphore] = None


async def _via_browser(url: str) -> Optional[CrawlOutcome]:
    """Headless Chromium (BROWSER_FETCH=1 on a worker); None when not enabled."""
    global _browser_sem
    import browser_fetch
    if not browser_fetch.enabled():
        return None
    if _browser_sem is None:
        _browser_sem = asyncio.Semaphore(int(os.getenv("BROWSER_CONCURRENCY", "3") or 3))
    async with _browser_sem:
        try:
            status, final, html = await browser_fetch.fetch_html(url)
        except Exception as exc:
            return CrawlOutcome(url=url, error=f"browser: {type(exc).__name__}: {str(exc)[:100]}", via="browser")
    return _classify(url, final, status, "text/html", html, "browser")


async def _fetch_one(client: primp.AsyncClient, proxy_client: Optional[primp.AsyncClient],
                     robots: Optional[_Robots], sems: Dict[str, asyncio.Semaphore],
                     url: str, timeout_s: float, unblockers: List[str], keys: dict) -> CrawlOutcome:
    if robots and not await robots.allowed(url):
        return CrawlOutcome(url=url, blocked=True, error="disallowed by robots.txt")
    outcome = await _paced_direct(client, sems, url, timeout_s)
    if outcome.ok or (outcome.error or "").startswith(("HTTP 404", "HTTP 410")):
        return outcome
    # Blocked, walled or failed directly → LinkedIn's public post embed, a real headless browser (workers),
    # then the unblocker services.
    errors = [outcome.error or "failed"]
    embed = await _linkedin_embed(client, url, timeout_s)
    if embed is not None:
        if embed.ok:
            return embed
        errors.append(embed.error or "LinkedIn embed failed")
    browser = await _via_browser(url)
    if browser is not None:
        if browser.ok:
            return browser
        errors.append(browser.error or "browser failed")
    for name in unblockers[:MAX_UNBLOCK_TRIES]:
        retry = await _unblock(name, keys, proxy_client, url)
        if retry.ok:
            return retry
        errors.append(retry.error or "failed")
        outcome = retry
    if len(errors) > 1:
        outcome.error = "; ".join(errors)
    return outcome


async def fetch_batch(urls: List[str], page_timeout_s: int = DEFAULT_PAGE_TIMEOUT_S,
                      respect_robots: bool = True, keys: Optional[dict] = None) -> List[CrawlOutcome]:
    """Fetch a batch of URLs concurrently; never raises."""
    if not urls:
        return []
    from integrations import summary
    keys = dict(keys or {})
    if scrapedo_token(keys):
        keys["scrapedo"] = scrapedo_token(keys)
    unblockers = summary(keys)["unblock"]
    client = primp.AsyncClient(impersonate="chrome", follow_redirects=True, max_redirects=5,
                               timeout=page_timeout_s, headers={"Accept-Language": "en-IN,en;q=0.9"})
    proxy_client = primp.AsyncClient(timeout=SCRAPEDO_TIMEOUT_S + 5) if "scrapedo" in unblockers else None
    robots = _Robots(client) if respect_robots else None
    sems: Dict[str, asyncio.Semaphore] = {}
    return list(await asyncio.gather(
        *(_fetch_one(client, proxy_client, robots, sems, u, page_timeout_s, unblockers, keys) for u in urls)))
