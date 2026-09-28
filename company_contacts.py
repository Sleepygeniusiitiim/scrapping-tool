"""
Company contact finder — the public-web part of how Hunter / Apollo / Lusha build business contacts.

For an organization lead (training centre, institute, agency, company):
  1. find its own website — the lead's page if it is the company's site, otherwise a search-engine result
     whose domain / title matches the organization name (directories like Justdial are skipped);
  2. crawl a few pages of that site: home + contact / about / team / admissions / careers pages;
  3. collect what the organization publishes: emails (text and mailto:), phones (text and tel:),
     WhatsApp links (wa.me), LinkedIn / Facebook / Instagram pages, and named decision makers
     ("Director: …", schema.org founder / employee / Person);
  4. check that each email's domain accepts mail (MX record via DNS-over-HTTPS). That proves the domain
     can receive mail, not that the mailbox exists — reported as "domain accepts mail".

Not done, on purpose: guessing emails from name patterns, SMTP mailbox probing, logged-in scraping.
Only public pages; the robots.txt setting of the run is respected.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
import primp
from bs4 import BeautifulSoup

import integrations
import rule_extractor
from fetcher import _Robots

DIRECTORIES = ("justdial", "indiamart", "sulekha", "tradeindia", "yellowpages", "facebook.com", "instagram.com",
               "linkedin.com", "google.", "youtube.com", "wikipedia.org", "quora.com", "reddit.com", "twitter.com",
               "x.com", "yelp.", "glassdoor", "naukri", "indeed", "shiksha", "collegedunia", "urbanpro", "magicpin")
_PAGE_HINT = re.compile(r"contact|about|team|reach|enquir|inquir|admission|career|staff|faculty|management|"
                        r"director|founder|leadership|people|who-we-are", re.IGNORECASE)
_ROLE = r"(?:founder|co-?founder|director|managing director|md|ceo|owner|proprietor|principal|chairman|" \
        r"partner|head|manager|hr manager|admissions? (?:head|officer|manager)|coordinator|president|secretary)"
_PERSON_ROLE = re.compile(rf"\b([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+){{1,2}})\s*[,\-–(|:]\s*({_ROLE})\b"
                          rf"|\b({_ROLE})\s*[:\-–]\s*(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{1,2}})",
                          re.IGNORECASE)
_SOCIAL = re.compile(r"https?://(?:[a-z]{2,3}\.)?(?:linkedin\.com/(?:company|in|school)/[^\s\"'<>]+|"
                     r"facebook\.com/[^\s\"'<>?]+|instagram\.com/[^\s\"'<>?]+)", re.IGNORECASE)


def is_directory(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(d in host for d in DIRECTORIES)


def _tokens(name: str) -> List[str]:
    stop = {"the", "and", "of", "pvt", "ltd", "private", "limited", "india", "institute", "school", "centre", "center",
            "academy", "training", "driving", "motor", "college", "services", "llp", "co", "company"}
    return [w for w in re.findall(r"[a-z0-9]+", (name or "").lower()) if len(w) > 2 and w not in stop]


def _matches(name: str, url: str, title: str) -> bool:
    toks = _tokens(name)
    if not toks:
        return False
    host = (urlparse(url).hostname or "").lower().replace("-", "")
    hay = f"{host} {title}".lower()
    # the brand word (usually first) in the domain is a strong match; otherwise most name words must appear
    return (len(toks[0]) >= 3 and toks[0] in host) or sum(t in hay for t in toks) >= max(1, (len(toks) + 1) // 2)


def find_website(keys: Dict[str, str], name: str, city: str = "") -> Optional[str]:
    """The organization's own site from a search API (or DuckDuckGo): first non-directory result matching it."""
    query = f'"{name}" {city} contact'.strip()
    apis = [a for a in integrations.search_available(keys) if a != "scrapedo"]
    hits = []
    if apis:
        hits, _, _ = integrations.web_search(apis[0], keys, query, 10, "in-en")
    if not hits:
        from search_module import search_query
        hits = [{"url": h.url, "title": h.title} for h in search_query(query, max_results=10).hits]
    for h in hits:
        if not is_directory(h["url"]) and _matches(name, h["url"], h.get("title", "")):
            p = urlparse(h["url"])
            return f"{p.scheme}://{p.netloc}/"
    return None


def _extract(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    hrefs = [a.get("href", "") for a in soup.find_all("a")]
    for t in soup(["script", "style", "noscript"]):
        if t.name == "script" and t.get("type") == "application/ld+json":
            continue
        t.decompose()
    text = soup.get_text(" ", strip=True)
    emails = set(rule_extractor.emails_in(text))
    emails |= {h[7:].split("?")[0].strip().lower() for h in hrefs if h.lower().startswith("mailto:") and "@" in h}
    phones = set(rule_extractor.phones_in(text))
    for h in hrefs:
        if h.lower().startswith("tel:"):
            phones.update(rule_extractor.phones_in("phone " + h[4:]))
    whatsapp = {re.sub(r"\D", "", m.group(1)) for h in hrefs for m in [re.search(r"wa\.me/(\+?\d{8,15})", h)] if m}
    whatsapp |= {re.sub(r"\D", "", m.group(1)) for h in hrefs
                 for m in [re.search(r"whatsapp\.com/send\?phone=(\+?\d{8,15})", h)] if m}
    social = {m.group(0).rstrip("/.,") for m in _SOCIAL.finditer(" ".join(hrefs))}
    people = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (TypeError, ValueError):
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack += node
            elif isinstance(node, dict):
                tel, mail = node.get("telephone"), node.get("email")
                for t in (tel if isinstance(tel, list) else [tel] if tel else []):
                    phones.update(rule_extractor.phones_in("phone " + str(t)))
                if isinstance(mail, str) and "@" in mail:
                    emails.add(mail.removeprefix("mailto:").lower())
                for key in ("founder", "employee", "member", "contactPoint"):
                    v = node.get(key)
                    for p in (v if isinstance(v, list) else [v] if v else []):
                        if isinstance(p, dict) and p.get("name"):
                            people.append({"name": p["name"], "role": p.get("jobTitle") or key,
                                           "email": p.get("email"), "phone": p.get("telephone")})
                        if isinstance(p, dict) and p.get("telephone"):
                            phones.update(rule_extractor.phones_in("phone " + str(p["telephone"])))
                        if isinstance(p, dict) and p.get("email"):
                            emails.add(str(p["email"]).removeprefix("mailto:").lower())
                stack += [v for v in node.values() if isinstance(v, (dict, list))]
    for m in _PERSON_ROLE.finditer(text[:30000]):
        name, role = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        if name and len(name.split()) >= 2:
            people.append({"name": name.strip(), "role": role.strip().title()})
    links = [urljoin(url, h) for h in hrefs if h and not h.startswith(("mailto:", "tel:", "#", "javascript:"))]
    return {"emails": emails, "phones": phones, "whatsapp": whatsapp, "social": social, "people": people,
            "links": links}


FREE_MAIL = ("gmail.", "yahoo.", "hotmail.", "outlook.", "rediffmail.", "ymail.", "icloud.", "live.", "aol.", "proton")


def email_domain(emails, site_domain: str) -> Optional[str]:
    """The company's mail domain: the website's own domain if it publishes addresses on it, else the domain
    most of its published (non-free-mail) addresses use."""
    doms = [e.split("@")[1].lower() for e in emails if "@" in e]
    if any(d == site_domain or d.endswith("." + site_domain) for d in doms):
        return site_domain
    own = [d for d in doms if not d.startswith(FREE_MAIL)]
    return max(set(own), key=own.count) if own else None


async def _mx_ok(domain: str) -> Optional[bool]:
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get("https://dns.google/resolve", params={"name": domain, "type": "MX"})
        return bool(r.json().get("Answer")) if r.status_code == 200 else None
    except Exception:
        return None


async def crawl(website: str, respect_robots: bool = True, max_pages: int = 6) -> dict:
    """Public contacts an organization publishes on its own site."""
    host = urlparse(website).netloc
    client = primp.AsyncClient(impersonate="chrome", follow_redirects=True, max_redirects=5, timeout=12)
    robots = _Robots(client) if respect_robots else None
    seen, queue = set(), [website]
    out = {"website": website, "emails": set(), "phones": set(), "whatsapp": set(), "social": set(), "people": [],
           "pages": 0, "status": "ok"}
    while queue and out["pages"] < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        if robots and not await robots.allowed(url):
            out["status"] = "robots"
            continue
        try:
            r = await asyncio.wait_for(client.get(url), timeout=15)
        except Exception:
            continue
        if r.status_code >= 400 or "html" not in r.headers.get("content-type", "html"):
            continue
        out["pages"] += 1
        x = _extract(str(r.url), r.text)
        for k in ("emails", "phones", "whatsapp", "social"):
            out[k] |= x[k]
        out["people"] += x["people"]
        if out["pages"] == 1:                    # queue the contact / about / team pages linked from home
            queue += [l.split("#")[0] for l in x["links"]
                      if urlparse(l).netloc == host and _PAGE_HINT.search(urlparse(l).path)][:max_pages * 2]
    site_domain = host.removeprefix("www.")
    emails = sorted(out["emails"], key=lambda e: (not e.endswith(site_domain), e))[:10]
    checks = await asyncio.gather(*(_mx_ok(e.split("@")[1]) for e in emails))
    people, names = [], set()
    for p in out["people"]:
        if p["name"].lower() not in names:
            names.add(p["name"].lower())
            people.append(p)
    # Learn the company's email format from what it publishes, then guess for named people without one.
    import email_patterns
    mail_domain = email_domain(out["emails"], site_domain)
    pattern = email_patterns.infer(out["emails"], mail_domain, people) if mail_domain else None
    domain_ok = next((ok for e, ok in zip(emails, checks) if e.endswith("@" + (mail_domain or ""))), None)
    for p in people:
        if not p.get("email") and pattern and domain_ok is not False:
            g = email_patterns.guess(p["name"], mail_domain, pattern)
            if g and g["email"] in out["emails"]:
                p["email"] = g["email"]            # published by the company — a real address, not a guess
            elif g:
                p["email_guess"] = g
    return {"website": website, "pages": out["pages"], "status": out["status"] if not out["pages"] else "ok",
            "emails": [{"email": e, "domain_accepts_mail": ok} for e, ok in zip(emails, checks)],
            "phones": sorted(out["phones"])[:10], "whatsapp": sorted(out["whatsapp"])[:5],
            "social": sorted(out["social"])[:8], "people": people[:10], "email_format": pattern,
            "all_emails": sorted(out["emails"])[:40]}


async def company_profile(keys: Dict[str, str], company: str, respect_robots: bool = True) -> Optional[dict]:
    """An employer's own site, mail domain, email format and published people / addresses (one crawl).
    `company` may be a name ("Fortis Hospital Mohali") or a domain / URL ("fortishealthcare.com")."""
    import email_patterns
    company = (company or "").strip()
    if not company:
        return None
    if re.fullmatch(r"(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+/?", company, re.I):
        site = company if company.startswith("http") else "https://" + company.rstrip("/") + "/"
    else:
        site = await asyncio.to_thread(find_website, keys, company, "")
    if not site:
        return None
    res = await crawl(site, respect_robots, max_pages=5)
    site_domain = urlparse(site).netloc.lower().removeprefix("www.")
    domain = email_domain(res.get("all_emails", []), site_domain)
    pattern = res.get("email_format") or (email_patterns.infer(res.get("all_emails", []), domain, res.get("people", []))
                                          if domain else None)
    return {"site": site, "site_domain": site_domain, "domain": domain or site_domain, "domain_from_emails": bool(domain),
            "format": pattern, "people": res.get("people", []), "emails": res.get("all_emails", []),
            "pages": res.get("pages", 0)}


async def guess_for_person(keys: Dict[str, str], full_name: str, company: str, respect_robots: bool = True
                           ) -> Optional[dict]:
    """A person's likely work email: find their employer's site, learn its email format from the addresses it
    publishes, apply it to their name. None when the site or the format evidence is missing."""
    import email_patterns
    if not email_patterns.name_parts(full_name) or not company:
        return None
    prof = await company_profile(keys, company, respect_robots)
    if not prof or not prof["domain_from_emails"]:
        return None
    g = email_patterns.guess(full_name, prof["domain"], prof["format"])
    if g:
        g["company_site"] = prof["site"]
    return g


async def for_organization(keys: Dict[str, str], name: str, city: str, page_url: str,
                           respect_robots: bool = True) -> Optional[dict]:
    """Contacts for one organization lead: its own site (found if needed), crawled."""
    site = None
    if page_url and not is_directory(page_url):
        p = urlparse(page_url)
        site = f"{p.scheme}://{p.netloc}/"
    if not site and name:
        site = await asyncio.to_thread(find_website, keys, name, city)
    if not site:
        return None
    return await crawl(site, respect_robots)
