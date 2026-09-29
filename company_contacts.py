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

# Sites that list many businesses (directories, data brokers, lead databases, city guides, social / map links):
# never a business's own website, and their contacts belong to the site, not to the business.
AGGREGATORS = ("placementindia", "idbf.in", "cybo.com", "worldorgs", "contactout", "rocketreach", "companydetails",
               "threebestrated", "findglocal", "infobel", "bharatbz", "localpunjab", "mybathinda", "karnalguide",
               "apnapanipat", "amritsarfirst", "labourbooking", "freelistingindia", "dnb.com", "zaubacorp", "tofler",
               "indiainfo.net", "buyersellerworld", "justvisitonline", "indosearch", "brandestate", "plenoemprego",
               "yappe.in", "jsdl.in", "asklaila", "grotal", "yelu.in", "infoisinfo", "joonsquare", "exportersindia",
               "whatsapp.com", "wa.me", "threads.net", "threads.com", "waze.com", "maps.app.goo.gl", "goo.gl",
               "linktr.ee", "bizdir", "companieslist", "thecompanycheck", "instafinancials", "zoominfo", "apollo.io",
               "lusha.com", "crunchbase", "glassdoor", "ambitionbox", "naukri", "shine.com", "timesjobs",
               "workindia", "apna.co", "indeed", "monster", "foundit", "quikr", "olx", "clickindia", "yellowpages",
               "tradeindia", "sulekha", "justdial", "indiamart", "grexa.site", "mapquest",
               "hotfrog", "cylex", "brownbook", "manta.com", "yelp", "trustpilot", "gov.in", "nic.in", "wikipedia")
DIRECTORIES = AGGREGATORS + ("justdial", "indiamart", "sulekha", "tradeindia", "yellowpages", "facebook.com", "instagram.com",
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
    """A directory / aggregator / social host. Dotted entries match the domain ("dnb.com", "gov.in"); bare names
    match the site's own label ("justdial" → justdial.com, not naukripoint.com for "naukri")."""
    host = (urlparse(url if "//" in url else "http://" + url).hostname or "").lower().removeprefix("www.")
    if not host:
        return False
    labels = host.split(".")
    site = labels[-3] if len(labels) >= 3 and labels[-2] in ("co", "org", "net", "gov", "ac", "com") else \
        labels[-2] if len(labels) >= 2 else host
    for d in DIRECTORIES:
        if "." in d:
            if host == d or host.endswith("." + d) or (d.endswith(".") and host.startswith(d)) or \
                    (d.endswith(".") and ("." + d) in host):
                return True
        elif site == d or site.startswith(d) and d in ("google", "yelp"):
            return True
    return False


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
    official = name.endswith(" official website")
    name = name.removesuffix(" official website")
    query = (f'"{name}" official website' if official else f'"{name}" {city} contact').strip()
    hits = integrations.search_first(keys, query, 10, "in-en")
    for h in hits:
        if not is_directory(h["url"]) and _matches(name, h["url"], h.get("title", "")):
            p = urlparse(h["url"])
            return f"{p.scheme}://{p.netloc}/"
    return None


_SUFFIX_WORDS = {"group", "services", "service", "travels", "travel", "consultants", "consultancy", "overseas",
                 "international", "enterprises", "solutions", "pvt", "private", "ltd", "limited", "llp", "india",
                 "company", "co", "and", "the", "of", "agency", "associates", "global"}


def guess_domains(name: str) -> List[str]:
    """Likely own domains for a business name — the Clearbit-style guess: "Gill Smart Group" → gillsmartgroup.com,
    gillsmart.com, gillsmartgroup.in, gillsmart.in, gillsmart.co.in …"""
    words = [w for w in re.findall(r"[a-z0-9]+", (name or "").lower()) if w not in ("pvt", "private", "ltd", "limited",
                                                                                   "llp", "the", "m", "s")]
    if not words:
        return []
    core = [w for w in words if w not in _SUFFIX_WORDS] or words
    light = [w for w in words if w not in ("group", "services", "service", "company", "co", "and", "the", "of")]
    stems = list(dict.fromkeys(["".join(words), "".join(light), "".join(core), "-".join(core)] +
                               (["".join(core[:2])] if len(core) > 2 else [])))
    # a single word ("rolex") names someone else's site far too often: only when the name is one word
    stems = [x for x in stems if 4 <= len(x) <= 40 and (len(words) == 1 or x not in words)]
    return [f"{st}{tld}" for st in stems for tld in (".com", ".in", ".co.in")][:12]


async def _probe_site(client, domain: str, name: str) -> Optional[str]:
    toks = _tokens(name) or re.findall(r"[a-z0-9]{3,}", name.lower())
    for url in (f"https://{domain}/", f"http://{domain}/"):
        try:
            r = await asyncio.wait_for(client.get(url), timeout=7)
        except Exception:
            continue
        if r.status_code >= 400:
            continue
        text = BeautifulSoup(r.text[:200000], "html.parser").get_text(" ", strip=True).lower()[:20000]
        if re.search(r"domain (?:is )?for sale|buy this domain|parked|godaddy|sedo|hugedomains", text):
            return None
        # every distinctive word of the name must be on the page (a "gillsmart.com" about something else fails)
        if toks and sum(t in text for t in toks) >= (len(toks) if len(toks) <= 3 else len(toks) - 1):
            final = urlparse(str(r.url))
            return f"{final.scheme}://{final.netloc}/"
    return None


async def discover_website(keys: Dict[str, str], name: str, city: str = "") -> Optional[str]:
    """The business's own site: likely domains checked directly (free), then web search."""
    if not name:
        return None
    client = primp.AsyncClient(impersonate="chrome", follow_redirects=True, max_redirects=4, timeout=8)
    doms = guess_domains(name)
    found = await asyncio.gather(*(_probe_site(client, d, name) for d in doms), return_exceptions=True)
    for site in found:
        if isinstance(site, str):
            return site
    site = await asyncio.to_thread(find_website, keys, name, city)
    if not site:
        site = await asyncio.to_thread(find_website, keys, f"{name} official website", "")
    return site


def _extract(url: str, html: str) -> dict:
    from fetcher import reveal_contacts
    soup = BeautifulSoup(html, "html.parser")
    hrefs = [a.get("href", "") for a in soup.find_all("a")]
    reveal_contacts(soup)                       # Cloudflare-protected emails, "Call now" / "Email us" links
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


_CONTACT_PATHS = ("contact-us", "contact", "contactus", "about-us", "about")


async def _reader_text(keys: Optional[Dict[str, str]], url: str) -> str:
    """Page text through the Jina reader (renders JavaScript sites that show nothing to a plain request)."""
    try:
        status, _, _, body, err = await integrations.unblock_fetch("jina", keys or {}, url)
    except Exception:
        return ""
    return body if not err and status and status < 400 else ""


async def crawl(website: str, respect_robots: bool = True, max_pages: int = 6,
                keys: Optional[Dict[str, str]] = None) -> dict:
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
        if out["pages"] == 1:
            head = BeautifulSoup(r.text[:300000], "html.parser")
            out["title"] = head.title.get_text(" ", strip=True) if head.title else ""
            out["text_sample"] = head.get_text(" ", strip=True)[:4000]
        x = _extract(str(r.url), r.text)
        for k in ("emails", "phones", "whatsapp", "social"):
            out[k] |= x[k]
        out["people"] += x["people"]
        if out["pages"] == 1:                    # queue the contact / about / team pages linked from home
            linked = [l.split("#")[0] for l in x["links"]
                      if urlparse(l).netloc == host and _PAGE_HINT.search(urlparse(l).path)][:max_pages * 2]
            queue += linked
            if not any(re.search(r"contact", urlparse(l).path, re.I) for l in linked):
                # menus built by JavaScript: try the usual contact page addresses directly
                base = f"{urlparse(str(r.url)).scheme}://{urlparse(str(r.url)).netloc}"
                queue += [f"{base}/{p}" for p in _CONTACT_PATHS[:3]] + [f"{base}/{p}/" for p in _CONTACT_PATHS[:1]]
    if not (out["emails"] or out["phones"] or out["whatsapp"]) and out["status"] != "robots":
        # nothing in the HTML (JavaScript-rendered site, or the request was refused): read it rendered
        base = website.rstrip("/")
        for url in (website, base + "/contact-us", base + "/contact"):
            text = await _reader_text(keys, url)
            if text:
                out["pages"] += 1
                out["emails"] |= set(rule_extractor.emails_in(text))
                out["phones"] |= set(rule_extractor.phones_in(text))
                out["whatsapp"] |= {m.group(1) for m in re.finditer(r"wa\.me/\+?(\d{8,15})", text)}
            if out["emails"] and out["phones"]:
                break
        out["via_reader"] = True
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
    return {"website": website, "title": out.get("title", ""), "text_sample": out.get("text_sample", ""),
            "pages": out["pages"], "status": out["status"] if not out["pages"] else "ok",
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
        site = await discover_website(keys, company, "")
    if not site:
        return None
    res = await crawl(site, respect_robots, max_pages=5, keys=keys)
    site_domain = urlparse(site).netloc.lower().removeprefix("www.")
    if not res.get("all_emails") or not res.get("email_format"):
        # theHarvester-style: addresses at this domain that search engines have indexed anywhere on the web
        harvested = await harvest_domain_emails(keys, site_domain)
        if harvested:
            res["all_emails"] = sorted(set(res.get("all_emails", [])) | set(harvested))
            res["email_format"] = None
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


def host_owned(name: str, url: str) -> bool:
    """The site's domain carries the business's own name ("gillinternational.in" for Gill International)."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if not host or is_directory(url):
        return False
    toks = [t for t in _tokens(name) if len(t) >= 3]
    joined = re.sub(r"[^a-z0-9]", "", host.split(".")[0])
    return bool(toks) and (any(t in joined for t in toks) or
                           "".join(t[0] for t in toks) == joined[:len(toks)] and len(toks) >= 3)


def page_names(res: dict, name: str) -> bool:
    """The crawled site names the business in its title / text (for domains like "hrintl.com")."""
    toks = [t for t in _tokens(name) if len(t) >= 3]
    text = (res.get("title", "") + " " + res.get("text_sample", "")).lower()
    return bool(toks) and sum(t in text for t in toks) >= (len(toks) if len(toks) <= 2 else len(toks) - 1)


def own_emails(emails: List[str], site: str) -> List[str]:
    """Addresses that belong to the business: on its own domain, or a personal mailbox (gmail …) it publishes on
    its own site — never an address of a directory / data broker (support@contactout.com, info@companydetails.in)."""
    dom = (urlparse(site).hostname or "").lower().removeprefix("www.")
    out = []
    for e in emails:
        d = e.split("@")[-1].lower()
        if any(a in d for a in AGGREGATORS if "." in a or len(a) > 6):
            continue
        if d == dom or d.endswith("." + dom) or dom.endswith("." + d) or d.startswith(FREE_MAIL) or \
                d.split(".")[0] == dom.split(".")[0]:
            out.append(e)
    return out


async def for_organization(keys: Dict[str, str], name: str, city: str, page_url: str,
                           respect_robots: bool = True) -> Optional[dict]:
    """Contacts for one organization lead: its own site (found if needed), crawled — only when the site is
    really the business's (its name in the domain, or on the site), and only the site's own contacts."""
    site = None
    if page_url and not is_directory(page_url) and host_owned(name, page_url):
        p = urlparse(page_url)
        site = f"{p.scheme}://{p.netloc}/"
    if not site and name:
        site = await discover_website(keys, name, city)
    if not site:
        return None
    res = await crawl(site, respect_robots, keys=keys)
    if not res or not (host_owned(name, site) or page_names(res, name)):
        return None                       # someone else's site: its phones / emails are not this business's
    res["emails"] = [e for e in res["emails"] if e["email"] in own_emails([x["email"] for x in res["emails"]], site)]
    res["people"] = [p for p in res["people"] if not p.get("email") or p["email"] in own_emails([p["email"]], site)]
    return res


_DM_ROLE = re.compile(r"\b(owner|co-?founder|founder|managing director|director|proprietor|partner|principal|"
                      r"chairman|ceo|md|chief executive|general manager|head)\b", re.IGNORECASE)


def _dm_search(keys: Dict[str, str], query: str) -> List[dict]:
    return integrations.search_first(keys, query, 10, "in-en")


async def decision_makers(keys: Dict[str, str], org_name: str, city: str = "") -> List[dict]:
    """Owners / directors / founders of one business from search-indexed LinkedIn profiles: a result counts
    only when its title or snippet names the business and a senior role ("Rajesh Kumar - Owner - Sharma
    Driving School | LinkedIn")."""
    toks = _tokens(org_name)
    if not toks:
        return []
    query = (f'site:linkedin.com/in "{org_name}" '
             "(owner OR founder OR director OR proprietor OR partner OR principal)")
    try:
        hits = await asyncio.to_thread(_dm_search, keys, query)
    except Exception:
        return []
    out, seen = [], set()
    for h in hits:
        url = (h.get("url") or "").split("?")[0]
        if not re.match(r"https?://([a-z]{2,3}\.)?linkedin\.com/in/", url):
            continue
        title, snippet = h.get("title") or "", h.get("snippet") or ""
        text = f"{title} {snippet}".lower()
        if sum(t in text for t in toks) < max(1, (len(toks) + 1) // 2):
            continue                                      # not about this business
        role = _DM_ROLE.search(f"{title} {snippet}")
        if not role:
            continue
        name = re.split(r"\s[-–|]\s", title)[0].strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z.' ]{2,50}", name) or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append({"name": name, "role": role.group(1).title(), "linkedin": url, "source": "linkedin search"})
        if len(out) == 3:
            break
    return out



def _harvest_sync(keys: Dict[str, str], domain: str) -> List[str]:
    query = f'"@{domain}"'
    hits = []
    try:
        hits = integrations.search_first(keys, query, 20, "in-en")
    except Exception:
        return []
    found = set()
    for h in hits:
        for e in rule_extractor.emails_in(f"{h.get('title', '')} {h.get('snippet', '')}"):
            if e.endswith("@" + domain):
                found.add(e)
    return sorted(found)


async def harvest_domain_emails(keys: Dict[str, str], domain: str) -> List[str]:
    """Addresses at a company's domain that appear anywhere search engines index (directories, PDFs, posts,
    other sites) — more evidence for the company's email format and sometimes the person's own address."""
    if not domain or domain.startswith(FREE_MAIL):
        return []
    return await asyncio.to_thread(_harvest_sync, keys, domain)


ROLE_ADDRESSES = ("info", "contact", "enquiry", "admissions", "office", "hello", "support", "sales")


async def probe_role_addresses(keys: Dict[str, str], domain: str) -> Optional[dict]:
    """A business with a website but no published email: test the usual shared inboxes (info@, contact@ …)
    with the zero-send SMTP check. Only a mailbox the server confirms is returned (never on catch-all
    domains, where every address "exists")."""
    import email_verify
    if not domain or domain.startswith(FREE_MAIL):
        return None
    cands = [f"{r}@{domain}" for r in ROLE_ADDRESSES]
    res = await email_verify.verify_many(cands, keys)
    for e in cands:
        v = res.get(e) or {}
        if v.get("status") == "valid":
            return {"email": e, **v}
    if any((res.get(e) or {}).get("status") == "catch_all" for e in cands):
        return {"email": None, "status": "catch_all"}
    return None
