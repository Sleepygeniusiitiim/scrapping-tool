"""
Google Maps business listings: every local business with its phone number, website, address and category.

Uses the first available of:
    * Google Places API (New) — official; key GOOGLE_PLACES_API_KEY (Google Cloud, "Places API (New)" enabled)
    * Serper.dev  /places     — SERPER_API_KEY (same key as Google search)
    * SerpApi google_maps     — SERPAPI_KEY
Each place becomes one document with one "organization" unit, so it flows through the normal organization
scoring, the website crawl (owners / directors / emails) and the government-list matching.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from ..models import QuerySpec, RawDocument, Unit
from .base import BaseProvider, Capability, ProviderConfig


def available(keys: Dict[str, str]) -> List[str]:
    out = []
    if keys.get("google_places") or os.getenv("GOOGLE_PLACES_API_KEY"):
        out.append("google_places")
    if keys.get("serper"):
        out.append("serper")
    if keys.get("serpapi"):
        out.append("serpapi")
    return out


def _norm(p: dict) -> dict:
    out = {k: (str(v).strip() if v is not None else "") for k, v in p.items()}
    if out.get("website"):
        import company_contacts
        if company_contacts.is_directory(out["website"]):     # a directory profile / WhatsApp link, not a website
            out["listing"], out["website"] = out["website"], ""
    return out


async def places(keys: Dict[str, str], query: str, limit: int = 20) -> List[dict]:
    """[{name, phone, website, address, category, rating, reviews, maps_url}] from the first working API."""
    errors = []
    for api in available(keys):
        try:
            async with httpx.AsyncClient(timeout=25) as c:
                if api == "google_places":
                    key = keys.get("google_places") or os.getenv("GOOGLE_PLACES_API_KEY", "")
                    r = await c.post("https://places.googleapis.com/v1/places:searchText",
                                     json={"textQuery": query, "regionCode": "IN", "pageSize": min(limit, 20)},
                                     headers={"X-Goog-Api-Key": key, "X-Goog-FieldMask":
                                              "places.id,places.displayName,places.formattedAddress,"
                                              "places.nationalPhoneNumber,places.internationalPhoneNumber,"
                                              "places.websiteUri,places.primaryTypeDisplayName,places.googleMapsUri,"
                                              "places.rating,places.userRatingCount"})
                    r.raise_for_status()
                    return [_norm({"name": (p.get("displayName") or {}).get("text"),
                                   "phone": p.get("internationalPhoneNumber") or p.get("nationalPhoneNumber"),
                                   "website": p.get("websiteUri"), "address": p.get("formattedAddress"),
                                   "category": (p.get("primaryTypeDisplayName") or {}).get("text"),
                                   "rating": p.get("rating"), "reviews": p.get("userRatingCount"),
                                   "maps_url": p.get("googleMapsUri")}) for p in r.json().get("places", [])]
                if api == "serper":
                    r = await c.post("https://google.serper.dev/places", headers={"X-API-KEY": keys["serper"]},
                                     json={"q": query, "gl": "in", "hl": "en"})
                    r.raise_for_status()
                    return [_norm({"name": p.get("title"), "phone": p.get("phoneNumber"), "website": p.get("website"),
                                   "address": p.get("address"), "category": p.get("category"),
                                   "rating": p.get("rating"), "reviews": p.get("ratingCount"),
                                   "maps_url": f"https://www.google.com/maps?cid={p['cid']}" if p.get("cid") else ""})
                            for p in r.json().get("places", [])][:limit]
                if api == "serpapi":
                    r = await c.get("https://serpapi.com/search.json",
                                    params={"engine": "google_maps", "q": query, "type": "search", "hl": "en",
                                            "gl": "in", "api_key": keys["serpapi"]})
                    r.raise_for_status()
                    return [_norm({"name": p.get("title"), "phone": p.get("phone"), "website": p.get("website"),
                                   "address": p.get("address"), "category": p.get("type"),
                                   "rating": p.get("rating"), "reviews": p.get("reviews"),
                                   "maps_url": (f"https://www.google.com/maps/place/?q=place_id:{p['place_id']}"
                                                if p.get("place_id") else "")})
                            for p in r.json().get("local_results", [])][:limit]
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "")
            errors.append(f"{api}: {type(exc).__name__} {status}".strip())
    if errors:
        raise RuntimeError("Maps lookup failed — " + "; ".join(errors))
    raise RuntimeError("No Maps key: add a Serper or SerpApi key (🔑 keys), or GOOGLE_PLACES_API_KEY")


def place_doc(p: dict, city: str = "") -> RawDocument:
    url = p.get("maps_url") or ("https://www.google.com/maps/search/?api=1&query=" +
                                quote_plus(f"{p['name']} {p.get('address', '')}"))
    lines = [f"{p['name']} — {p.get('category') or 'business'}."]
    for label, k in (("Address", "address"), ("Phone", "phone"), ("Website", "website")):
        if p.get(k):
            lines.append(f"{label}: {p[k]}.")
    if p.get("rating"):
        lines.append(f"Google rating {p['rating']} ({p.get('reviews') or 0} reviews).")
    doc = RawDocument(url=url, source="maps", title=f"{p['name']} — Google Maps", via="api",
                      metadata={"website": p.get("website") or "", "phone": p.get("phone") or "",
                                "address": p.get("address") or "", "city": city})
    doc.units.append(Unit("organization", p["name"], " ".join(lines), p.get("website") or None, None))
    return doc


class MapsProvider(BaseProvider):
    name = "maps"
    capability = Capability(search=True, fetch=False, comments=False, access="api")
    config = ProviderConfig("maps", requests_per_second=3, max_concurrency=3)

    async def search(self, query: str, spec: QuerySpec, limit: int) -> List[dict]:
        city = (re.search(r"\bin\s+(.+)$", query) or [None, ""])[1]
        async with self.limiter:
            try:
                found = await places(self.keys, query, limit)
            except RuntimeError:
                self.note("failed")
                raise
        self.note("ok")
        hits = []
        for p in found:
            if not p.get("name"):
                continue
            doc = place_doc(p, city)
            hits.append({"url": doc.url, "title": doc.title, "snippet": doc.units[0].text[:300], "date": None,
                         "doc": doc})
        return hits

    async def fetch(self, url: str, hit: Optional[dict] = None) -> RawDocument:
        doc = (hit or {}).get("doc")
        return doc if isinstance(doc, RawDocument) else RawDocument(url=url, source="maps", status="skipped",
                                                                    error="Maps listings come with the search")


async def lookup(keys: Dict[str, str], name: str, city: str = "") -> Optional[dict]:
    """The Maps listing of one named business (to get its phone / website), only when the name clearly matches."""
    if not available(keys) or not name:
        return None
    import gov_registry
    try:
        found = await places(keys, f"{name} {city}".strip(), 5)
    except RuntimeError:
        return None
    best = max(found, key=lambda p: gov_registry.name_similarity(name, p.get("name", "")), default=None)
    if best and gov_registry.name_similarity(name, best.get("name", "")) >= 0.8:
        return best
    return None
