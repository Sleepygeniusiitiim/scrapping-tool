"""
Source planner: which kinds of sites are most likely to give the contacts THIS command asks for, and which
places to search.

A B2B command ("owners of truck driving schools in North India") needs business listings — Google Maps
places (name, phone, website, address), directories (JustDial, IndiaMART, Sulekha …), the businesses' own
websites, and LinkedIn for owners / directors. A job-seeker command needs the opposite: comments under
recruitment posts and videos, forums, Reddit, Quora. The AI ranks the sources inside the understanding call
(QuerySpec.source_plan); these rules are the fallback and a sanity check, and they add the template searches
for sources the AI under-planned (Maps / directories need one search per city).
"""

from __future__ import annotations

import re
from typing import Dict, List

from .models import QuerySpec, SourceChoice, SourcedQuery

CATALOG: Dict[str, str] = {
    "maps": "Google Maps business listings — every local business with name, PHONE NUMBER, website, address, "
            "category. Best for local businesses / institutes / shops / clinics / schools in named cities.",
    "directories": "Business directories: JustDial, IndiaMART, Sulekha, TradeIndia, Yellow Pages, OLX / Quikr "
                   "classifieds — business names, addresses, sometimes phones (often hidden behind login).",
    "search": "Open web search — businesses' own websites (contact / about / team pages with phones, emails, "
              "owner and director names), news, any public page.",
    "linkedin": "LinkedIn profiles / posts (search-indexed) — owners, founders, directors, HR of companies; "
                "job seekers' posts. Contacts rarely public; names and titles are.",
    "facebook": "Facebook pages / groups (search-indexed) — small businesses' pages with phone numbers; "
                "comments under recruitment posts with job seekers' numbers.",
    "youtube": "YouTube comments — viewers of recruitment / jobs-abroad / visa videos commenting 'interested' "
               "with phone numbers. People, not businesses.",
    "reddit": "Reddit — individuals discussing plans (moving abroad, careers). Rarely contact details.",
    "quora": "Quora — questions / answers by individuals; snippet only (Quora blocks crawlers).",
    "forums": "Forums / Q&A / job-seeker boards — individuals in their own words, sometimes with numbers.",
    "blogs": "Blog posts with comment sections — readers asking for jobs and sharing numbers.",
}
PEOPLE_DEFAULT = {"facebook": 80, "youtube": 80, "linkedin": 70, "blogs": 65, "forums": 60, "search": 60,
                  "reddit": 45, "quora": 40, "maps": 5, "directories": 10}
ORG_DEFAULT = {"maps": 95, "directories": 85, "search": 80, "linkedin": 60, "facebook": 55, "blogs": 10,
               "forums": 15, "youtube": 5, "reddit": 5, "quora": 10}
REASONS = {
    "maps": "business listings with phone number, website and address for each place",
    "directories": "JustDial / IndiaMART / Sulekha listings name the businesses in each city",
    "search": "the businesses' own websites: contact pages, owner / director names, emails",
    "linkedin": "owners, founders and directors by name and title",
    "facebook": "small businesses' pages often show a phone number",
}

REGIONS: Dict[str, List[str]] = {
    "north india": ["Delhi", "Chandigarh", "Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda", "Mohali",
                    "Gurugram", "Faridabad", "Panipat", "Ambala", "Karnal", "Hisar", "Rohtak", "Sonipat",
                    "Noida", "Ghaziabad", "Meerut", "Lucknow", "Kanpur", "Agra", "Varanasi", "Prayagraj",
                    "Bareilly", "Aligarh", "Moradabad", "Gorakhpur", "Dehradun", "Haridwar", "Haldwani",
                    "Shimla", "Mandi", "Jammu", "Srinagar", "Jaipur", "Jodhpur", "Bikaner", "Sri Ganganagar"],
    "south india": ["Chennai", "Bengaluru", "Hyderabad", "Kochi", "Thiruvananthapuram", "Kozhikode", "Coimbatore",
                    "Madurai", "Tiruchirappalli", "Salem", "Mysuru", "Mangaluru", "Hubballi", "Visakhapatnam",
                    "Vijayawada", "Guntur", "Tirupati", "Warangal", "Thrissur", "Kollam"],
    "west india": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Thane", "Ahmedabad", "Surat", "Vadodara",
                   "Rajkot", "Goa", "Kolhapur", "Solapur", "Bhavnagar"],
    "east india": ["Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri", "Patna", "Gaya", "Muzaffarpur",
                   "Bhubaneswar", "Cuttack", "Rourkela", "Ranchi", "Jamshedpur", "Dhanbad"],
    "central india": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Raipur", "Bilaspur", "Bhilai"],
    "northeast india": ["Guwahati", "Shillong", "Imphal", "Agartala", "Aizawl", "Dimapur", "Itanagar", "Gangtok"],
    "punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda", "Mohali", "Hoshiarpur", "Moga",
               "Firozpur", "Pathankot", "Sangrur", "Kapurthala"],
    "haryana": ["Gurugram", "Faridabad", "Panipat", "Ambala", "Karnal", "Hisar", "Rohtak", "Sonipat", "Yamunanagar",
                "Sirsa", "Kurukshetra", "Rewari"],
    "uttar pradesh": ["Lucknow", "Kanpur", "Noida", "Ghaziabad", "Agra", "Meerut", "Varanasi", "Prayagraj",
                      "Bareilly", "Aligarh", "Moradabad", "Gorakhpur", "Saharanpur", "Jhansi"],
    "rajasthan": ["Jaipur", "Jodhpur", "Kota", "Bikaner", "Ajmer", "Udaipur", "Sri Ganganagar", "Alwar", "Sikar"],
    "himachal pradesh": ["Shimla", "Mandi", "Dharamshala", "Solan", "Kullu", "Hamirpur", "Una"],
    "uttarakhand": ["Dehradun", "Haridwar", "Haldwani", "Roorkee", "Rudrapur", "Kashipur", "Rishikesh"],
    "jammu and kashmir": ["Jammu", "Srinagar", "Kathua", "Udhampur", "Anantnag", "Baramulla"],
    "kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur", "Kollam", "Kannur", "Kottayam", "Malappuram"],
    "tamil nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli", "Erode", "Vellore"],
    "gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Jamnagar", "Gandhinagar", "Anand"],
    "maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Thane", "Kolhapur", "Solapur"],
    "delhi ncr": ["Delhi", "Gurugram", "Noida", "Ghaziabad", "Faridabad", "Greater Noida"],
}
_ALIAS = {"ncr": "delhi ncr", "up": "uttar pradesh", "hp": "himachal pradesh", "j&k": "jammu and kashmir",
          "jk": "jammu and kashmir", "northern india": "north india", "southern india": "south india",
          "western india": "west india", "eastern india": "east india", "north-east india": "northeast india",
          "north east india": "northeast india"}
MAX_PLACES = 40
MAX_MAPS_QUERIES = 50          # one Maps search ≈ 20 businesses with phone numbers
MAX_DIRECTORY_QUERIES = 30


def expand_places(places: List[str], command: str = "") -> List[str]:
    """Regions → their main cities; cities stay as they are."""
    out: List[str] = []
    text = " ".join(places) + " " + command
    low = text.lower()
    for name in sorted(set(list(REGIONS) + list(_ALIAS)), key=len, reverse=True):
        if re.search(r"\b" + re.escape(name) + r"\b", low):
            out += REGIONS[_ALIAS.get(name, name)]
            low = re.sub(r"\b" + re.escape(name) + r"\b", " ", low)
    for p in places:
        pl = p.lower().strip()
        if pl and pl not in REGIONS and pl not in _ALIAS and pl not in ("india", "abroad", "overseas"):
            out.append(p.strip())
    return list(dict.fromkeys(out))[:MAX_PLACES]


def catalog_text() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in CATALOG.items())


def finalize(spec: QuerySpec, command: str, auto: bool, user_sources: List[str], num_queries: int) -> QuerySpec:
    """Merge the AI's source ranking with the rules, pick the sources to run, expand places, and add the
    per-city template searches that local-business sources need."""
    orgs = spec.target == "organizations"
    base = ORG_DEFAULT if orgs else PEOPLE_DEFAULT
    ai = {c.source: c for c in spec.source_plan if c.source in CATALOG}
    plan = []
    for src in CATALOG:
        w = base.get(src, 30)
        if src in ai:                                   # the AI's judgement for this command, kept sane by rules
            w = round(0.6 * ai[src].weight + 0.4 * w)
        reason = (ai[src].reason if src in ai and ai[src].reason else REASONS.get(src) if orgs else "") or \
            CATALOG[src].split(" — ")[0]
        plan.append(SourceChoice(source=src, weight=w, reason=reason))
    plan.sort(key=lambda c: -c.weight)
    spec.source_plan = plan
    chosen = [c.source for c in plan if c.weight >= 50][:6] if auto else list(user_sources or CATALOG)
    places = expand_places(spec.places + spec.origin + spec.destination, command)
    spec.places = places
    profs = [p for p in spec.professions if len(p) <= 60][:3] or [spec.industry or "business"]
    extra: List[SourcedQuery] = []
    cities = places or spec.origin[:3] or ["India"]
    if "maps" in chosen:                           # every city with the main term, big cities with a synonym too
        extra += [SourcedQuery(source="maps", query=f"{profs[0]} in {c}") for c in cities]
        extra += [SourcedQuery(source="maps", query=f"{p} in {c}") for p in profs[1:2] for c in cities[:10]]
    if "directories" in chosen and orgs:
        sites = ["justdial.com", "indiamart.com", "sulekha.com", "olx.in"]
        extra += [SourcedQuery(source="directories", query=f"site:{s} {profs[0]} {c}")
                  for c in cities[:10] for s in sites[:3]]
        extra += [SourcedQuery(source="directories", query=f"site:{sites[3]} {profs[0]} {c}") for c in cities[:4]]
    if "linkedin" in chosen and orgs:
        regions = spec.origin[:2] or cities[:3]
        extra += [SourcedQuery(source="linkedin",
                               query=f'site:linkedin.com/in (owner OR director OR founder OR proprietor) "{p}" {r}')
                  for p in profs[:2] for r in regions]
    have = {q.query.lower() for q in spec.queries}
    queries = [q for q in spec.queries if q.source in chosen or (q.source == "search" and "search" in chosen)]
    queries += [q for q in extra if q.query.lower() not in have]
    # keep the AI's own queries within the requested size; template searches are cheap per-city lookups
    own = [q for q in queries if q.source not in ("maps", "directories")][:max(num_queries, 4) + 4]
    spec.queries = own + [q for q in queries if q.source == "maps"][:MAX_MAPS_QUERIES] + \
        [q for q in queries if q.source == "directories"][:MAX_DIRECTORY_QUERIES]
    return spec
