"""
Rule-based candidate extraction — no AI tokens.

Reads the text that fetcher.html_to_text() produces and finds the people on the
page with regular expressions and keyword lists:

* Pages with a post/comment thread (schema.org data): one person per author line.
* Profile / CV pages (linkedin.com/in, /profile, /cv, /resume …): the page is one person.
* Any other page: every email / phone number, with the lines around it, is a
  possible person — kept only if the text reads like the person talking about
  themselves, not a recruiter's "send your CV to …".

Role words come from the search plan's queries and the intent, so the only AI
call in a run is the plan. The dicts returned here go through the same
grounding and contact-ownership checks as the AI extraction in pipeline.py.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional
from urllib.parse import urlparse

from fetcher import EMAIL_RE, PHONE_RE

_THREAD_LINE = re.compile(r"^(POST|COMMENT) by (.+?): (.*)$", re.MULTILINE)

_SELF = re.compile(
    r"\b(?:i|i'?m|iam|my|me|mine|myself|interested|intrested|intersted|my cv|resume|cv attached|"
    r"years? (?:of )?(?:exp|experience)|\d+\s*(?:yrs?|years?)\b|exp\b|experienced|looking for (?:a )?"
    r"(?:job|work|opportunit\w*)|open to work|ready to (?:join|relocate)|available|please consider|"
    r"kindly consider|contact me|call me|whatsapp me|dm me)\b", re.IGNORECASE)
_HIRING = re.compile(
    r"\b(?:we are hiring|we're hiring|hiring|urgent(?:ly)? (?:requirement|required|need)|requirement for|"
    r"vacanc(?:y|ies)|walk[- ]?in|interview (?:on|at|in|date)|send (?:your )?(?:cv|resume)|share (?:your )?"
    r"(?:cv|resume)|apply now|apply (?:at|to|via)|free recruitment|salary|accommodation|recruit(?:ment|ing|er)|"
    r"hr (?:team|manager|executive)|job (?:code|id|opening)|openings?|immediate joiners?)\b", re.IGNORECASE)
_INTEREST = re.compile(r"\b(?:interested|intrested|intersted|i am interested|please consider|my cv|resume|"
                       r"looking for (?:a )?(?:job|work|opportunit\w*)|open to work|ready to (?:join|relocate))\b",
                       re.IGNORECASE)

_PROFILE_PATH = re.compile(r"/(?:in|pub|profile|profiles|cv|cvs|resume|resumes|candidate|candidates|jobseeker|"
                           r"people|user|users|member|members)/", re.IGNORECASE)
_BAD_EMAIL = re.compile(r"(?:noreply|no-reply|donotreply|example\.|sentry|wixpress|\.(?:png|jpe?g|gif|webp|svg)$|"
                        r"@(?:domain|email|yourmail|mail)\.com$)", re.IGNORECASE)
_PHONE_CC = ("91", "971", "966", "974", "965", "968", "973", "977", "880", "92", "94", "49", "44", "48", "40",
             "420", "36", "31", "32", "33", "34", "39", "351", "353", "358", "46", "47", "45", "1", "61", "64", "65",
             "60", "7", "20", "27", "234", "254", "63", "62", "81", "82", "86")

_STOP = set("""
a an and or the of for to in on at by with from as is are was were be been being this that these those who whom
whose which what where when why how find finding search get looking look candidates candidate people person persons
profile profiles post posts comment comments group groups page pages job jobs hiring hire hired recruit recruitment
recruiting recruiter recruiters vacancy vacancies urgent urgently requirement required need needed opening openings
abroad overseas foreign international gulf europe european middle east relocation relocate relocating visa work
working opportunities opportunity career careers apply application interested interest indian india indians
cv cvs resume resumes gmail yahoo com whatsapp contact number mobile phone email mail open experience experienced
my me i we our you your their them they new latest free best top good salary interview walk site inurl intitle
""".split())
_GENERIC_ROLE = {"operator", "operators", "engineer", "engineers", "technician", "technicians", "worker", "workers",
                 "helper", "helpers", "staff", "supervisor", "supervisors", "heavy", "commercial", "vehicle",
                 "senior", "junior", "skilled", "trade", "trades"}

TARGET_COUNTRIES = {
    "UAE": r"uae|dubai|abu dhabi|sharjah|ajman|united arab emirates", "Saudi Arabia": r"saudi(?: arabia)?|ksa|riyadh|jeddah|dammam|jubail",
    "Qatar": r"qatar|doha", "Kuwait": r"kuwait", "Oman": r"oman|muscat", "Bahrain": r"bahrain|manama",
    "Germany": r"germany|deutschland", "Poland": r"poland", "Romania": r"romania", "Czech Republic": r"czech",
    "Hungary": r"hungary", "Netherlands": r"netherlands|holland", "Portugal": r"portugal", "Croatia": r"croatia",
    "Serbia": r"serbia", "Malta": r"malta", "Ireland": r"ireland", "United Kingdom": r"\buk\b|united kingdom|england",
    "Canada": r"canada", "Australia": r"australia", "New Zealand": r"new zealand", "Japan": r"japan",
    "Singapore": r"singapore", "Malaysia": r"malaysia", "Russia": r"russia", "Israel": r"israel", "Europe": r"europe",
    "Gulf": r"gulf|middle east",
}
_TARGET_RE = {k: re.compile(rf"\b(?:{v})\b", re.IGNORECASE) for k, v in TARGET_COUNTRIES.items()}
INDIAN_PLACES = [
    "Mumbai", "Delhi", "New Delhi", "Bangalore", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad",
    "Surat", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna", "Vadodara", "Ludhiana", "Agra",
    "Nashik", "Rajkot", "Coimbatore", "Madurai", "Kochi", "Cochin", "Thiruvananthapuram", "Trivandrum", "Kozhikode",
    "Calicut", "Visakhapatnam", "Vijayawada", "Guwahati", "Bhubaneswar", "Ranchi", "Jamshedpur", "Dehradun",
    "Chandigarh", "Amritsar", "Jalandhar", "Noida", "Gurgaon", "Gurugram", "Faridabad", "Ghaziabad", "Aurangabad",
    "Kolhapur", "Hosur", "Tiruppur", "Salem", "Trichy", "Mangalore", "Mysore", "Hubli", "Belgaum", "Goa",
    "Kerala", "Tamil Nadu", "Karnataka", "Maharashtra", "Gujarat", "Punjab", "Haryana", "Rajasthan", "Uttar Pradesh",
    "Bihar", "West Bengal", "Odisha", "Orissa", "Andhra Pradesh", "Telangana", "Madhya Pradesh", "Jharkhand",
    "Assam", "Uttarakhand", "Himachal", "Chhattisgarh",
]
_INDIA_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in INDIAN_PLACES) + r")\b", re.IGNORECASE)
_FROM_INDIA = re.compile(r"\b(?:from|in|based in|located in|living in|experience in) india\b", re.IGNORECASE)
_NAME_SAID = re.compile(r"\b(?:my name is|i am|i'm|this is|name\s*[:\-])\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")
_POSTED_BY = re.compile(r"\b(?:posted by|asked by|answered by|reply from|comment by)\s+@?([\w.-]{3,40})", re.IGNORECASE)
# Common trade skills, machines, controllers and software (added to the plan's own keywords).
_SKILLS = re.compile(r"(?<!\w)(?:fanuc|siemens|sinumerik|heidenhain|haas|mazak|dmg mori|okuma|mitsubishi|hurco|"
                     r"doosan|makino|brother|amada|trumpf|bystronic|delem|cybelec|mastercam|autocad|solidworks|"
                     r"catia|nx cam|fusion 360|g[- ]?code|cnc turning|cnc milling|vmc|hmc|lathe|milling|grinding|"
                     r"welding|tig|mig|arc welding|fabrication|fitter|electrician|plumber|hvac|diesel engine|"
                     r"hydraulics?|pneumatics?|plc|scada|forklift|crane|excavator|gd&t|iti|diploma|b\.?tech|"
                     r"gulf return(?:ed)?|ecnr|passport)(?!\w)", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


# ---------------------------------------------------------------------------
# Keywords from the plan (the only AI output this extractor uses)
# ---------------------------------------------------------------------------
def keywords_from(intent: str, queries: Iterable[str]) -> List[str]:
    """Role / skill phrases: quoted phrases and runs of non-stop words in the intent and queries."""
    phrases: dict[str, None] = {}
    for text in [intent, *queries]:
        text = text or ""
        for q in re.findall(r'"([^"]{2,60})"', text):
            words = [w for w in re.findall(r"[\w+#./-]+", q.lower()) if w not in _STOP]
            if words and not EMAIL_RE.search(q) and "." not in q:
                phrases[" ".join(re.findall(r"[\w+#/-]+", q.lower()))] = None
        text = re.sub(r'"[^"]*"|\b(?:site|inurl|intitle):\S+|(?:^|\s)-\S+|\bOR\b', " ", text)
        run: List[str] = []
        for w in re.findall(r"[A-Za-z][\w+#/-]*", text) + [""]:
            lw = w.lower()
            if lw and lw not in _STOP and not _TARGET_RE_any(lw) and not _INDIA_RE.fullmatch(lw):
                run.append(lw)
                continue
            if run:
                phrases[" ".join(run)] = None
                run = []
    return [p for p in phrases if len(p) >= 2][:80]


def _TARGET_RE_any(word: str) -> bool:
    return any(rx.fullmatch(word) for rx in _TARGET_RE.values())


def _kw_regex(keywords: List[str]) -> tuple[Optional[re.Pattern], Optional[re.Pattern]]:
    """(phrase regex, distinctive single-word regex)."""
    phrases = sorted({k for k in keywords}, key=len, reverse=True)
    words = sorted({w for k in keywords for w in k.split() if len(w) >= 3 and w not in _GENERIC_ROLE},
                   key=len, reverse=True)
    mk = lambda xs: re.compile(r"(?<!\w)(?:" + "|".join(re.escape(x) for x in xs) + r")(?!\w)", re.IGNORECASE) \
        if xs else None
    return mk(phrases), mk(words)


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------
_AT = re.compile(r"\s*(?:\[at\]|\(at\)|\{at\}|\s+at\s+)\s*", re.IGNORECASE)
_DOT = re.compile(r"\s*(?:\[dot\]|\(dot\)|\{dot\}|\s+dot\s+)\s*", re.IGNORECASE)
_SPELLED_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+\s*(?:\[at\]|\(at\)|\{at\}|\s+at\s+)\s*(?:gmail|yahoo|hotmail|outlook|"
                            r"rediffmail|ymail|live|icloud)\s*(?:\[dot\]|\(dot\)|\{dot\}|\s+dot\s+|\.)\s*(?:com|co\.in|in)\b",
                            re.IGNORECASE)


def emails_in(text: str) -> List[str]:
    found = [m.group() for m in EMAIL_RE.finditer(text)]
    found += [_DOT.sub(".", _AT.sub("@", m.group())).replace(" ", "") for m in _SPELLED_EMAIL.finditer(text)]
    out: dict[str, None] = {}
    for e in found:
        e = e.strip(".").lower()
        if not _BAD_EMAIL.search(e):
            out[e] = None
    return list(out)


def _plausible_phone(raw: str) -> Optional[str]:
    digits = re.sub(r"\D", "", raw)
    intl = raw.strip().startswith(("+", "00"))
    if raw.strip().startswith("00"):
        digits = digits[2:]
    if not 9 <= len(digits) <= 15 or len(set(digits)) < 4:
        return None
    if intl and digits.startswith(_PHONE_CC):
        return "+" + digits
    if len(digits) == 10 and digits[0] in "6789":                 # Indian mobile
        return "+91" + digits
    if len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        return "+91" + digits[1:]
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return "+" + digits
    if len(digits) in (12, 13) and digits.startswith(("971", "966", "974", "965", "968", "973")):
        return "+" + digits
    return None


def phones_in(text: str) -> List[str]:
    out: dict[str, None] = {}
    for m in PHONE_RE.finditer(text):
        # Skip dates, prices, job codes and IDs.
        before = text[max(0, m.start() - 14):m.start()].lower()
        if re.search(r"(?:id|ref|code|no\.?|#|rs\.?|inr|aed|sar|usd|\$)\s*[:.-]?\s*$", before):
            continue
        p = _plausible_phone(m.group())
        if p:
            out[p] = None
    return list(out)


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
def _evidence(text: str, *patterns: Optional[re.Pattern]) -> str:
    sentences = [s.strip() for s in _SENTENCE.findall(text) if len(s.strip()) > 3]
    for rx in (*patterns, _SELF):
        if rx is None:
            continue
        for s in sentences:
            if rx.search(s):
                return s[:300]
    return text.strip()[:300]


def _location(text: str) -> Optional[str]:
    m = _INDIA_RE.search(text)
    if m:
        place = next(p for p in INDIAN_PLACES if p.lower() == m.group().lower())
        return f"{place}, India"
    return "India" if _FROM_INDIA.search(text) else None


def _targets(text: str) -> List[str]:
    return [k for k, rx in _TARGET_RE.items() if rx.search(text)]


def _matches(rx: Optional[re.Pattern], text: str) -> List[str]:
    if not rx:
        return []
    out: dict[str, None] = {}
    for m in rx.finditer(text):
        out[m.group().strip()] = None
    return list(out)


def _name_from_title(title: str, url: str) -> Optional[str]:
    """'Ravi Kumar - CNC Operator - ABC | LinkedIn' → 'Ravi Kumar'."""
    head = re.split(r"\s[-–|]\s|\s\|", title or "")[0].strip()
    if 2 <= len(head.split()) <= 4 and all(w[:1].isupper() for w in head.split()) and not _HIRING.search(head):
        return head
    return None


def _name_said(text: str) -> Optional[str]:
    m = _NAME_SAID.search(text)
    if m and m.group(1).split()[0].lower() not in _STOP:
        return m.group(1)
    m = _POSTED_BY.search(text)
    return m.group(1) if m else None


def _person(name, text, role_rx, word_rx, fallback_targets=None, context_for_role="") -> Optional[dict]:
    emails, phones = emails_in(text), phones_in(text)
    roles = _matches(role_rx, text) or _matches(word_rx, text)
    return {
        "name": name,
        "current_role": roles[0] if roles else None,
        "skills": list(dict.fromkeys(roles + _matches(_SKILLS, text)))[:12],
        "current_location": _location(text),
        "target_countries": _targets(text) or list(fallback_targets or []),
        "evidence_snippet": _evidence(text, _INTEREST, role_rx, word_rx),
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "_relevant": bool(roles) or bool(context_for_role),
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def extract_people(content: str, url: str, keywords: List[str], snippet_only: bool = False) -> List[dict]:
    """People on the page as ExtractedCandidate-shaped dicts (without source_url / platform)."""
    role_rx, word_rx = _kw_regex(keywords)
    title = (re.match(r"#\s*(.+)", content) or re.search(r"^Title:\s*(.+)$", content, re.MULTILINE))
    title = title.group(1).strip() if title else ""
    page_text = content.split("## Page text", 1)[-1]
    page_relevant = bool(_matches(role_rx, content) or _matches(word_rx, content))
    people: List[dict] = []

    # 1. Author-attributed thread: each commenter (and a non-recruiter post author) is a person.
    thread = _THREAD_LINE.findall(content)
    if thread:
        post_targets = _targets(" ".join(b for k, _, b in thread if k == "POST"))
        for kind, author, body in thread:
            author = author.strip()
            if author.lower() == "unknown":
                author = _name_said(body)
            if _HIRING.search(body) and not _INTEREST.search(body):
                continue                              # the recruiter / job ad itself
            if not (_SELF.search(body) or emails_in(body) or phones_in(body)):
                continue
            p = _person(author, body, role_rx, word_rx, post_targets if kind == "COMMENT" else None,
                        context_for_role="page" if page_relevant and kind == "COMMENT" else "")
            if p["_relevant"] and (p["name"] or p["email"] or p["phone"]):
                people.append(p)
        if people:
            return _clean(people)

    # 2. A profile / CV page is one person.
    if not snippet_only and _PROFILE_PATH.search(urlparse(url).path or ""):
        name = _name_from_title(title, url) or _name_said(page_text)
        head = page_text[:6000]
        if name and not _HIRING.search(title):
            p = _person(name, head, role_rx, word_rx)
            if p["_relevant"]:
                return _clean([p])

    # 3. Anything else: each contact with its surrounding lines, if it reads like the person themselves.
    lines = [l for l in page_text.splitlines() if l.strip()]
    seen_contacts: set[str] = set()
    for i, line in enumerate(lines):
        if not (emails_in(line) or phones_in(line)):
            continue
        window = " ".join(lines[max(0, i - 2):i + 2])
        own = " ".join(lines[max(0, i - 1):i + 1])
        contacts = set(emails_in(line) + phones_in(line))
        if contacts <= seen_contacts:
            continue
        seen_contacts |= contacts
        if _HIRING.search(own) and not _INTEREST.search(own):
            continue                                  # "send your CV to hr@…" — the recruiter's contact
        if not _SELF.search(window):
            continue
        p = _person(_name_said(window), window, role_rx, word_rx,
                    context_for_role="page" if page_relevant else "")
        p["email"] = next((e for e in emails_in(line)), p["email"])
        p["phone"] = next((x for x in phones_in(line)), p["phone"])
        if p["_relevant"]:
            if not p["name"] and p["email"]:
                p["name"] = p["email"].split("@")[0]  # public handle
            people.append(p)
    return _clean(people)


def _clean(people: List[dict]) -> List[dict]:
    out, seen = [], set()
    for p in people:
        p.pop("_relevant", None)
        key = (p.get("email") or "", p.get("phone") or "", (p.get("name") or "").lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
