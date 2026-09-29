"""
Does each business found actually fit the command? ("MEA-registered recruitment agents of North India" must not
return a construction company in Dubai, or a Mumbai travel agent.)

1. rules (free):   location — the target region's cities / states against the lead's address, text and phone
                   area code (022 = Mumbai, 0161 = Ludhiana …); a business plainly based abroad or in another
                   Indian region is rejected. Licence patterns (eMigrate RA "B-1234/DEL/PER/…") confirm an MEA
                   registration.
2. AI (reasoning): the remaining leads, 20 per call, judged against the command and its hard requirements —
                   type of business, location, each requirement met / not met / not shown by the evidence.
3. decision:       not a fit → removed (listed in the log with the reason). Requirement not shown → kept but
                   marked "unverified", scored lower, and removed when "strict requirements" is on.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

import rule_extractor

from . import planner
from .models import QuerySpec

REGION_STATES = {
    "north india": ["Delhi", "Punjab", "Haryana", "Chandigarh", "Uttar Pradesh", "Uttarakhand", "Himachal Pradesh",
                    "Himachal", "Jammu", "Kashmir", "Ladakh", "Rajasthan", "NCR", "UP"],
    "south india": ["Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh", "Telangana", "Puducherry"],
    "west india": ["Maharashtra", "Gujarat", "Goa"],
    "east india": ["West Bengal", "Bihar", "Odisha", "Orissa", "Jharkhand"],
    "central india": ["Madhya Pradesh", "Chhattisgarh"],
    "northeast india": ["Assam", "Meghalaya", "Manipur", "Tripura", "Mizoram", "Nagaland", "Arunachal", "Sikkim"],
}
# landline area codes of the larger cities (a business's phone says where it is)
STD = {"11": "Delhi", "22": "Mumbai", "33": "Kolkata", "44": "Chennai", "80": "Bengaluru", "40": "Hyderabad",
       "20": "Pune", "79": "Ahmedabad", "161": "Ludhiana", "172": "Chandigarh", "183": "Amritsar", "181": "Jalandhar",
       "175": "Patiala", "164": "Bathinda", "522": "Lucknow", "512": "Kanpur", "141": "Jaipur", "291": "Jodhpur",
       "135": "Dehradun", "177": "Shimla", "191": "Jammu", "194": "Srinagar", "124": "Gurugram", "129": "Faridabad",
       "120": "Noida", "121": "Meerut", "562": "Agra", "542": "Varanasi", "532": "Prayagraj", "657": "Jamshedpur",
       "612": "Patna", "651": "Ranchi", "674": "Bhubaneswar", "361": "Guwahati", "484": "Kochi", "471": "Thiruvananthapuram",
       "495": "Kozhikode", "422": "Coimbatore", "452": "Madurai", "712": "Nagpur", "731": "Indore", "755": "Bhopal",
       "261": "Surat", "265": "Vadodara", "281": "Rajkot", "832": "Goa", "824": "Mangaluru", "821": "Mysuru"}
_RA_LICENCE = re.compile(r"\bB\s?-?\s?\d{2,5}\s?/\s?[A-Z]{2,6}\s?/\s?(?:PER|PART|COM|PROP|CO|FIRM|OTH)[A-Z]*\b|"
                         r"\bRA\s*(?:licen[cs]e|reg(?:istration)?)\s*(?:no\.?|number)?\s*[:\-]?\s*B\s?-?\s?\d{2,5}",
                         re.IGNORECASE)
_MEA_WORDS = re.compile(r"\bmea\b|ministry of external affairs|e-?migrate|protector general of emigrants|\bpge\b|"
                        r"recruiting agent licen[cs]e|emigration act", re.IGNORECASE)


def _states_for(spec: QuerySpec, command: str) -> List[str]:
    text = " ".join(spec.origin + spec.destination + [spec.summary, command]).lower()
    out = []
    for region, states in REGION_STATES.items():
        if region in text or region.replace(" india", "ern india") in text:
            out += states
    for st in sum(REGION_STATES.values(), []):
        if re.search(r"\b" + re.escape(st.lower()) + r"\b", " ".join(spec.origin).lower()):
            out.append(st)
    return list(dict.fromkeys(out))


def targets(spec: QuerySpec, command: str) -> List[str]:
    """Places a lead must be in (empty = no location requirement)."""
    places = list(spec.places) + _states_for(spec, command)
    for p in spec.origin:
        if p.lower() not in planner.REGIONS and p.lower() not in ("india",):
            places.append(p)
    return list(dict.fromkeys(p for p in places if p))


def phone_city(phone: Optional[str]) -> Optional[str]:
    d = re.sub(r"\D", "", phone or "")
    if d.startswith("91") and len(d) == 12:
        d = d[2:]
    elif d.startswith("0"):
        d = d[1:]
    if len(d) != 10 or d[0] in "6789":            # mobiles carry no location
        return None
    for n in (3, 2):
        if d[:n] in STD:
            return STD[d[:n]]
    return None


def location_check(text: str, phone: Optional[str], places: List[str]) -> Tuple[str, str]:
    """('ok' | 'mismatch' | 'unknown', reason)."""
    if not places:
        return "ok", ""
    own, text = (text or "").split(" ||| ", 1) if " ||| " in (text or "") else ("", text or "")
    # the business's own name / address / city first: an interview "in Delhi" mentioned in a post does not
    # move a Jamshedpur agency or a Dubai company into North India
    if own.strip():
        o = own.lower()
        if any(re.search(r"\b" + re.escape(p.lower()) + r"\b", o) for p in places):
            return "ok", "in " + next(p for p in places if re.search(r"\b" + re.escape(p.lower()) + r"\b", o))
        abroad = [k for k, rx in rule_extractor._TARGET_RE.items() if rx.search(own) and k not in ("Europe", "Gulf")]
        if abroad:
            return "mismatch", f"based in {abroad[0]} (its own name / address), not in the target region"
        other = rule_extractor._INDIA_RE.search(own)
        if other:
            return "mismatch", f"based in {other.group(0)}, outside {', '.join(places[:3])}…"
    low = text.lower()
    hit = next((p for p in places if re.search(r"\b" + re.escape(p.lower()) + r"\b", low)), None)
    if hit:
        return "ok", f"in {hit}"
    abroad = [k for k, rx in rule_extractor._TARGET_RE.items() if rx.search(text or "") and k not in ("Europe", "Gulf")]
    india = rule_extractor._INDIA_RE.search(text or "") or re.search(r"\bindia\b", low)
    if abroad and not india:
        return "mismatch", f"based in {abroad[0]}, not in the target region"
    other = rule_extractor._INDIA_RE.search(text or "")
    if other:
        return "mismatch", f"based in {other.group(0)}, outside {', '.join(places[:3])}…"
    city = phone_city(phone)
    if city:
        return ("ok", f"phone area code of {city}") if any(city.lower() == p.lower() for p in places) else \
            ("mismatch", f"phone area code is {city}, outside the target region")
    return "unknown", "location not stated"


def mea_evidence(text: str) -> Optional[str]:
    m = _RA_LICENCE.search(text or "")
    return m.group(0) if m else None


class _Verdict(BaseModel):
    i: int
    fits: bool = Field(..., description="Is it the kind of business the command asks for, where it asks?")
    reason: str = ""
    requirements_met: List[str] = Field(default_factory=list)
    requirements_unknown: List[str] = Field(default_factory=list)
    requirements_failed: List[str] = Field(default_factory=list)


class _Verdicts(BaseModel):
    items: List[_Verdict] = Field(default_factory=list)


SYSTEM = """You check leads found online against a B2B search command. For each numbered business decide:
- fits: true only if it IS the kind of organisation asked for (e.g. an overseas recruitment agency — not an employer,
  construction company, travel agent, visa consultant or job portal, unless the command asks for those) AND it is
  located where the command asks (use the address, city, phone area code, website domain). A company abroad
  (e.g. in Dubai) is not an Indian agency; an office in Mumbai is not in North India.
- for each hard requirement: met (the evidence shows it, e.g. an RA licence number for MEA registration),
  failed (the evidence contradicts it) or unknown (not shown). Never assume a registration that is not shown.
Judge only from the evidence given. Be strict: when the business type is unclear, fits = false."""


async def check(ai, spec: QuerySpec, command: str, leads: List[dict], texts: Dict[str, str],
                strict: bool = False) -> Tuple[List[dict], List[Tuple[dict, str]]]:
    """(kept leads, [(rejected lead, reason)])."""
    places = targets(spec, command)
    needs_mea = any(_MEA_WORDS.search(r) for r in spec.requirements) or bool(_MEA_WORDS.search(command))
    kept, rejected, ask = [], [], []
    for L in leads:
        text = f"{L.get('display_name') or ''} {L.get('origin') or ''} {L.get('address') or ''} ||| " \
               f"{texts.get(L['lead_key'], '')}"
        name = L.get("display_name") or ""
        abroad_name = [k for k, rx in rule_extractor._TARGET_RE.items() if rx.search(name) and k not in ("Europe", "Gulf")]
        if places and abroad_name and not rule_extractor._INDIA_RE.search(name) and \
                not re.search(r"\b(?:overseas|manpower|recruit|placement|consultan|travels?|agency|agencies)\b", name, re.I):
            # "Dubai DUTCO Construction Co. LLC": a company abroad, whatever city its post mentions
            rejected.append((L, f"a company in {abroad_name[0]} (named so), not a business in the target region"))
            continue
        loc, why = location_check(text, L.get("phone"), places)
        if loc == "mismatch":
            rejected.append((L, why))
            continue
        if why:
            L["why"] = L.get("why", []) + [f"✓ Location: {why}"]
        if needs_mea:
            lic = mea_evidence(text)
            L["requirement_evidence"] = {"MEA registration": lic} if lic else {}
            if lic:
                L["why"] = L.get("why", []) + [f"✓ MEA / eMigrate RA licence shown: {lic}"]
        ask.append((L, text, loc))
    if ai is not None and ask:
        for start in range(0, len(ask), 20):
            chunk = ask[start:start + 20]
            lines = []
            for n, (L, text, loc) in enumerate(chunk):
                pc = phone_city(L.get("phone"))
                ev = re.sub(r"\s+", " ", text.replace(" ||| ", " "))[:400]
                lines.append(f"[{n}] {L.get('display_name')} | type: {L.get('profession') or '?'} | "
                             f"place: {L.get('origin') or L.get('address') or '?'}"
                             f"{' | phone area: ' + pc if pc else ''} | website: {L.get('website') or '-'} | "
                             f"evidence: {ev}")
            prompt = (f"Command: {command}\nHard requirements: {'; '.join(spec.requirements) or 'none'}\n"
                      f"Target places: {', '.join(places[:25]) or 'any'}\n\nBusinesses:\n" + "\n".join(lines))
            try:
                res: _Verdicts = await ai.generate_structured(prompt, _Verdicts, system_instruction=SYSTEM,
                                                              temperature=0.0, max_retries=2)
                verdicts = {v.i: v for v in res.items}
            except Exception:
                verdicts = {}
            for n, (L, text, loc) in enumerate(chunk):
                v = verdicts.get(n)
                if v is None:
                    kept.append(L)
                    continue
                if not v.fits or v.requirements_failed:
                    rejected.append((L, v.reason or ("fails: " + ", ".join(v.requirements_failed))))
                    continue
                unknown = [r for r in v.requirements_unknown
                           if not (needs_mea and L.get("requirement_evidence") and _MEA_WORDS.search(r))]
                L["why"] = L.get("why", []) + [f"✓ AI check: {v.reason}"[:300]]
                if unknown:
                    if strict:
                        rejected.append((L, "not shown: " + ", ".join(unknown)))
                        continue
                    L["why"].append("⚠ Unverified: " + ", ".join(unknown))
                    L["lead_score"] = max(0, int(L.get("lead_score", 0)) - 10)
                kept.append(L)
    else:
        for L, text, loc in ask:
            if needs_mea and not L.get("requirement_evidence") and strict:
                rejected.append((L, "MEA registration not shown"))
                continue
            if needs_mea and not L.get("requirement_evidence"):
                L["why"] = L.get("why", []) + ["⚠ Unverified: MEA registration"]
            kept.append(L)
    return kept, rejected


# ---------------------------------------------------------------------------
# Score calibration: HIGH only with evidence for the command's requirements
# ---------------------------------------------------------------------------
_OVERSEAS_CMD = re.compile(r"overseas|abroad|foreign|gulf|emigrat|\bmea\b|external affairs|recruiting agent|\bra\b",
                           re.IGNORECASE)
# large domestic staffing / executive-search brands: not emigration recruiting agents
DOMESTIC_BRANDS = re.compile(r"\b(?:randstad|adecco|michael page|teamlease|quess|manpowergroup|kelly services|"
                             r"ciel hr|abc consultants|hunt partners|korn ferry|egon zehnder|heidrick|spencer stuart|"
                             r"allegis|hays|robert half|persol|naukri|workindia|apna|indeed|monster|foundit|"
                             r"placementindia|shine\.com)\b", re.IGNORECASE)
_OFF_TYPE = re.compile(r"\b(?:it recruit|bpo placement|software|visa agent|visa consult|travel agen|tours|"
                       r"immigration|employment exchange|employment office|unemployment office|training cent|"
                       r"csc|facilitation cent|study visa|education consult)", re.IGNORECASE)
_OVERSEAS_NAME = re.compile(r"overseas|international|intl|abroad|gulf|global|foreign|manpower|emigra|world|"
                            r"expat|middle east|arabia|qatar|dubai|kuwait|oman|saudi", re.IGNORECASE)


def _has_requirement_evidence(L: dict, texts: str) -> bool:
    gm = L.get("gov_match") or {}
    if gm.get("status") == "matched" and re.search(r"mea|emigrat|recruit|\bra\b|register", gm.get("dataset", ""), re.I):
        return True
    return bool(L.get("requirement_evidence")) or bool(mea_evidence(texts)) or \
        bool(re.search(r"(?:approved|licen[cs]ed|registered|certified)\s+(?:by|with|under)\s+(?:the\s+)?"
                       r"(?:mea|ministry of external affairs|government of india|govt\.? of india|poe)", texts, re.I) or
                  re.search(r"(?:\bmea\b|ministry of external affairs|e-?migrate)[^.]{0,40}?"
                            r"(?:approved|licen[cs]ed|registered|certified|licen[cs]e)", texts, re.I))


def calibrate(leads: List[dict], spec: QuerySpec, command: str) -> Tuple[List[dict], List[Tuple[dict, str]]]:
    """Business leads after the fit check and the government-list match:
    * overseas / MEA commands: big domestic staffing brands removed; IT / BPO placement, visa / travel agents,
      employment exchanges without an overseas word in the name scored down (−15);
    * a command with hard requirements: a lead reaches HIGH (80+) only when the evidence shows them (RA licence,
      match on an official MEA list, "approved by MEA" on its own pages); otherwise capped at 79 and marked."""
    from . import scoring
    overseas = bool(_OVERSEAS_CMD.search(command + " " + " ".join(spec.requirements)))
    kept, dropped = [], []
    for L in leads:
        name = L.get("display_name") or ""
        blob = " ".join(str(x) for x in [name, L.get("profession"), " ".join(map(str, L.get("evidence") or [])),
                                         " ".join(map(str, L.get("why") or []))] if x)
        if overseas and DOMESTIC_BRANDS.search(name):
            dropped.append((L, "domestic staffing / job-portal brand, not an emigration recruiting agent"))
            continue
        score = int(L.get("lead_score") or 0)
        if overseas and _OFF_TYPE.search(f"{name} {L.get('profession') or ''}") and not _OVERSEAS_NAME.search(name):
            score -= 15
            L["why"] = (L.get("why") or []) + ["− Business type is not clearly overseas recruitment"]
        if spec.requirements or overseas:
            if _has_requirement_evidence(L, blob):
                L["why"] = (L.get("why") or []) + ["✓ Requirement shown (licence / official list / approval)"]
            elif score >= 80:
                score = 79
                L["why"] = (L.get("why") or []) + ["⚠ Capped below HIGH: no licence number, official-list match or "
                                                   "approval statement found"]
        L["lead_score"] = max(0, score)
        L["tier"] = scoring.tier(L["lead_score"])
        kept.append(L)
    return kept, dropped
