"""
Company email-format inference (the Hunter.io approach), evidence-based.

1. Learn the format from addresses the company itself publishes on its website:
   - best evidence: a published address next to a named person ("Harpreet Kaur" ↔ harpreet.kaur@…);
   - weaker evidence: the shape of published personal-looking addresses (two name-like parts → first.last).
2. Apply the format to other named people at that company → a GUESSED address, with a confidence level.

Guesses are never mixed with real contacts: they are labelled "guessed", carry the evidence, and are only
made when the company's own published addresses show the format. The domain's MX record is checked (can
receive mail) — that still does not prove the mailbox exists, so send to guessed addresses sparingly.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

FORMATS: Dict[str, str] = {
    "first.last": "{first}.{last}", "firstlast": "{first}{last}", "first_last": "{first}_{last}",
    "first-last": "{first}-{last}", "flast": "{f}{last}", "f.last": "{f}.{last}", "first": "{first}",
    "last": "{last}", "last.first": "{last}.{first}", "lastfirst": "{last}{first}", "firstl": "{first}{l}",
    "first.l": "{first}.{l}", "lastf": "{last}{f}",
}
GENERIC = re.compile(r"^(?:info|contact|admin|admissions?|enquir(?:y|ies)|inquir(?:y|ies)|hr|jobs|careers|sales|"
                     r"support|help|office|hello|mail|team|accounts?|billing|marketing|noreply|no-reply|principal|"
                     r"director|webmaster|reception|booking|bookings|service|services|training|academy|school)\d*$",
                     re.IGNORECASE)
_HONORIFIC = re.compile(r"^(?:mr|mrs|ms|miss|dr|prof|er|shri|smt|sri)\.?$", re.IGNORECASE)


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def name_parts(full_name: str) -> Optional[Tuple[str, str]]:
    """'Mrs. Harpreet Kaur' → ('harpreet', 'kaur'); middle names are ignored."""
    words = [w for w in re.findall(r"[A-Za-zÀ-ÿ]+", full_name or "") if not _HONORIFIC.match(w)]
    words = [_ascii(w) for w in words if len(_ascii(w)) >= 2]
    if len(words) < 2:
        return None
    return words[0], words[-1]


def render(fmt: str, first: str, last: str) -> str:
    return FORMATS[fmt].format(first=first, last=last, f=first[0], l=last[0])


def infer(emails: Iterable[str], domain: str, people: Iterable[dict] = ()) -> Optional[dict]:
    """The company's email format from its published addresses. Returns
    {format, confidence: high|medium|low, evidence: [...]} or None when there is no evidence."""
    domain = domain.lower().removeprefix("www.")
    locals_ = [e.split("@")[0].lower() for e in emails
               if e.lower().endswith("@" + domain) and not GENERIC.match(e.split("@")[0])]
    if not locals_:
        return None
    votes: Counter = Counter()
    evidence: List[str] = []
    for p in people:
        parts = name_parts(p.get("name", ""))
        if not parts:
            continue
        for loc in locals_:
            for fmt in FORMATS:
                if render(fmt, *parts) == loc:
                    votes[fmt] += 2
                    evidence.append(f"{p['name']} ↔ {loc}@{domain}")
    if votes:
        fmt, score = votes.most_common(1)[0]
        return {"format": fmt, "confidence": "high" if score >= 4 else "medium", "evidence": evidence[:4]}
    # No name pairs: judge by shape of personal-looking local parts.
    shapes: Counter = Counter()
    for loc in locals_:
        if re.fullmatch(r"[a-z]{2,}\.[a-z]{2,}", loc):
            shapes["first.last"] += 1
        elif re.fullmatch(r"[a-z]{2,}_[a-z]{2,}", loc):
            shapes["first_last"] += 1
        elif re.fullmatch(r"[a-z]\.[a-z]{2,}", loc):
            shapes["f.last"] += 1
    if not shapes:
        return None
    fmt, n = shapes.most_common(1)[0]
    return {"format": fmt, "confidence": "low" if n == 1 else "medium",
            "evidence": [f"{n} published address(es) shaped like {fmt}@{domain}"]}


def guess(full_name: str, domain: str, pattern: Optional[dict]) -> Optional[dict]:
    parts = name_parts(full_name)
    if not parts or not pattern:
        return None
    return {"email": f"{render(pattern['format'], *parts)}@{domain.lower().removeprefix('www.')}",
            "status": "guessed", "format": pattern["format"], "confidence": pattern["confidence"],
            "evidence": pattern["evidence"]}
