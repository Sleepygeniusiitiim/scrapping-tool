"""Stage 3: LLM intent classification of the shortlisted units of one document (structured output)."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .models import INTENT_TYPES, DocumentIntent, QuerySpec, UnitIntent

SYSTEM = f"""You classify the intent of people in ONE online discussion for a lead-intelligence tool.
For each numbered unit (a post, comment or answer by one author) decide whether the AUTHOR matches the
request and shows the requested intent. Judge only what the author themselves writes.

Return {{"units": [...]}} with one object per unit you were given:
- unit: its number.  relevant: does the author match the request (role / product, place, intent)?
- intent_type: one of {", ".join(INTENT_TYPES)}. A recruiter, agency or company advertising a job is
  "recruitment", not "job_search". Someone only giving advice is "informational".
- intent_strength 0–100: how clearly the author wants this for themselves, now.
- explicit_need: true only if the author states their own need / plan ("I am looking for…", "I want to…").
- buying_stage (awareness / research / consideration / evaluation / ready / unknown), urgency.
- profession, origin (where they are), destination (where they want to go), timeline, budget — only if stated.
- confidence 0–1 in your classification.
- evidence: 1–3 VERBATIM quotes from that unit that prove the intent. Never paraphrase.
Contact details are replaced by tags like [PHONE_1]; ignore them.
"""


ORG_SYSTEM = """You qualify BUSINESS / ORGANIZATION leads for a B2B sales tool from ONE web page.
For each numbered unit (a listing entry, a profile, a page section or a post) decide whether it describes
an ORGANIZATION (business, institute, school, training centre, agency, company — or its owner / director
speaking for it) that matches the request: its type of business / services and its location.

Return {"units": [...]} with one object per unit you were given:
- unit: its number. relevant: true only if the unit is about a real organization matching the request.
  News, job ads for drivers, government notices, blog advice and directories' own boilerplate are false.
- organization: the organization's name as written. org_type: what kind of organization it is.
- city: its city / area if stated. intent_type: "vendor_search" for a matching business, "partnership"
  if they invite partners / franchise / tie-ups, otherwise "irrelevant".
- intent_strength 0–100: how well it matches the request. explicit_need: true if the unit shows the
  organization is actively operating / enrolling / seeking partners (admissions open, call now, enquire).
- confidence 0–1. evidence: 1–3 VERBATIM quotes proving the match. Never paraphrase.
Contact details are replaced by tags like [PHONE_1]; ignore them.
"""


def build_prompt(spec: QuerySpec, doc_title: str, units: List[Tuple[int, str, str, str]]) -> str:
    lines = [f"Request: {spec.summary or ', '.join(spec.professions)}",
             f"Intent wanted: {spec.intent_type}; role/product: {', '.join(spec.professions[:8])}; "
             f"origin: {', '.join(spec.origin) or 'any'}; destination: {', '.join(spec.destination) or 'any'}"
             + (f"; timeline ≤ {spec.timeline_months} months" if spec.timeline_months else ""),
             f"Discussion title: {doc_title[:200]}", ""]
    for i, kind, author, text in units:
        lines.append(f"[{i}] {kind} by {author or 'unknown'}: {text[:1500]}")
    return "\n".join(lines)


def _grounded(quote: str, text: str) -> bool:
    words = re.findall(r"\w+", quote.lower())
    if not words:
        return False
    page = set(re.findall(r"\w+", text.lower()))
    return sum(w in page for w in words) / len(words) >= 0.8


async def classify(ai, spec: QuerySpec, doc_title: str, units: List[Tuple[int, str, str, str]]
                   ) -> Dict[int, UnitIntent]:
    if not units:
        return {}
    res: DocumentIntent = await ai.generate_structured(
        build_prompt(spec, doc_title, units), DocumentIntent,
        system_instruction=ORG_SYSTEM if spec.target == "organizations" else SYSTEM, temperature=0.1,
        max_retries=2)
    texts = {i: t for i, _, _, t in units}
    out: Dict[int, UnitIntent] = {}
    for r in res.units:
        if r.unit not in texts:
            continue
        r.evidence = [q.strip()[:300] for q in r.evidence if q and _grounded(q, texts[r.unit])][:3]
        out[r.unit] = r
    return out
