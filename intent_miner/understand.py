"""Query understanding: the user's command → a structured QuerySpec (one LLM call)."""

from __future__ import annotations

import re
from typing import List, Optional

from .models import INTENT_TYPES, QuerySpec, SourcedQuery

SYSTEM = f"""You turn a recruiter's / sales person's natural-language command into a structured search plan
for finding PEOPLE who publicly show that intent in online discussions (Reddit, Quora, forums, public
posts, search-indexed pages).

Return JSON with:
- summary: one sentence restating the request.
- intent_type: one of {", ".join(INTENT_TYPES)}.
- industry, professions (job titles / products / services plus real-world synonyms, local-language forms
  and qualifications, e.g. nursing → "staff nurse", "GNM", "BSc Nursing", "Pflegefachkraft").
- origin: where the people are now (only if stated). destination: where they want to work / move / buy.
- high_intent_terms: 10–20 short phrases people use when they have this intent ("looking for", "want to
  work in", "how can I apply", "planning to move", local-language equivalents).
- negative_terms: words that mark irrelevant pages (news, salary statistics, exam prep, job ads …).
- languages: ISO codes of languages the discussions are likely in.
- timeline_months: the planning horizon if the command states one (e.g. "within 12 months" → 12), else null.
- max_age_days: how recent discussions must be if stated ("last 30 days" → 30), else null.
- subreddits: 3–8 real subreddit names where such people post (no "r/").
- queries: 12–20 SHORT search queries (3–8 terms), each with a source:
    * "search": open web, e.g. "CNC operator" Germany "from India"
    * "reddit": plain keywords for Reddit search (no site: operator)
    * "quora": site:quora.com …
    * "forums": inurl:forum / inurl:thread / known forums for this field …
    * "linkedin": site:linkedin.com/posts …   * "facebook": site:facebook.com …
  At most one quoted phrase per query; use the professions, places and intent terms; vary them.
"""


def prompt(command: str, sources: List[str], max_age_days: Optional[int]) -> str:
    lines = [f"Command: {command.strip()}", "",
             f"Sources the user enabled: {', '.join(sources) or 'all'} — only generate queries for these."]
    if max_age_days:
        lines.append(f"The user wants discussions from the last {max_age_days} days.")
    return "\n".join(lines)


async def understand(ai, command: str, sources: List[str], max_age_days: Optional[int]) -> QuerySpec:
    spec = await ai.generate_structured(prompt(command, sources, max_age_days), QuerySpec,
                                        system_instruction=SYSTEM, temperature=0.4, thinking_budget=512,
                                        max_retries=3)
    if max_age_days:
        spec.max_age_days = max_age_days
    allowed = set(sources or [])
    seen, queries = set(), []
    for q in spec.queries:
        src = q.source.strip().lower()
        src = src if src in ("search", "reddit", "quora", "forums", "linkedin", "facebook") else "search"
        if allowed and src not in allowed:
            continue
        text = re.sub(r"\s+", " ", q.query).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            queries.append(SourcedQuery(source=src, query=text))
    spec.queries = queries[:30]
    spec.subreddits = [re.sub(r"^/?r/", "", s).strip("/ ") for s in spec.subreddits][:8]
    return spec
