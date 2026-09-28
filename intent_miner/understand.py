"""Query understanding: the user's command → a structured QuerySpec (one LLM call)."""

from __future__ import annotations

import re
from typing import List, Optional

from .models import INTENT_TYPES, QuerySpec, SourcedQuery, concrete_places

SYSTEM = f"""You turn a recruiter's / sales person's natural-language command into a structured search plan
for finding PEOPLE who publicly show that intent in online discussions (Reddit, Quora, forums, public
posts, search-indexed pages).

Return JSON with:
- summary: one sentence restating the request.
- target: "organizations" when the user wants businesses / institutes / schools / training centres /
  agencies / companies or their owners, directors or contact persons (B2B leads: find WHO they are and
  their public business contact details); "people" when the user wants individuals who show a need or
  interest (job seekers, buyers, students, …).
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
  origin / destination must be real countries, regions or cities — if the command only says "abroad" or
  "overseas", leave destination empty (put "abroad" in high_intent_terms instead).
  For target "people", most "search" queries should find pages where such people write in their own
  words, ideally with contact details: comments under recruiter / agency posts on instagram.com,
  facebook.com and linkedin.com/posts ("interested", "how to apply", phone numbers), "jobs wanted" /
  "looking for job" classifieds, job-seeker forums and Q&A threads — not articles or guides ABOUT the topic.
  For target "organizations": professions = the kinds of organization and their services (e.g. "driving
  school", "HMV driver training institute", "commercial vehicle training centre"); high_intent_terms =
  words that mark a real business page ("contact us", "call", "address", "admission", "enquiry", "fees",
  "courses", "owner", "director", "founder"); queries target business directories and listings
  (justdial.com, indiamart.com, sulekha.com, tradeindia.com, yellow pages, Google-indexed business sites,
  "contact us" pages, LinkedIn company / founder posts) in the places named.
"""


def prompt(command: str, sources: List[str], max_age_days: Optional[int], num_queries: int = 16,
           exclude: Optional[List[str]] = None) -> str:
    lines = [f"Command: {command.strip()}", "",
             f"Sources the user enabled: {', '.join(sources) or 'all'} — only generate queries for these.",
             f"Generate about {num_queries} queries in total, spread across those sources."]
    if max_age_days:
        lines.append(f"The user wants discussions from the last {max_age_days} days.")
    if exclude:
        lines.append("These queries were already run — do not repeat them or near-copies; use other synonyms, "
                     "places and phrasings:")
        lines += [f"- {q}" for q in exclude[-120:]]
    return "\n".join(lines)


async def understand(ai, command: str, sources: List[str], max_age_days: Optional[int], num_queries: int = 16,
                     exclude: Optional[List[str]] = None) -> QuerySpec:
    spec = await ai.generate_structured(prompt(command, sources, max_age_days, num_queries, exclude), QuerySpec,
                                        system_instruction=SYSTEM, temperature=0.4, thinking_budget=512,
                                        max_retries=3)
    if max_age_days:
        spec.max_age_days = max_age_days
    # "abroad / overseas" is a wish, not a place: keep it as an intent phrase, not a location filter
    vague = [p for p in spec.destination + spec.origin if p not in concrete_places([p])]
    spec.destination, spec.origin = concrete_places(spec.destination), concrete_places(spec.origin)
    if vague:
        spec.high_intent_terms = list(dict.fromkeys(spec.high_intent_terms + ["abroad", "overseas", "foreign"]))
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
    done = {q.strip().lower() for q in exclude or []}
    spec.queries = [q for q in queries if q.query.lower() not in done][:max(num_queries, 4) + 4]
    spec.subreddits = [re.sub(r"^/?r/", "", s).strip("/ ") for s in spec.subreddits][:8]
    return spec
