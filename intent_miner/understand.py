"""Query understanding: the user's command → a structured QuerySpec (one LLM call)."""

from __future__ import annotations

import re
from typing import List, Optional

from . import planner
from .models import INTENT_TYPES, QuerySpec, SourcedQuery, concrete_places

SYSTEM = f"""You turn a recruiter's / sales person's natural-language command into a structured search plan
for finding PEOPLE who publicly show that intent in online discussions (Reddit, Quora, forums, public
posts, search-indexed pages).

Return JSON with:
- summary: one sentence restating the request.
- target: "organizations" when the user wants businesses / institutes / schools / training centres /
  agencies / companies or their owners, directors or contact persons (B2B leads: find WHO they are and
  their public business contact details) — INCLUDING employers and their HR managers, recruiters, talent
  acquisition heads, CEOs / directors / top management who are HIRING (e.g. "HRs of foreign companies hiring
  Indian candidates" = organizations: the employers, with their decision makers). "people" only when the user
  wants individuals who show a need or interest for themselves (job seekers, buyers, students, …).
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
    * "youtube": plain keywords for YouTube video search (no site:) — recruitment / "jobs abroad" / visa /
      how-to-apply videos whose comment sections are full of "interested" viewers, e.g. "Germany nursing
      jobs for Indian nurses", "Dubai driver vacancy apply"
    * "blogs": blog posts / articles on ANY website whose comment sections hold readers asking for the job or
      sharing their number — recruitment-agency and consultancy blogs (inurl:blog …), job-news sites, and
      blog platforms (site:blogspot.com, site:wordpress.com, site:medium.com). E.g. inurl:blog "nursing jobs
      in Germany" apply, "Dubai driver vacancy" "leave a reply", site:blogspot.com gulf jobs interested
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
  EMPLOYERS WHO ARE HIRING (a recruitment agency looking for client companies): professions = the employer
  types / industries and the roles they hire; high_intent_terms = hiring phrases that name the candidates'
  country, e.g. "hiring from India", "Indian candidates", "Indian nationals", "recruitment from India",
  "manpower from India", "Indian workers", "we are hiring", "urgent requirement", "HR manager", "talent
  acquisition"; negative_terms add "freelancer", "upwork", "fiverr", "bid", "proposal", "course", "training";
  queries = hiring posts and job ads by the employers themselves: site:linkedin.com/posts "hiring" "from India"
  <country>, site:linkedin.com/posts "Indian candidates" <industry> <country>, site:linkedin.com/in "HR Manager"
  <country> "India", site:facebook.com "hiring" "from India" <country>, naukrigulf.com / bayt.com /
  gulftalent.com / indeed <country> "Indian", "recruitment agency in India" <industry> <country> — never
  freelance marketplaces.
- source_plan: rank EVERY source id of the catalog below by how likely it gives what the user wants for THIS
  command (weight 0-100, one-line reason naming what it yields, e.g. "phone numbers of each driving school").
  Businesses in named places → Google Maps listings and directories first; individuals showing intent →
  comment sections (Facebook / LinkedIn / YouTube / blogs) and forums first.
- requirements: hard conditions EVERY result must meet that public information can confirm, in plain words —
  licences / registrations ("registered with the Ministry of External Affairs (MEA) as a Recruiting Agent — eMigrate
  RA licence number"), the kind of business ("an overseas recruitment agency, not an employer or a travel agent"),
  the location ("based in North India"). Empty when the command has none.
  For MEA / eMigrate / "registered recruiting agent" commands add queries for site:emigrate.gov.in and for
  "RA licence" / "registration no" pages of the agencies.
- places: when the command names a region ("North India", "Punjab", "Gulf"), list its main cities / districts
  (up to 25) to search one by one; for named cities, those cities.
  Query sources may also be "maps" (plain "<business type> in <city>", no site:) and "directories"
  (site:justdial.com / site:indiamart.com / site:sulekha.com / site:olx.in + business type + city).

SOURCE CATALOG
""" + planner.catalog_text()


def prompt(command: str, sources: List[str], max_age_days: Optional[int], num_queries: int = 16,
           exclude: Optional[List[str]] = None, auto: bool = False) -> str:
    lines = [f"Command: {command.strip()}", ""]
    if auto:
        lines += ["The user lets you choose the sources: rank them in source_plan and generate queries for the "
                  "best ones (weight 50+).",
                  f"Generate about {num_queries} queries in total, most of them for the top-ranked sources."]
    else:
        lines += [f"Sources the user enabled: {', '.join(sources) or 'all'} — only generate queries for these "
                  "(still rank all sources in source_plan).",
                  f"Generate about {num_queries} queries in total, spread across those sources."]
    if max_age_days:
        lines.append(f"The user wants discussions from the last {max_age_days} days.")
    if exclude:
        lines.append("These queries were already run — do not repeat them or near-copies; use other synonyms, "
                     "places and phrasings:")
        lines += [f"- {q}" for q in exclude[-120:]]
    return "\n".join(lines)


_HIRING_SIDE = re.compile(
    r"\b(?:hrs?|human resources?|recruiters?|talent acquisition|hiring managers?|top management|management|"
    r"ceos?|cxos?|directors?|founders?|owners?|decision[- ]makers?|employers?|compan(?:y|ies)|firms?|"
    r"organi[sz]ations?|businesses)\b", re.IGNORECASE)
_WANTS_TO_HIRE = re.compile(
    r"\b(?:hiring|recruit(?:ing|ment)?|interested (?:in|for) hiring|looking (?:for|to hire)|want(?:s)? to hire|"
    r"need(?:s)? (?:\w+ ){0,2}(?:candidates|workers|staff|manpower))\b", re.IGNORECASE)
_JOB_SEEKER = re.compile(r"\b(?:job ?seekers?|looking for (?:a )?jobs?|want(?:s)? (?:a )?jobs?|"
                         r"interested (?:in|for) (?:jobs?|work|abroad opportunit\w*))\b", re.IGNORECASE)


def hiring_side(command: str) -> bool:
    """The command asks for the employers / HR / management who hire — not the candidates."""
    return bool(_HIRING_SIDE.search(command or "") and _WANTS_TO_HIRE.search(command or "")
                and not _JOB_SEEKER.search(command or ""))


async def understand(ai, command: str, sources: List[str], max_age_days: Optional[int], num_queries: int = 16,
                     exclude: Optional[List[str]] = None, auto_sources: bool = False,
                     target: str = "") -> QuerySpec:
    """target: "people" / "organizations" chosen on the page ("Looking for"), or "" to let the AI decide."""
    ask = prompt(command, sources, max_age_days, num_queries, exclude, auto_sources)
    if target == "organizations":
        ask += ("\n\nThe user chose: LOOKING FOR ORGANIZATIONS (employers / businesses and their decision makers, "
                "HR, management) — target must be \"organizations\"; plan queries for them, not for job seekers.")
    elif target == "people":
        ask += ("\n\nThe user chose: LOOKING FOR PEOPLE (individual candidates showing interest) — target must be "
                "\"people\"; plan queries where such people write in their own words.")
    spec = await ai.generate_structured(ask,
                                        QuerySpec,
                                        system_instruction=SYSTEM, temperature=0.4, thinking_budget=512,
                                        max_retries=3)
    if max_age_days:
        spec.max_age_days = max_age_days
    if target in ("people", "organizations"):
        spec.target = target                  # the page's choice wins
    elif spec.target != "organizations" and hiring_side(command):
        # "HRs / top management of companies hiring …": the employers are the leads, not job seekers
        spec.target = "organizations"
    # "abroad / overseas" is a wish, not a place: keep it as an intent phrase, not a location filter
    vague = [p for p in spec.destination + spec.origin if p not in concrete_places([p])]
    spec.destination, spec.origin = concrete_places(spec.destination), concrete_places(spec.origin)
    if vague:
        spec.high_intent_terms = list(dict.fromkeys(spec.high_intent_terms + ["abroad", "overseas", "foreign"]))
    allowed = set() if auto_sources else set(sources or [])
    seen, queries = set(), []
    for q in spec.queries:
        src = q.source.strip().lower()
        src = src if src in planner.CATALOG else "search"
        if allowed and src not in allowed:
            continue
        text = re.sub(r"\s+", " ", q.query).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            queries.append(SourcedQuery(source=src, query=text))
    done = {q.strip().lower() for q in exclude or []}
    spec.queries = [q for q in queries if q.query.lower() not in done]
    spec = planner.finalize(spec, command, auto_sources, sources, num_queries)
    spec.queries = [q for q in spec.queries if q.query.lower() not in done]
    spec.subreddits = [re.sub(r"^/?r/", "", s).strip("/ ") for s in spec.subreddits][:8]
    return spec
