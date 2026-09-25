# Autonomous Candidate Sourcing Agent — local Streamlit version

> This is the original full-featured version (headless-Chromium crawling, editable grid).
> It runs on your own machine. The Vercel deployment lives in the repo root.
> Shared modules (`schema.py`, `search_module.py`, `supabase_db.py`, `gemini_client.py`) are in the repo root.

Streamlit app that turns an open intent ("Find Indian candidates who are interested
for abroad opportunities in CNC") into a multi-wave search plan, crawls only
never-before-seen pages, and stores structured candidate records in Supabase.

```
intent ─► Gemini 2.5 Flash: multi-wave dork plan
          Wave 1 LinkedIn · Wave 2 Reddit & Quora · Wave 3 Forums & Job portals (· up to 6)
             │  (waves run strictly one after another)
             ▼
   DuckDuckGo search (1.0–2.5 s jitter per query, 429 cool-down)
   ─► URL canonicalisation (utm_*, fbclid, trk… stripped)
   ─► Supabase `scraped_urls` dedup  (already-seen → skipped, logged)
   ─► record URLs ─► Crawl4AI headless Chromium, 5 pages/batch, per-page timeout
   ─► Gemini structured extraction ─► grounding check (quote must be on the page)
   ─► Supabase `candidates` upsert (per batch — nothing lost on Stop)
   ─► live table in Streamlit, CSV / JSON export
```

## 1. Supabase

1. Create a project at supabase.com.
2. **SQL Editor** → paste `schema.sql` → **Run**.
   `"current_role"` is double-quoted because `CURRENT_ROLE` is a reserved word
   in PostgreSQL. Without the quotes the script fails with a syntax error.
3. **Project Settings → API**: copy the Project URL and the `service_role` key.
   (With the anon key and RLS enabled, also run the optional policies at the
   bottom of `schema.sql`.)

## 2. Install (Windows PowerShell)

```powershell
cd candidate-sourcing-agent\legacy-streamlit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # from inside legacy-streamlit
crawl4ai-setup                 # installs Playwright Chromium for Crawl4AI
# if that fails:  python -m playwright install chromium
copy .env.example ..\.env      # .env goes in the repo root; fill in the three values
```

macOS / Linux: same steps with `source .venv/bin/activate` and `cp`.

Python 3.10–3.12 recommended.

## 3. Run

```powershell
streamlit run app.py
```

Sidebar → **Test connections** first. Then enter the intent, adjust wave depth, **Start**.
**Stop** finishes the current batch and exits; everything already extracted is in Supabase.

## Files

| File | Purpose |
|---|---|
| `schema.sql` | Supabase tables + indexes (+ optional RLS policies) |
| `schema.py` | `CandidateRecord`, `SearchWave`, `ComprehensiveSearchPlan` (+ internal extraction models) |
| `supabase_db.py` | `filter_fresh_urls`, `record_scraped_urls`, `save_candidates`, `fetch_all_candidates` |
| `search_module.py` | DDGS search with jitter/back-off, URL canonicalisation, platform tagging |
| `crawler_module.py` | Crawl4AI batching, per-page timeouts, login-wall detection |
| `gemini_client.py` | Gemini structured output, retries, AI Studio vs Vertex-express key handling |
| `agent_pipeline.py` | Wave orchestration (async generator) + background runner for Streamlit |
| `app.py` | Streamlit UI |

## Design notes

- **URL recording order.** Each batch of fresh URLs is written to `scraped_urls` right
  before that batch is crawled. URLs you never reach because you pressed Stop stay unmarked
  and are picked up next run.
- **Walled pages (LinkedIn, some Facebook).** These usually block logged-out crawlers or
  disallow them in robots.txt. When a page is blocked, the agent can instead extract from
  the search result's title + snippet ("Use search snippet…" toggle). That snippet often
  holds the headline, location and "open to work" text. The URL is still marked as seen.
- **Several people on one page.** `candidates.source_url` is UNIQUE. A Reddit/Quora thread
  can hold several interested people. The first keeps the page URL and the others are
  stored as `page-url#candidate-<name>`, which still opens the same page.
- **Keeping out made-up records.** Gemini must return a verbatim `evidence_snippet`.
  If most of its words aren't on the page, the record is rejected. The
  "Rejected (ungrounded)" column counts these.
- **Gemini key types.** AI Studio keys (`AIza…`) use the Gemini API. Vertex AI express keys
  (`AQ.…`) use Vertex. `auto` picks one from the key's format and falls back to the other
  if the key is refused.
- **Search backend.** `ddgs` is the renamed `duckduckgo-search` package. `auto` rotates
  across engines inside the library, so you hit fewer rate limits. `duckduckgo` forces DDG
  only.

## Compliance

You are collecting personal data about people. Use only publicly visible information.
Keep "Respect robots.txt" on. Note that LinkedIn's User Agreement prohibits automated
scraping. Handle stored records in line with India's DPDP Act 2023: have a purpose,
keep only what you need, and delete on request. Before you contact anyone, check each
lead manually.
