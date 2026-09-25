# Autonomous Candidate Sourcing Agent

Turns an open intent ("Find Indian candidates who are interested for abroad opportunities
in CNC") into a multi-wave search plan, fetches only never-before-seen pages, and stores
structured candidate records in Supabase.

This repo deploys to **Vercel** (a static page + a Python serverless API). The original
Streamlit version with headless-Chromium crawling is in [`legacy-streamlit/`](legacy-streamlit/)
for running on your own machine.

```
browser (public/index.html) drives the run step by step:
  POST /api/plan     Gemini 2.5 Flash: multi-wave dork plan
  POST /api/search   one DuckDuckGo query → canonicalised URLs     (×N, 1–2.5 s jitter)
  POST /api/dedup    drop URLs already in Supabase `scraped_urls`
  POST /api/process  ≈5 URLs: record → HTTP fetch → Gemini extraction
                     → grounding check → Supabase `candidates` upsert
  GET  /api/candidates, POST /api/candidates   view / edit stored records
```

Each request finishes well inside Vercel's 60 s function limit. Waves still run strictly one
after another, and Stop takes effect after the current batch.

## 1. Supabase

1. Create a project at supabase.com.
2. **SQL Editor** → paste `schema.sql` → **Run**.
3. **Project Settings → API**: copy the Project URL and the `service_role` key.

## 2. Deploy on Vercel

1. vercel.com → **Add New… → Project** → import this GitHub repo. Framework preset: **Other**
   (no build command; `public/` is served as the site, `api/index.py` as the function).
2. **Settings → Environment Variables**, add:

   | Name | Value |
   |---|---|
   | `GEMINI_API_KEY` | AI Studio (`AIza…`) or Vertex express (`AQ.…`) key |
   | `SUPABASE_URL` | Supabase project URL |
   | `SUPABASE_KEY` | `service_role` key |
   | `APP_PASSWORD` | any long password — the page asks for it; the API refuses everything without it |

3. Redeploy so the variables take effect. Open the site, enter the password, **Test connections**.

The keys stay on the server — the browser only ever sends `APP_PASSWORD`.

## Differences from the local Streamlit version

- **No headless browser.** Vercel functions can't run Chromium, so pages are fetched over
  plain HTTP. Pages that need JavaScript or a login (LinkedIn, most of Facebook) come back as
  "blocked", and the agent extracts from the search result's title + snippet instead
  (the "Use search snippet…" toggle). Reddit, Quora, forums and job portals mostly work.
- **Search from datacenter IPs.** DuckDuckGo rate-limits cloud IPs more often than home
  connections. Rate-limited queries are skipped and logged; the `auto` backend helps.
- **Time limits.** Page timeout is capped at 30 s and batches at 8 URLs so one request fits
  in 60 s.

## Local development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt uvicorn python-dotenv
copy .env.example .env         # fill in the values
uvicorn api.index:app --reload --env-file .env
```

Open http://127.0.0.1:8000 — the page and the API are served together locally.

## Files

| File | Purpose |
|---|---|
| `schema.sql` | Supabase tables + indexes (+ optional RLS policies) |
| `schema.py` | `CandidateRecord`, `SearchWave`, `ComprehensiveSearchPlan` |
| `supabase_db.py` | dedup ledger + candidate upserts |
| `search_module.py` | DDGS search with jitter/back-off, URL canonicalisation, platform tagging |
| `gemini_client.py` | Gemini structured output, retries, AI Studio vs Vertex-express key handling |
| `fetcher.py` | HTTP fetch, robots.txt, HTML → text, login-wall detection |
| `pipeline.py` | plan / search / dedup / process-batch steps |
| `api/index.py` | FastAPI app (Vercel function), password check |
| `public/index.html` | UI: run controls, live progress, filters, editing, CSV / JSON export |
| `vercel.json` | function duration + `/api/*` routing |

## Compliance

You are collecting personal data about people. Use only publicly visible information.
Keep "Respect robots.txt" on. LinkedIn's User Agreement prohibits automated scraping.
Handle stored records in line with India's DPDP Act 2023: have a purpose, keep only what
you need, and delete on request. Before you contact anyone, check each lead manually.
