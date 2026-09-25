# Autonomous Candidate Sourcing Agent

Turns an open intent ("Find Indian candidates who are interested for abroad opportunities
in CNC") into a multi-wave search plan, fetches only never-before-seen pages, and stores
structured candidate records in **Neon DB (PostgreSQL)**.

This repo deploys to **Vercel** (a static page + a Python serverless API). The original
Streamlit version with headless-Chromium crawling is in [`legacy-streamlit/`](legacy-streamlit/)
for running on your own machine.

```
browser (public/index.html) drives the run step by step:
  POST /api/plan     Gemini 2.5 Flash: multi-wave dork plan
  POST /api/search   one DuckDuckGo query → canonicalised URLs     (×N, 1–2.5 s jitter)
  POST /api/dedup    drop URLs already in Neon DB `scraped_urls`
  POST /api/process  ≈5 URLs: record → HTTP fetch → Gemini extraction
                     → grounding check → Neon DB `candidates` upsert
  GET  /api/candidates, POST /api/candidates   view / edit stored records
```

Each request finishes well inside Vercel's 60 s function limit. Waves still run strictly one
after another, and Stop takes effect after the current batch.

## 1. Neon DB (PostgreSQL)

The app is pre-configured to connect to Neon DB using `psycopg2-binary` and automatically initializes the `scraped_urls` and `candidates` tables (`schema.sql`) on startup.

## 2. Deploy on Vercel

1. vercel.com → **Add New… → Project** → import this GitHub repo (`Sleepygeniusiitiim/scrapping-tool`). Framework preset: **Other**
   (no build command; `public/` is served as the site, `api/index.py` as the function).
2. **Settings → Environment Variables**, add:

   | Name | Value |
   |---|---|
   | `DATABASE_URL` | `postgresql://neondb_owner:npg_jBms9Rc4oHgD@ep-fancy-dust-b5rdd2ee-pooler.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require` |
   | `APP_PASSWORD` | `CSA-Neon-Vercel-2026!` (also configured as default fallback) |
   | `GEMINI_API_KEY` | AI Studio (`AIza…`) or Vertex express (`AQ.…`) key (can also be entered directly in the UI) |

3. Redeploy so the variables take effect. Open the site, enter the password (`CSA-Neon-Vercel-2026!`), and click **Test connections**.

## Contact details, comments and Scrape.do

- **email / phone columns.** Filled only when the candidate wrote the detail themselves — in their
  own post, profile or comment. Every value must appear verbatim on the page, and on pages with a
  comment thread it must be in that person's own lines (a recruiter's number is never attached to a
  commenter). Invented values are discarded.
- **Comment sections.** LinkedIn posts, Quora answers and many forums embed the post and its comments
  (with authors) as schema.org data; the app reads that first. Logged-out LinkedIn shows about the
  first 10 comments of a post.
- **LinkedIn / Facebook.** Their robots.txt disallows crawlers. Untick "Respect robots.txt" to read
  public posts — check that this fits your use and their terms. These hosts are fetched one page at a
  time with a pause, because parallel requests get a login wall.
- **Scrape.do (optional).** Add `SCRAPEDO_TOKEN` in Vercel → Environment Variables. Pages that come
  back blocked are fetched again through Scrape.do (residential proxies for LinkedIn/Facebook), and
  when DuckDuckGo returns fewer than 3 results, Google results are added via Scrape.do's SERP API.
  Choose "google (Scrape.do)" as the search backend to use Google only. Both use Scrape.do credits.

## AI provider: OpenRouter (or Gemini)

Enter an OpenRouter key (`sk-or-…`) on the home page, or set `OPENROUTER_API_KEY` in Vercel.
The default model is `google/gemini-3.8-flash`; OpenRouter falls back to `openai/gpt-6-luna` and
`deepseek/deepseek-v4.1-flash` if it is unavailable. Choose another model in the "AI model" box
(any OpenRouter model id). Without an OpenRouter key the app uses `GEMINI_API_KEY` as before.
Out of credits (402) or daily limits stop the run cleanly; unread pages are retried next run.

## Assisted outreach (Outreach tab)

1. Fill in your name, agency, the role and location, then **Load candidates** (default filter: no contact yet).
2. Tick candidates → **Draft messages** (AI writes a short, honest message citing the post where they
   showed interest, asking them to share phone/email if interested, with an opt-out line). Edit freely.
3. **Copy & open** copies the message and opens their post/profile — you paste it as a reply or DM.
   The app never sends messages itself: LinkedIn and Facebook forbid automated messaging.
4. When they answer, paste the reply and **Save reply**. The AI reads it; any phone/email that is
   actually in the reply is saved with `contact_source = shared_in_reply` and the date — a record
   that the candidate shared it with you. "Not interested" replies are marked so.

## Local development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt uvicorn python-dotenv
copy .env.example .env         # fill in GEMINI_API_KEY if desired
uvicorn api.index:app --reload --env-file .env
```

Open http://127.0.0.1:8000 — the page and the API are served together locally.

## Files

| File | Purpose |
|---|---|
| `schema.sql` | PostgreSQL tables (`scraped_urls`, `candidates`) + indexes |
| `schema.py` | `CandidateRecord`, `SearchWave`, `ComprehensiveSearchPlan` |
| `supabase_db.py` | Neon PostgreSQL dedup ledger + candidate upserts |
| `search_module.py` | DDGS search with jitter/back-off, URL canonicalisation, platform tagging |
| `gemini_client.py` | Gemini structured output, retries, AI Studio vs Vertex-express key handling |
| `fetcher.py` | HTTP fetch, robots.txt, HTML → text, login-wall detection |
| `pipeline.py` | plan / search / dedup / process-batch steps |
| `api/index.py` | FastAPI app (Vercel function), password & Gemini header check |
| `public/index.html` | UI: run controls, live progress, filters, editing, CSV / JSON export |
| `vercel.json` | function duration + `/api/*` routing |
